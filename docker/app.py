# app.py

import os
import io
import json
import base64
import torch
import boto3
import librosa
import soundfile
import numpy as np
from flask import Flask, request, jsonify
from utils import (
    compute_melspec,
    mono_to_color,
    crop_or_pad,
)  # break these into utils.py
from urllib.parse import unquote_plus
import logging
from flask.logging import default_handler

app = Flask(__name__)

# Remove the default Flask handler to avoid duplicate logs
app.logger.removeHandler(default_handler)

# Set up logging to stdout
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)  # or INFO
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)

app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

sagemaker_runtime = boto3.client("sagemaker-runtime")
model, utils = torch.hub.load(
    "snakers4/silero-vad", model="silero_vad", force_reload=False
)
get_speech_timestamps, _, read_audio, _, _ = utils


@app.before_request
def log_request_info():
    print("Received request.")
    app.logger.debug("Headers: %s", request.headers)
    app.logger.debug("Body: %s", request.get_data())


@app.route("/")
def hello():
    return "OK", 200


@app.route("/healthz")
def health():
    return "healthy", 200


@app.route("/process", methods=["POST"])
def process_audio():
    print("Received request to process audio.")
    app.logger.debug("Audio data: %s", request.get_data())
    data = request.get_json()
    app.logger.debug("Parsed data: %s", data)
    audio_base64 = data.get("audio_base64")
    endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME")

    app.logger.debug("Endpoint: %s", endpoint)

    if not audio_base64 or not endpoint:
        app.logger.error(
            "Missing required input (audio_base64) or config (SAGEMAKER_ENDPOINT_NAME)."
        )
        return jsonify({"error": "Missing required input or config."}), 400

    try:
        app.logger.info("Starting audio processing.")
        app.logger.debug("Decoding base64 audio data.")
        audio_bytes = base64.b64decode(audio_base64)
        buffer = io.BytesIO(audio_bytes)
        app.logger.debug("Reading audio data with soundfile.")
        y, sr = soundfile.read(buffer, dtype="float32")
        app.logger.info(f"Audio loaded. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")

        app.logger.debug("Converting audio to tensor for VAD.")
        wav_vad = torch.from_numpy(y)
        app.logger.info("Running Silero VAD to get speech timestamps.")
        timestamps = get_speech_timestamps(
            wav_vad,
            model,
            return_seconds=True,
            sampling_rate=16000,  # Assuming VAD model expects 16kHz
        )
        app.logger.info(f"VAD finished. Found {len(timestamps)} speech segments.")
        app.logger.debug(f"Speech timestamps: {timestamps}")

        buffer.seek(
            0
        )  # Reset buffer if needed elsewhere, though not strictly necessary here
        clean_audio = y.copy()

        if timestamps:
            app.logger.info("Applying VAD timestamps to remove non-speech sections.")
            mask = np.ones(len(y), dtype=bool)
            # Add padding around speech segments before removing silence
            vad_padding_seconds = 0.25
            for ts in timestamps:
                s = max(0, int((ts["start"] - vad_padding_seconds) * sr))
                e = min(len(y), int((ts["end"] + vad_padding_seconds) * sr))
                if s < e:  # Ensure start is before end
                    mask[s:e] = (
                        False  # Mark speech segments (plus padding) to *keep* silence
                    )
            # Invert mask logic: keep speech, remove silence
            # Create a mask where True means keep the sample
            keep_mask = np.zeros(len(y), dtype=bool)
            for ts in timestamps:
                s = max(0, int((ts["start"] - vad_padding_seconds) * sr))
                e = min(len(y), int((ts["end"] + vad_padding_seconds) * sr))
                if s < e:
                    keep_mask[s:e] = True
            clean_audio = y[keep_mask]  # Keep only the parts marked True
            app.logger.info(
                f"Audio cleaned based on VAD. New duration: {len(clean_audio)/sr:.2f}s"
            )
        else:
            app.logger.info("No speech detected by VAD, processing original audio.")

        config = {
            "sampling_rate": 32000,
            "duration": 5,
            "fmin": 0,
            "fmax": 16000,
            "n_mels": 128,
            "res_type": "kaiser_fast",
        }
        app.logger.debug(f"Using config: {config}")

        if sr != config["sampling_rate"]:
            app.logger.info(
                f"Resampling audio from {sr} Hz to {config['sampling_rate']} Hz."
            )
            clean_audio = librosa.resample(
                clean_audio.astype(np.float32),
                orig_sr=sr,
                target_sr=config["sampling_rate"],
                res_type=config["res_type"],
            )
            sr = config["sampling_rate"]  # Update sample rate
            app.logger.info(
                f"Resampling complete. New duration: {len(clean_audio)/sr:.2f}s"
            )
        else:
            app.logger.info("Audio already at target sample rate. Skipping resampling.")
            clean_audio = clean_audio.astype(np.float32)  # Ensure correct dtype

        audio_length = config["duration"] * sr
        step = int(config["duration"] * 0.666 * sr)  # Overlapping window step

        num_chunks = max(1, len(clean_audio) - audio_length + step) // step + (
            1 if (len(clean_audio) - audio_length + step) % step > 0 else 0
        )

        app.logger.info(
            f"Starting chunk processing. Window size: {config['duration']}s, Step: {step/sr:.2f}s, Expected chunks: ~{num_chunks}"
        )
        start = 0
        end = start + audio_length
        chunk = clean_audio[start:end]

        # Handle the last chunk carefully
        if len(chunk) < audio_length:
            # If this is potentially the last segment and it's shorter than needed
            app.logger.debug(
                f"Padding last chunk (length {len(chunk)/sr:.2f}s) to {config['duration']}s."
            )
            chunk = crop_or_pad(chunk, audio_length, sr)  # Pad the last chunk

        app.logger.debug(f"Computing melspectrogram for chunk.")
        melspec = compute_melspec(
            chunk, sr, config["n_mels"], config["fmin"], config["fmax"]
        )
        app.logger.debug(f"Converting melspectrogram to image format for chunk.")
        image = mono_to_color(melspec).astype(np.uint8)  # Ensure uint8 type
        payload_io = io.BytesIO()
        np.save(payload_io, image)  # Save as numpy array
        payload_io.seek(0)

        app.logger.debug(f"Encoding payload for chunk.")
        b64 = base64.b64encode(payload_io.read()).decode("utf-8")
        payload = json.dumps({"array": b64})  # Structure expected by endpoint

        app.logger.info(f"Invoking SageMaker endpoint for chunk.")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=payload,
        )

        app.logger.debug(
            f'SageMaker response status: {response.get("ResponseMetadata", {}).get("HTTPStatusCode")}'
        )
        result = json.loads(response["Body"].read().decode())
        app.logger.debug(f"SageMaker result for chunk: {result}")

        app.logger.info(f"Finished processing chunk.")

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=8080)
