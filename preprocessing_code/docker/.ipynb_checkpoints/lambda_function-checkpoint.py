# File that is deployed to FarGate for Preprocessing input and calling endpoint


import librosa
librosa.cache.enable = False

import numpy as np
import torch
import boto3
import base64
import io
import os
import json
from urllib.parse import unquote_plus


print("Initializing clients and loading VAD model...")
s3_client = boto3.client("s3")
sagemaker_runtime = boto3.client("sagemaker-runtime")


def compute_melspec(audio, sr, n_mels, fmin, fmax):
    """Compute a mel-spectrogram."""
    melspec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, hop_length=1024
    )
    melspec = librosa.power_to_db(melspec)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """Convert mono audio to color image format"""
    X = X.astype(np.float32)
    mean = mean if mean is not None else X.mean()
    std = std if std is not None else X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
    else:
        V = np.zeros_like(X)
    V = V.astype(np.uint8)
    return np.stack([V, V, V], axis=2)


def crop_or_pad(audio, length, sr):
    """Crop or pad an audio sample to a fixed length."""
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    elif len(audio) > length:
        audio = audio[:length]
    return audio


# Preprocess the data, and call endpoint
def process_and_invoke(audio_buffer, sagemaker_endpoint_name):
    """
    Processes a single audio file from S3, invokes SageMaker endpoint, and deletes original file on success.
    """

    try:
        os.makedirs(os.environ.get("TORCH_HOME", "/tmp/torch_cache"), exist_ok=True)
        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (get_speech_timestamps, _, read_audio, _, _) = utils
        print("VAD model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading VAD model: {e}")
        raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    try:
        # --- Download and Initial Checks ---
        audio_buffer.seek(0)

        # --- VAD Processing ---
        # Silero VAD expects 16kHz. Load/resample audio specifically for VAD.
        print("Loading audio for VAD...")
        try:
            y_vad_loader, sr_vad_loader = librosa.load(
                audio_buffer, sr=16000
            )  # Load directly at 16kHz
            wav_vad = torch.from_numpy(y_vad_loader)
            print(f"Audio loaded for VAD. Shape: {wav_vad.shape}")
        except Exception as load_err:
            print(
                f"Error using librosa for VAD loading: {load_err}. Trying torchaudio.read_audio..."
            )
            audio_buffer.seek(0)  # Reset buffer
            try:
                wav_vad = read_audio(
                    audio_buffer, sampling_rate=16000
                )  # Use torchaudio utility
                print(f"Audio loaded via read_audio for VAD. Shape: {wav_vad.shape}")
            except Exception as read_audio_err:
                raise

        print("Getting speech timestamps...")
        speech_timestamps = get_speech_timestamps(
            wav_vad, model, return_seconds=True, threshold=0.4, sampling_rate=16000
        )
        print(f"Found {len(speech_timestamps)} speech segments.")

        # --- Load Audio for Main Processing ---
        print("Loading audio for main processing...")
        audio_buffer.seek(0)  # Reset buffer again
        y, sr = librosa.load(audio_buffer, sr=None)
        print(f"Audio loaded for processing. Sample Rate: {sr}, Duration: {len(y)/sr:.2f}s")

        # --- Apply VAD Mask ---
        clean_audio = y.copy()
        if speech_timestamps:
            keep_mask = np.ones(len(y), dtype=bool)
            for segment in speech_timestamps:
                buffer = 0.25
                start_sample = max(0, int((segment["start"] - buffer) * sr))
                end_sample = min(len(y), int((segment["end"] + buffer) * sr))
                if start_sample < end_sample:
                    keep_mask[start_sample:end_sample] = False

            clean_audio = y[keep_mask]
            percent_retained = (len(clean_audio) / len(y)) * 100 if len(y) > 0 else 0
            print(
                f"Audio length after VAD: {len(clean_audio)} samples. Retained: {percent_retained:.2f}%"
            )
        else:
            print("No speech detected or VAD failed, using original audio.")


        # --- Spectrogram Configuration and Resampling ---
        config = {
            "sampling_rate": 32000,
            "duration": 5,
            "fmin": 0,
            "fmax": 16000,  # Set fmax explicitly
            "n_mels": 128,
            "res_type": "kaiser_fast",
        }

        target_sr = config["sampling_rate"]
        audio_length = config["duration"] * target_sr
        step = int(config["duration"] * 0.666 * target_sr)

        if sr != target_sr:
            print(f"Resampling clean audio from {sr}Hz to {target_sr}Hz...")
            clean_audio = librosa.resample(
                clean_audio.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr,
                res_type=config["res_type"],
            )
            sr = target_sr  # Update sr
            print(f"Resampled audio length: {len(clean_audio)}")
        else:
            clean_audio = clean_audio.astype(np.float32)  # Ensure float32

        all_sagemaker_results = []

        # --- Chunking, Spectrogram Generation, and Data Collection ---
        print(f"Processing {len(clean_audio)} samples in chunks...")
        for i in range(0, max(1, len(clean_audio) - audio_length + step), step):
            start = i
            end = start + audio_length
            chunk = clean_audio[start:end].copy()

            if len(chunk) < audio_length:
                if start + step >= len(clean_audio):  # Only pad the very last segment
                    chunk = crop_or_pad(chunk, audio_length, sr)
                else:
                    print(
                        f"Skipping intermediate short chunk at index {i}, length {len(chunk)}"
                    )
                    continue

            if len(chunk) != audio_length:
                continue

            # --- Process the chunk (Spectrogram + Color) ---
            melspec = compute_melspec(
                chunk, sr, config["n_mels"], config["fmin"], config["fmax"]
            )
            # Ensure image is uint8 as per your example that worked
            image = mono_to_color(melspec).astype(np.uint8)

            # --- Serialize chunk ---
            payload_io = io.BytesIO()
            np.save(payload_io, image) 
            payload_io.seek(0)
            raw_bytes = payload_io.read()
            b64_bytes = base64.b64encode(raw_bytes)
            b64_string = b64_bytes.decode("utf-8")

            # Create the JSON payload for the chunk
            data_for_sagemaker = {"array": b64_string}
            payload_json = json.dumps(data_for_sagemaker)
            del payload_io, raw_bytes, b64_bytes, b64_string  # Memory cleanup

            # --- Invoke SageMaker Endpoint for the chunk ---
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=sagemaker_endpoint_name,
                ContentType="application/json",
                Body=payload_json,
            )
            # Add response to all_sagemaker_results. 
            result = json.loads(response["Body"].read().decode())
            all_sagemaker_results.append(result)
        
        return {
            "sagemaker_results": all_sagemaker_results
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "error"
        }


# Entry Point for incoming data
def lambda_handler(event, context):
    print("Lambda handler started.")
    print("Received event:", json.dumps(event))

    # Get the incoming data
    audio_base64 = event.get("audio_base64")

    if not audio_base64:
        raise ValueError("Missing 'audio_base64' in event.")

    # Decode the incoming data back to .ogg, store it in memory
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    # Get SageMaker endpoint name from environment variable
    sagemaker_endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not sagemaker_endpoint:
        print("ERROR: SAGEMAKER_ENDPOINT_NAME environment variable not set.")
        return {
            "statusCode": 500,
            "body": json.dumps("Configuration error: SageMaker endpoint name not set."),
        }

    # Preprocess the data, and call endpoint
    result = process_and_invoke(audio_buffer, sagemaker_endpoint)
    final_status_code = 200

    return {
        "statusCode": final_status_code,
        "body": json.dumps({"data": result}),
    }
