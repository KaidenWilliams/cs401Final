# lambda_function.py

import librosa
import numpy as np
import torch
import boto3
import base64
import io
import os  # For environment variables
import json  # For SageMaker payload/response
from urllib.parse import unquote_plus  # For handling S3 keys

# Global Initializations (clients, model) for potential reuse across invocations
print("Initializing clients and loading VAD model...")
s3_client = boto3.client("s3")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# --- Load VAD model ---
# Ensure TORCH_HOME is set in Lambda environment variables (e.g., /tmp/torch_cache)
# for persistent cache across warm starts within the same container instance.
os.makedirs(os.environ.get("TORCH_HOME", "/tmp/torch_cache"), exist_ok=True)
torch.set_num_threads(1)
try:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils
    print("VAD model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading VAD model: {e}")
    # If the model can't load, the function likely can't proceed.
    # You might want to raise an exception here to cause the Lambda invocation to fail clearly.
    raise RuntimeError(f"Failed to load Silero VAD model: {e}")


# --- Helper Functions (compute_melspec, mono_to_color, crop_or_pad) ---
# (These functions remain the same as in your provided script)
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


# --- Core Processing Logic for One File ---
def process_and_invoke(bucket_name, s3_key, sagemaker_endpoint_name, max_size_mb=1.2):
    """
    Processes a single audio file from S3, invokes SageMaker endpoint, and deletes original file on success.
    """
    print(f"Processing s3://{bucket_name}/{s3_key}")
    audio_buffer = None
    processed_chunks_data = (
        []
    )  # List to hold processed numpy arrays (as lists) for SageMaker
    sagemaker_invoked = False
    sagemaker_successful = False
    file_deleted = False

    try:
        # --- Download and Initial Checks ---
        base_filename = os.path.basename(s3_key)
        print(f"Base filename: {base_filename}")

        audio_buffer = BytesIO()
        s3_client.download_fileobj(bucket_name, s3_key, audio_buffer)
        file_size_bytes = audio_buffer.getbuffer().nbytes
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"Downloaded file size: {file_size_mb:.2f} MB")

        if file_size_mb > max_size_mb:
            print(
                f"Skipping {base_filename}: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit"
            )
            # Optionally delete oversized files or move them
            # s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            # print(f"Deleted oversized file: {s3_key}")
            return {"status": "skipped_oversize"}

        audio_buffer.seek(0)

        # --- VAD Processing ---
        # Silero VAD expects 16kHz. Load/resample audio specifically for VAD.
        print("Loading audio for VAD...")
        try:
            # Try loading directly if torchaudio's read_audio handles resampling/format well
            # Note: read_audio might be sensitive to format; librosa is often more robust.
            # Alternative: Load with librosa first to ensure correct format/sr.
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
            # This might fail if the format isn't easily handled by torchaudio's backend without ffmpeg readily available
            try:
                wav_vad = read_audio(
                    audio_buffer, sampling_rate=16000
                )  # Use torchaudio utility
                print(f"Audio loaded via read_audio for VAD. Shape: {wav_vad.shape}")
            except Exception as read_audio_err:
                print(
                    f"FATAL: Both librosa and read_audio failed for VAD preprocessing on {s3_key}: {read_audio_err}"
                )
                raise  # Fail the invocation if audio can't be read for VAD

        print("Getting speech timestamps...")
        speech_timestamps = get_speech_timestamps(
            wav_vad, model, return_seconds=True, threshold=0.4, sampling_rate=16000
        )
        print(f"Found {len(speech_timestamps)} speech segments.")

        # --- Load Audio for Main Processing ---
        print("Loading audio for main processing...")
        audio_buffer.seek(0)  # Reset buffer again
        y, sr = librosa.load(audio_buffer, sr=None)  # Load with original sample rate
        print(
            f"Audio loaded for processing. Sample Rate: {sr}, Duration: {len(y)/sr:.2f}s"
        )

        # --- Apply VAD Mask ---
        clean_audio = y.copy()  # Start with original audio
        if speech_timestamps:
            keep_mask = np.ones(len(y), dtype=bool)
            for segment in speech_timestamps:
                buffer = 0.25  # seconds buffer around speech
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

        if len(clean_audio) == 0:
            print(f"Skipping {base_filename}: No audio left after VAD removal.")
            # Decide whether to delete files with no audio left
            # s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            # print(f"Deleted file with no audio after VAD: {s3_key}")
            return {"status": "skipped_no_audio_after_vad"}

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
                print(
                    f"Warning: Chunk {num_chunks_processed} has unexpected length {len(chunk)}. Skipping."
                )
                continue

            # --- Process the chunk (Spectrogram + Color) ---
            melspec = compute_melspec(
                chunk, sr, config["n_mels"], config["fmin"], config["fmax"]
            )
            # Ensure image is uint8 as per your example that worked
            image = mono_to_color(melspec).astype(np.uint8)
            num_chunks_processed += 1

            # --- Serialize ONE chunk using the required method ---
            payload_io = io.BytesIO()
            np.save(payload_io, image)  # Save numpy array using np.save
            payload_io.seek(0)
            raw_bytes = payload_io.read()
            b64_bytes = base64.b64encode(raw_bytes)  # Encode bytes to base64
            b64_string = b64_bytes.decode("utf-8")  # Decode base64 bytes to string

            # Create the JSON payload for THIS chunk
            data_for_sagemaker = {"array": b64_string}
            payload_json = json.dumps(data_for_sagemaker)
            del payload_io, raw_bytes, b64_bytes, b64_string  # Memory cleanup

            # --- Invoke SageMaker Endpoint FOR THIS CHUNK ---
            print(f"Invoking SageMaker for chunk {num_chunks_processed}...")
            try:
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=sagemaker_endpoint_name,
                    ContentType="application/json",  # Endpoint expects JSON
                    Body=payload_json,
                )
                # Process response if needed
                result = json.loads(response["Body"].read().decode())
                print(
                    f"SageMaker Response (Chunk {num_chunks_processed}): {json.dumps(result)}"
                )
                all_sagemaker_results.append(result)  # Optionally store results

                # Check if your result indicates success/failure for this chunk
                # if not result.get("success_flag"): # Example check
                #     sagemaker_overall_success = False
                #     print(f"SageMaker inference failed for chunk {num_chunks_processed}")

            except Exception as invoke_error:
                print(
                    f"ERROR invoking SageMaker for chunk {num_chunks_processed}: {invoke_error}"
                )
                sagemaker_overall_success = False  # Mark overall process as failed
                # Optional: break the loop if one chunk fails? Or process all chunks?
                # break

            # Aggressive memory cleanup in loop
            del chunk
            del melspec
            del image
            del payload_json
            if num_chunks_processed % 5 == 0:
                gc.collect()  # Collect garbage more frequently

        print(f"Finished processing {num_chunks_processed} chunks.")
        if "clean_audio" in locals():
            del clean_audio
        gc.collect()

        # --- Delete Original S3 File only if ALL chunks succeeded ---
        file_deleted = False
        if num_chunks_processed == 0:
            print(
                "No chunks were processed (e.g., audio too short). Treating as success for cleanup."
            )
            sagemaker_overall_success = True  # Or decide if empty files should be kept

        if sagemaker_overall_success:
            try:
                print(
                    f"All processing successful, deleting original file: s3://{bucket_name}/{s3_key}"
                )
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                file_deleted = True
                print("Original file deleted.")
            except Exception as delete_error:
                print(
                    f"ERROR deleting original file s3://{bucket_name}/{s3_key}: {delete_error}"
                )
                # Log error but don't necessarily fail the function if SM was okay
        else:
            print(
                "SageMaker invocation failed for one or more chunks. Original file NOT deleted."
            )

        return {
            "status": "success" if sagemaker_overall_success else "processing_error",
            "chunks_processed": num_chunks_processed,
            "original_file_deleted": file_deleted,
            # "sagemaker_results": all_sagemaker_results # Optionally return results
        }

    except Exception as e:
        print(f"FATAL Error processing {s3_key}: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "fatal_error", "error": str(e)}


# --- Lambda Handler Entry Point ---
def lambda_handler(event, context):
    print("Lambda handler started.")
    print("Received event:", json.dumps(event))  # Log the incoming event

    # Get SageMaker endpoint name from environment variable
    sagemaker_endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not sagemaker_endpoint:
        print("ERROR: SAGEMAKER_ENDPOINT_NAME environment variable not set.")
        return {
            "statusCode": 500,
            "body": json.dumps("Configuration error: SageMaker endpoint name not set."),
        }

    # Process all records in the event (usually only one for S3 triggers)
    results = []
    for record in event.get("Records", []):
        if "s3" not in record:
            print("WARN: Record does not contain S3 info, skipping.")
            continue

        s3_info = record["s3"]
        bucket = s3_info["bucket"]["name"]
        # Handle potential spaces/special characters in keys
        key = unquote_plus(s3_info["object"]["key"])

        print(f"Processing record for: Bucket={bucket}, Key={key}")

        try:
            result = process_and_invoke(bucket, key, sagemaker_endpoint)
            results.append({"file": f"s3://{bucket}/{key}", **result})
        except Exception as e:
            # Catch errors raised from process_and_invoke or before it
            print(f"FATAL exception for s3://{bucket}/{key} in handler: {e}")
            results.append(
                {
                    "file": f"s3://{bucket}/{key}",
                    "status": "handler_error",
                    "error": str(e),
                }
            )

    # Determine overall status code
    # If any file failed, maybe return 500, otherwise 200
    # Or always return 200 and let caller inspect the body for individual errors
    final_status_code = 200
    print("Lambda processing finished. Results:", json.dumps(results))

    return {
        "statusCode": final_status_code,
        "body": json.dumps({"message": "Processing complete.", "results": results}),
    }
