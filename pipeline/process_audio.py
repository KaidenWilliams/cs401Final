# Code for Step 1: Pre-Processing. Turns .ogg files into Spectograms


import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])

import librosa
import numpy as np
import torch
# import soundfile as sf
import boto3
from io import BytesIO
import gc  # For garbage collection
import psutil # For memory usage
import os # For memory usage
import argparse

# Initialize S3 client
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')


input_bucket = 'cs401finalpipelineinput'
output_bucket = 'cs401finalpipelineprocessingdata'
output_prefix = 'data'





# Limit Set for Testing, we don't want to process all 12 GB of audio files, would be way too expensive
TEST_COUNT_LIMIT = 1000

# Load VAD model
torch.set_num_threads(1)
print("Loading VAD model...")
try:
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    print("VAD model loaded successfully.")
except Exception as e:
    print(f"Error loading VAD model: {e}")
    # Handle error appropriately, maybe exit or raise
    raise

# Was used for debugging memory leaks with silero-vad
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return f"{mem_bytes / (1024 * 1024):.2f} mb"

# Get all .ogg files to process
def list_s3_files(bucket, prefix):
    """List files in an S3 bucket with given prefix"""
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    file_list = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                file_list.append(obj["Key"])

    return file_list


def compute_melspec(audio, sr, n_mels, fmin, fmax):
    """Compute a mel-spectrogram."""
    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=1024
    )
    melspec = librosa.power_to_db(melspec)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """Convert mono audio to color image format"""
    # Ensure X is float for calculations
    X = X.astype(np.float32)

    mean = mean if mean is not None else X.mean()
    std = std if std is not None else X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
    else:
        # Handle potential division by zero if range is too small
        V = np.zeros_like(X)

    # Convert to uint8
    V = V.astype(np.uint8)

    # Create RGB channels (stack the same array 3 times)
    return np.stack([V, V, V], axis=2)


def crop_or_pad(audio, length, sr):
    """Crop or pad an audio sample to a fixed length."""
    if len(audio) < length:
        # Pad with zeros
        audio = np.pad(audio, (0, length - len(audio)))
        
    # Handle cropping
    elif len(audio) > length:
        # Crop to length
        audio = audio[:length]
    return audio

def process_single_file(s3_key, max_size_mb=1.2):
    audio_buffer = None
    wav = None
    y = None
    clean_audio = None
    melspec = None
    image = None
    npy_buffer = None
    chunk = None

    try:
        # Extract class from directory structure
        # - directory structure is important, .ogg files are stored in the folder of the class / label they are
        path_parts = s3_key.split('/')
        if len(path_parts) < 2:
             print(f"Skipping invalid key format: {s3_key}")
             return
        parent_dir = path_parts[-2]  # Class name
        base_filename = path_parts[-1]  # Filename
        base_name = base_filename.split('.')[0] if '.' in base_filename else base_filename

        # Download audio file to memory
        audio_buffer = BytesIO()
        s3_client.download_fileobj(input_bucket, s3_key, audio_buffer)
        audio_buffer.seek(0)

        # Check file size
        file_size_mb = len(audio_buffer.getvalue()) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            print(f"Skipping {base_filename}: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit")
            return

        # Use BytesIO for VAD model
        audio_buffer.seek(0)
        wav = read_audio(audio_buffer, sampling_rate=16000)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.4, sampling_rate=16000)

        # Load audio at original sample rate for processing
        audio_buffer.seek(0)
        y, sr = librosa.load(audio_buffer, sr=None)

        clean_audio = y

        # Create clean audio by removing human voice segments
        if speech_timestamps:
            keep_mask = np.ones(len(y), dtype=bool)
            for segment in speech_timestamps:
                buffer = 0.5 # seconds
                start_sample = max(0, int((segment["start"] - buffer) * sr))
                end_sample = min(len(y), int((segment["end"] + buffer) * sr))
                if start_sample < end_sample:
                    keep_mask[start_sample:end_sample] = False
    
            # Apply mask to get clean audio
            clean_audio = y[keep_mask]

        # If it was all human voices, don't do anything with it
        if len(clean_audio) == 0:
            print(f"Skipping {base_filename}: No audio left after VAD removal.")
            return

        # Configuration for spectrogram generation
        config = {
            'sampling_rate': 32000,
            'duration': 5, # seconds
            'fmin': 0,
            'fmax': None,
            'n_mels': 128,
            'res_type': "kaiser_fast"
        }
        target_sr = config['sampling_rate']
        audio_length = config['duration'] * target_sr
        step = int(config['duration'] * 0.666 * target_sr)

        # Resample clean_audio once before chunking
        if sr != target_sr:
            clean_audio = librosa.resample(
                clean_audio.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr,
                res_type=config['res_type']
            )
            sr = target_sr 
            print(f"Resampled clean_audio to {target_sr} Hz")
        else:
             # Ensure clean_audio is float32 for compute_melspec consistency
             clean_audio = clean_audio.astype(np.float32)

        # Iterate through chunks without creating a list of all chunks first
        num_chunks_processed = 0
        for i in range(0, max(1, len(clean_audio) - audio_length + step), step):
            start = i
            end = start + audio_length

            # Extract chunk - this creates a view or copy depending on numpy version/context
            chunk = clean_audio[start:end].copy()

            # Pad the last chunk if necessary
            if len(chunk) < audio_length:
                 # Check if padding is actually needed 
                 if start + step >= len(clean_audio):
                     chunk = crop_or_pad(chunk, audio_length, sr) # Pad
                 else:
                      chunk = crop_or_pad(chunk, audio_length, sr)


            # Ensure chunk has the correct length after potential padding
            if len(chunk) != audio_length:
                 print(f"Warning: Chunk {num_chunks_processed} has unexpected length {len(chunk)} after padding. Skipping.")
                 continue


            # Create spectrogram
            melspec = compute_melspec(
                chunk.astype(np.float32),
                sr, # Use the target_sr here
                config['n_mels'],
                config['fmin'],
                config['fmax'] or sr // 2
            )
            image = mono_to_color(melspec)

            # Save as npy file and upload to S3
            npy_buffer = BytesIO()
            np.save(npy_buffer, image)
            npy_buffer.seek(0)

            # Upload spectograms to S3
            spec_s3_key = f"{output_prefix}/audio_specs/{parent_dir}/{base_name}_chunk_{num_chunks_processed}.npy"
            
            s3_client.upload_fileobj(npy_buffer, output_bucket, spec_s3_key)
            num_chunks_processed += 1



    except Exception as e:
        print(f"Error processing {s3_key}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # try to handle memory management stuff, wasn't really needed at the end of the day
        collected = gc.collect() 



def main():
    # List files from S3
    s3_files = list_s3_files(input_bucket, 'train_audio/')
    # Filter for .ogg or other relevant audio files
    files = sorted([f for f in s3_files if f.endswith(('.ogg', '.wav', '.mp3', '.flac')) and not f.endswith('/')]) # Ensure it's not a directory-like key

    print(f"Found {len(files)} files to process.")

    # List existing output spectrograms
    existing_spec_keys = list_s3_files(output_bucket, 'data/audio_specs/')
    existing_base_names = set()

    for key in existing_spec_keys:
        if key.endswith('.npy'):
            filename = key.split('/')[-1]  # e.g., dog_bark_chunk_0.npy
            base_name = filename.split('_chunk_')[0]  # e.g., dog_bark
            existing_base_names.add(base_name)

    print(f"Found {len(existing_base_names)} files already processed.")

    # Process files one by one
    count = 0
    for s3_key in files:
        audio_filename = s3_key.split('/')[-1]
        base_name = audio_filename.split('.')[0]

        # Skip if already processed
        if base_name in existing_base_names:
            print(f"Skipping {s3_key} (already processed)")
            continue
        count += 1
        print(f"Processing file {count}/{len(files)}: {s3_key}")
        if count > TEST_COUNT_LIMIT:
             print(f"Reached TEST_COUNT_LIMIT of {TEST_COUNT_LIMIT}, stopping.")
             break

        process_single_file(s3_key)

        # Explicitly clean up memory after each file in the main loop as well
        gc.collect()

    print(f"All done, processed {count} files")

if __name__ == "__main__":
    main()