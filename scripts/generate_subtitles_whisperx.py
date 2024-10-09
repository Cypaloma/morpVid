#!/usr/bin/env python

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch
import whisperx
import argparse
import json

# Import the centralized logger
from centralized_logger import logger

# Define the path to the conda environment
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent  # Adjust the relative path as needed
ENV_DIR = PROJECT_ROOT / 'dependencies/envs/whisperx_env'
CONDA_DIR = PROJECT_ROOT / 'dependencies/miniconda'
CONDA_PYTHON = ENV_DIR / 'bin' / 'python'

METADATA_FILE = 'metadata.json'

# Configuration settings
CONFIG = {
    'model': {
        'name': "large-v3",
    },
    'input': {
        'folder': Path("./input").resolve(),
        'video_extensions': ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
    },
    'output': {
        'folder': Path("./output/subtitles").resolve()
    }
}

def create_metadata_file():
    """Create metadata.json if it doesn't exist."""
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({}, f, indent=4)
        logger.info("Created metadata.json file.")

def load_metadata():
    """Load metadata from metadata.json."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            logger.info("Loaded metadata from metadata.json.")
            return json.load(f)
    logger.info("No existing metadata found.")
    return {}

def save_metadata(metadata):
    """Save metadata to metadata.json."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved metadata to metadata.json.")

def get_device():
    """Get the appropriate device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU for inference.")
    return device

def format_time(seconds):
    """Format time in SRT format."""
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def setup_whisperx_model():
    """Load the WhisperX model."""
    logger.info("Loading WhisperX model...")
    device = get_device()
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(CONFIG['model']['name'], device=device, compute_type=compute_type)
    logger.info("WhisperX model loaded successfully.")
    return model

def extract_audio(video_path, output_audio_path):
    """Extract audio from video file using FFmpeg."""
    logger.info(f"Extracting audio from {video_path} to {output_audio_path}")
    command = [
        'ffmpeg', '-y', '-i', str(video_path), '-vn',
        '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
        str(output_audio_path)
    ]
    subprocess.run(command, check=True)
    logger.info(f"Audio extracted to {output_audio_path}")

def transcribe_audio_file(model, audio_path, subtitle_path):
    """Transcribe and align the audio using WhisperX."""
    logger.info(f"Starting transcription for {audio_path}")
    device = get_device()

    try:
        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe audio
        result = model.transcribe(audio, batch_size=16)
        logger.info(f"Transcription completed for {audio_path}")

        # Load alignment model
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        logger.info("Alignment model loaded.")

        # Perform alignment
        result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
        logger.info("Alignment completed.")

        # Write subtitles
        write_subtitles(result_aligned, subtitle_path)
        logger.info(f"Transcription written to {subtitle_path}")
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_path}: {e}")

def write_subtitles(result, subtitle_path):
    """Write the transcribed segments into SRT and ASS format."""
    srt_path = subtitle_path.with_suffix(".srt")

    # Write SRT subtitles
    logger.info(f"Writing SRT subtitles to {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as subtitle_file:
        for idx, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()

            subtitle_file.write(f"{idx + 1}\n")
            subtitle_file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            subtitle_file.write(f"{text}\n\n")

    logger.info("SRT subtitle writing completed.")

    # Convert SRT to ASS format
    ass_path = subtitle_path.with_suffix(".ass")
    logger.info(f"Converting SRT subtitles to ASS format at {ass_path}")
    subprocess.run(['ffmpeg', '-y', '-i', str(srt_path), str(ass_path)], check=True)
    logger.info("Subtitle conversion to ASS format completed.")

def process_videos(video_files):
    """Process each video file for transcription."""
    logger.info("Starting video processing...")
    model = setup_whisperx_model()

    for video_file in tqdm(video_files, desc="Processing videos"):
        logger.info(f"Processing video: {video_file}")
        audio_path = CONFIG['output']['folder'] / f"{video_file.stem}.wav"
        subtitle_path = CONFIG['output']['folder'] / f"{video_file.stem}_subs"

        # Extract audio to WAV file
        extract_audio(video_file, audio_path)

        # Transcribe the audio file
        transcribe_audio_file(model, audio_path, subtitle_path)

        # Update metadata
        metadata = load_metadata()
        video_key = str(video_file.resolve())
        if video_key not in metadata:
            metadata[video_key] = {}

        # Reference the .ass subtitle file
        ass_subtitle_path = subtitle_path.with_suffix('.ass')
        metadata[video_key]['subtitles'] = str(ass_subtitle_path.resolve())
        save_metadata(metadata)

    logger.info("All videos processed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input video file or directory containing video files')
    args = parser.parse_args()

    create_metadata_file()
    CONFIG['output']['folder'].mkdir(parents=True, exist_ok=True)

    # Determine input files
    if args.input:
        input_path = Path(args.input).resolve()
        if input_path.is_file():
            video_files = [input_path]
        elif input_path.is_dir():
            video_files = [f for f in input_path.iterdir() if f.suffix.lower() in CONFIG['input']['video_extensions']]
        else:
            logger.error(f"Input path {input_path} is not valid.")
            return
    else:
        # Use default input folder
        video_files = [f for f in CONFIG['input']['folder'].iterdir()
                    if f.suffix.lower() in CONFIG['input']['video_extensions']]

    if not video_files:
        logger.info(f"No video files found in {CONFIG['input']['folder']}")
        return

    process_videos(video_files)
    logger.info("Main process completed.")

if __name__ == "__main__":
    main()
