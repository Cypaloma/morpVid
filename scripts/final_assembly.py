import os
import logging
import subprocess
from pathlib import Path
import argparse

# Setup logging
log_folder = Path("./output/logs").resolve()
log_folder.mkdir(parents=True, exist_ok=True)
log_file = log_folder / "final_assembly.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CentralLogger')

def get_video_fps(video_path):
    """Get the frames per second (fps) of a video."""
    cmd = [
        'ffprobe', '-v', '0', '-select_streams', 'v:0',
        '-of', 'csv=p=0', '-show_entries', 'stream=r_frame_rate',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    nums = output.split('/')
    if len(nums) == 2 and int(nums[1]) != 0:
        fps = float(nums[0]) / float(nums[1])
    else:
        fps = float(nums[0])
    logger.info(f"Detected FPS: {fps}")
    return fps

def extract_audio(input_video_path, output_audio_path):
    """Extract and re-encode audio to AAC format."""
    logger.info(f"Extracting audio from {input_video_path} to {output_audio_path}")
    cmd = [
        'ffmpeg', '-i', str(input_video_path), '-vn',
        '-acodec', 'aac', '-b:a', '320k', '-y', str(output_audio_path)
    ]
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully extracted audio to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        raise

def assemble_final_video(input_video_path, frames_dir, output_video_path):
    logger.info(f"Assembling final video for {input_video_path}")

    # Get FPS from input video
    fps = get_video_fps(input_video_path)
    logger.info(f"FPS of input video: {fps}")

    # Extract audio from the original video
    audio_path = Path(output_video_path).with_suffix('.m4a')
    extract_audio(input_video_path, audio_path)

    # Stitch frames and encode video
    try:
        stitch_frames_to_video(frames_dir, output_video_path, fps, audio_path=audio_path)
    except Exception as e:
        logger.error(f"Error during stitching and encoding: {e}")
        return

    logger.info(f"Final assembly completed for {input_video_path}")

def stitch_frames_to_video(frames_dir, output_video_path, fps, audio_path=None):
    """Stitch upscaled frames to a video file using ffmpeg."""
    logger.info(f"Stitching frames from {frames_dir} to {output_video_path}")

    frame_pattern = os.path.join(frames_dir, 'frame%08d.png')

    cmd = [
        'ffmpeg', '-framerate', str(fps), '-i', frame_pattern,
        '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '18',
        '-pix_fmt', 'yuv420p'
    ]

    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', str(audio_path), '-c:a', 'aac', '-b:a', '320k'])
        cmd.append('-shortest')  # Ensure the output duration matches the shortest input
    else:
        cmd.extend(['-an'])  # No audio

    cmd.extend(['-y', str(output_video_path)])  # Overwrite output file if exists

    try:
        logger.info(f"Running command: {' '.join(map(str, cmd))}")
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully reconstructed video: {output_video_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error reconstructing video: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Assemble final video from frames and audio.')
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Path to the original input video file.')
    parser.add_argument('-f', '--frames_dir', type=str, required=True, help='Directory containing the upscaled frames.')
    parser.add_argument('-o', '--output_video', type=str, required=True, help='Path to the output assembled video file.')
    args = parser.parse_args()

    logger.info("Starting final assembly process...")

    assemble_final_video(args.input_video, args.frames_dir, args.output_video)

    logger.info("Final assembly process completed.")

if __name__ == "__main__":
    main()
