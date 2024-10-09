import os
import logging
import subprocess
from pathlib import Path
import json
from datetime import datetime
import sys
from tqdm import tqdm

# Import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Please install colorama for colored CLI output: pip install colorama")
    exit(1)

# Setup logging with date and time stamps in the log file name
log_folder = Path("./output/logs").resolve()
log_folder.mkdir(parents=True, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_folder / f"encode_video_{current_time}.log"

# Set up the logger
logger = logging.getLogger(__name__)

# We will set the logging level later based on user input
logger.setLevel(logging.DEBUG)  # Set to DEBUG initially; will adjust later

# Create file handler which logs even debug messages
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)  # Log everything to file

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Default level; will adjust later

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)

METADATA_FILE = 'metadata.json'

def load_metadata():
    """Load metadata from metadata.json."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def detect_cuda_device():
    """Detect if a CUDA device is available."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_encoder_available(encoder_name):
    """Check if a given encoder is available in FFmpeg."""
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        encoders = result.stdout
        return encoder_name in encoders
    except Exception as e:
        logger.error(f"Error checking for encoder {encoder_name}: {e}")
        return False

def run_shell_command(command):
    """Run a shell command with logging and tqdm progress bar."""
    try:
        logger.info(f"Running command: {' '.join(map(str, command))}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Read stdout and stderr
        stdout_lines = []
        stderr_lines = []

        # Use tqdm for progress bar
        with tqdm(total=100, desc="Processing", unit="%", ncols=80) as pbar:
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output:
                    stdout_lines.append(output)
                    tqdm.write(output.strip())
                if error:
                    stderr_lines.append(error)
                    tqdm.write(error.strip())
                if output == '' and error == '' and process.poll() is not None:
                    break

                pbar.update(0)  # Keeps the progress bar displayed

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(''.join(stderr_lines))
            raise subprocess.CalledProcessError(process.returncode, command)

        # Log the outputs
        logger.debug(''.join(stdout_lines))
        logger.debug(''.join(stderr_lines))

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {' '.join(map(str, command))}")
        logger.error(e)
        raise

def get_video_fps(video_path):
    """Get the frames per second (fps) of a video."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
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

def get_video_resolution(video_path):
    """Get the resolution (width and height) of a video."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json',
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    return width, height

def get_video_bitrate(video_path):
    """Get the video stream bitrate in bps."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    if output == 'N/A' or output == '':
        logger.warning("Video bitrate not available, defaulting to 5000000 bps")
        return 5000000  # Default to 5 Mbps
    bitrate = int(output)
    return bitrate

def get_audio_bitrate(video_path):
    """Get the audio stream bitrate in bps."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    if output == 'N/A' or output == '':
        logger.warning("Audio bitrate not available, defaulting to 128000 bps")
        return 128000  # Default to 128 kbps
    bitrate = int(output)
    return bitrate

def set_encoder_settings(config, input_video):
    """Set encoder settings based on codec, quality preset, resolution, and framerate."""
    video_codec = config['ffmpeg']['video_codec']
    quality_preset = config['ffmpeg']['quality_preset']
    width, height = get_video_resolution(input_video)
    fps = get_video_fps(input_video)
    config['resolution'] = f"{width}x{height}"
    config['fps'] = fps

    # Define hardware and software encoders
    hardware_encoders = {
        'AV1': 'av1_nvenc',
        'h264': 'h264_nvenc',
        'h265': 'hevc_nvenc'
    }

    software_encoders = {
        'AV1': 'libaom-av1',
        'h264': 'libx264',
        'h265': 'libx265'
    }

    # Determine if hardware encoder is available
    encoder = None
    preset = 'medium'
    use_crf = True

    hardware_encoder_name = hardware_encoders.get(video_codec)
    hardware_encoder_available = hardware_encoder_name and is_encoder_available(hardware_encoder_name)

    if hardware_encoder_available:
        encoder = hardware_encoder_name
        logger.info(f"{Fore.GREEN}Hardware encoder {encoder} is available and will be used.")
        preset = 'slow'  # Adjust as needed
        use_crf = False  # For hardware encoders, we might use other quality controls
    else:
        encoder = software_encoders.get(video_codec)
        if not encoder:
            logger.error(f"Unsupported video codec: {video_codec}")
            raise ValueError(f"Unsupported video codec: {video_codec}")
        preset = 'veryslow' if quality_preset == 'High' else 'medium'
        logger.info(f"{Fore.YELLOW}Hardware encoder not available. Using software encoder {encoder} with preset {preset}.")
        use_crf = True  # Software encoders can use CRF

    config['ffmpeg']['encoder'] = encoder
    config['ffmpeg']['preset'] = preset
    config['ffmpeg']['use_crf'] = use_crf

    # Now, set the encoding parameters based on the encoder and codec
    if encoder in hardware_encoders.values():
        # Hardware encoder settings
        if video_codec == 'h265':
            # Use constqp mode for hevc_nvenc
            config['ffmpeg']['rc'] = 'constqp'
            qp_values = {'Low': 28, 'Regular': 23, 'High': 18}
            qp = qp_values[quality_preset]
            config['ffmpeg']['qp'] = qp
            logger.info(f"Using QP {qp} for quality preset {quality_preset}")
        elif video_codec == 'h264':
            # Use constqp mode for h264_nvenc
            config['ffmpeg']['rc'] = 'constqp'
            qp_values = {'Low': 28, 'Regular': 23, 'High': 18}
            qp = qp_values[quality_preset]
            config['ffmpeg']['qp'] = qp
            logger.info(f"Using QP {qp} for quality preset {quality_preset}")
        elif video_codec == 'AV1':
            # Use vbr_hq mode for av1_nvenc
            config['ffmpeg']['rc'] = 'vbr_hq'
            cq_values = {'Low': 35, 'Regular': 28, 'High': 20}
            cq = cq_values[quality_preset]
            config['ffmpeg']['cq'] = cq
            logger.info(f"Using CQ {cq} for quality preset {quality_preset}")

        # Set bitrate for hardware encoders if necessary
        bitrate_settings = {
            '480p': {'Low': 2000, 'Regular': 3000, 'High': 4000},
            '720p': {'Low': 4000, 'Regular': 6000, 'High': 8000},
            '1080p': {'Low': 8000, 'Regular': 12000, 'High': 16000},
            '1440p': {'Low': 16000, 'Regular': 24000, 'High': 32000},
            '2160p': {'Low': 24000, 'Regular': 36000, 'High': 48000}
        }

        # Determine resolution category
        if height <= 480:
            res_name = '480p'
        elif height <= 720:
            res_name = '720p'
        elif height <= 1080:
            res_name = '1080p'
        elif height <= 1440:
            res_name = '1440p'
        else:
            res_name = '2160p'

        target_bitrate_kbps = bitrate_settings[res_name][quality_preset]
        # Adjust bitrate based on framerate
        if fps > 30:
            fps_factor = fps / 30.0
            target_bitrate_kbps = int(target_bitrate_kbps * fps_factor)

        config['ffmpeg']['target_bitrate'] = target_bitrate_kbps * 1000  # Convert to bps

        # Adjust target bitrate if source bitrate is lower
        source_bitrate = get_video_bitrate(input_video)
        if source_bitrate and source_bitrate < config['ffmpeg']['target_bitrate']:
            buffer = int(source_bitrate * 0.15)  # 15% buffer
            adjusted_bitrate = source_bitrate + buffer
            config['ffmpeg']['target_bitrate'] = adjusted_bitrate
            logger.info(f"Adjusted video bitrate to source bitrate + buffer: {adjusted_bitrate} bps")
        else:
            logger.info(f"Using target bitrate {config['ffmpeg']['target_bitrate']} bps for quality preset {quality_preset}")

    else:
        # Software encoder settings
        # Use CRF mode
        if video_codec == 'AV1':
            crf_values = {'Low': 35, 'Regular': 28, 'High': 20}
        else:
            crf_values = {'Low': 28, 'Regular': 23, 'High': 18}
        config['ffmpeg']['crf'] = crf_values[quality_preset]
        logger.info(f"Using CRF {config['ffmpeg']['crf']} for quality preset {quality_preset}")

    # Set audio bitrate
    source_audio_bitrate = get_audio_bitrate(input_video)
    target_audio_bitrate = 350000  # Default to 350 kbps

    if source_audio_bitrate and source_audio_bitrate < 300000:
        buffer = int(source_audio_bitrate * 0.15)  # 15% buffer
        adjusted_audio_bitrate = source_audio_bitrate + buffer
        config['ffmpeg']['audio_bitrate'] = adjusted_audio_bitrate
        logger.info(f"Adjusted audio bitrate to source bitrate + buffer: {adjusted_audio_bitrate} bps")
    else:
        config['ffmpeg']['audio_bitrate'] = target_audio_bitrate  # 350 kbps

    # Log the final settings
    logger.info(f"Final encoder settings: {config['ffmpeg']}")

def extract_and_convert_subtitles(input_video, subtitles_dir):
    """Extract subtitles from the input video and convert them to ASS format."""
    logger.info(f"Extracting subtitles from {input_video}")

    # Create subtitles directory
    os.makedirs(subtitles_dir, exist_ok=True)

    # Check if the input video has subtitle streams
    cmd_check = [
        'ffprobe', '-v', 'error', '-select_streams', 's',
        '-show_entries', 'stream=index', '-of', 'compact=p=0:nk=1',
        input_video
    ]
    result = subprocess.run(cmd_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout.strip() == '':
        logger.info("No subtitle streams found in the input video.")
        return []

    # FFmpeg command to extract subtitles
    cmd = [
        'ffmpeg', '-y', '-i', input_video, '-map', '0:s?', '-c:s', 'ass',
        os.path.join(subtitles_dir, 'subtitle_%d.ass')
    ]

    try:
        run_shell_command(cmd)
    except Exception as e:
        logger.error(f"Error extracting subtitles: {e}")
        return []

    # Return the list of extracted subtitles
    subtitle_files = [os.path.join(subtitles_dir, f) for f in os.listdir(subtitles_dir) if f.endswith('.ass')]
    return subtitle_files

def encode_video(input_video, frames_dir, output_video, config, subtitles_path=None):
    logger.info(f"Starting encoding for {input_video}")

    # Set encoder settings dynamically
    set_encoder_settings(config, input_video)

    # Extract and convert subtitles
    subtitles_dir = os.path.join(config['output']['folder'], 'subtitles')
    extracted_subtitles = extract_and_convert_subtitles(input_video, subtitles_dir)

    # Include external subtitles if provided
    if subtitles_path and os.path.exists(subtitles_path):
        extracted_subtitles.append(subtitles_path)
        logger.info(f"Including external subtitles: {subtitles_path}")

    if frames_dir and os.path.exists(frames_dir):
        # Get FPS from input video
        fps = get_video_fps(input_video)

        # Stitch frames and encode video
        try:
            stitch_frames_to_video(frames_dir, output_video, fps, extracted_subtitles, config)
        except Exception as e:
            logger.error(f"Error during stitching and encoding: {e}")
            return
    else:
        # No upscaled frames, re-encode original video
        try:
            assemble_video_without_upscaling(input_video, output_video, extracted_subtitles, config)
        except Exception as e:
            logger.error(f"Error during assembling without upscaling: {e}")
            return

    logger.info(f"{Fore.GREEN}Encoding completed for {input_video}")

def stitch_frames_to_video(frames_dir, output_video_path, fps, extracted_subtitles, config):
    """Stitch upscaled frames to a video file using ffmpeg."""
    logger.info(f"Stitching frames from {frames_dir} to {output_video_path}")

    frame_pattern = os.path.join(frames_dir, 'frame%08d.png')

    encoder = config['ffmpeg']['encoder']
    preset = config['ffmpeg']['preset']
    audio_codec = config['ffmpeg']['audio_codec']
    audio_bitrate = config['ffmpeg']['audio_bitrate']

    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', frame_pattern, '-i', config['input_video']]

    # Include subtitles
    for subtitle_file in extracted_subtitles:
        cmd.extend(['-i', subtitle_file])

    # Map streams
    cmd.extend(['-map', '0:v', '-map', '1:a?'])
    for idx, subtitle_file in enumerate(extracted_subtitles):
        cmd.extend(['-map', f'{idx + 2}:0'])

    # Video encoding settings
    cmd.extend(['-c:v', encoder, '-preset', preset])

    # Set encoding parameters based on config
    if 'rc' in config['ffmpeg']:
        cmd.extend(['-rc:v', config['ffmpeg']['rc']])

    if 'qp' in config['ffmpeg']:
        cmd.extend(['-qp', str(config['ffmpeg']['qp'])])
    elif 'cq' in config['ffmpeg']:
        cmd.extend(['-cq', str(config['ffmpeg']['cq'])])
    elif 'crf' in config['ffmpeg']:
        cmd.extend(['-crf', str(config['ffmpeg']['crf'])])

    if 'target_bitrate' in config['ffmpeg']:
        cmd.extend(['-b:v', str(config['ffmpeg']['target_bitrate'])])
        # Optionally set maxrate and bufsize
        maxrate = int(config['ffmpeg']['target_bitrate'] * 1.5)
        bufsize = int(config['ffmpeg']['target_bitrate'] * 2)
        cmd.extend(['-maxrate', str(maxrate), '-bufsize', str(bufsize)])

    # Audio encoding settings
    cmd.extend(['-c:a', audio_codec, '-b:a', str(audio_bitrate)])

    # Subtitle codec
    cmd.extend(['-c:s', 'copy'])

    # Preserve chapters and metadata
    cmd.extend(['-map_metadata', '1', '-map_chapters', '1'])

    # Output file
    cmd.extend([str(output_video_path)])

    run_shell_command(cmd)

def assemble_video_without_upscaling(input_video, output_video_path, extracted_subtitles, config):
    """Assemble video without upscaling frames."""
    logger.info(f"Assembling video without upscaling for {input_video}")

    encoder = config['ffmpeg']['encoder']
    preset = config['ffmpeg']['preset']
    audio_codec = config['ffmpeg']['audio_codec']
    audio_bitrate = config['ffmpeg']['audio_bitrate']

    cmd = ['ffmpeg', '-y', '-i', input_video]

    # Include subtitles
    for subtitle_file in extracted_subtitles:
        cmd.extend(['-i', subtitle_file])

    # Map streams
    cmd.extend(['-map', '0:v', '-map', '0:a?'])
    for idx, subtitle_file in enumerate(extracted_subtitles):
        cmd.extend(['-map', f'{idx + 1}:0'])

    # Video encoding settings
    cmd.extend(['-c:v', encoder, '-preset', preset])

    # Set encoding parameters based on config
    if 'rc' in config['ffmpeg']:
        cmd.extend(['-rc:v', config['ffmpeg']['rc']])

    if 'qp' in config['ffmpeg']:
        cmd.extend(['-qp', str(config['ffmpeg']['qp'])])
    elif 'cq' in config['ffmpeg']:
        cmd.extend(['-cq', str(config['ffmpeg']['cq'])])
    elif 'crf' in config['ffmpeg']:
        cmd.extend(['-crf', str(config['ffmpeg']['crf'])])

    if 'target_bitrate' in config['ffmpeg']:
        cmd.extend(['-b:v', str(config['ffmpeg']['target_bitrate'])])
        # Optionally set maxrate and bufsize
        maxrate = int(config['ffmpeg']['target_bitrate'] * 1.5)
        bufsize = int(config['ffmpeg']['target_bitrate'] * 2)
        cmd.extend(['-maxrate', str(maxrate), '-bufsize', str(bufsize)])

    # Audio encoding settings
    cmd.extend(['-c:a', audio_codec, '-b:a', str(audio_bitrate)])

    # Subtitle codec
    cmd.extend(['-c:s', 'copy'])

    # Preserve chapters and metadata
    cmd.extend(['-map_metadata', '0', '-map_chapters', '0'])

    # Output file
    cmd.extend([str(output_video_path)])

    run_shell_command(cmd)

def process_videos(config, input_video, frames_dir):
    output_dir = config['output']['folder']
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_video_path = output_dir / f"{Path(input_video).stem}_encoded.mkv"
    config['input_video'] = input_video

    # Load metadata
    metadata = load_metadata()
    video_key = str(Path(input_video).resolve())
    subtitles_path = metadata.get(video_key, {}).get('subtitles', None)

    encode_video(input_video, frames_dir, encoded_video_path, config, subtitles_path)

    logger.info(f"Video encoding completed for {input_video}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Encode video with upscaled frames.")
    parser.add_argument('-i', '--input_video', type=str, required=True, help="Input video file")
    parser.add_argument('-f', '--frames_dir', type=str, help="Directory containing upscaled frames")
    parser.add_argument('-o', '--output', type=str, default='./output', help="Output directory for encoded video")
    parser.add_argument('--quality_preset', type=str, choices=['Low', 'Regular', 'High'], required=True, help="Quality preset (Low, Regular, High)")
    parser.add_argument('--codec', type=str, choices=['AV1', 'h264', 'h265'], required=True, help="Video codec (AV1, h264, h265)")
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="Set the logging level")
    args = parser.parse_args()

    # Set logging level based on user input
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {args.log_level}")
        sys.exit(1)
    logger.setLevel(numeric_level)
    ch.setLevel(numeric_level)

    CONFIG = {
        'output': {
            'folder': Path(args.output).resolve(),
        },
        'ffmpeg': {
            'video_codec': args.codec,
            'audio_codec': 'libopus',
            'audio_bitrate': None,  # Will be set dynamically
            'preset': 'veryslow',   # Will be adjusted based on codec
            'encoder': '',          # Will be set dynamically
            'crf': None,            # Will be set dynamically
            'target_bitrate': None, # Will be set dynamically
            'use_crf': True,        # Will be set dynamically
            'quality_preset': args.quality_preset
        }
    }

    logger.info("Starting video encoding process...")

    process_videos(CONFIG, args.input_video, args.frames_dir)

    logger.info(f"{Fore.GREEN}Encoding process completed.")

if __name__ == "__main__":
    main()
