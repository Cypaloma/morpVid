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
logger.setLevel(logging.DEBUG)  # Set to DEBUG initially; will adjust later

# Create file handler which logs even debug messages
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)  # Log everything to file

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Default level; will adjust later

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)
else:
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

METADATA_FILE = 'metadata.json'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
        return {}

def detect_cuda_device():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_encoder_available(encoder_name):
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        encoders_output = result.stdout
        lines = encoders_output.split('\n')
        for line in lines:
            if line.startswith(' '):  # Encoder lines start with a space
                columns = line.strip().split()
                if len(columns) >= 2 and columns[1] == encoder_name:
                    logger.debug(f"Found encoder: {encoder_name}")
                    return True
        logger.debug(f"Encoder {encoder_name} not found.")
        return False
    except Exception as e:
        logger.error(f"Error checking for encoder {encoder_name}: {e}")
        return False

def run_shell_command(command):
    try:
        logger.info(f"Running command: {' '.join(map(str, command))}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        stdout_lines = []
        with tqdm(total=100, desc="Processing", unit="%", ncols=80) as pbar:
            for line in process.stdout:
                stdout_lines.append(line)
                tqdm.write(line.strip())
                pbar.update(0)
        process.wait()
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(''.join(stdout_lines))
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                output=''.join(stdout_lines),
                stderr=''
            )
        logger.debug(''.join(stdout_lines))
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {' '.join(map(str, command))}")
        logger.error(e.output)
        raise

def get_video_resolution(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json', str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = json.loads(result.stdout)
    width = output['streams'][0]['width']
    height = output['streams'][0]['height']
    return width, height

def get_video_fps(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-print_format', 'json', '-show_entries',
        'stream=r_frame_rate', str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = json.loads(result.stdout)
    fps_str = output['streams'][0]['r_frame_rate']
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom
    logger.info(f"Detected FPS: {fps}")
    return fps

def get_video_duration(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout.strip()
    return float(output)

def get_video_bitrate(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout.strip()
    if output == 'N/A' or output == '':
        # Try to estimate bitrate using file size and duration
        try:
            duration = get_video_duration(video_path)
            file_size = os.path.getsize(video_path) * 8  # Convert bytes to bits
            bitrate = int(file_size / duration)
            logger.info(f"Estimated video bitrate: {bitrate} bps")
            return bitrate
        except Exception as e:
            logger.warning(f"Could not estimate video bitrate: {e}")
            return 5000000  # Default to 5 Mbps
    return int(float(output))

def get_audio_bitrate(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a',
        '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    outputs = result.stdout.strip().split('\n')
    if not outputs or outputs[0] == 'N/A' or outputs[0] == '':
        # Default to 128 kbps if bitrate not available
        logger.warning("Audio bitrate not available, defaulting to 256000 bps")
        return 256000
    # Sum the bitrates of all audio streams
    total_bitrate = sum(int(float(bitrate)) for bitrate in outputs if bitrate and bitrate != 'N/A')
    logger.info(f"Total audio bitrate: {total_bitrate} bps")
    return total_bitrate

def calculate_target_bitrate(width, height, fps, quality_preset, video_codec):
    # Base bitrate calculation based on resolution and fps
    base_bitrate = width * height * fps * 0.07  # This factor can be adjusted
    # Quality multipliers
    quality_multipliers = {'Low': 0.5, 'Regular': 1.0, 'High': 1.5}
    codec_multipliers = {'h264': 1.0, 'h265': 0.7, 'AV1': 0.5}
    adjusted_bitrate = base_bitrate * quality_multipliers[quality_preset] * codec_multipliers[video_codec]
    maxrate = adjusted_bitrate * 1.5
    bufsize = adjusted_bitrate * 2
    return int(adjusted_bitrate), int(maxrate), int(bufsize)  # Return as integers

def set_encoder_settings(config, input_video):
    video_codec = config['ffmpeg']['video_codec']
    quality_preset = config['ffmpeg']['quality_preset']
    width, height = get_video_resolution(input_video)
    fps = get_video_fps(input_video)
    config['resolution'] = f"{width}x{height}"
    config['fps'] = fps

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

    hardware_encoder_name = hardware_encoders.get(video_codec)
    hardware_encoder_available = hardware_encoder_name and is_encoder_available(hardware_encoder_name)

    software_encoder_name = software_encoders.get(video_codec)
    if not software_encoder_name:
        logger.error(f"Unsupported video codec: {video_codec}")
        raise ValueError(f"Unsupported video codec: {video_codec}")

    config['ffmpeg']['hardware_encoder'] = hardware_encoder_name
    config['ffmpeg']['software_encoder'] = software_encoder_name

    if hardware_encoder_available:
        config['ffmpeg']['encoder'] = hardware_encoder_name
        config['ffmpeg']['preset'] = 'slow'
        config['ffmpeg']['use_crf'] = False
        logger.info(f"{Fore.GREEN}Hardware encoder {hardware_encoder_name} is available and will be used.")
    else:
        config['ffmpeg']['encoder'] = software_encoder_name
        config['ffmpeg']['preset'] = 'veryslow' if quality_preset == 'High' else 'medium'
        config['ffmpeg']['use_crf'] = True
        logger.info(f"{Fore.YELLOW}Hardware encoder not available. Using software encoder {software_encoder_name} with preset {config['ffmpeg']['preset']}.")

    # Calculate target bitrate
    target_bitrate, maxrate, bufsize = calculate_target_bitrate(
        width, height, fps, quality_preset, video_codec
    )
    config['ffmpeg']['target_bitrate'] = target_bitrate
    config['ffmpeg']['maxrate'] = maxrate
    config['ffmpeg']['bufsize'] = bufsize

    # Adjust target bitrate if source bitrate is lower
    source_bitrate = get_video_bitrate(input_video)
    if source_bitrate and source_bitrate < config['ffmpeg']['target_bitrate']:
        buffer = int(source_bitrate * 0.15)
        adjusted_bitrate = source_bitrate + buffer
        config['ffmpeg']['target_bitrate'] = adjusted_bitrate
        logger.info(f"Adjusted video bitrate to source bitrate + buffer: {adjusted_bitrate} bps")
    else:
        logger.info(f"Using target bitrate {config['ffmpeg']['target_bitrate']} bps for quality preset {quality_preset}")

    if hardware_encoder_available:
        if video_codec == 'h265' or video_codec == 'h264':
            config['ffmpeg']['rc'] = 'vbr_hq'
            config['ffmpeg']['cq'] = None
            cq_values = {'Low': 28, 'Regular': 23, 'High': 18}
            cq = cq_values[quality_preset]
            config['ffmpeg']['cq'] = cq
            logger.info(f"Using CQ {cq} for quality preset {quality_preset}")
        elif video_codec == 'AV1':
            config['ffmpeg']['rc'] = 'vbr'
            cq_values = {'Low': 40, 'Regular': 35, 'High': 30}
            cq = cq_values[quality_preset]
            config['ffmpeg']['cq'] = cq
            logger.info(f"Using CQ {cq} for quality preset {quality_preset}")
    else:
        # Only set 'crf' if using software encoder
        if video_codec == 'AV1':
            crf_values = {'Low': 40, 'Regular': 35, 'High': 30}
        else:
            crf_values = {'Low': 28, 'Regular': 23, 'High': 18}
        config['ffmpeg']['crf'] = crf_values[quality_preset]
        logger.info(f"Using CRF {config['ffmpeg']['crf']} for quality preset {quality_preset}")

    # Get audio bitrate
    source_audio_bitrate = get_audio_bitrate(input_video)
    target_audio_bitrate = source_audio_bitrate + int(source_audio_bitrate * 0.15)
    config['ffmpeg']['audio_bitrate'] = target_audio_bitrate
    logger.info(f"Adjusted audio bitrate to source bitrate + buffer: {target_audio_bitrate} bps")

    logger.info(f"Final encoder settings: {config['ffmpeg']}")

def extract_stream_metadata(input_video):
    """Extract metadata for all streams using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-print_format', 'json',
        '-show_entries', 'stream', str(input_video)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    return info.get('streams', [])

def extract_and_convert_subtitles(input_video, subtitles_dir):
    logger.info(f"Extracting subtitles from {input_video}")
    os.makedirs(subtitles_dir, exist_ok=True)
    streams_info = extract_stream_metadata(input_video)
    subtitle_streams = [s for s in streams_info if s['codec_type'] == 'subtitle']
    if not subtitle_streams:
        logger.info("No subtitle streams found in the input video.")
        return [], []
    extracted_subtitles = []
    subtitles_metadata = []
    for idx, stream in enumerate(subtitle_streams):
        stream_index = stream['index']
        subtitle_file = os.path.join(subtitles_dir, f'subtitle_{idx + 1}.ass')
        cmd = [
            'ffmpeg', '-y', '-i', input_video, '-map', f'0:{stream_index}', '-c:s', 'ass', subtitle_file
        ]
        try:
            run_shell_command(cmd)
            extracted_subtitles.append(subtitle_file)
            metadata = {
                'index': idx,
                'title': stream.get('tags', {}).get('title', ''),
                'language': stream.get('tags', {}).get('language', ''),
                'default': '1' if stream.get('disposition', {}).get('default') else '0',
                'forced': '1' if stream.get('disposition', {}).get('forced') else '0',
            }
            subtitles_metadata.append(metadata)
            logger.info(f"Extracted subtitle stream 0:{stream_index} to {subtitle_file}")
        except Exception as e:
            logger.error(f"Error extracting subtitle stream 0:{stream_index}: {e}")
    return extracted_subtitles, subtitles_metadata

def encode_video(input_video, frames_dir, output_video, config, subtitles_path=None):
    logger.info(f"Starting encoding for {input_video}")
    set_encoder_settings(config, input_video)
    subtitles_dir = os.path.join(config['output']['folder'], 'subtitles')
    extracted_subtitles, subtitles_metadata = extract_and_convert_subtitles(input_video, subtitles_dir)
    if subtitles_path and os.path.exists(subtitles_path):
        extracted_subtitles.append(subtitles_path)
        subtitles_metadata.append({'index': len(subtitles_metadata), 'title': '', 'language': '', 'default': '0', 'forced': '0'})
        logger.info(f"Including external subtitles: {subtitles_path}")
    if frames_dir and os.path.exists(frames_dir):
        fps = get_video_fps(input_video)
        try:
            stitch_frames_to_video(frames_dir, output_video, fps, extracted_subtitles, subtitles_metadata, config)
        except Exception as e:
            logger.error(f"Error during stitching and encoding: {e}")
            return
    else:
        try:
            assemble_video_without_upscaling(input_video, output_video, extracted_subtitles, subtitles_metadata, config)
        except Exception as e:
            logger.error(f"Error during assembling without upscaling: {e}")
            return
    logger.info(f"{Fore.GREEN}Encoding completed for {input_video}")

def assemble_video_without_upscaling(input_video, output_video_path, extracted_subtitles, subtitles_metadata, config):
    logger.info(f"Assembling video without upscaling for {input_video}")
    audio_codec = config['ffmpeg']['audio_codec']
    audio_bitrate = config['ffmpeg']['audio_bitrate']

    # Start constructing the FFmpeg command
    cmd = ['ffmpeg', '-y', '-i', input_video]
    for subtitle_file in extracted_subtitles:
        cmd.extend(['-i', subtitle_file])

    # Mapping inputs
    cmd.extend(['-map', '0:v', '-map', '0:a?'])
    for idx, _ in enumerate(extracted_subtitles):
        cmd.extend(['-map', f'{idx + 1}:0'])
    cmd.extend(['-map', '0:t?'])  # Map attachments (e.g., fonts)
    cmd.extend(['-map_metadata', '0', '-map_chapters', '0'])

    # Video encoding options (must be before output file)
    encoder = config['ffmpeg']['encoder']
    preset = config['ffmpeg']['preset']
    cmd.extend(['-c:v', encoder, '-preset', preset])

    if 'rc' in config['ffmpeg']:
        cmd.extend(['-rc:v', config['ffmpeg']['rc']])
    if 'cq' in config['ffmpeg'] and config['ffmpeg']['cq'] is not None:
        cmd.extend(['-cq', str(config['ffmpeg']['cq'])])
    elif 'crf' in config['ffmpeg'] and config['ffmpeg']['crf'] is not None:
        cmd.extend(['-crf', str(config['ffmpeg']['crf'])])

    if 'target_bitrate' in config['ffmpeg']:
        cmd.extend(['-b:v', str(config['ffmpeg']['target_bitrate'])])
        cmd.extend(['-maxrate', str(config['ffmpeg']['maxrate']), '-bufsize', str(config['ffmpeg']['bufsize'])])

    # Audio and subtitle options
    cmd.extend(['-c:a', audio_codec, '-b:a', str(audio_bitrate)])
    cmd.extend(['-c:s', 'copy'])
    cmd.extend(['-c:t', 'copy'])  # Copy attachments

    # Set metadata for subtitle streams
    for idx, metadata in enumerate(subtitles_metadata):
        if metadata['title']:
            cmd.extend(['-metadata:s:s:{}'.format(idx), f"title={metadata['title']}"])
        if metadata['language']:
            cmd.extend(['-metadata:s:s:{}'.format(idx), f"language={metadata['language']}"])
        # Set dispositions (default, forced)
        dispositions = []
        if metadata['default'] == '1':
            dispositions.append('default')
        if metadata['forced'] == '1':
            dispositions.append('forced')
        if dispositions:
            cmd.extend(['-disposition:s:{}'.format(idx), ','.join(dispositions)])

    # Place the output filename at the end
    cmd.extend([str(output_video_path)])

    try:
        logger.info("Starting encoding with FFmpeg...")
        run_shell_command(cmd)
        logger.info("Encoding completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Encoding failed: {e.stderr}")
        raise

def stitch_frames_to_video(frames_dir, output_video_path, fps, extracted_subtitles, subtitles_metadata, config):
    logger.info(f"Stitching frames from {frames_dir} to {output_video_path}")
    frame_pattern = os.path.join(frames_dir, 'frame%08d.png')
    audio_codec = config['ffmpeg']['audio_codec']
    audio_bitrate = config['ffmpeg']['audio_bitrate']
    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', frame_pattern, '-i', config['input_video']]
    for subtitle_file in extracted_subtitles:
        cmd.extend(['-i', subtitle_file])
    cmd.extend(['-map', '0:v', '-map', '1:a?'])
    for idx, _ in enumerate(extracted_subtitles):
        cmd.extend(['-map', f'{idx + 2}:0'])
    cmd.extend(['-map', '1:t?'])
    cmd.extend(['-map_metadata', '1', '-map_chapters', '1'])

    # Video encoding options (must be before output file)
    encoder = config['ffmpeg']['encoder']
    preset = config['ffmpeg']['preset']
    cmd.extend(['-c:v', encoder, '-preset', preset])

    if 'rc' in config['ffmpeg']:
        cmd.extend(['-rc:v', config['ffmpeg']['rc']])
    if 'cq' in config['ffmpeg'] and config['ffmpeg']['cq'] is not None:
        cmd.extend(['-cq', str(config['ffmpeg']['cq'])])
    elif 'crf' in config['ffmpeg'] and config['ffmpeg']['crf'] is not None:
        cmd.extend(['-crf', str(config['ffmpeg']['crf'])])

    if 'target_bitrate' in config['ffmpeg']:
        cmd.extend(['-b:v', str(config['ffmpeg']['target_bitrate'])])
        cmd.extend(['-maxrate', str(config['ffmpeg']['maxrate']), '-bufsize', str(config['ffmpeg']['bufsize'])])

    # Audio and subtitle options
    cmd.extend(['-c:a', audio_codec, '-b:a', str(audio_bitrate)])
    cmd.extend(['-c:s', 'copy'])
    cmd.extend(['-c:t', 'copy'])  # Copy attachments

    # Set metadata for subtitle streams
    for idx, metadata in enumerate(subtitles_metadata):
        if metadata['title']:
            cmd.extend(['-metadata:s:s:{}'.format(idx), f"title={metadata['title']}"])
        if metadata['language']:
            cmd.extend(['-metadata:s:s:{}'.format(idx), f"language={metadata['language']}"])
        # Set dispositions (default, forced)
        dispositions = []
        if metadata['default'] == '1':
            dispositions.append('default')
        if metadata['forced'] == '1':
            dispositions.append('forced')
        if dispositions:
            cmd.extend(['-disposition:s:{}'.format(idx), ','.join(dispositions)])

    # Place the output filename at the end
    cmd.extend([str(output_video_path)])

    try:
        logger.info("Starting encoding with FFmpeg...")
        run_shell_command(cmd)
        logger.info("Encoding completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Encoding failed: {e.stderr}")
        raise

def process_videos(config, input_video, frames_dir):
    output_dir = config['output']['folder']
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_video_path = output_dir / f"{Path(input_video).stem}_encoded.mkv"
    config['input_video'] = input_video
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
    parser.add_argument('-o', '--output', type=str, default='./output/encoded', help="Output directory for encoded video")
    parser.add_argument('--quality_preset', type=str, choices=['Low', 'Regular', 'High'], required=True, help="Quality preset (Low, Regular, High)")
    parser.add_argument('--codec', type=str, choices=['AV1', 'h264', 'h265'], required=True, help="Video codec (AV1, h264, h265)")
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="Set the logging level")
    args = parser.parse_args()

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
            'audio_bitrate': None,
            'preset': 'veryslow',
            'encoder': '',
            'hardware_encoder': '',
            'software_encoder': '',
            'crf': None,
            'target_bitrate': None,
            'maxrate': None,
            'bufsize': None,
            'use_crf': True,
            'quality_preset': args.quality_preset
        }
    }

    logger.info("Starting video encoding process...")
    process_videos(CONFIG, args.input_video, args.frames_dir)
    logger.info(f"{Fore.GREEN}Encoding process completed.")

if __name__ == "__main__":
    main()
