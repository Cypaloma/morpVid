import os
import subprocess
import threading
import json
from pathlib import Path
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Import the centralized logger
from centralized_logger import logger, log_info, log_warning, log_error, log_debug

# Paths and Constants
METADATA_FILE = Path('metadata.json')
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Assuming main.py is in /scripts
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'  # Directory containing scripts

# Anime upscale environment
ANIME_ENV_DIR = PROJECT_ROOT / 'dependencies/envs/anime_upscale_env'
ANIME_PYTHON_EXECUTABLE = ANIME_ENV_DIR / 'bin' / 'python'

# WhisperX environment
WHISPERX_ENV_DIR = PROJECT_ROOT / 'dependencies/envs/whisperx_env'
WHISPERX_PYTHON_EXECUTABLE = WHISPERX_ENV_DIR / 'bin' / 'python'

# Miniconda activation path
CONDA_DIR = PROJECT_ROOT / 'dependencies/miniconda'
CONDA_ACTIVATE = CONDA_DIR / 'bin' / 'activate'


def activate_anime_upscale_env():
    """Activate the anime upscale environment using subprocess."""
    log_info(f"Activating the 'anime_upscale_env' environment...")
    command = f"source {CONDA_ACTIVATE} {ANIME_ENV_DIR}"
    process = subprocess.Popen(command, shell=True, executable="/bin/bash")
    process.communicate()  # Ensure the environment is activated

    if process.returncode != 0:
        log_error(f"Failed to activate environment at {ANIME_ENV_DIR}")
        sys.exit(1)
    else:
        log_info(f"Successfully activated 'anime_upscale_env' environment.")


def create_metadata_file():
    """Create metadata.json if it doesn't exist."""
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, 'w') as f:
            json.dump({}, f, indent=4)


def load_metadata():
    """Load metadata from metadata.json."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    """Save metadata to metadata.json."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)


def run_command(command, env_name):
    """Run a command and log its output in real-time."""
    log_info(f"Running command in {env_name} environment: {' '.join(map(str, command))}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)

    # Read the output in a thread
    def read_output(pipe):
        for line in iter(pipe.readline, ''):
            if line:
                line = line.strip()
                if "error" in line.lower():
                    log_error(line)
                else:
                    log_info(line)

    output_thread = threading.Thread(target=read_output, args=(process.stdout,))
    output_thread.start()

    process.wait()
    output_thread.join()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def ensure_environment(env_path, env_name):
    """Ensure that the specified environment is activated."""
    current_env = os.environ.get('CONDA_PREFIX')
    if current_env != str(env_path):
        log_info(f"Activating the '{env_name}' environment at {env_path}...")
        activate_script = f"source {CONDA_ACTIVATE} {env_path}"
        shell = os.environ.get('SHELL', '/bin/bash')
        # Activate the environment in the subprocess
        command = f'{activate_script} && echo "Environment activated"'
        result = subprocess.run(command, shell=True, executable=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Environment activated" in result.stdout:
            log_info(f"Successfully activated environment: {env_path}")
        else:
            log_error(f"Failed to activate environment: {env_path}")
            raise EnvironmentError(f"Could not activate environment at {env_path}")
    else:
        log_info(f"The '{env_name}' environment is already active.")


def extract_video_info(video_path):
    """Extract resolution and codec information from a video file."""
    log_info(f"Extracting video info for {video_path}")
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,codec_name', '-of', 'json', str(video_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        codec = stream['codec_name']
        return width, height, codec
    except Exception as e:
        log_error(f"Failed to extract video info for {video_path}: {e}")
        return None, None, None


def determine_scaling_options(width, height):
    """Determine possible scaling options based on current resolution."""
    # Supported resolutions
    target_resolutions = {
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '2160p': (3840, 2160)  # 4K
    }

    scaling_options = {}
    for name, (target_width, target_height) in target_resolutions.items():
        scale_x = target_width / width
        scale_y = target_height / height
        scale_factor = min(scale_x, scale_y)
        # Round scale factor to nearest integer
        scale_factor = int(round(scale_factor))
        if scale_factor >= 1:
            scaling_options[name] = scale_factor

    return scaling_options


def main():
    # Activate anime upscale environment at the very beginning
    activate_anime_upscale_env()

    log_info("Starting the media processing workflow.")

    # Ensure metadata.json exists
    create_metadata_file()

    # Load metadata (if any) from previous stages
    metadata = load_metadata()

    # Ask the user to provide the input folder
    input_folder = input("Enter the path to the folder containing the media files: ").strip()
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        log_error(f"The path {input_folder} is not a valid directory.")
        print(f"{Fore.RED}The path {input_folder} is not a valid directory.")
        return

    # Process all media files in the folder
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
    input_files = [file for file in input_folder.iterdir() if file.suffix.lower() in supported_extensions]

    if not input_files:
        log_info(f"No supported media files found in {input_folder}")
        print(f"{Fore.YELLOW}No supported media files found in the specified folder.")
        return

    # Collect user preferences
    log_info("Collecting user preferences.")

    # Upscaling options
    upscale_choice = input("Do you want to upscale the videos? (yes/no): ").strip().lower()
    if upscale_choice in ["yes", "y"]:
        scaling_factors = {}
        for input_file in input_files:
            width, height, codec = extract_video_info(input_file)
            if width is None or height is None:
                print(f"{Fore.YELLOW}Skipping file {input_file} due to extraction error.")
                continue

            print(f"\nCurrent resolution of {input_file.name}: {width}x{height}")
            scaling_options = determine_scaling_options(width, height)
            if not scaling_options:
                print(f"{Fore.YELLOW}No upscaling options available for this video.")
                continue

            print("Select target resolution:")
            for idx, (res_name, scale_factor) in enumerate(scaling_options.items(), start=1):
                print(f"{idx}. {res_name} ({scale_factor}x upscale)")

            choice = input("Enter the number corresponding to your choice: ").strip()
            try:
                choice_idx = int(choice)
                res_name = list(scaling_options.keys())[choice_idx - 1]
                scale_factor = scaling_options[res_name]
                scaling_factors[input_file] = scale_factor
            except (ValueError, IndexError):
                print(f"{Fore.YELLOW}Invalid choice. Skipping this video.")
                log_error(f"Invalid resolution choice provided for {input_file}.")
                continue

    else:
        scaling_factors = {}

    # Subtitles generation
    subtitles_choice = input("Do you want to generate subtitles using WhisperX? (yes/no): ").strip().lower()
    generate_subtitles = subtitles_choice in ["yes", "y"]

    # Ask for quality preset
    print("\nSelect the quality preset:")
    print("1. Low")
    print("2. Regular")
    print("3. High")
    quality_choice = input("Enter the number corresponding to your choice: ").strip()
    if quality_choice not in ['1', '2', '3']:
        print(f"{Fore.YELLOW}Invalid choice. Defaulting to 'Regular'.")
        log_error("Invalid quality choice provided. Defaulting to 'Regular'.")
        quality_choice = '2'  # Default to Regular
    quality_presets = {'1': 'Low', '2': 'Regular', '3': 'High'}
    quality_preset = quality_presets[quality_choice]

    # Ask for video codec
    print("\nSelect the video codec:")
    print("1. AV1")
    print("2. h264")
    print("3. h265")
    codec_choice = input("Enter the number corresponding to your choice: ").strip()
    if codec_choice not in ['1', '2', '3']:
        print(f"{Fore.YELLOW}Invalid choice. Defaulting to 'h265'.")
        log_error("Invalid codec choice provided. Defaulting to 'h265'.")
        codec_choice = '3'  # Default to h265
    codec_options = {'1': 'AV1', '2': 'h264', '3': 'h265'}
    selected_codec = codec_options[codec_choice]

    # Paths to scripts
    ai_upscale_script = SCRIPTS_DIR / 'ai_upscale_anime.py'
    generate_subtitles_script = SCRIPTS_DIR / 'generate_subtitles_whisperx.py'
    encode_video_script = SCRIPTS_DIR / 'encode_video.py'

    # Process each video file
    for input_file in input_files:
        log_info(f"Processing file: {input_file}")
        video_name = input_file.stem

        # Upscaling
        if upscale_choice in ["yes", "y"] and input_file in scaling_factors:
            scale_factor = scaling_factors[input_file]
            ensure_environment(ANIME_ENV_DIR, "anime_upscale_env")

            output_frames_dir = PROJECT_ROOT / 'output' / video_name
            output_frames_dir.mkdir(parents=True, exist_ok=True)

            upscale_command = [
                str(ANIME_PYTHON_EXECUTABLE), str(ai_upscale_script),
                '-i', str(input_file),
                '-s', str(scale_factor),
                '-o', str(output_frames_dir)
            ]
            run_command(upscale_command, "anime_upscale_env")

            metadata[str(input_file.resolve())] = {
                'original_video_path': str(input_file),
                'upscaled_frames_dir': str(output_frames_dir),
                'scale_factor': scale_factor
            }
            save_metadata(metadata)

        # Subtitles generation
        if generate_subtitles:
            ensure_environment(WHISPERX_ENV_DIR, "whisperx_env")
            subtitles_command = [
                str(WHISPERX_PYTHON_EXECUTABLE), str(generate_subtitles_script),
                '-i', str(input_file)
            ]
            run_command(subtitles_command, "whisperx_env")

        # Encoding
        ensure_environment(ANIME_ENV_DIR, "anime_upscale_env")

        if upscale_choice in ["yes", "y"] and input_file in scaling_factors:
            frames_dir_path = PROJECT_ROOT / 'output' / video_name / 'upscaled_frames'
        else:
            frames_dir_path = None  # No upscaled frames, use original video

        output_encoded_dir = PROJECT_ROOT / 'output' / 'encoded'

        encode_command = [
            str(ANIME_PYTHON_EXECUTABLE), str(encode_video_script),
            '-i', str(input_file),
            '-o', str(output_encoded_dir)
        ]

        if frames_dir_path and frames_dir_path.exists():
            encode_command.extend(['-f', str(frames_dir_path)])

        encode_command.extend(['--quality_preset', quality_preset, '--codec', selected_codec])

        run_command(encode_command, "anime_upscale_env")

    log_info(f"{Fore.GREEN}All files processed.")


if __name__ == "__main__":
    main()
