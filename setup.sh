#!/bin/bash

# Get the absolute path of the script and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Define the path to the `scripts` directory
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Check if the scripts directory exists
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "Error: scripts directory not found at $SCRIPTS_DIR."
    exit 1
fi

# Paths to environment setup scripts inside the /scripts folder
ANIME_ENV_SETUP="$SCRIPTS_DIR/setup_anime_upscale_env.sh"
WHISPERX_ENV_SETUP="$SCRIPTS_DIR/setup_whisperx_env.sh"

# Check if setup scripts exist
if [ ! -f "$ANIME_ENV_SETUP" ]; then
    echo "Error: Anime upscale setup script not found at $ANIME_ENV_SETUP."
    exit 1
fi

if [ ! -f "$WHISPERX_ENV_SETUP" ]; then
    echo "Error: WhisperX setup script not found at $WHISPERX_ENV_SETUP."
    exit 1
fi

# Run the setup scripts for both environments
echo "Setting up the anime upscale environment..."
bash "$ANIME_ENV_SETUP"

echo "Setting up the WhisperX environment..."
bash "$WHISPERX_ENV_SETUP"

# Activate anime upscale environment and install colorama
echo "Activating anime upscale environment and installing colorama..."
source "$PROJECT_ROOT/dependencies/miniconda/bin/activate" "$PROJECT_ROOT/dependencies/envs/anime_upscale_env"
if [ "$CONDA_PREFIX" != "$PROJECT_ROOT/dependencies/envs/anime_upscale_env" ]; then
    echo "Failed to activate anime upscale environment."
    exit 1
else
    pip install colorama
fi

# Activate WhisperX environment and install colorama
echo "Activating WhisperX environment and installing colorama..."
source "$PROJECT_ROOT/dependencies/miniconda/bin/activate" "$PROJECT_ROOT/dependencies/envs/whisperx_env"
if [ "$CONDA_PREFIX" != "$PROJECT_ROOT/dependencies/envs/whisperx_env" ]; then
    echo "Failed to activate WhisperX environment."
    exit 1
else
    pip install colorama
fi

echo "All environments set up and colorama installed."
