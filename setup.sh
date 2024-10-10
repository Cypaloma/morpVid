#!/bin/bash

# Define the installation directory relative to the script location
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPENDENCIES_DIR="$PROJECT_DIR/dependencies"
MINICONDA_DIR="$DEPENDENCIES_DIR/miniconda3"

# Download Miniconda installer
if [ ! -f "$DEPENDENCIES_DIR/Miniconda3-latest-Linux-x86_64.sh" ]; then
    mkdir -p "$DEPENDENCIES_DIR"
    wget -O "$DEPENDENCIES_DIR/Miniconda3-latest-Linux-x86_64.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

# Install Miniconda to the specified directory
if [ ! -d "$MINICONDA_DIR" ]; then
    bash "$DEPENDENCIES_DIR/Miniconda3-latest-Linux-x86_64.sh" -b -p "$MINICONDA_DIR"
fi

# Initialize conda from the local installation
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Update conda
conda update -n base -c defaults conda -y

# Create the anime_upscale_env environment in the project directory
ENV_DIR="$DEPENDENCIES_DIR/envs/anime_upscale_env"
if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.10 -y
fi

# Activate the anime_upscale_env environment
conda activate "$ENV_DIR"

# Install necessary packages in anime_upscale_env
echo "Installing required packages in anime_upscale_env..."
pip install colorama

# Deactivate the anime_upscale_env environment
conda deactivate

# Call the setup_whisperx_env.sh script
WHISPERX_SETUP_SCRIPT="$PROJECT_DIR/scripts/setup_whisperx_env.sh"

if [ -f "$WHISPERX_SETUP_SCRIPT" ]; then
    echo "Running setup_whisperx_env.sh script..."
    # Make sure the script is executable
    chmod +x "$WHISPERX_SETUP_SCRIPT"
    # Execute the script
    bash "$WHISPERX_SETUP_SCRIPT"
else
    echo "Error: setup_whisperx_env.sh script not found at $WHISPERX_SETUP_SCRIPT."
    exit 1
fi

echo "Setup completed successfully."
