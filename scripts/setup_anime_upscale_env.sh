#!/bin/bash

# Get the absolute path of the script and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set paths for Miniconda and the environment
CONDA_DIR="$PROJECT_ROOT/dependencies/miniconda"  # Miniconda will be installed in /dependencies at the project root
ENV_DIR="$PROJECT_ROOT/dependencies/envs/anime_upscale_env"  # Environment will be created in /dependencies/envs
CONDA_EXEC="$CONDA_DIR/bin/conda"

# Check if Miniconda is already installed in the specified directory
if [ ! -f "$CONDA_EXEC" ]; then
    echo "Miniconda not found. Downloading Miniconda..."
    MINICONDA_INSTALLER="$SCRIPT_DIR/miniconda_installer.sh"
    wget -O "$MINICONDA_INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
    rm "$MINICONDA_INSTALLER"
else
    echo "Miniconda already installed."
fi

# Check if the conda environment already exists
if [ -d "$ENV_DIR" ]; then
    read -p "Conda environment at $ENV_DIR already exists. Do you want to delete it and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing environment..."
        "$CONDA_EXEC" remove -y -p "$ENV_DIR" --all
    else
        echo "Exiting setup."
        exit 1
    fi
fi

# Create a new conda environment with necessary packages
echo "Creating a new conda environment for Anime Upscale..."
"$CONDA_EXEC" create -y -p "$ENV_DIR" python=3.10

# Activate the environment
source "$CONDA_DIR/bin/activate" "$ENV_DIR"

# Check if environment activation was successful
if [[ "$CONDA_PREFIX" != "$ENV_DIR" ]]; then
    echo "Failed to activate the environment."
    exit 1
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
"$CONDA_EXEC" install -y -p "$ENV_DIR" -c pytorch -c nvidia pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.1 pytorch-cuda=11.8

# Install other required Python packages
echo "Installing required Python packages..."
"$ENV_DIR/bin/pip" install numpy<2.0 opencv-python-headless tqdm colorama

# Install basicsr and realesrgan from GitHub to get the latest versions
echo "Installing basicsr and realesrgan from GitHub..."
"$ENV_DIR/bin/pip" install git+https://github.com/xinntao/BasicSR.git
"$ENV_DIR/bin/pip" install git+https://github.com/xinntao/Real-ESRGAN.git

# Install FFmpeg if not already installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    "$CONDA_EXEC" install -y -p "$ENV_DIR" -c conda-forge ffmpeg
else
    echo "FFmpeg is already installed."
fi

# Install any other required dependencies into /dependencies
echo "Installing other dependencies into /dependencies..."
# Example: Install centralized_logger if needed
# "$ENV_DIR/bin/pip" install -e "$PROJECT_ROOT/dependencies/centralized_logger"

echo "Environment created successfully at $ENV_DIR"

# Check and display installed packages
echo "Installed packages in the environment:"
"$CONDA_EXEC" list -p "$ENV_DIR"

# Optional: Export environment variables or provide usage instructions for further steps
echo "To use this environment, run the following command:"
echo "source $CONDA_DIR/bin/activate $ENV_DIR"
