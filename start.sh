#!/bin/bash

# Get the absolute path of the script and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Define the path to the main Python script
MAIN_SCRIPT="$PROJECT_ROOT/scripts/main.py"

# Check if the main Python script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: main.py not found at $MAIN_SCRIPT."
    exit 1
fi

# Activate the anime upscale environment
echo "Activating anime upscale environment..."
source "$PROJECT_ROOT/dependencies/miniconda/bin/activate" "$PROJECT_ROOT/dependencies/envs/anime_upscale_env"
if [ "$CONDA_PREFIX" != "$PROJECT_ROOT/dependencies/envs/anime_upscale_env" ]; then
    echo "Failed to activate anime upscale environment."
    exit 1
fi

# Run the main.py script using the active environment
echo "Running main.py..."
python "$MAIN_SCRIPT"

# Deactivate the environment after the script finishes
echo "Deactivating the environment..."
conda deactivate

echo "main.py script execution finished."
