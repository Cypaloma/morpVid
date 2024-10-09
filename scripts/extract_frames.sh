#!/bin/bash

# Ensure the frames/original directory exists
mkdir -p frames/original

# Extract frames from the video
ffmpeg -i "$1" frames/original/frame%04d.png
