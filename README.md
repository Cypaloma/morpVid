Here's the updated version with the thank you section and relevant links to documentation:

---

# morpVid: Streamlined Video Management 🎥🛠️

morpVid is a partially automated tool designed to simplify the management of the average consumer's local video libraries. It handles encoding, anime upscaling, subtitle generation, and packaging into standardized MKV files with OPUS audio and ASS subtitles. Built on Arch Linux and CUDA 11.8, morpVid ensures consistent quality and reasonable file sizes across your entire collection.

## Overview 🌟

With morpVid, you can automate the processing of all incoming video files. The tool allows you to standardize quality, ensure efficient compression, and optionally generate subtitles or upscale anime videos. morpVid asks for minimal input and then processes each valid video file in the specified directory based on your preferences.

## Key Features 🚀

- **Quality Presets**: Select from Low, Medium, or High quality settings, designed to balance file size and visual fidelity.
- **Anime Upscaling**: Supports the `realesr-animevideov3.pth` model for upscaling anime videos using ESRGAN, with customizable scaling factors. 🎨
- **Subtitle Generation**: Uses WhisperX with Whisper Large v3 to transcribe audio to text and create accurate subtitles. 📝
- **Codec Selection**: Choose your preferred codec, and morpVid will handle the encoding specifics based on the video’s resolution and format. 🎬
- **Batch Processing**: Automatically processes every valid video file within the input directory, ensuring a uniform output across the entire collection. 🔄

## How It Works 🛠️

morpVid takes care of multiple steps in video processing:

1. **Input Selection**: Prompts you to select an input directory. 📁
2. **Anime Upscaling (Optional)**: Offers to upscale anime videos, allowing you to specify the scaling factor. 🌸
3. **Subtitle Generation (Optional)**: Generates subtitles using WhisperX for transcription. 🗣️
4. **Quality Preset Choice**: Lets you choose a quality preset—Low, Medium, or High—each with different bitrate and compression profiles. 🎚️
5. **Codec Selection**: Allows you to select the codec, and morpVid manages the rest, dynamically adjusting encoding settings to maintain consistent quality across different formats and resolutions. 🎞️

## Installation 💻

1. **Install CUDA 11.8**  
   *Optional, but recommended for faster GPU processing; the tool will fall back to CPU if CUDA is unavailable, at a reduced speed.* ⚡

2. **Download the ZIP** 📦  
   Extract morpVid anywhere you prefer, as it’s portable.

3. **Run `setup.sh`** 🛠️  
   This installs two conda environments—one for anime upscaling and general processing, and another for WhisperX transcription.

4. **Run `start.sh`** 🎉  
   Just follow the instructions, and morpVid will take care of the rest! 🚀

*Please note that morpVid requires the installation of several machine learning libraries and models, including proprietary NVIDIA software and multiple versions of PyTorch for compatibility with different tasks. The setup process is resource-intensive, takes a considerably large amount of storage space, and will automatically download the required dependencies without further prompts.* 💾

## Quality Presets ⚙️

- **Low**: Prioritizes compression and minimal file size while maintaining reasonable visual quality. 📉
- **Medium**: Offers a balanced approach with bitrates optimized for each resolution, plus a small buffer for extra quality. ⚖️
- **High**: Focuses on maximizing visual fidelity, approaching the limits of diminishing returns. 📈

## Why I Built morpVid 💡

Managing a diverse video library can be tedious—dealing with different formats, codecs, inconsistent resolutions, and missing or incorrect subtitles. I built morpVid to automate and standardize these processes. It allows me to optimize file size without compromising quality and ensures subtitles are consistently generated across my collection. The tool is especially useful for local backups, self-hosted media servers, or anyone needing an efficient, automated solution for video management. 💼

## Thank You and Resources 🙏

A huge thank you to all the developers and open-source projects that made morpVid possible! 🎉 If you’re interested in learning more or contributing to these amazing libraries, check out their documentation below:

- **WhisperX** (for subtitle generation):  
  [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)  
  WhisperX is used to transcribe audio and generate subtitles with high accuracy, powered by Whisper Large v3.

- **faster-whisper** (for optimized Whisper inference):  
  [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)  
  An optimized Whisper model for faster transcription, great for batch processing large video libraries.

- **Real-ESRGAN** (for anime upscaling):  
  [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)  
  [Anime Video Model Documentation](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)  
  Real-ESRGAN is used to upscale anime videos with exceptional quality, especially when paired with the anime-specific model.

- **CUDA 11.8** (for GPU-accelerated processing):  
  [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)  
  For faster GPU processing, morpVid leverages CUDA 11.8, but it will also work in CPU-only mode for those without NVIDIA GPUs.

- **FFmpeg** (for video encoding and format conversion):  
  [https://ffmpeg.org/](https://ffmpeg.org/)  
  morpVid uses FFmpeg under the hood to handle all the video encoding and format conversions, ensuring high-quality results.

Feel free to explore these resources for a deeper understanding of how morpVid integrates these technologies! 💻✨
