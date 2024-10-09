# centralized_logger.py

import logging
import sys
from datetime import datetime
from pathlib import Path

# Initialize colorama
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Please install colorama for colored CLI output: pip install colorama")
    sys.exit(1)

# Create log folder
log_folder = Path("./output/logs")
log_folder.mkdir(parents=True, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_folder / f"morpvid_{current_time}.log"

# Create logger
logger = logging.getLogger('CentralLogger')
logger.setLevel(logging.DEBUG)

# File handler for logging to file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)

# Stream handler for logging to console
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to logger
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)

# Functions to log with colors
def log_info(message):
    logger.info(Fore.GREEN + message)

def log_warning(message):
    logger.warning(Fore.YELLOW + message)

def log_error(message):
    logger.error(Fore.RED + message)

def log_debug(message):
    logger.debug(message)
