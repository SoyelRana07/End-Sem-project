"""
Configuration file for surveillance system - Crime Detection
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "saved_model"
ALERT_DIR = BASE_DIR / "alerts"
ALERT_DIR.mkdir(exist_ok=True)

# Camera Settings
DEFAULT_CAMERA = 0  # Use 0 for webcam, or RTSP URL for IP camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Performance Settings
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for performance
FRAME_BUFFER_SIZE = 10

# GPU Configuration
ENABLE_GPU = True  # Set to False if no GPU available
if ENABLE_GPU:
    try:
        import torch
        if torch.cuda.is_available():
            ENABLE_GPU = True
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
        else:
            ENABLE_GPU = False
            print("GPU acceleration disabled: CUDA is not available")
            print("  - Make sure you have a compatible NVIDIA GPU")
            print("  - Install CUDA toolkit and cuDNN")
            print("  - Install PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        ENABLE_GPU = False
        print("PyTorch not found, GPU acceleration disabled")
    except Exception as e:
        ENABLE_GPU = False
        print(f"Error checking GPU availability: {e}")
        print("Falling back to CPU")

MAX_WORKERS = 4  # Number of threads for processing

# Model Settings
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
BERT_MODEL_PATH = str(MODEL_DIR)
BERT_MAX_LENGTH = 128
BATCH_SIZE = 1  # Increase if processing multiple cameras

# Cache Settings
ENABLE_CAPTION_CACHE = True
CACHE_SIZE = 100
CACHE_SIMILARITY_THRESHOLD = 0.95  # Image similarity threshold for cache hit

# Event Types - Crime Detection
EVENT_TYPES = [
    "Abuse",
    "Arrest", 
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Normal_Videos_event",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism"
]

# Event Severity Levels
EVENT_SEVERITY = {
    # CRITICAL - Immediate danger to life
    "Shooting": "CRITICAL",
    "Explosion": "CRITICAL",
    "Arson": "CRITICAL",
    
    # HIGH - Violence or major crime
    "Assault": "HIGH",
    "Fighting": "HIGH",
    "Robbery": "HIGH",
    "Burglary": "HIGH",
    
    # MEDIUM - Property crime or minor violence
    "Stealing": "MEDIUM",
    "Shoplifting": "MEDIUM",
    "Vandalism": "MEDIUM",
    "Abuse": "MEDIUM",
    
    # LOW - Non-emergency events
    "Arrest": "LOW",
    "RoadAccidents": "LOW",
    
    # NORMAL - No alert needed
    "Normal_Videos_event": "NORMAL"
}

# Alert Settings by Severity
ALERT_THRESHOLDS = {
    "CRITICAL": 0.60,   # Lower threshold for critical events
    "HIGH": 0.70,       # Medium threshold
    "MEDIUM": 0.75,     # Higher threshold
    "LOW": 0.80,        # Very high threshold
    "NORMAL": 1.00      # Never alert
}

# Alert behavior
SAVE_ALERT_IMAGES = True
SAVE_ALERT_VIDEO = True
ALERT_VIDEO_DURATION = 10  # seconds before and after alert
MAX_ALERTS_PER_MINUTE = 10  # Rate limiting

# Alert cooldown by severity (seconds between same event type)
ALERT_COOLDOWN = {
    "CRITICAL": 3,   # Alert frequently for critical events
    "HIGH": 5,
    "MEDIUM": 8,
    "LOW": 10,
    "NORMAL": 999999  # Never alert
}

# Display Settings
DISPLAY_FPS = True
DISPLAY_CAPTION = True
DISPLAY_PREDICTIONS = True
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Color coding by severity (BGR format for OpenCV)
SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),      # Red
    "HIGH": (0, 100, 255),         # Orange
    "MEDIUM": (0, 255, 255),       # Yellow
    "LOW": (255, 200, 0),          # Cyan
    "NORMAL": (0, 255, 0)          # Green
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "surveillance.log"

# Notification Settings (for future integration)
ENABLE_SOUND_ALERT = True
ENABLE_EMAIL_ALERT = False  # Set to True and configure email
EMAIL_RECIPIENTS = ["adarsh.rounak.043@gmail.com"]
