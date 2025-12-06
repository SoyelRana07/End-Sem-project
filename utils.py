"""
Utility functions for surveillance system
"""
import cv2
import numpy as np
from pathlib import Path
import json

def create_video_from_frames(frames, output_path, fps=30):
    """
    Create video from list of frames
    
    Args:
        frames: list of numpy arrays
        output_path: path to save video
        fps: frames per second
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def load_config_from_file(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def calculate_image_similarity(img1, img2):
    """
    Calculate similarity between two images using histogram comparison
    
    Args:
        img1, img2: numpy arrays
        
    Returns:
        float: similarity score (0-1)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Compare
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity

class PerformanceMonitor:
    """Monitor system performance"""
    def __init__(self):
        self.metrics = {
            'frame_times': [],
            'caption_times': [],
            'classify_times': [],
            'total_frames': 0
        }
    
    def log_frame_time(self, duration):
        """Log frame processing time"""
        self.metrics['frame_times'].append(duration)
        self.metrics['total_frames'] += 1
    
    def log_caption_time(self, duration):
        """Log caption generation time"""
        self.metrics['caption_times'].append(duration)
    
    def log_classify_time(self, duration):
        """Log classification time"""
        self.metrics['classify_times'].append(duration)
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        
        for key, times in self.metrics.items():
            if key == 'total_frames':
                stats[key] = times
            elif times:
                stats[f'{key}_avg'] = np.mean(times)
                stats[f'{key}_max'] = np.max(times)
                stats[f'{key}_min'] = np.min(times)
        
        return stats
