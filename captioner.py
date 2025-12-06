"""
Optimized BLIP Image Captioning with caching and GPU support
"""
import torch
import hashlib
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from functools import lru_cache
import config

class OptimizedCaptioner:
    def __init__(self, model_name=config.BLIP_MODEL_NAME):
        """Initialize BLIP model with GPU support"""
        print(f"Loading BLIP model: {model_name}")
        
        # Set device
        self.device = "cuda" if config.ENABLE_GPU and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)

        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Enable half precision on GPU for faster inference
        if self.device == "cuda":
            self.model = self.model.half()
        
        # Cache for similar images
        self.caption_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        print("BLIP model loaded successfully")
    
    def _compute_image_hash(self, image_array):
        """Compute hash for image caching"""
        # Downsample image for faster hashing
        small_image = Image.fromarray(image_array).resize((32, 32))
        image_bytes = np.array(small_image).tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _check_cache(self, image_hash):
        """Check if caption exists in cache"""
        if not config.ENABLE_CAPTION_CACHE:
            return None
        
        if image_hash in self.caption_cache:
            self.cache_hits += 1
            return self.caption_cache[image_hash]
        
        self.cache_misses += 1
        return None
    
    def _update_cache(self, image_hash, caption):
        """Update caption cache with LRU policy"""
        if not config.ENABLE_CAPTION_CACHE:
            return
        
        # Simple LRU: remove oldest if cache is full
        if len(self.caption_cache) >= config.CACHE_SIZE:
            oldest_key = next(iter(self.caption_cache))
            del self.caption_cache[oldest_key]
        
        self.caption_cache[image_hash] = caption
    
    @torch.no_grad()  # Disable gradient computation for inference
    def generate_caption(self, image_array):
        """
        Generate caption for an image with caching
        
        Args:
            image_array: numpy array (BGR format from OpenCV)
            
        Returns:
            caption: string
        """
        # Check cache first
        image_hash = self._compute_image_hash(image_array)
        cached_caption = self._check_cache(image_hash)
        
        if cached_caption is not None:
            return cached_caption
        
        # Convert BGR to RGB
        image_rgb = Image.fromarray(image_array[:, :, ::-1])
        
        # Process image
        inputs = self.processor(image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Convert to half precision if using GPU
        if self.device == "cuda":
            inputs['pixel_values'] = inputs['pixel_values'].half()
        
        # Generate caption
        outputs = self.model.generate(
            **inputs,
            max_length=50,
            num_beams=3,  # Reduced from 5 for speed
            early_stopping=True
        )
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Update cache
        self._update_cache(image_hash, caption)
        
        return caption
    
    def get_cache_stats(self):
        """Return cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.caption_cache)
        }
