"""
Utility functions for OpenCam Auto Exposure Python package
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Union
import logging

try:
    import opencam_auto_exposure as _cpp
except ImportError:
    _cpp = None

logger = logging.getLogger(__name__)

def create_test_frame(width: int, height: int, brightness: float = 0.5, 
                     pattern: str = "uniform") -> np.ndarray:
    """
    Create a test image with specified properties.
    
    Args:
        width: Image width
        height: Image height
        brightness: Brightness level (0.0-1.0)
        pattern: Pattern type ("uniform", "gradient", "checkerboard", "noise")
        
    Returns:
        Test image as numpy array
    """
    if not 0.0 <= brightness <= 1.0:
        raise ValueError("Brightness must be between 0.0 and 1.0")
    
    if pattern == "uniform":
        image = np.full((height, width, 3), int(brightness * 255), dtype=np.uint8)
    
    elif pattern == "gradient":
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int((y / height) * brightness * 255)
            image[y, :, :] = intensity
    
    elif pattern == "checkerboard":
        image = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = min(width, height) // 8
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    intensity = int(brightness * 255)
                else:
                    intensity = int(brightness * 128)
                y_end = min(y + square_size, height)
                x_end = min(x + square_size, width)
                image[y:y_end, x:x_end, :] = intensity
    
    elif pattern == "noise":
        mean = brightness * 255
        std = 30
        image = np.random.normal(mean, std, (height, width, 3))
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return image

def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        target_size: Optional target size (width, height)
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image cannot be loaded
    """
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if target_size:
            image = cv2.resize(image, target_size)
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        raise

def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        path: Output path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        success = cv2.imwrite(path, image_bgr)
        if success:
            logger.info(f"Image saved to {path}")
        else:
            logger.error(f"Failed to save image to {path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Exception saving image to {path}: {e}")
        return False

def validate_image(image: np.ndarray) -> bool:
    """
    Validate that an image array is suitable for processing.
    
    Args:
        image: Image to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if image.ndim not in [2, 3]:
        return False
    
    if image.ndim == 3 and image.shape[2] not in [1, 3]:
        return False
    
    if image.size == 0:
        return False
    
    return True

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 format.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image
    
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            # Assume normalized float
            return (image * 255).astype(np.uint8)
        else:
            # Assume unnormalized float
            return np.clip(image, 0, 255).astype(np.uint8)
    
    # For other integer types
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    
    return image.astype(np.uint8)

def create_backlit_scene(width: int, height: int) -> np.ndarray:
    """
    Create a synthetic backlit scene for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Backlit test image
    """
    image = np.full((height, width, 3), 240, dtype=np.uint8)  # Bright background
    
    # Dark subject in center
    center_x, center_y = width // 2, height // 2
    subject_w, subject_h = width // 3, height // 3
    
    x1 = center_x - subject_w // 2
    x2 = center_x + subject_w // 2
    y1 = center_y - subject_h // 2
    y2 = center_y + subject_h // 2
    
    image[y1:y2, x1:x2, :] = 50  # Dark subject
    
    return image

def create_low_light_scene(width: int, height: int) -> np.ndarray:
    """
    Create a synthetic low light scene for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Low light test image
    """
    # Base low light level
    mean_intensity = 30
    noise_std = 15
    
    # Generate noisy low light image
    image = np.random.normal(mean_intensity, noise_std, (height, width, 3))
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

def create_high_contrast_scene(width: int, height: int) -> np.ndarray:
    """
    Create a synthetic high contrast scene for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        High contrast test image
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create random bright and dark regions
    np.random.seed(42)  # For reproducible results
    
    num_regions = 20
    for i in range(num_regions):
        x = np.random.randint(0, width - width // 4)
        y = np.random.randint(0, height - height // 4)
        w = np.random.randint(width // 8, width // 4)
        h = np.random.randint(height // 8, height // 4)
        
        intensity = 220 if i % 2 == 0 else 30
        image[y:y+h, x:x+w, :] = intensity
    
    return image

def compute_image_statistics(image: np.ndarray) -> dict:
    """
    Compute basic statistics for an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image statistics
    """
    if not validate_image(image):
        raise ValueError("Invalid image format")
    
    # Convert to grayscale for analysis
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    stats = {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': int(np.min(gray)),
        'max': int(np.max(gray)),
        'median': float(np.median(gray)),
        'brightness': float(np.mean(gray) / 255.0),
        'contrast': float(np.std(gray) / 255.0),
    }
    
    # Histogram statistics
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()  # Normalize
    
    # Check for low light (lots of dark pixels)
    stats['low_light_ratio'] = float(np.sum(hist[:64]))
    
    # Check for overexposure (lots of bright pixels)
    stats['overexposed_ratio'] = float(np.sum(hist[240:]))
    
    # Check for high contrast (bimodal distribution)
    dark_peak = np.sum(hist[:85])
    bright_peak = np.sum(hist[170:])
    stats['high_contrast'] = dark_peak > 0.3 and bright_peak > 0.3
    
    return stats

def benchmark_image_processing(images: list, processor_func, iterations: int = 100) -> dict:
    """
    Benchmark image processing function performance.
    
    Args:
        images: List of test images
        processor_func: Function to benchmark
        iterations: Number of iterations per image
        
    Returns:
        Benchmark results dictionary
    """
    import time
    
    if not images:
        raise ValueError("Images list cannot be empty")
    
    results = []
    
    for i, image in enumerate(images):
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                processor_func(image)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Processing failed for image {i}: {e}")
                continue
        
        if times:
            results.append({
                'image_index': i,
                'image_shape': image.shape,
                'iterations': len(times),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'fps': 1.0 / np.mean(times),
            })
    
    # Overall statistics
    all_times = [r['mean_time'] for r in results]
    all_fps = [r['fps'] for r in results]
    
    summary = {
        'total_images': len(images),
        'successful_images': len(results),
        'iterations_per_image': iterations,
        'overall_mean_time': np.mean(all_times) if all_times else 0,
        'overall_fps': np.mean(all_fps) if all_fps else 0,
        'per_image_results': results,
    }
    
    return summary 