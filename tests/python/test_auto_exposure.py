#!/usr/bin/env python3
"""
Python tests for OpenCam Auto Exposure Algorithm
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock

# Mock implementations for when OpenCam is not available
class MockParameters:
    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'INTELLIGENT')
        self.target_brightness = kwargs.get('target_brightness', 0.5)
        self.convergence_speed = kwargs.get('convergence_speed', 0.15)

class MockAutoExposure:
    def __init__(self, parameters=None):
        self.parameters = parameters or MockParameters()
    
    def compute_exposure(self, image, **kwargs):
        # Simple mock: return inverse of brightness
        if isinstance(image, np.ndarray):
            brightness = np.mean(image) / 255.0
            return max(0.1, min(1.0, 0.5 / max(brightness, 0.1)))
        return 0.5
    
    def is_converged(self):
        return True
    
    def reset(self):
        pass

@pytest.fixture
def test_image():
    """Create a test image for testing"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def controller():
    """Create a controller (mock or real)"""
    try:
        from opencam import AutoExposure
        return AutoExposure()
    except ImportError:
        return MockAutoExposure()

class TestBasicFunctionality:
    """Test basic auto exposure functionality"""
    
    def test_controller_initialization(self, controller):
        """Test controller can be initialized"""
        assert controller is not None
    
    def test_compute_exposure_basic(self, controller, test_image):
        """Test basic exposure computation"""
        exposure = controller.compute_exposure(test_image)
        assert isinstance(exposure, float)
        assert 0.0 <= exposure <= 2.0
    
    def test_compute_exposure_different_images(self, controller):
        """Test exposure computation with different image types"""
        # Dark image
        dark_image = np.full((100, 100, 3), 50, dtype=np.uint8)
        dark_exposure = controller.compute_exposure(dark_image)
        
        # Bright image
        bright_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        bright_exposure = controller.compute_exposure(bright_image)
        
        # Dark images should generally get higher exposure
        assert dark_exposure >= bright_exposure
    
    def test_convergence_check(self, controller):
        """Test convergence detection"""
        converged = controller.is_converged()
        assert isinstance(converged, bool)
    
    def test_reset_functionality(self, controller):
        """Test controller reset"""
        controller.reset()  # Should not raise exception

class TestImageValidation:
    """Test image input validation"""
    
    def test_valid_image_formats(self, controller):
        """Test valid image formats are accepted"""
        # RGB image
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        exposure = controller.compute_exposure(rgb_image)
        assert isinstance(exposure, float)
        
        # Grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        exposure = controller.compute_exposure(gray_image)
        assert isinstance(exposure, float)
    
    def test_different_data_types(self, controller):
        """Test different numpy data types"""
        # uint8
        uint8_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        exposure = controller.compute_exposure(uint8_image)
        assert isinstance(exposure, float)
        
        # float32
        float32_image = np.random.rand(50, 50, 3).astype(np.float32)
        exposure = controller.compute_exposure(float32_image)
        assert isinstance(exposure, float)

class TestPerformance:
    """Test performance characteristics"""
    
    def test_processing_speed(self, controller):
        """Test processing speed is reasonable"""
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Measure time for multiple iterations
        start_time = time.perf_counter()
        iterations = 10
        
        for i in range(iterations):
            controller.compute_exposure(test_image)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / iterations
        
        # Should process faster than 100ms per frame
        assert avg_time < 0.1
    
    def test_memory_stability(self, controller):
        """Test memory usage doesn't grow excessively"""
        test_image = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
        
        # Process many frames - should not cause memory issues
        for i in range(100):
            exposure = controller.compute_exposure(test_image)
            assert isinstance(exposure, float)

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extreme_brightness_values(self, controller):
        """Test with extreme brightness values"""
        # Very dark image
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dark_exposure = controller.compute_exposure(dark_image)
        assert isinstance(dark_exposure, float)
        assert dark_exposure > 0
        
        # Very bright image
        bright_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        bright_exposure = controller.compute_exposure(bright_image)
        assert isinstance(bright_exposure, float)
        assert bright_exposure > 0
    
    def test_small_images(self, controller):
        """Test with very small images"""
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        exposure = controller.compute_exposure(tiny_image)
        assert isinstance(exposure, float)
    
    def test_large_images(self, controller):
        """Test with large images"""
        # Create a reasonably large image
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        exposure = controller.compute_exposure(large_image)
        assert isinstance(exposure, float)

class TestUtilityFunctions:
    """Test utility functions if available"""
    
    def test_create_test_frame(self):
        """Test test frame creation utility"""
        try:
            from opencam.utils import create_test_frame
            image = create_test_frame(320, 240, 0.5)
            assert image.shape == (240, 320, 3)
            assert image.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCam utils not available")
    
    def test_image_statistics(self):
        """Test image statistics computation"""
        try:
            from opencam.utils import compute_image_statistics
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            stats = compute_image_statistics(test_image)
            
            assert 'mean' in stats
            assert 'brightness' in stats
            assert isinstance(stats['mean'], (int, float))
            assert 0 <= stats['brightness'] <= 1.0
        except ImportError:
            pytest.skip("OpenCam utils not available")

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 