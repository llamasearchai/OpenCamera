"""
Benchmarking utilities for OpenCam Auto Exposure Algorithm
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import json

from .auto_exposure import AutoExposure, Parameters, MeteringMode
from .utils import create_test_frame, create_backlit_scene, create_low_light_scene

logger = logging.getLogger(__name__)

class AutoExposureBenchmark:
    """
    Comprehensive benchmarking suite for auto exposure algorithm.
    
    This class provides methods to benchmark performance, accuracy, and
    convergence behavior of the auto exposure algorithm.
    """
    
    def __init__(self, controller: Optional[AutoExposure] = None):
        """
        Initialize benchmark suite.
        
        Args:
            controller: Auto exposure controller to benchmark. If None, creates default.
        """
        self.controller = controller or AutoExposure()
        self.results = {}
        
    def benchmark_performance(self, 
                            resolutions: List[tuple] = None,
                            modes: List[MeteringMode] = None,
                            iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark processing performance across different configurations.
        
        Args:
            resolutions: List of (width, height) tuples to test
            modes: List of metering modes to test
            iterations: Number of iterations per configuration
            
        Returns:
            Dictionary with performance results
        """
        if resolutions is None:
            resolutions = [
                (320, 240),   # QVGA
                (640, 480),   # VGA
                (1280, 720),  # HD
                (1920, 1080), # Full HD
                (3840, 2160)  # 4K
            ]
        
        if modes is None:
            modes = [
                MeteringMode.AVERAGE,
                MeteringMode.CENTER_WEIGHTED,
                MeteringMode.SPOT,
                MeteringMode.MULTI_ZONE,
                MeteringMode.INTELLIGENT
            ]
        
        results = {
            'test_info': {
                'iterations': iterations,
                'resolutions': resolutions,
                'modes': [mode.value for mode in modes],
                'timestamp': time.time(),
            },
            'results': []
        }
        
        logger.info(f"Starting performance benchmark with {len(resolutions)} resolutions "
                   f"and {len(modes)} modes")
        
        for width, height in resolutions:
            for mode in modes:
                result = self._benchmark_single_config(width, height, mode, iterations)
                results['results'].append(result)
                
                logger.info(f"Completed {width}x{height} {mode.value}: "
                           f"{result['throughput_fps']:.1f} FPS")
        
        self.results['performance'] = results
        return results
    
    def _benchmark_single_config(self, width: int, height: int, 
                                mode: MeteringMode, iterations: int) -> Dict[str, Any]:
        """Benchmark a single configuration"""
        # Set parameters
        params = Parameters(mode=mode)
        self.controller.parameters = params
        self.controller.reset()
        
        # Create test image
        test_image = create_test_frame(width, height, 0.5)
        
        # Warm up
        for _ in range(min(10, iterations)):
            self.controller.compute_exposure(test_image)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            self.controller.compute_exposure(test_image, frame_number=i)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        return {
            'resolution': {'width': width, 'height': height},
            'mode': mode.value,
            'iterations': iterations,
            'timing': {
                'avg_seconds': avg_time,
                'std_seconds': std_time,
                'min_seconds': min_time,
                'max_seconds': max_time,
                'avg_ms': avg_time * 1000,
                'avg_us': avg_time * 1000000,
            },
            'throughput_fps': fps,
            'pixels_per_second': (width * height) * fps,
        }
    
    def benchmark_convergence(self, 
                            scene_types: List[str] = None,
                            brightness_levels: List[float] = None) -> Dict[str, Any]:
        """
        Benchmark convergence behavior for different scenes.
        
        Args:
            scene_types: List of scene types to test
            brightness_levels: List of brightness levels to test
            
        Returns:
            Dictionary with convergence results
        """
        if scene_types is None:
            scene_types = ['uniform', 'backlit', 'low_light', 'high_contrast']
        
        if brightness_levels is None:
            brightness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {
            'test_info': {
                'scene_types': scene_types,
                'brightness_levels': brightness_levels,
                'timestamp': time.time(),
            },
            'results': []
        }
        
        logger.info(f"Starting convergence benchmark with {len(scene_types)} scene types "
                   f"and {len(brightness_levels)} brightness levels")
        
        for scene_type in scene_types:
            for brightness in brightness_levels:
                result = self._benchmark_convergence_single(scene_type, brightness)
                results['results'].append(result)
                
                logger.info(f"Convergence {scene_type} @ {brightness:.1f}: "
                           f"{result['frames_to_convergence']} frames, "
                           f"{result['convergence_time_ms']:.1f}ms")
        
        self.results['convergence'] = results
        return results
    
    def _benchmark_convergence_single(self, scene_type: str, brightness: float) -> Dict[str, Any]:
        """Benchmark convergence for a single scene configuration"""
        # Reset controller
        params = Parameters(mode=MeteringMode.INTELLIGENT)
        self.controller.parameters = params
        self.controller.reset()
        
        # Create test scene
        if scene_type == 'uniform':
            image = create_test_frame(640, 480, brightness)
        elif scene_type == 'backlit':
            image = create_backlit_scene(640, 480)
        elif scene_type == 'low_light':
            image = create_low_light_scene(640, 480)
        elif scene_type == 'high_contrast':
            image = create_test_frame(640, 480, brightness, pattern='checkerboard')
        else:
            image = create_test_frame(640, 480, brightness)
        
        # Track convergence
        start_time = time.perf_counter()
        frame_count = 0
        max_frames = 100
        
        exposures = []
        while frame_count < max_frames:
            exposure = self.controller.compute_exposure(image, frame_number=frame_count)
            exposures.append(exposure)
            frame_count += 1
            
            if self.controller.is_converged():
                break
        
        end_time = time.perf_counter()
        convergence_time = (end_time - start_time) * 1000  # ms
        
        # Analyze convergence behavior
        exposures = np.array(exposures)
        final_exposure = exposures[-1] if exposures.size > 0 else 0.0
        
        # Calculate settling characteristics
        if len(exposures) > 5:
            # Look for when exposure stabilizes (last 5 frames have low variance)
            for i in range(5, len(exposures)):
                recent_exposures = exposures[i-5:i]
                if np.std(recent_exposures) < 0.001:  # Stable
                    settled_frame = i
                    break
            else:
                settled_frame = len(exposures)
        else:
            settled_frame = len(exposures)
        
        return {
            'scene_type': scene_type,
            'brightness': brightness,
            'frames_to_convergence': frame_count,
            'frames_to_settle': settled_frame,
            'convergence_time_ms': convergence_time,
            'final_exposure': final_exposure,
            'exposure_trajectory': exposures.tolist(),
            'converged': self.controller.is_converged(),
        }
    
    def benchmark_accuracy(self, test_images: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Benchmark exposure accuracy against reference values.
        
        Args:
            test_images: List of test images with known optimal exposures
            
        Returns:
            Dictionary with accuracy results
        """
        if test_images is None:
            # Create standard test images
            test_images = [
                create_test_frame(640, 480, 0.2),  # Dark
                create_test_frame(640, 480, 0.5),  # Normal
                create_test_frame(640, 480, 0.8),  # Bright
                create_backlit_scene(640, 480),    # Backlit
                create_low_light_scene(640, 480), # Low light
            ]
        
        results = {
            'test_info': {
                'num_images': len(test_images),
                'timestamp': time.time(),
            },
            'results': []
        }
        
        logger.info(f"Starting accuracy benchmark with {len(test_images)} test images")
        
        for i, image in enumerate(test_images):
            result = self._benchmark_accuracy_single(image, i)
            results['results'].append(result)
        
        # Calculate overall accuracy metrics
        exposures = [r['computed_exposure'] for r in results['results']]
        brightnesses = [r['measured_brightness'] for r in results['results']]
        
        results['summary'] = {
            'mean_exposure': np.mean(exposures),
            'std_exposure': np.std(exposures),
            'mean_brightness': np.mean(brightnesses),
            'std_brightness': np.std(brightnesses),
            'exposure_range': [np.min(exposures), np.max(exposures)],
            'brightness_range': [np.min(brightnesses), np.max(brightnesses)],
        }
        
        self.results['accuracy'] = results
        return results
    
    def _benchmark_accuracy_single(self, image: np.ndarray, image_id: int) -> Dict[str, Any]:
        """Benchmark accuracy for a single image"""
        # Reset and process
        self.controller.reset()
        
        # Process until convergence
        exposure = None
        for frame_num in range(50):  # Max 50 frames
            exposure = self.controller.compute_exposure(image, frame_number=frame_num)
            if self.controller.is_converged():
                break
        
        # Get analysis results
        stats = self.controller.get_statistics()
        scene_analysis = self.controller.get_scene_analysis()
        
        return {
            'image_id': image_id,
            'image_shape': image.shape,
            'computed_exposure': exposure,
            'measured_brightness': stats.average_brightness,
            'frames_processed': stats.frame_count,
            'converged': stats.is_converged,
            'scene_analysis': {
                'scene_type': scene_analysis.scene_type,
                'confidence': scene_analysis.confidence,
                'is_low_light': scene_analysis.is_low_light,
                'is_backlit': scene_analysis.is_backlit,
                'is_high_contrast': scene_analysis.is_high_contrast,
            }
        }
    
    def benchmark_memory_usage(self, duration_seconds: float = 10.0) -> Dict[str, Any]:
        """
        Benchmark memory usage over time.
        
        Args:
            duration_seconds: How long to run the test
            
        Returns:
            Dictionary with memory usage results
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            'test_info': {
                'duration_seconds': duration_seconds,
                'baseline_memory_mb': baseline_memory,
                'timestamp': time.time(),
            },
            'memory_samples': [],
            'processing_stats': {'frames_processed': 0}
        }
        
        logger.info(f"Starting memory benchmark for {duration_seconds} seconds")
        
        # Create test image
        test_image = create_test_frame(1280, 720, 0.5)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Process frame
            self.controller.compute_exposure(test_image, frame_number=frame_count)
            frame_count += 1
            
            # Sample memory every 100 frames
            if frame_count % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                results['memory_samples'].append({
                    'frame': frame_count,
                    'time': time.time() - start_time,
                    'memory_mb': current_memory,
                    'memory_delta_mb': current_memory - baseline_memory,
                })
        
        results['processing_stats']['frames_processed'] = frame_count
        
        # Calculate memory statistics
        if results['memory_samples']:
            memory_values = [s['memory_mb'] for s in results['memory_samples']]
            memory_deltas = [s['memory_delta_mb'] for s in results['memory_samples']]
            
            results['summary'] = {
                'max_memory_mb': max(memory_values),
                'min_memory_mb': min(memory_values),
                'avg_memory_mb': np.mean(memory_values),
                'max_memory_delta_mb': max(memory_deltas),
                'avg_memory_delta_mb': np.mean(memory_deltas),
                'memory_leak_detected': max(memory_deltas) > 50,  # > 50MB increase
            }
        
        self.results['memory'] = results
        return results
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Starting full benchmark suite")
        
        suite_start_time = time.time()
        
        # Run all benchmarks
        self.benchmark_performance()
        self.benchmark_convergence()
        self.benchmark_accuracy()
        self.benchmark_memory_usage()
        
        suite_end_time = time.time()
        
        # Compile results
        full_results = {
            'suite_info': {
                'start_time': suite_start_time,
                'end_time': suite_end_time,
                'duration_seconds': suite_end_time - suite_start_time,
                'timestamp': time.time(),
            },
            'benchmarks': self.results
        }
        
        logger.info(f"Full benchmark suite completed in {suite_end_time - suite_start_time:.1f} seconds")
        
        return full_results
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def load_results(self, filepath: str):
        """
        Load benchmark results from JSON file.
        
        Args:
            filepath: Path to load results from
        """
        try:
            with open(filepath, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Benchmark results loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def generate_report(self) -> str:
        """
        Generate a text report of benchmark results.
        
        Returns:
            Formatted text report
        """
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report = []
        report.append("OpenCam Auto Exposure Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Performance summary
        if 'performance' in self.results:
            perf_results = self.results['performance']['results']
            report.append("Performance Summary:")
            report.append("-" * 20)
            
            for result in perf_results:
                res = result['resolution']
                report.append(f"{res['width']}x{res['height']} {result['mode']}: "
                             f"{result['throughput_fps']:.1f} FPS "
                             f"({result['timing']['avg_ms']:.2f}ms)")
            report.append("")
        
        # Convergence summary
        if 'convergence' in self.results:
            conv_results = self.results['convergence']['results']
            report.append("Convergence Summary:")
            report.append("-" * 20)
            
            for result in conv_results:
                report.append(f"{result['scene_type']} @ {result['brightness']:.1f}: "
                             f"{result['frames_to_convergence']} frames "
                             f"({result['convergence_time_ms']:.1f}ms)")
            report.append("")
        
        # Memory summary
        if 'memory' in self.results and 'summary' in self.results['memory']:
            mem_summary = self.results['memory']['summary']
            report.append("Memory Summary:")
            report.append("-" * 15)
            report.append(f"Max memory: {mem_summary['max_memory_mb']:.1f} MB")
            report.append(f"Avg memory: {mem_summary['avg_memory_mb']:.1f} MB")
            report.append(f"Memory delta: {mem_summary['avg_memory_delta_mb']:.1f} MB")
            report.append(f"Leak detected: {mem_summary['memory_leak_detected']}")
            report.append("")
        
        return "\n".join(report) 