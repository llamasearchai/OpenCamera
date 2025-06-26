#!/usr/bin/env python3
"""
Benchmark comparison script for Auto Exposure performance analysis.
Compares current benchmark results with baseline to detect performance regressions.
"""

import json
import sys
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BenchmarkResult:
    test_name: str
    resolution: Tuple[int, int]
    mode: str
    avg_time_us: float
    throughput_fps: float
    iterations: int

class BenchmarkComparator:
    def __init__(self, baseline_file: str, current_file: str):
        self.baseline_file = Path(baseline_file)
        self.current_file = Path(current_file)
        self.baseline_results = {}
        self.current_results = {}
        self.regression_threshold = 0.1  # 10% performance regression threshold
        self.improvement_threshold = 0.05  # 5% improvement threshold
        
    def load_results(self) -> bool:
        """Load benchmark results from JSON files."""
        try:
            # Load baseline results
            if self.baseline_file.exists():
                with open(self.baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    self.baseline_results = self._parse_results(baseline_data)
            else:
                print(f"Warning: Baseline file {self.baseline_file} not found")
                return False
            
            # Load current results
            if self.current_file.exists():
                with open(self.current_file, 'r') as f:
                    current_data = json.load(f)
                    self.current_results = self._parse_results(current_data)
            else:
                print(f"Error: Current results file {self.current_file} not found")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error loading benchmark results: {e}")
            return False
    
    def _parse_results(self, data: Dict) -> Dict[str, BenchmarkResult]:
        """Parse JSON data into BenchmarkResult objects."""
        results = {}
        
        for result in data.get('results', []):
            test_name = result['test_name']
            resolution = (result['resolution']['width'], result['resolution']['height'])
            mode = result['mode']
            avg_time_us = result['timing']['avg_us']
            throughput_fps = result['throughput_fps']
            iterations = result['iterations']
            
            benchmark_result = BenchmarkResult(
                test_name=test_name,
                resolution=resolution,
                mode=mode,
                avg_time_us=avg_time_us,
                throughput_fps=throughput_fps,
                iterations=iterations
            )
            
            results[test_name] = benchmark_result
            
        return results
    
    def compare_results(self) -> Dict:
        """Compare current results with baseline and identify regressions/improvements."""
        comparison = {
            'regressions': [],
            'improvements': [],
            'new_tests': [],
            'missing_tests': [],
            'summary': {}
        }
        
        # Find regressions and improvements
        for test_name, current in self.current_results.items():
            if test_name in self.baseline_results:
                baseline = self.baseline_results[test_name]
                
                # Calculate performance change
                fps_change = (current.throughput_fps - baseline.throughput_fps) / baseline.throughput_fps
                time_change = (current.avg_time_us - baseline.avg_time_us) / baseline.avg_time_us
                
                if fps_change < -self.regression_threshold:
                    comparison['regressions'].append({
                        'test_name': test_name,
                        'baseline_fps': baseline.throughput_fps,
                        'current_fps': current.throughput_fps,
                        'fps_change_percent': fps_change * 100,
                        'baseline_time_us': baseline.avg_time_us,
                        'current_time_us': current.avg_time_us,
                        'time_change_percent': time_change * 100
                    })
                elif fps_change > self.improvement_threshold:
                    comparison['improvements'].append({
                        'test_name': test_name,
                        'baseline_fps': baseline.throughput_fps,
                        'current_fps': current.throughput_fps,
                        'fps_change_percent': fps_change * 100,
                        'baseline_time_us': baseline.avg_time_us,
                        'current_time_us': current.avg_time_us,
                        'time_change_percent': time_change * 100
                    })
            else:
                comparison['new_tests'].append(test_name)
        
        # Find missing tests
        for test_name in self.baseline_results:
            if test_name not in self.current_results:
                comparison['missing_tests'].append(test_name)
        
        # Generate summary statistics
        comparison['summary'] = self._generate_summary(comparison)
        
        return comparison
    
    def _generate_summary(self, comparison: Dict) -> Dict:
        """Generate summary statistics for the comparison."""
        total_tests = len(self.current_results)
        regressions_count = len(comparison['regressions'])
        improvements_count = len(comparison['improvements'])
        new_tests_count = len(comparison['new_tests'])
        missing_tests_count = len(comparison['missing_tests'])
        
        # Calculate average performance change
        all_changes = []
        for test_name, current in self.current_results.items():
            if test_name in self.baseline_results:
                baseline = self.baseline_results[test_name]
                fps_change = (current.throughput_fps - baseline.throughput_fps) / baseline.throughput_fps
                all_changes.append(fps_change)
        
        avg_change = sum(all_changes) / len(all_changes) if all_changes else 0
        
        return {
            'total_tests': total_tests,
            'regressions_count': regressions_count,
            'improvements_count': improvements_count,
            'new_tests_count': new_tests_count,
            'missing_tests_count': missing_tests_count,
            'avg_performance_change_percent': avg_change * 100,
            'regression_rate_percent': (regressions_count / total_tests * 100) if total_tests > 0 else 0,
            'improvement_rate_percent': (improvements_count / total_tests * 100) if total_tests > 0 else 0
        }
    
    def print_comparison_report(self, comparison: Dict):
        """Print a detailed comparison report."""
        print("=" * 80)
        print("BENCHMARK COMPARISON REPORT")
        print("=" * 80)
        
        # Summary
        summary = comparison['summary']
        print(f"\nSUMMARY:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Regressions: {summary['regressions_count']} ({summary['regression_rate_percent']:.1f}%)")
        print(f"  Improvements: {summary['improvements_count']} ({summary['improvement_rate_percent']:.1f}%)")
        print(f"  New tests: {summary['new_tests_count']}")
        print(f"  Missing tests: {summary['missing_tests_count']}")
        print(f"  Average performance change: {summary['avg_performance_change_percent']:+.2f}%")
        
        # Regressions
        if comparison['regressions']:
            print(f"\n🔴 PERFORMANCE REGRESSIONS ({len(comparison['regressions'])}):")
            print("-" * 80)
            for reg in comparison['regressions']:
                print(f"  {reg['test_name']}")
                print(f"    FPS: {reg['baseline_fps']:.1f} → {reg['current_fps']:.1f} ({reg['fps_change_percent']:+.1f}%)")
                print(f"    Time: {reg['baseline_time_us']:.1f}μs → {reg['current_time_us']:.1f}μs ({reg['time_change_percent']:+.1f}%)")
                print()
        
        # Improvements
        if comparison['improvements']:
            print(f"\n🟢 PERFORMANCE IMPROVEMENTS ({len(comparison['improvements'])}):")
            print("-" * 80)
            for imp in comparison['improvements']:
                print(f"  {imp['test_name']}")
                print(f"    FPS: {imp['baseline_fps']:.1f} → {imp['current_fps']:.1f} ({imp['fps_change_percent']:+.1f}%)")
                print(f"    Time: {imp['baseline_time_us']:.1f}μs → {imp['current_time_us']:.1f}μs ({imp['time_change_percent']:+.1f}%)")
                print()
        
        # New tests
        if comparison['new_tests']:
            print(f"\n🆕 NEW TESTS ({len(comparison['new_tests'])}):")
            print("-" * 80)
            for test in comparison['new_tests']:
                current = self.current_results[test]
                print(f"  {test}: {current.throughput_fps:.1f} FPS")
        
        # Missing tests
        if comparison['missing_tests']:
            print(f"\n⚠️  MISSING TESTS ({len(comparison['missing_tests'])}):")
            print("-" * 80)
            for test in comparison['missing_tests']:
                baseline = self.baseline_results[test]
                print(f"  {test}: was {baseline.throughput_fps:.1f} FPS")
        
        print("=" * 80)
    
    def save_comparison_report(self, comparison: Dict, output_file: str):
        """Save comparison results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"Comparison report saved to {output_file}")
        except Exception as e:
            print(f"Error saving comparison report: {e}")
    
    def check_for_critical_regressions(self, comparison: Dict) -> bool:
        """Check if there are critical performance regressions."""
        critical_threshold = 0.25  # 25% regression is considered critical
        
        critical_regressions = [
            reg for reg in comparison['regressions']
            if reg['fps_change_percent'] < -critical_threshold * 100
        ]
        
        if critical_regressions:
            print(f"\n🚨 CRITICAL REGRESSIONS DETECTED ({len(critical_regressions)}):")
            print("=" * 80)
            for reg in critical_regressions:
                print(f"  {reg['test_name']}: {reg['fps_change_percent']:.1f}% performance loss")
            print("=" * 80)
            return True
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('baseline', help='Baseline benchmark JSON file')
    parser.add_argument('current', help='Current benchmark JSON file')
    parser.add_argument('--output', '-o', help='Output comparison report file')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                       help='Regression threshold (default: 0.1 = 10%)')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with error code if regressions are found')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = BenchmarkComparator(args.baseline, args.current)
    comparator.regression_threshold = args.threshold
    
    # Load and compare results
    if not comparator.load_results():
        sys.exit(1)
    
    comparison = comparator.compare_results()
    
    # Print report
    comparator.print_comparison_report(comparison)
    
    # Save report if requested
    if args.output:
        comparator.save_comparison_report(comparison, args.output)
    
    # Check for critical regressions
    has_critical_regressions = comparator.check_for_critical_regressions(comparison)
    
    # Exit with error if regressions found and flag is set
    if args.fail_on_regression and (comparison['regressions'] or has_critical_regressions):
        print("\nExiting with error due to performance regressions.")
        sys.exit(1)
    
    # Exit with warning code if there are any regressions
    if comparison['regressions']:
        sys.exit(2)  # Warning exit code
    
    print("\nNo performance regressions detected.")
    sys.exit(0)

if __name__ == '__main__':
    main()