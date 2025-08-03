#!/usr/bin/env python3
"""
Script to automatically update cSpell configuration with OpenCV functions
found in the codebase.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import json
import re
import os
import glob
from pathlib import Path
from typing import Set, List, Dict, Any


def find_opencv_functions(directory: str = ".") -> Set[str]:
    """Find all OpenCV function names used in the codebase."""
    opencv_functions = set()
    
    # Patterns to match OpenCV functions and constants
    patterns = [
        r'cv::([a-zA-Z_][a-zA-Z0-9_]*)',  # cv::function_name
        r'CV_([A-Z0-9_]+)',               # CV_CONSTANT
        r'CAP_PROP_([A-Z_]+)',           # CAP_PROP_PROPERTY
        r'DNN_([A-Z_]+)',                # DNN_CONSTANT
        r'INTER_([A-Z_]+)',              # INTER_CONSTANT
        r'COLOR_([A-Z0-9_]+)',           # COLOR_CONSTANT
        r'NORM_([A-Z_]+)',               # NORM_CONSTANT
        r'WINDOW_([A-Z_]+)',             # WINDOW_CONSTANT
        r'FONT_([A-Z_]+)',               # FONT_CONSTANT
    ]
    
    # File extensions to search
    extensions = ['*.cpp', '*.h', '*.hpp', '*.cc', '*.c']
    
    for ext in extensions:
        for file_path in glob.glob(f"{directory}/**/{ext}", recursive=True):
            # Skip build directories and generated files
            if any(skip_dir in file_path for skip_dir in ['build', 'htmlcov', '__pycache__', '.pytest_cache']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if pattern.startswith('cv::'):
                            opencv_functions.add(match)
                        else:
                            # Reconstruct the full constant name
                            prefix = pattern.split('(')[0].split('_')[0]
                            opencv_functions.add(f"{prefix}_{match}")
                            
            except (UnicodeDecodeError, IOError):
                continue
    
    return opencv_functions


def load_cspell_config(config_path: str = "cspell.json") -> Dict[str, Any]:
    """Load existing cSpell configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "version": "0.2",
            "language": "en",
            "words": [],
            "ignoreWords": [],
            "patterns": [],
            "ignorePaths": []
        }


def update_cspell_config(opencv_functions: Set[str], config_path: str = "cspell.json"):
    """Update cSpell configuration with new OpenCV functions."""
    config = load_cspell_config(config_path)
    
    # Get existing words
    existing_words = set(config.get("words", []))
    
    # Add new OpenCV functions
    new_words = existing_words.union(opencv_functions)
    
    # Sort words for consistent output
    config["words"] = sorted(list(new_words))
    
    # Ensure we have the basic ignore paths
    if "ignorePaths" not in config:
        config["ignorePaths"] = [
            "build*/**",
            "htmlcov/**",
            "*.egg-info/**",
            "__pycache__/**",
            ".pytest_cache/**",
            ".benchmarks/**",
            "Testing/**",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.exe"
        ]
    
    # Ensure we have the basic patterns
    if "patterns" not in config:
        config["patterns"] = [
            {
                "name": "cpp-includes",
                "pattern": "#include\\s*<[^>]+>",
                "description": "C++ include statements"
            },
            {
                "name": "cpp-namespace",
                "pattern": "cv::[a-zA-Z_][a-zA-Z0-9_]*",
                "description": "OpenCV namespace functions"
            },
            {
                "name": "cpp-constants",
                "pattern": "CV_[A-Z0-9_]+",
                "description": "OpenCV constants"
            },
            {
                "name": "cpp-cap-props",
                "pattern": "CAP_PROP_[A-Z_]+",
                "description": "OpenCV capture properties"
            }
        ]
    
    # Write updated configuration
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    
    return len(opencv_functions - existing_words)


def main():
    """Main function to update cSpell configuration."""
    print("Scanning codebase for OpenCV functions...")
    
    # Find OpenCV functions
    opencv_functions = find_opencv_functions()
    
    print(f"Found {len(opencv_functions)} OpenCV functions/constants:")
    for func in sorted(opencv_functions):
        print(f"  - {func}")
    
    # Update cSpell configuration
    new_count = update_cspell_config(opencv_functions)
    
    print(f"\nUpdated cSpell configuration with {new_count} new functions.")
    print("Configuration saved to cspell.json")


if __name__ == "__main__":
    main() 