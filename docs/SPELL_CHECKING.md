# Spell Checking Configuration

This document describes the spell checking setup for the OpenCam project.

## Overview

The project uses cSpell (Code Spell Checker) to maintain consistent spelling across the codebase. This is particularly important for a computer vision project that uses many OpenCV functions and technical terms.

## Configuration Files

### Root Configuration (`cspell.json`)
The main cSpell configuration file is located at the root of the project. It contains:
- Allowed technical terms and OpenCV function names
- Patterns to ignore specific code constructs
- Paths to exclude from spell checking

### VS Code Settings (`.vscode/settings.json`)
VS Code-specific settings that reference the root configuration and enable spell checking for various file types.

## OpenCV Functions

The configuration includes a comprehensive list of OpenCV functions and constants used throughout the codebase, including:

- Core functions: `Mat`, `Scalar`, `Rect`, `Point`, `Size`
- Image processing: `cvtColor`, `resize`, `GaussianBlur`, `normalize`
- Drawing functions: `rectangle`, `circle`, `line`, `putText`
- Window management: `imshow`, `waitKey`, `namedWindow`
- Constants: `CV_8UC3`, `COLOR_BGR2GRAY`, `INTER_AREA`

## Maintenance

### Automatic Updates
Use the provided script to automatically detect and add new OpenCV functions:

```bash
python scripts/update_cspell.py
```

This script:
1. Scans all C++ source files for OpenCV function usage
2. Extracts function names using regex patterns
3. Updates the cSpell configuration with new functions
4. Preserves existing configuration

### Manual Updates
To manually add new terms:

1. Edit `cspell.json` and add terms to the `words` array
2. Keep the array sorted alphabetically for consistency
3. Test by running the spell checker in VS Code

## Ignored Patterns

The configuration ignores:
- C++ include statements (`#include <...>`)
- OpenCV namespace functions (`cv::function_name`)
- OpenCV constants (`CV_CONSTANT`)
- Build artifacts and generated files

## File Types

Spell checking is enabled for:
- C++ source files (`.cpp`, `.h`, `.hpp`, `.cc`, `.c`)
- Python files (`.py`)
- Documentation (`.md`, `.txt`)
- Configuration files (`.json`, `.cmake`)

## Troubleshooting

### False Positives
If you encounter false positives:

1. Add the term to the `words` array in `cspell.json`
2. Use inline comments to suppress specific instances:
   ```cpp
   // cspell:ignore specificterm
   ```

### Missing OpenCV Functions
If new OpenCV functions are flagged:

1. Run the update script: `python scripts/update_cspell.py`
2. If the function isn't detected, manually add it to `cspell.json`

## Best Practices

1. **Keep the configuration updated**: Run the update script regularly
2. **Use consistent naming**: Follow OpenCV naming conventions
3. **Document additions**: Add comments for non-obvious terms
4. **Test changes**: Verify spell checking works after configuration changes

## Author

- **Author**: Nik Jois
- **Email**: nikjois@llamasearch.ai

## Related Files

- `cspell.json` - Main configuration
- `.vscode/settings.json` - VS Code settings
- `scripts/update_cspell.py` - Automatic update script 