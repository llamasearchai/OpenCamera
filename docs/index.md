# OpenCam Auto Exposure

A high-performance, intelligent auto exposure system for computer vision applications.

**Author:** Nik Jois <nikjois@llamasearch.ai>

## Overview

OpenCam Auto Exposure is a comprehensive C++ library that provides advanced automatic exposure control for camera systems. It features multiple metering modes, intelligent scene analysis, face detection, and real-time performance optimization.

## Features

### Multiple Metering Modes
- **Average**: Traditional average brightness metering
- **Center-weighted**: Prioritizes center of frame
- **Spot**: Precise spot metering
- **Multi-zone**: Advanced multi-zone analysis
- **Intelligent**: AI-powered scene analysis

### AI-Powered Features
- Machine learning-based scene classification
- Face detection and priority metering
- Backlit scene detection
- Low-light optimization

### High Performance
- Real-time processing up to 12,000+ FPS
- Optimized for multiple resolutions
- Thread-safe implementation
- Memory-efficient design

### Developer Friendly
- Comprehensive C++ API
- Python bindings
- Extensive test suite
- Detailed documentation
- Command-line demo application

## Quick Start

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-username/opencam.git
cd opencam

# Build the project
make build

# Run tests
make test

# Run benchmarks
make benchmark
```

### Using the Demo Application

```bash
# Show help
./build/auto_exposure_demo --help

# Run with default camera
./build/auto_exposure_demo

# Run with specific camera and debug logging
./build/auto_exposure_demo --debug 1
```

### Basic C++ Usage

```cpp
#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"

using namespace opencam;
using namespace opencam::algorithms;

// Initialize auto exposure controller
AutoExposureController controller;

// Configure parameters
AutoExposureController::Parameters params;
params.mode = AutoExposureController::Mode::INTELLIGENT;
params.targetBrightness = 0.5f;
params.convergenceSpeed = 0.15f;
controller.setParameters(params);

// Process frames
while (auto frame = camera->getNextFrame()) {
    float exposure = controller.computeExposure(*frame);
    controller.applyToCamera(*camera);
}
```

## Performance

### Benchmark Results

| Resolution | Mode | Performance |
|------------|------|-------------|
| 320x240 | Average | 12,648 FPS |
| 640x480 | Intelligent | 1,889 FPS |
| 1920x1080 | Spot | 471 FPS |
| 3840x2160 | Center-weighted | 51 FPS |

### Real-time Capability

All metering modes achieve real-time performance (≥30 FPS) across common resolutions:

- **Average**: 7/7 configs ≥ 30 FPS, 7/7 configs ≥ 60 FPS
- **Center-weighted**: 7/7 configs ≥ 30 FPS, 6/7 configs ≥ 60 FPS
- **Intelligent**: 7/7 configs ≥ 30 FPS, 6/7 configs ≥ 60 FPS
- **Multi-zone**: 7/7 configs ≥ 30 FPS, 7/7 configs ≥ 60 FPS
- **Spot**: 7/7 configs ≥ 30 FPS, 7/7 configs ≥ 60 FPS

## API Reference

### AutoExposureController

The main class for auto exposure control.

#### Constructor
```cpp
AutoExposureController();
```

#### Methods

##### `computeExposure(const CameraFrame& frame)`
Computes optimal exposure value for the given frame.

**Parameters:**
- `frame`: Input camera frame

**Returns:** Optimal exposure value (0.0 to 1.0)

##### `setParameters(const Parameters& params)`
Sets controller parameters.

**Parameters:**
- `params`: Configuration parameters

##### `getParameters() const`
Gets current parameters.

**Returns:** Current parameter configuration

##### `applyToCamera(CameraDevice& camera)`
Applies computed exposure to camera device.

**Parameters:**
- `camera`: Target camera device

### Parameters Structure

```cpp
struct Parameters {
    Mode mode = Mode::INTELLIGENT;
    float targetBrightness = 0.5f;
    float convergenceSpeed = 0.15f;
    float minExposure = 0.001f;
    float maxExposure = 1.0f;
    float exposureCompensation = 0.0f;
    bool lockExposure = false;
    bool enableFaceDetection = true;
    bool enableSceneAnalysis = true;
};
```

### Metering Modes

```cpp
enum class Mode {
    AVERAGE,           // Traditional average metering
    CENTER_WEIGHTED,   // Center-weighted metering
    SPOT,             // Spot metering
    MULTI_ZONE,       // Multi-zone analysis
    INTELLIGENT       // AI-powered analysis
};
```

## Configuration

### Environment Variables

- `OPENCAM_ENABLE_ML=1`: Enable ML-based scene classification
- `OPENCAM_CASCADE=/path/to/cascade`: Set face detection cascade path

### Build Options

```bash
# Enable ML features
cmake -DENABLE_ML=ON ..

# Enable testing
cmake -DENABLE_TESTING=ON ..

# Enable benchmarks
cmake -DENABLE_BENCHMARKS=ON ..
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test
./build/test_auto_exposure --gtest_filter="AutoExposureTest.BasicExposureComputationTest"

# Run with coverage
make test-coverage
```

### Test Coverage

- **Unit Tests**: 22 auto exposure tests, 10 debayer tests
- **Integration Tests**: Camera integration, parameter validation
- **Performance Tests**: Real-time performance validation
- **Memory Tests**: Memory leak detection

## Benchmarks

### Running Benchmarks

```bash
# Run performance benchmarks
make benchmark

# View results
cat benchmark_results/auto_exposure_benchmark.csv
```

### Benchmark Metrics

- **Throughput**: Frames per second (FPS)
- **Latency**: Average processing time per frame
- **Memory Usage**: Memory consumption analysis
- **Convergence**: Time to reach optimal exposure

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Set up development environment
make dev-setup

# Run quality checks
make quality

# Run continuous integration simulation
make ci
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/opencam/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/opencam/discussions)
- **Email**: nikjois@llamasearch.ai

## Acknowledgments

- OpenCV community for the excellent computer vision library
- Google Test for the testing framework
- spdlog for fast logging
- All contributors and users

---

**OpenCam Auto Exposure v1.0.0** - Built by Nik Jois 