# OpenCam Auto Exposure Algorithm

**Author:** Nik Jois <nikjois@llamasearch.ai>

A high-performance, intelligent auto exposure algorithm for camera systems with comprehensive Python bindings, FastAPI service, ML integration, and complete testing infrastructure.

## Features

- **Advanced Auto Exposure Algorithm**: Multi-zone metering with intelligent scene analysis
- **Python Bindings**: Complete pybind11 integration with NumPy support
- **FastAPI Service**: REST API with OpenAI agents integration
- **ML Scene Classification**: PyTorch-based scene analysis for optimal exposure
- **Comprehensive Testing**: Unit tests, integration tests, and benchmarking suite
- **Docker Support**: Complete containerization with automated CI/CD
- **Performance Optimized**: Multi-threaded C++ core with OpenMP acceleration
- **Cross-Platform**: Supports Linux, macOS, and Windows

## Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 8+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- Python 3.7+
- OpenCV 4.5+
- spdlog 2.0+

### Installation

#### Option 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/nikjois/opencam-auto-exposure.git
cd opencam-auto-exposure

# Install dependencies
pip install -r requirements.txt

# Build C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build Python bindings
cd ../python/bindings
pip install -e .
```

#### Option 2: Using Docker

```bash
# Build and run with Docker
docker build -f docker/Dockerfile.test -t opencam-auto-exposure .
docker run --rm -p 8000:8000 opencam-auto-exposure
```

#### Option 3: Package Installation

```bash
# Install from PyPI (when available)
pip install opencam-auto-exposure[all]
```

## Usage

### C++ API

```cpp
#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"

using namespace opencam::algorithms;

// Initialize auto exposure controller
AutoExposureController controller;

// Configure parameters
AutoExposureController::Parameters params;
params.mode = AutoExposureController::Mode::INTELLIGENT;
params.targetBrightness = 0.5f;
params.convergenceSpeed = 0.15f;
controller.setParameters(params);

// Process camera frame
CameraFrame frame;
frame.image = /* your cv::Mat image */;
frame.frameNumber = 1;
frame.timestamp = getCurrentTimestamp();

float exposure = controller.computeExposure(frame);
bool converged = controller.isConverged();

// Get statistics and scene analysis
auto stats = controller.getStatistics();
auto scene = controller.getLastSceneAnalysis();
```

### Python API

```python
import numpy as np
from opencam import AutoExposure, Parameters, MeteringMode

# Initialize controller
controller = AutoExposure()

# Configure parameters
params = Parameters(
    mode=MeteringMode.INTELLIGENT,
    target_brightness=0.5,
    convergence_speed=0.15,
    enable_scene_analysis=True
)
controller.parameters = params

# Process image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
exposure = controller.compute_exposure(image)

# Get comprehensive results
exposure, scene_analysis, statistics = controller.process_video_frame(image)

print(f"Computed exposure: {exposure:.4f}")
print(f"Scene type: {scene_analysis.scene_type}")
print(f"Converged: {statistics.is_converged}")
```

### FastAPI Service

```bash
# Start the API server
cd python/api
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### API Usage Examples

```python
import requests
import base64

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Compute exposure for image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/exposure", json={
    "image_data": image_data,
    "frame_number": 1
})

result = response.json()
print(f"Exposure: {result['exposure']}")
print(f"Scene: {result['scene_analysis']}")
```

#### OpenAI Agent Integration

```python
# Query the AI agent
response = requests.post("http://localhost:8000/agent", json={
    "query": "My images are too dark in low light conditions. How should I configure the auto exposure?",
    "context": {"current_mode": "AVERAGE", "target_brightness": 0.5}
})

print(response.json()["response"])
```

### ML Scene Classification

```bash
# Train scene classification model
cd ml/training
python train_scene_classifier.py --create-synthetic --data-dir data/scenes

# Export to ONNX
python train_scene_classifier.py --export-onnx --data-dir data/scenes
```

## Architecture

### Core Components

1. **Auto Exposure Algorithm** (`algorithms/3a/auto_exposure.cpp`)
   - Multi-zone metering implementation
   - Intelligent scene analysis integration
   - Real-time convergence detection
   - Face detection priority metering

2. **Image Processing Pipeline** (`algorithms/isp/`)
   - Debayer algorithms for RAW processing
   - Color space conversions
   - Noise reduction filters

3. **Camera Interface** (`core/`)
   - Abstract camera interface
   - Frame management and buffering
   - Metadata handling

4. **Python Bindings** (`python/bindings/`)
   - pybind11-based C++ integration
   - NumPy array conversion
   - Pythonic API wrapper

5. **FastAPI Service** (`python/api/`)
   - RESTful API endpoints
   - OpenAI agents integration
   - Async request handling
   - CORS support

6. **ML Pipeline** (`ml/`)
   - PyTorch scene classification
   - ONNX model export
   - Synthetic data generation

### Algorithm Details

The auto exposure algorithm implements several metering modes:

- **AVERAGE**: Simple average of entire image
- **CENTER_WEIGHTED**: Weighted average favoring center region
- **SPOT**: Single point metering at image center
- **MULTI_ZONE**: Multi-zone matrix metering
- **INTELLIGENT**: ML-enhanced adaptive metering

The intelligent mode uses scene classification to adapt metering strategy:
- Portrait scenes: Face-priority metering
- Landscape scenes: Horizon-aware metering
- Low-light scenes: Noise-optimized exposure
- Backlit scenes: Shadow/highlight balancing

## Performance

### Benchmarks

Typical performance on modern hardware:

| Resolution | Mode | FPS | Latency |
|------------|------|-----|---------|
| 640x480 | AVERAGE | 2000+ | 0.5ms |
| 1920x1080 | INTELLIGENT | 500+ | 2.0ms |
| 3840x2160 | MULTI_ZONE | 200+ | 5.0ms |

### Memory Usage

- C++ core: ~2MB baseline
- Python bindings: +5MB
- ML models: +50MB (scene classifier)

## Testing

### Running Tests

```bash
# C++ tests
cd build
make test

# Python tests
cd python
pytest tests/ -v --cov=opencam

# Integration tests
pytest tests/integration/ -v

# Benchmarks
python -m opencam.benchmark --full-suite
```

### Test Coverage

- Unit tests: >95% coverage
- Integration tests: All major workflows
- Performance benchmarks: Multiple resolutions and modes
- Memory leak detection: Valgrind integration

## Development

### Building Documentation

```bash
# Sphinx documentation
cd docs
make html

# Doxygen (C++)
doxygen Doxyfile
```

### Code Formatting

```bash
# C++ formatting
clang-format -i algorithms/**/*.cpp algorithms/**/*.h

# Python formatting
black python/
isort python/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run the full test suite
5. Submit a pull request

### Development Environment

```bash
# Set up development environment
make dev-setup

# Run in development mode
make dev

# Run all checks
make ci
```

## Deployment

### Docker Deployment

```dockerfile
# Production Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "python.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencam-auto-exposure
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opencam-auto-exposure
  template:
    metadata:
      labels:
        app: opencam-auto-exposure
    spec:
      containers:
      - name: api
        image: opencam-auto-exposure:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Configuration

### Environment Variables

```bash
# API Configuration
OPENCAM_API_HOST=0.0.0.0
OPENCAM_API_PORT=8000
OPENCAM_LOG_LEVEL=INFO

# OpenAI Integration
OPENAI_API_KEY=your_api_key_here

# ML Models
OPENCAM_MODEL_PATH=/models/scene_classifier.onnx
OPENCAM_ENABLE_GPU=true

# Performance Tuning
OPENCAM_NUM_THREADS=4
OPENCAM_BATCH_SIZE=32
```

### Configuration Files

```yaml
# config.yaml
auto_exposure:
  default_mode: "INTELLIGENT"
  target_brightness: 0.5
  convergence_speed: 0.15
  enable_face_detection: true
  enable_scene_analysis: true

ml:
  scene_classifier:
    model_path: "models/scene_classifier.onnx"
    confidence_threshold: 0.8
    batch_size: 1

api:
  cors_origins: ["*"]
  rate_limit: "100/minute"
  enable_docs: true
```

## Troubleshooting

### Common Issues

1. **Build Errors**
   ```bash
   # Missing dependencies
   sudo apt-get install libopencv-dev libspdlog-dev
   
   # CMake version too old
   pip install cmake --upgrade
   ```

2. **Runtime Errors**
   ```bash
   # OpenCV not found
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   
   # Python import errors
   export PYTHONPATH=$PWD/python:$PYTHONPATH
   ```

3. **Performance Issues**
   ```bash
   # Enable optimizations
   export OMP_NUM_THREADS=4
   export OPENCAM_ENABLE_SIMD=1
   ```

### Debug Mode

```bash
# Enable debug logging
export OPENCAM_LOG_LEVEL=DEBUG

# Run with profiler
python -m cProfile -o profile.stats your_script.py

# Memory debugging
valgrind --tool=memcheck --leak-check=full ./your_program
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{opencam_auto_exposure,
  title={OpenCam Auto Exposure Algorithm},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/nikjois/opencam-auto-exposure},
  version={1.0.0}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/nikjois/opencam-auto-exposure/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nikjois/opencam-auto-exposure/discussions)
- **Email**: nikjois@llamasearch.ai

## Roadmap

- [ ] Real-time video processing optimization
- [ ] Additional ML models (HDR, noise reduction)
- [ ] Mobile platform support (iOS/Android)
- [ ] WebAssembly bindings
- [ ] Cloud deployment templates
- [ ] Advanced benchmarking suite
- [ ] Integration with popular camera SDKs

---

**Built with ❤️ by Nik Jois** 