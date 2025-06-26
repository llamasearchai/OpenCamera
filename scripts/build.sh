#!/bin/bash
# Comprehensive build script for OpenCam Auto Exposure
# Author: Nik Jois <nikjois@llamasearch.ai>

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_TESTS=${BUILD_TESTS:-ON}
BUILD_BENCHMARKS=${BUILD_BENCHMARKS:-ON}
BUILD_PYTHON=${BUILD_PYTHON:-ON}
BUILD_DOCS=${BUILD_DOCS:-OFF}
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}

# Directories
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
INSTALL_DIR="${ROOT_DIR}/install"

echo -e "${BLUE}OpenCam Auto Exposure Build Script${NC}"
echo -e "${BLUE}Author: Nik Jois <nikjois@llamasearch.ai>${NC}"
echo "=================================="
echo "Build Type: ${BUILD_TYPE}"
echo "Build Tests: ${BUILD_TESTS}"
echo "Build Benchmarks: ${BUILD_BENCHMARKS}"
echo "Build Python: ${BUILD_PYTHON}"
echo "Build Docs: ${BUILD_DOCS}"
echo "Parallel Jobs: ${PARALLEL_JOBS}"
echo "Root Directory: ${ROOT_DIR}"
echo "Build Directory: ${BUILD_DIR}"
echo "Install Directory: ${INSTALL_DIR}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_deps+=("make")
    fi
    
    if ! command_exists g++; then
        missing_deps+=("g++")
    fi
    
    if ! command_exists pkg-config; then
        missing_deps+=("pkg-config")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    # Check for optional dependencies
    if ! pkg-config --exists opencv4; then
        print_warning "OpenCV not found via pkg-config. Make sure OpenCV is installed."
    fi
    
    if ! pkg-config --exists spdlog; then
        print_warning "spdlog not found via pkg-config. Will use header-only mode."
    fi
    
    print_status "Dependencies check completed."
}

# Clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    if [ -d "${BUILD_DIR}" ]; then
        rm -rf "${BUILD_DIR}"
    fi
    mkdir -p "${BUILD_DIR}"
}

# Configure CMake
configure_cmake() {
    print_status "Configuring CMake..."
    
    cd "${BUILD_DIR}"
    
    cmake "${ROOT_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DBUILD_TESTS="${BUILD_TESTS}" \
        -DBUILD_BENCHMARKS="${BUILD_BENCHMARKS}" \
        -DBUILD_PYTHON_BINDINGS="${BUILD_PYTHON}" \
        -DBUILD_DOCUMENTATION="${BUILD_DOCS}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native" \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -DDEBUG" \
        -DUSE_OPENMP=ON \
        -DUSE_SIMD=ON
    
    print_status "CMake configuration completed."
}

# Build C++ library
build_cpp() {
    print_status "Building C++ library..."
    
    cd "${BUILD_DIR}"
    make -j"${PARALLEL_JOBS}"
    
    print_status "C++ library build completed."
}

# Run tests
run_tests() {
    if [ "${BUILD_TESTS}" = "ON" ]; then
        print_status "Running C++ tests..."
        
        cd "${BUILD_DIR}"
        ctest --output-on-failure -j"${PARALLEL_JOBS}"
        
        print_status "C++ tests completed."
    else
        print_status "Skipping tests (BUILD_TESTS=OFF)."
    fi
}

# Build Python bindings
build_python() {
    if [ "${BUILD_PYTHON}" = "ON" ]; then
        print_status "Building Python bindings..."
        
        # Check if Python is available
        if ! command_exists python3; then
            print_error "Python 3 not found. Cannot build Python bindings."
            return 1
        fi
        
        # Install Python dependencies
        print_status "Installing Python dependencies..."
        cd "${ROOT_DIR}"
        python3 -m pip install -r requirements.txt
        
        # Build and install Python package
        cd "${ROOT_DIR}/python/bindings"
        python3 setup.py build_ext --inplace
        python3 -m pip install -e .
        
        print_status "Python bindings build completed."
    else
        print_status "Skipping Python bindings (BUILD_PYTHON=OFF)."
    fi
}

# Run Python tests
run_python_tests() {
    if [ "${BUILD_PYTHON}" = "ON" ]; then
        print_status "Running Python tests..."
        
        cd "${ROOT_DIR}"
        python3 -m pytest tests/python/ -v --tb=short
        
        print_status "Python tests completed."
    else
        print_status "Skipping Python tests (BUILD_PYTHON=OFF)."
    fi
}

# Build documentation
build_documentation() {
    if [ "${BUILD_DOCS}" = "ON" ]; then
        print_status "Building documentation..."
        
        # C++ documentation with Doxygen
        if command_exists doxygen; then
            cd "${ROOT_DIR}"
            doxygen Doxyfile
            print_status "Doxygen documentation generated."
        else
            print_warning "Doxygen not found. Skipping C++ documentation."
        fi
        
        # Python documentation with Sphinx
        if command_exists sphinx-build; then
            cd "${ROOT_DIR}/docs"
            make html
            print_status "Sphinx documentation generated."
        else
            print_warning "Sphinx not found. Skipping Python documentation."
        fi
        
        print_status "Documentation build completed."
    else
        print_status "Skipping documentation (BUILD_DOCS=OFF)."
    fi
}

# Install the project
install_project() {
    print_status "Installing project..."
    
    cd "${BUILD_DIR}"
    make install
    
    print_status "Installation completed."
}

# Create package
create_package() {
    print_status "Creating packages..."
    
    cd "${BUILD_DIR}"
    
    # Create DEB package if available
    if command_exists cpack; then
        cpack -G DEB
        print_status "DEB package created."
    fi
    
    # Create Python wheel
    if [ "${BUILD_PYTHON}" = "ON" ]; then
        cd "${ROOT_DIR}/python/bindings"
        python3 setup.py bdist_wheel
        print_status "Python wheel created."
    fi
    
    print_status "Package creation completed."
}

# Run benchmarks
run_benchmarks() {
    if [ "${BUILD_BENCHMARKS}" = "ON" ]; then
        print_status "Running benchmarks..."
        
        cd "${BUILD_DIR}"
        
        # Run C++ benchmarks
        if [ -f "./benchmarks/auto_exposure_benchmark" ]; then
            ./benchmarks/auto_exposure_benchmark
        fi
        
        # Run Python benchmarks
        if [ "${BUILD_PYTHON}" = "ON" ]; then
            cd "${ROOT_DIR}"
            python3 -c "
from opencam.benchmark import AutoExposureBenchmark
from opencam import AutoExposure
benchmark = AutoExposureBenchmark(AutoExposure())
results = benchmark.run_performance_benchmark(iterations=100)
print('Python Benchmark Results:')
print(f'Average FPS: {results[\"fps\"]:.1f}')
print(f'Average time: {results[\"avg_time_seconds\"]*1000:.2f} ms')
"
        fi
        
        print_status "Benchmarks completed."
    else
        print_status "Skipping benchmarks (BUILD_BENCHMARKS=OFF)."
    fi
}

# Main build function
main() {
    print_status "Starting OpenCam Auto Exposure build process..."
    
    # Check dependencies
    check_dependencies
    
    # Clean and configure
    clean_build
    configure_cmake
    
    # Build C++ library
    build_cpp
    
    # Run tests
    run_tests
    
    # Build Python bindings
    build_python
    
    # Run Python tests
    run_python_tests
    
    # Build documentation
    build_documentation
    
    # Install project
    install_project
    
    # Create packages
    create_package
    
    # Run benchmarks
    run_benchmarks
    
    print_status "Build process completed successfully!"
    echo ""
    echo -e "${GREEN}Build Summary:${NC}"
    echo "- C++ library: ${BUILD_DIR}/lib/"
    echo "- Headers: ${INSTALL_DIR}/include/"
    echo "- Executables: ${BUILD_DIR}/bin/"
    if [ "${BUILD_PYTHON}" = "ON" ]; then
        echo "- Python package: Installed in site-packages"
    fi
    if [ "${BUILD_DOCS}" = "ON" ]; then
        echo "- Documentation: ${ROOT_DIR}/docs/_build/html/"
    fi
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Test the installation: make test"
    echo "2. Run benchmarks: make benchmark"
    echo "3. Start the API server: cd python/api && python main.py"
    echo "4. Train ML models: cd ml/training && python train_scene_classifier.py"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --no-python)
            BUILD_PYTHON=OFF
            shift
            ;;
        --with-docs)
            BUILD_DOCS=ON
            shift
            ;;
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build-type TYPE    Build type (Release, Debug, RelWithDebInfo)"
            echo "  --no-tests          Skip building and running tests"
            echo "  --no-python         Skip Python bindings"
            echo "  --with-docs         Build documentation"
            echo "  --jobs N            Number of parallel jobs"
            echo "  --help              Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_TYPE          Build type (default: Release)"
            echo "  BUILD_TESTS         Build tests (default: ON)"
            echo "  BUILD_PYTHON        Build Python bindings (default: ON)"
            echo "  BUILD_DOCS          Build documentation (default: OFF)"
            echo "  PARALLEL_JOBS       Number of parallel jobs (default: nproc)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Run main function
main 