cmake_minimum_required(VERSION 3.16)
project(OpenCamAutoExposure VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# Enable coverage if requested
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()
endif()

# Enable sanitizers if requested
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SPDLOG REQUIRED spdlog)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Core library
add_library(opencam_core
    core/src/camera.cpp
)

target_link_libraries(opencam_core
    ${OpenCV_LIBS}
    ${SPDLOG_LIBRARIES}
)

target_include_directories(opencam_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# Auto Exposure algorithm library
add_library(auto_exposure
    algorithms/3a/auto_exposure.cpp
)

target_link_libraries(auto_exposure
    opencam_core
    ${OpenCV_LIBS}
    ${SPDLOG_LIBRARIES}
)

target_include_directories(auto_exposure PUBLIC
    ${CMAKE_SOURCE_DIR}/algorithms
    ${CMAKE_SOURCE_DIR}/include
)

# ISP algorithms library
add_library(isp_algorithms
    algorithms/isp/debayer.cpp
)

target_link_libraries(isp_algorithms
    ${OpenCV_LIBS}
    ${SPDLOG_LIBRARIES}
)

# Testing
if(ENABLE_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)
    
    # Unit tests
    add_executable(test_auto_exposure
        tests/test_auto_exposure.cpp
        tests/test_camera.cpp
        tests/test_debayer.cpp
    )
    
    target_link_libraries(test_auto_exposure
        auto_exposure
        isp_algorithms
        opencam_core
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )
    
    # Register tests with CTest
    add_test(NAME AutoExposureTests COMMAND test_auto_exposure)
    
    # Set test properties
    set_tests_properties(AutoExposureTests PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "GTEST_OUTPUT=xml:test_results.xml"
    )
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_executable(auto_exposure_benchmark
        benchmarks/auto_exposure_benchmark.cpp
    )
    
    target_link_libraries(auto_exposure_benchmark
        auto_exposure
        opencam_core
        ${OpenCV_LIBS}
        ${SPDLOG_LIBRARIES}
    )
endif()

# Examples
if(BUILD_EXAMPLES)
    # Auto exposure demo
    add_executable(auto_exposure_demo
        examples/auto_exposure_demo.cpp
    )
    
    target_link_libraries(auto_exposure_demo
        auto_exposure
        opencam_core
        ${OpenCV_LIBS}
        ${SPDLOG_LIBRARIES}
    )
    
    # Camera test utility
    add_executable(camera_test
        examples/camera_test.cpp
    )
    
    target_link_libraries(camera_test
        opencam_core
        ${OpenCV_LIBS}
        ${SPDLOG_LIBRARIES}
    )
endif()

# Installation
install(TARGETS opencam_core auto_exposure isp_algorithms
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

if(BUILD_EXAMPLES)
    install(TARGETS auto_exposure_demo camera_test
        RUNTIME DESTINATION bin
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenCamConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCamConfig.cmake"
    INSTALL_DESTINATION lib/cmake/OpenCam
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCamConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCamConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCamConfigVersion.cmake"
    DESTINATION lib/cmake/OpenCam
)

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "OpenCamAutoExposure")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Auto Exposure Algorithm for OpenCam")
set(CPACK_PACKAGE_VENDOR "OpenCam Team")
set(CPACK_PACKAGE_CONTACT "opencam-team@example.com")

set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_GENERATOR "DEB;RPM;TGZ")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev, libspdlog-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "opencv-devel, spdlog-devel")

include(CPack)

# Custom targets for development
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/algorithms ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/tests 
            -name "*.cpp" -o -name "*.h" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(lint
    COMMAND find ${CMAKE_SOURCE_DIR}/algorithms ${CMAKE_SOURCE_DIR}/core 
            -name "*.cpp" -o -name "*.h" | xargs clang-tidy -p ${CMAKE_BINARY_DIR}
    COMMENT "Running static analysis"
)

# Print configuration summary
message(STATUS "")
message(STATUS "OpenCam Auto Exposure Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Testing: ${ENABLE_TESTING}")
message(STATUS "  Benchmarks: ${ENABLE_BENCHMARKS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")