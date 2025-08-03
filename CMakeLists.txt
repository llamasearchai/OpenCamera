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
# Prefer pkg-config spdlog; fall back to header-only if not linkable
pkg_check_modules(SPDLOG spdlog)
if(SPDLOG_FOUND)
    message(STATUS "Found spdlog via pkg-config: ${SPDLOG_VERSION}")
else()
    message(WARNING "spdlog not found with pkg-config; using header-only mode")
endif()
# Homebrew fallback for spdlog on macOS (header-only or keg-only installs)
if(APPLE AND NOT SPDLOG_FOUND)
    # Common Homebrew include directory for header-only spdlog
    if(EXISTS "/opt/homebrew/include/spdlog/spdlog.h")
        set(SPDLOG_INCLUDE_DIRS "/opt/homebrew/include")
        message(STATUS "Using spdlog header-only from /opt/homebrew/include")
    endif()
endif()

# Include directories
# Ensure headers under core/include/opencam are visible to all targets.
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/algorithms
    ${OpenCV_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# Core library
add_library(opencam_core
    core/src/camera.cpp
)

# Link spdlog only if libraries were found; otherwise rely on header-only usage
# Prefer header-only spdlog on macOS to avoid missing dylib issues
if(APPLE)
    target_link_libraries(opencam_core
        ${OpenCV_LIBS}
    )
    if(SPDLOG_INCLUDE_DIRS)
        target_include_directories(opencam_core PUBLIC ${SPDLOG_INCLUDE_DIRS})
    endif()
    target_compile_definitions(opencam_core PUBLIC SPDLOG_HEADER_ONLY)
else()
    if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
        target_link_libraries(opencam_core
            ${OpenCV_LIBS}
            ${SPDLOG_LIBRARIES}
        )
    else()
        target_link_libraries(opencam_core
            ${OpenCV_LIBS}
        )
        target_compile_definitions(opencam_core PUBLIC SPDLOG_HEADER_ONLY)
    endif()
endif()

target_include_directories(opencam_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core/include
    ${OpenCV_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# Auto Exposure algorithm library
add_library(auto_exposure
    algorithms/3a/auto_exposure.cpp
)

if(APPLE)
    target_link_libraries(auto_exposure
        opencam_core
        ${OpenCV_LIBS}
    )
    if(SPDLOG_INCLUDE_DIRS)
        target_include_directories(auto_exposure PUBLIC ${SPDLOG_INCLUDE_DIRS})
    endif()
    target_compile_definitions(auto_exposure PUBLIC SPDLOG_HEADER_ONLY)
else()
    if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
        target_link_libraries(auto_exposure
            opencam_core
            ${OpenCV_LIBS}
            ${SPDLOG_LIBRARIES}
        )
    else()
        target_link_libraries(auto_exposure
            opencam_core
            ${OpenCV_LIBS}
        )
        target_compile_definitions(auto_exposure PUBLIC SPDLOG_HEADER_ONLY)
    endif()
endif()

target_include_directories(auto_exposure PUBLIC
    ${CMAKE_SOURCE_DIR}/algorithms
    ${CMAKE_SOURCE_DIR}/core/include
    ${OpenCV_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# ISP algorithms library
add_library(isp_algorithms
    algorithms/isp/debayer.cpp
)

target_include_directories(isp_algorithms PUBLIC
    ${CMAKE_SOURCE_DIR}/algorithms
    ${CMAKE_SOURCE_DIR}/core/include
    ${OpenCV_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)
# Avoid linking spdlog explicitly on Apple; keep header-only mode consistent
if(APPLE)
    target_link_libraries(isp_algorithms
        ${OpenCV_LIBS}
    )
    if(SPDLOG_INCLUDE_DIRS)
        target_include_directories(isp_algorithms PUBLIC ${SPDLOG_INCLUDE_DIRS})
    endif()
    target_compile_definitions(isp_algorithms PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)
else()
    if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
        target_link_libraries(isp_algorithms
            ${OpenCV_LIBS}
            ${SPDLOG_LIBRARIES}
        )
    elseif(SPDLOG_INCLUDE_DIRS)
        target_link_libraries(isp_algorithms
            ${OpenCV_LIBS}
        )
        target_include_directories(isp_algorithms PUBLIC ${SPDLOG_INCLUDE_DIRS})
        target_compile_definitions(isp_algorithms PUBLIC SPDLOG_HEADER_ONLY)
    else()
        target_link_libraries(isp_algorithms
            ${OpenCV_LIBS}
        )
        target_compile_definitions(isp_algorithms PUBLIC SPDLOG_HEADER_ONLY)
    endif()
endif()

# Testing
if(ENABLE_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)
    
    # Unit tests
    add_executable(test_auto_exposure
        tests/test_auto_exposure.cpp
    )
    
    target_link_libraries(test_auto_exposure
        auto_exposure
        isp_algorithms
        opencam_core
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )
    if(APPLE)
        # Ensure header-only fmt/spdlog are consistent on Apple
        if(EXISTS "/opt/homebrew/include/fmt/format.h")
            target_include_directories(test_auto_exposure PUBLIC "/opt/homebrew/include")
        endif()
        if(SPDLOG_INCLUDE_DIRS)
            target_include_directories(test_auto_exposure PUBLIC ${SPDLOG_INCLUDE_DIRS})
        endif()
        target_compile_definitions(test_auto_exposure PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)

        # Fix runtime loading of GTest dylibs by embedding rpath to Homebrew (Conda path already in CMake config)
        # Try common Homebrew lib first
        if(EXISTS "/opt/homebrew/lib")
            set_target_properties(test_auto_exposure PROPERTIES
                BUILD_RPATH "/opt/homebrew/lib"
                INSTALL_RPATH "/opt/homebrew/lib"
            )
        endif()
        # If GTest came from conda, also add that lib path
        if(EXISTS "/opt/miniconda3/lib")
            # Append to any existing rpath to be safe
            get_target_property(_curr_rpath test_auto_exposure BUILD_RPATH)
            if(NOT _curr_rpath)
                set(_curr_rpath "")
            endif()
            set(_new_rpath "${_curr_rpath};/opt/miniconda3/lib")
            set_target_properties(test_auto_exposure PROPERTIES
                BUILD_RPATH "${_new_rpath}"
                INSTALL_RPATH "${_new_rpath}"
            )
        endif()
    endif()
    
    # Register tests with CTest
    add_test(NAME AutoExposureTests COMMAND test_auto_exposure)
    
    # Set test properties
    set_tests_properties(AutoExposureTests PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "GTEST_OUTPUT=xml:test_results.xml"
    )
    
    # Add debayer test
    add_executable(test_debayer
        tests/test_debayer.cpp
    )
    
    target_link_libraries(test_debayer
        isp_algorithms
        ${OpenCV_LIBS}
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )
    
    # Set runtime path for macOS to find GTest libraries
    if(APPLE)
        set_target_properties(test_debayer PROPERTIES
            BUILD_WITH_INSTALL_RPATH FALSE
            LINK_FLAGS "-Wl,-rpath,@loader_path/../lib -Wl,-rpath,/opt/homebrew/lib -Wl,-rpath,/opt/miniconda3/lib"
        )
    endif()
    if(APPLE)
        # Ensure header-only fmt/spdlog are consistent on Apple
        if(EXISTS "/opt/homebrew/include/fmt/format.h")
            target_include_directories(test_debayer PUBLIC "/opt/homebrew/include")
        endif()
        if(SPDLOG_INCLUDE_DIRS)
            target_include_directories(test_debayer PUBLIC ${SPDLOG_INCLUDE_DIRS})
        endif()
        target_compile_definitions(test_debayer PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)
    else()
        # Non-Apple platforms
        if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
            target_link_libraries(test_debayer ${SPDLOG_LIBRARIES})
        else()
            if(SPDLOG_INCLUDE_DIRS)
                target_include_directories(test_debayer PUBLIC ${SPDLOG_INCLUDE_DIRS})
            endif()
            target_compile_definitions(test_debayer PUBLIC SPDLOG_HEADER_ONLY)
        endif()
    endif()
    
    # Register debayer test with CTest
    add_test(NAME DebayerTests COMMAND test_debayer)
    
    # Set test properties
    set_tests_properties(DebayerTests PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "GTEST_OUTPUT=xml:test_debayer_results.xml"
    )
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_executable(auto_exposure_benchmark
        benchmarks/auto_exposure_benchmark.cpp
    )
    # Always use header-only spdlog with explicit fmt include on Apple to avoid missing dylib/symbols
    if(APPLE)
        target_link_libraries(auto_exposure_benchmark
            auto_exposure
            opencam_core
            ${OpenCV_LIBS}
        )
        # Add Homebrew fmt include path if present to satisfy spdlog's fmt references
        if(EXISTS "/opt/homebrew/include/fmt/format.h")
            target_include_directories(auto_exposure_benchmark PUBLIC "/opt/homebrew/include")
        endif()
        if(SPDLOG_INCLUDE_DIRS)
            target_include_directories(auto_exposure_benchmark PUBLIC ${SPDLOG_INCLUDE_DIRS})
        endif()
        target_compile_definitions(auto_exposure_benchmark PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)
    else()
        if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
            target_link_libraries(auto_exposure_benchmark
                auto_exposure
                opencam_core
                ${OpenCV_LIBS}
                ${SPDLOG_LIBRARIES}
            )
        else()
            target_link_libraries(auto_exposure_benchmark
                auto_exposure
                opencam_core
                ${OpenCV_LIBS}
            )
            if(SPDLOG_INCLUDE_DIRS)
                target_include_directories(auto_exposure_benchmark PUBLIC ${SPDLOG_INCLUDE_DIRS})
            endif()
            target_compile_definitions(auto_exposure_benchmark PUBLIC SPDLOG_HEADER_ONLY)
        endif()
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    # Auto exposure demo
    add_executable(auto_exposure_demo
        examples/auto_exposure_demo.cpp
    )
    
    if(APPLE)
        target_link_libraries(auto_exposure_demo
            auto_exposure
            opencam_core
            ${OpenCV_LIBS}
        )
        if(EXISTS "/opt/homebrew/include/fmt/format.h")
            target_include_directories(auto_exposure_demo PUBLIC "/opt/homebrew/include")
        endif()
        if(SPDLOG_INCLUDE_DIRS)
            target_include_directories(auto_exposure_demo PUBLIC ${SPDLOG_INCLUDE_DIRS})
        endif()
        target_compile_definitions(auto_exposure_demo PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)
    else()
        if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
            target_link_libraries(auto_exposure_demo
                auto_exposure
                opencam_core
                ${OpenCV_LIBS}
                ${SPDLOG_LIBRARIES}
            )
        else()
            target_link_libraries(auto_exposure_demo
                auto_exposure
                opencam_core
                ${OpenCV_LIBS}
            )
            if(SPDLOG_INCLUDE_DIRS)
                target_include_directories(auto_exposure_demo PUBLIC ${SPDLOG_INCLUDE_DIRS})
            endif()
            target_compile_definitions(auto_exposure_demo PUBLIC SPDLOG_HEADER_ONLY)
        endif()
    endif()
    
    # Camera test utility (optional if source exists)
    if(EXISTS "${CMAKE_SOURCE_DIR}/examples/camera_test.cpp")
        add_executable(camera_test
            examples/camera_test.cpp
        )
        if(APPLE)
            # Avoid linking spdlog on macOS; use header-only with optional Homebrew includes
            target_link_libraries(camera_test
                opencam_core
                ${OpenCV_LIBS}
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
            )
            if(EXISTS "/opt/homebrew/include/fmt/format.h")
                target_include_directories(camera_test PUBLIC "/opt/homebrew/include")
            endif()
            if(SPDLOG_INCLUDE_DIRS)
                target_include_directories(camera_test PUBLIC ${SPDLOG_INCLUDE_DIRS})
            endif()
            target_compile_definitions(camera_test PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY=1 SPDLOG_FMT_EXTERNAL=1)
            # Set runtime path for macOS to find GTest libraries
            set_target_properties(camera_test PROPERTIES
                BUILD_WITH_INSTALL_RPATH FALSE
                LINK_FLAGS "-Wl,-rpath,@loader_path/../lib -Wl,-rpath,/opt/homebrew/lib -Wl,-rpath,/opt/miniconda3/lib"
            )
        else()
            if(SPDLOG_FOUND AND SPDLOG_LIBRARIES)
                target_link_libraries(camera_test
                    opencam_core
                    ${OpenCV_LIBS}
                    ${SPDLOG_LIBRARIES}
                    GTest::gtest
                    GTest::gtest_main
                    GTest::gmock
                )
            else()
                target_link_libraries(camera_test
                    opencam_core
                    ${OpenCV_LIBS}
                    GTest::gtest
                    GTest::gtest_main
                    GTest::gmock
                )
                if(SPDLOG_INCLUDE_DIRS)
                    target_include_directories(camera_test PUBLIC ${SPDLOG_INCLUDE_DIRS})
                endif()
                target_compile_definitions(camera_test PUBLIC SPDLOG_HEADER_ONLY)
            endif()
        endif()
    else()
        message(STATUS "examples/camera_test.cpp not found; skipping camera_test target")
    endif()
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
    # Always install auto_exposure_demo
    install(TARGETS auto_exposure_demo
        RUNTIME DESTINATION bin
    )
    # Conditionally install camera_test if it was created
    if(TARGET camera_test)
        install(TARGETS camera_test
            RUNTIME DESTINATION bin
        )
    else()
        message(STATUS "camera_test target not available; skipping installation")
    endif()
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