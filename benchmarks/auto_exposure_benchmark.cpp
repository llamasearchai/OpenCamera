#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace opencam;
using namespace opencam::algorithms;

class AutoExposureBenchmark {
public:
    struct BenchmarkResult {
        std::string testName;
        size_t iterations;
        double avgTimeUs;
        double minTimeUs;
        double maxTimeUs;
        double stdDevUs;
        double throughputFPS;
        size_t imageWidth;
        size_t imageHeight;
        AutoExposureController::Mode mode;
    };
    
    AutoExposureBenchmark() {
        controller_ = std::make_unique<AutoExposureController>();
        
        // Set up random number generator
        rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    
    void runAllBenchmarks() {
        spdlog::info("Starting Auto Exposure Benchmarks");
        
        results_.clear();
        
        // Test different image sizes
        std::vector<std::pair<int, int>> resolutions = {
            {320, 240},    // QVGA
            {640, 480},    // VGA
            {800, 600},    // SVGA
            {1280, 720},   // HD
            {1920, 1080},  // Full HD
            {2560, 1440},  // QHD
            {3840, 2160}   // 4K
        };
        
        // Test different metering modes
        std::vector<AutoExposureController::Mode> modes = {
            AutoExposureController::Mode::AVERAGE,
            AutoExposureController::Mode::CENTER_WEIGHTED,
            AutoExposureController::Mode::SPOT,
            AutoExposureController::Mode::MULTI_ZONE,
            AutoExposureController::Mode::INTELLIGENT
        };
        
        // Benchmark each resolution with each mode
        for (const auto& [width, height] : resolutions) {
            for (auto mode : modes) {
                auto result = benchmarkResolutionAndMode(width, height, mode);
                results_.push_back(result);
                
                spdlog::info("Completed: {}x{} {} - {:.2f} FPS", 
                           width, height, getModeString(mode), result.throughputFPS);
            }
        }
        
        // Benchmark convergence behavior
        benchmarkConvergence();
        
        // Benchmark scene change detection
        benchmarkSceneChangeDetection();
        
        // Benchmark memory usage
        benchmarkMemoryUsage();
        
        // Generate report
        generateReport();
    }

private:
    BenchmarkResult benchmarkResolutionAndMode(int width, int height, 
                                              AutoExposureController::Mode mode) {
        AutoExposureController::Parameters params;
        params.mode = mode;
        params.targetBrightness = 0.5f;
        params.convergenceSpeed = 0.15f;
        controller_->setParameters(params);
        controller_->reset();
        
        const size_t iterations = getIterationsForResolution(width, height);
        std::vector<double> times;
        times.reserve(iterations);
        
        // Generate test images
        std::vector<CameraFrame> testFrames;
        testFrames.reserve(iterations);
        
        for (size_t i = 0; i < iterations; i++) {
            testFrames.push_back(generateTestFrame(width, height, i));
        }
        
        // Warm up
        for (size_t i = 0; i < std::min(size_t(10), iterations); i++) {
            controller_->computeExposure(testFrames[i]);
        }
        
        // Benchmark
        for (size_t i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            controller_->computeExposure(testFrames[i]);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // Convert to microseconds
        }
        
        // Calculate statistics
        BenchmarkResult result;
        result.testName = fmt::format("{}x{}_{}", width, height, getModeString(mode));
        result.iterations = iterations;
        result.imageWidth = width;
        result.imageHeight = height;
        result.mode = mode;
        
        std::sort(times.begin(), times.end());
        result.minTimeUs = times.front();
        result.maxTimeUs = times.back();
        result.avgTimeUs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        result.throughputFPS = 1000000.0 / result.avgTimeUs;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : times) {
            double diff = time - result.avgTimeUs;
            variance += diff * diff;
        }
        result.stdDevUs = std::sqrt(variance / times.size());
        
        return result;
    }
    
    void benchmarkConvergence() {
        spdlog::info("Benchmarking convergence behavior");
        
        AutoExposureController::Parameters params;
        params.mode = AutoExposureController::Mode::INTELLIGENT;
        params.convergenceSpeed = 0.15f;
        controller_->setParameters(params);
        
        // Test convergence with different scene changes
        std::vector<float> brightnesses = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
        
        for (float brightness : brightnesses) {
            controller_->reset();
            
            auto start = std::chrono::high_resolution_clock::now();
            int frameCount = 0;
            
            // Process frames until convergence
            while (!controller_->isConverged() && frameCount < 100) {
                auto frame = generateTestFrame(1280, 720, frameCount, brightness);
                controller_->computeExposure(frame);
                frameCount++;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            spdlog::info("Convergence for brightness {:.1f}: {} frames, {} ms", 
                        brightness, frameCount, duration.count());
        }
    }
    
    void benchmarkSceneChangeDetection() {
        spdlog::info("Benchmarking scene change detection");
        
        const int width = 1280, height = 720;
        const size_t iterations = 1000;
        
        std::vector<double> times;
        times.reserve(iterations);
        
        // Generate alternating scenes
        for (size_t i = 0; i < iterations; i++) {
            float brightness = (i % 2 == 0) ? 0.2f : 0.8f;
            auto frame = generateTestFrame(width, height, i, brightness);
            
            auto start = std::chrono::high_resolution_clock::now();
            controller_->computeExposure(frame);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        spdlog::info("Scene change detection avg time: {:.2f} μs", avgTime);
    }
    
    void benchmarkMemoryUsage() {
        spdlog::info("Benchmarking memory usage");
        
        // This is a simplified memory benchmark
        // In a real implementation, you'd use tools like valgrind or custom memory tracking
        
        const size_t iterations = 10000;
        std::vector<std::unique_ptr<AutoExposureController>> controllers;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create many controllers to test memory usage
        for (size_t i = 0; i < 100; i++) {
            controllers.push_back(std::make_unique<AutoExposureController>());
        }
        
        // Process frames with all controllers
        auto frame = generateTestFrame(640, 480, 0);
        for (size_t i = 0; i < iterations; i++) {
            for (auto& controller : controllers) {
                controller->computeExposure(frame);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        spdlog::info("Memory benchmark: {} controllers, {} iterations, {} ms", 
                    controllers.size(), iterations, duration.count());
    }
    
    CameraFrame generateTestFrame(int width, int height, size_t frameNumber, 
                                 float brightness = -1.0f) {
        CameraFrame frame;
        frame.frameNumber = frameNumber;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Generate realistic test image
        frame.image = cv::Mat::zeros(height, width, CV_8UC3);
        
        if (brightness < 0) {
            // Generate varied scene
            generateRealisticScene(frame.image, frameNumber);
        } else {
            // Generate uniform brightness
            cv::Scalar color(brightness * 255, brightness * 255, brightness * 255);
            frame.image.setTo(color);
            
            // Add some noise for realism
            cv::Mat noise;
            cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(10));
            frame.image += noise;
        }
        
        return frame;
    }
    
    void generateRealisticScene(cv::Mat& image, size_t frameNumber) {
        // Create different types of realistic scenes based on frame number
        int sceneType = frameNumber % 6;
        
        switch (sceneType) {
            case 0: // Uniform scene
                image.setTo(cv::Scalar(128, 128, 128));
                break;
                
            case 1: // Gradient scene
                for (int y = 0; y < image.rows; y++) {
                    int intensity = (y * 255) / image.rows;
                    cv::line(image, cv::Point(0, y), cv::Point(image.cols, y), 
                            cv::Scalar(intensity, intensity, intensity));
                }
                break;
                
            case 2: // Center bright scene (backlit simulation)
                image.setTo(cv::Scalar(50, 50, 50));
                cv::circle(image, cv::Point(image.cols/2, image.rows/2), 
                          std::min(image.cols, image.rows)/4, cv::Scalar(200, 200, 200), -1);
                break;
                
            case 3: // High contrast scene
                image.setTo(cv::Scalar(30, 30, 30));
                for (int i = 0; i < 10; i++) {
                    cv::Rect rect(dist_(rng_) % (image.cols/2), dist_(rng_) % (image.rows/2),
                                 image.cols/4, image.rows/4);
                    cv::rectangle(image, rect, cv::Scalar(220, 220, 220), -1);
                }
                break;
                
            case 4: // Low light with noise
                {
                    cv::Mat noise;
                    cv::randn(noise, cv::Scalar(40, 40, 40), cv::Scalar(20, 20, 20));
                    image = noise;
                }
                break;
                
            case 5: // Textured scene
                for (int y = 0; y < image.rows; y += 20) {
                    for (int x = 0; x < image.cols; x += 20) {
                        int intensity = ((x/20 + y/20) % 2) ? 80 : 180;
                        cv::Rect rect(x, y, std::min(20, image.cols - x), std::min(20, image.rows - y));
                        cv::rectangle(image, rect, cv::Scalar(intensity, intensity, intensity), -1);
                    }
                }
                break;
        }
        
        // Add some random noise to all scenes
        cv::Mat noise;
        cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(5));
        image += noise;
    }
    
    size_t getIterationsForResolution(int width, int height) {
        // Fewer iterations for higher resolutions to keep benchmark time reasonable
        size_t pixels = width * height;
        if (pixels <= 320 * 240) return 1000;
        if (pixels <= 640 * 480) return 500;
        if (pixels <= 1280 * 720) return 200;
        if (pixels <= 1920 * 1080) return 100;
        return 50;
    }
    
    std::string getModeString(AutoExposureController::Mode mode) {
        switch (mode) {
            case AutoExposureController::Mode::AVERAGE: return "Average";
            case AutoExposureController::Mode::CENTER_WEIGHTED: return "CenterWeighted";
            case AutoExposureController::Mode::SPOT: return "Spot";
            case AutoExposureController::Mode::MULTI_ZONE: return "MultiZone";
            case AutoExposureController::Mode::INTELLIGENT: return "Intelligent";
            default: return "Unknown";
        }
    }
    
    void generateReport() {
        spdlog::info("Generating benchmark report");
        
        // Console report
        printConsoleReport();
        
        // CSV report
        generateCSVReport();
        
        // JSON report
        generateJSONReport();
        
        // Performance analysis
        analyzePerformance();
    }
    
    void printConsoleReport() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "AUTO EXPOSURE BENCHMARK RESULTS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << std::left << std::setw(20) << "Resolution"
                  << std::setw(15) << "Mode"
                  << std::setw(12) << "Avg (μs)"
                  << std::setw(12) << "Min (μs)"
                  << std::setw(12) << "Max (μs)"
                  << std::setw(12) << "StdDev"
                  << std::setw(10) << "FPS" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results_) {
            std::string resolution = fmt::format("{}x{}", result.imageWidth, result.imageHeight);
            
            std::cout << std::left << std::setw(20) << resolution
                      << std::setw(15) << getModeString(result.mode)
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.avgTimeUs
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.minTimeUs
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.maxTimeUs
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.stdDevUs
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.throughputFPS
                      << std::endl;
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void generateCSVReport() {
        std::ofstream csvFile("auto_exposure_benchmark.csv");
        if (!csvFile.is_open()) {
            spdlog::error("Failed to create CSV report file");
            return;
        }
        
        // Header
        csvFile << "Resolution,Width,Height,Mode,Iterations,AvgTime_us,MinTime_us,MaxTime_us,StdDev_us,Throughput_FPS\n";
        
        // Data
        for (const auto& result : results_) {
            csvFile << result.imageWidth << "x" << result.imageHeight << ","
                    << result.imageWidth << ","
                    << result.imageHeight << ","
                    << getModeString(result.mode) << ","
                    << result.iterations << ","
                    << std::fixed << std::setprecision(2) << result.avgTimeUs << ","
                    << std::fixed << std::setprecision(2) << result.minTimeUs << ","
                    << std::fixed << std::setprecision(2) << result.maxTimeUs << ","
                    << std::fixed << std::setprecision(2) << result.stdDevUs << ","
                    << std::fixed << std::setprecision(2) << result.throughputFPS << "\n";
        }
        
        csvFile.close();
        spdlog::info("CSV report saved to auto_exposure_benchmark.csv");
    }
    
    void generateJSONReport() {
        std::ofstream jsonFile("auto_exposure_benchmark.json");
        if (!jsonFile.is_open()) {
            spdlog::error("Failed to create JSON report file");
            return;
        }
        
        jsonFile << "{\n";
        jsonFile << "  \"benchmark_info\": {\n";
        jsonFile << "    \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
        jsonFile << "    \"total_tests\": " << results_.size() << "\n";
        jsonFile << "  },\n";
        jsonFile << "  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); i++) {
            const auto& result = results_[i];
            jsonFile << "    {\n";
            jsonFile << "      \"test_name\": \"" << result.testName << "\",\n";
            jsonFile << "      \"resolution\": {\n";
            jsonFile << "        \"width\": " << result.imageWidth << ",\n";
            jsonFile << "        \"height\": " << result.imageHeight << "\n";
            jsonFile << "      },\n";
            jsonFile << "      \"mode\": \"" << getModeString(result.mode) << "\",\n";
            jsonFile << "      \"iterations\": " << result.iterations << ",\n";
            jsonFile << "      \"timing\": {\n";
            jsonFile << "        \"avg_us\": " << std::fixed << std::setprecision(2) << result.avgTimeUs << ",\n";
            jsonFile << "        \"min_us\": " << std::fixed << std::setprecision(2) << result.minTimeUs << ",\n";
            jsonFile << "        \"max_us\": " << std::fixed << std::setprecision(2) << result.maxTimeUs << ",\n";
            jsonFile << "        \"stddev_us\": " << std::fixed << std::setprecision(2) << result.stdDevUs << "\n";
            jsonFile << "      },\n";
            jsonFile << "      \"throughput_fps\": " << std::fixed << std::setprecision(2) << result.throughputFPS << "\n";
            jsonFile << "    }";
            if (i < results_.size() - 1) jsonFile << ",";
            jsonFile << "\n";
        }
        
        jsonFile << "  ]\n";
        jsonFile << "}\n";
        
        jsonFile.close();
        spdlog::info("JSON report saved to auto_exposure_benchmark.json");
    }
    
    void analyzePerformance() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PERFORMANCE ANALYSIS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Find best and worst performing configurations
        auto bestResult = *std::max_element(results_.begin(), results_.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.throughputFPS < b.throughputFPS;
            });
            
        auto worstResult = *std::min_element(results_.begin(), results_.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.throughputFPS < b.throughputFPS;
            });
        
        std::cout << "Best Performance:\n";
        std::cout << "  " << bestResult.imageWidth << "x" << bestResult.imageHeight 
                  << " " << getModeString(bestResult.mode) 
                  << " - " << std::fixed << std::setprecision(1) << bestResult.throughputFPS << " FPS\n";
        
        std::cout << "Worst Performance:\n";
        std::cout << "  " << worstResult.imageWidth << "x" << worstResult.imageHeight 
                  << " " << getModeString(worstResult.mode) 
                  << " - " << std::fixed << std::setprecision(1) << worstResult.throughputFPS << " FPS\n";
        
        // Analyze scaling with resolution
        std::cout << "\nResolution Scaling Analysis:\n";
        std::map<std::string, std::vector<BenchmarkResult>> modeResults;
        for (const auto& result : results_) {
            modeResults[getModeString(result.mode)].push_back(result);
        }
        
        for (const auto& [mode, results] : modeResults) {
            if (results.size() >= 2) {
                auto minRes = *std::min_element(results.begin(), results.end(),
                    [](const BenchmarkResult& a, const BenchmarkResult& b) {
                        return a.imageWidth * a.imageHeight < b.imageWidth * b.imageHeight;
                    });
                auto maxRes = *std::max_element(results.begin(), results.end(),
                    [](const BenchmarkResult& a, const BenchmarkResult& b) {
                        return a.imageWidth * a.imageHeight < b.imageWidth * b.imageHeight;
                    });
                
                double scalingFactor = (double)(maxRes.imageWidth * maxRes.imageHeight) / 
                                     (minRes.imageWidth * minRes.imageHeight);
                double performanceRatio = minRes.throughputFPS / maxRes.throughputFPS;
                
                std::cout << "  " << mode << ": " << std::fixed << std::setprecision(1) 
                          << scalingFactor << "x pixels -> " << std::setprecision(2) 
                          << performanceRatio << "x performance ratio\n";
            }
        }
        
        // Real-time capability analysis
        std::cout << "\nReal-time Capability (30 FPS threshold):\n";
        for (const auto& [mode, results] : modeResults) {
            int count30fps = 0;
            int count60fps = 0;
            for (const auto& result : results) {
                if (result.throughputFPS >= 30.0) count30fps++;
                if (result.throughputFPS >= 60.0) count60fps++;
            }
            std::cout << "  " << mode << ": " << count30fps << "/" << results.size() 
                      << " configs >= 30 FPS, " << count60fps << "/" << results.size() 
                      << " configs >= 60 FPS\n";
        }
        
        std::cout << std::string(60, '=') << std::endl;
    }
    
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

private:
    std::unique_ptr<AutoExposureController> controller_;
    std::vector<BenchmarkResult> results_;
    std::mt19937 rng_;
    std::uniform_int_distribution<int> dist_{0, 255};
};

int main(int argc, char* argv[]) {
    // Set up logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S.%e] [%l] %v");
    
    try {
        AutoExposureBenchmark benchmark;
        benchmark.runAllBenchmarks();
        
        spdlog::info("Benchmark completed successfully");
        
    } catch (const std::exception& e) {
        spdlog::error("Exception in benchmark: {}", e.what());
        return -1;
    }
    
    return 0;
}