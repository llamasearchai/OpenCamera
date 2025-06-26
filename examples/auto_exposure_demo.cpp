#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <thread>
#include <chrono>

using namespace opencam;
using namespace opencam::algorithms;

class AutoExposureDemo {
public:
    AutoExposureDemo() : running_(false) {
        // Initialize auto exposure controller
        controller_ = std::make_unique<AutoExposureController>();
        
        // Set up parameters
        AutoExposureController::Parameters params;
        params.mode = AutoExposureController::Mode::INTELLIGENT;
        params.targetBrightness = 0.5f;
        params.convergenceSpeed = 0.15f;
        params.enableFaceDetection = true;
        params.enableSceneAnalysis = true;
        controller_->setParameters(params);
        
        spdlog::info("Auto Exposure Demo initialized");
    }
    
    ~AutoExposureDemo() {
        stop();
    }
    
    bool initialize(const std::string& cameraId = "0") {
        // Get camera manager and initialize camera
        auto& manager = CameraManager::getInstance();
        camera_ = manager.getCamera(cameraId);
        
        if (!camera_) {
            spdlog::error("Failed to create camera device");
            return false;
        }
        
        if (!camera_->connect()) {
            spdlog::error("Failed to connect to camera");
            return false;
        }
        
        if (!camera_->startStreaming()) {
            spdlog::error("Failed to start camera streaming");
            return false;
        }
        
        spdlog::info("Camera initialized successfully");
        return true;
    }
    
    void run() {
        if (!camera_) {
            spdlog::error("Camera not initialized");
            return;
        }
        
        running_ = true;
        
        // Create windows for display
        cv::namedWindow("Camera Feed", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Controls", cv::WINDOW_AUTOSIZE);
        createControlPanel();
        
        spdlog::info("Starting auto exposure demo. Press 'q' to quit.");
        
        auto lastStatsTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        
        while (running_) {
            auto frame = camera_->getNextFrame();
            if (!frame || frame->image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            // Process auto exposure
            float exposure = controller_->computeExposure(*frame);
            controller_->applyToCamera(*camera_);
            
            // Update display
            updateDisplay(*frame, exposure);
            
            // Handle keyboard input
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
            handleKeyPress(key);
            
            // Update statistics periodically
            frameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatsTime).count() >= 2) {
                updateStatistics(frameCount, now - lastStatsTime);
                lastStatsTime = now;
                frameCount = 0;
            }
        }
        
        cv::destroyAllWindows();
        spdlog::info("Demo stopped");
    }
    
    void stop() {
        running_ = false;
        if (camera_) {
            camera_->stopStreaming();
            camera_->disconnect();
        }
    }

private:
    void createControlPanel() {
        // Create trackbars for parameter adjustment
        cv::createTrackbar("Target Brightness", "Controls", nullptr, 100, onTargetBrightnessChange, this);
        cv::createTrackbar("Convergence Speed", "Controls", nullptr, 100, onConvergenceSpeedChange, this);
        cv::createTrackbar("Exposure Compensation", "Controls", nullptr, 200, onExposureCompensationChange, this);
        cv::createTrackbar("Metering Mode", "Controls", nullptr, 4, onMeteringModeChange, this);
        
        // Set initial values
        auto params = controller_->getParameters();
        cv::setTrackbarPos("Target Brightness", "Controls", static_cast<int>(params.targetBrightness * 100));
        cv::setTrackbarPos("Convergence Speed", "Controls", static_cast<int>(params.convergenceSpeed * 100));
        cv::setTrackbarPos("Exposure Compensation", "Controls", static_cast<int>((params.exposureCompensation + 2.0f) * 50));
        cv::setTrackbarPos("Metering Mode", "Controls", static_cast<int>(params.mode));
    }
    
    void updateDisplay(const CameraFrame& frame, float exposure) {
        cv::Mat displayImage = frame.image.clone();
        
        // Add overlay information
        addOverlayInfo(displayImage, frame, exposure);
        
        // Show the image
        cv::imshow("Camera Feed", displayImage);
    }
    
    void addOverlayInfo(cv::Mat& image, const CameraFrame& frame, float exposure) {
        // Get statistics
        auto stats = controller_->getStatistics();
        auto sceneAnalysis = controller_->getLastSceneAnalysis();
        auto params = controller_->getParameters();
        
        // Prepare overlay text
        std::vector<std::string> overlayText;
        overlayText.push_back(fmt::format("Frame: {}", frame.frameNumber));
        overlayText.push_back(fmt::format("Exposure: {:.4f}", exposure));
        overlayText.push_back(fmt::format("Target: {:.2f}", params.targetBrightness));
        overlayText.push_back(fmt::format("Avg Brightness: {:.3f}", stats.averageBrightness));
        overlayText.push_back(fmt::format("Converged: {}", stats.isConverged ? "Yes" : "No"));
        overlayText.push_back(fmt::format("Mode: {}", getModeString(params.mode)));
        
        if (params.enableSceneAnalysis) {
            overlayText.push_back(fmt::format("Scene: {}", sceneAnalysis.sceneType));
            overlayText.push_back(fmt::format("Low Light: {}", sceneAnalysis.isLowLight ? "Yes" : "No"));
            overlayText.push_back(fmt::format("Backlit: {}", sceneAnalysis.isBacklit ? "Yes" : "No"));
            overlayText.push_back(fmt::format("Faces: {}", sceneAnalysis.faceRegions.size()));
        }
        
        // Draw overlay
        int y = 30;
        for (const auto& text : overlayText) {
            cv::putText(image, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 
                       0.6, cv::Scalar(0, 255, 0), 2);
            y += 25;
        }
        
        // Draw face rectangles if detected
        if (params.enableFaceDetection) {
            for (const auto& face : sceneAnalysis.faceRegions) {
                cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);
                cv::putText(image, "Face", cv::Point(face.x, face.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }
        
        // Draw histogram
        drawHistogram(image);
        
        // Draw convergence indicator
        drawConvergenceIndicator(image, stats.isConverged);
    }
    
    void drawHistogram(cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // Calculate histogram
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
        
        // Normalize histogram
        cv::normalize(hist, hist, 0, 100, cv::NORM_MINMAX);
        
        // Draw histogram
        int histWidth = 256;
        int histHeight = 100;
        int binWidth = cvRound((double)histWidth / histSize);
        
        cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        
        for (int i = 1; i < histSize; i++) {
            cv::line(histImage,
                    cv::Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
                    cv::Point(binWidth * i, histHeight - cvRound(hist.at<float>(i))),
                    cv::Scalar(255, 255, 255), 1, 8, 0);
        }
        
        // Overlay histogram on main image
        cv::Rect histRect(image.cols - histWidth - 10, 10, histWidth, histHeight);
        cv::Mat roi = image(histRect);
        cv::addWeighted(roi, 0.3, histImage, 0.7, 0, roi);
        
        // Draw target brightness line
        auto params = controller_->getParameters();
        int targetX = static_cast<int>(params.targetBrightness * histWidth);
        cv::line(image, 
                cv::Point(image.cols - histWidth - 10 + targetX, 10),
                cv::Point(image.cols - histWidth - 10 + targetX, 10 + histHeight),
                cv::Scalar(0, 255, 0), 2);
    }
    
    void drawConvergenceIndicator(cv::Mat& image, bool converged) {
        cv::Scalar color = converged ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string text = converged ? "CONVERGED" : "ADJUSTING";
        
        cv::putText(image, text, cv::Point(image.cols - 150, image.rows - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
    
    void updateStatistics(int frameCount, std::chrono::high_resolution_clock::duration duration) {
        auto stats = controller_->getStatistics();
        double fps = frameCount / std::chrono::duration<double>(duration).count();
        
        spdlog::info("FPS: {:.1f}, Avg Exposure: {:.4f}, Avg Brightness: {:.3f}, Converged: {}", 
                    fps, stats.averageExposure, stats.averageBrightness, stats.isConverged);
    }
    
    void handleKeyPress(int key) {
        auto params = controller_->getParameters();
        
        switch (key) {
            case '1':
                params.mode = AutoExposureController::Mode::AVERAGE;
                spdlog::info("Switched to Average metering");
                break;
            case '2':
                params.mode = AutoExposureController::Mode::CENTER_WEIGHTED;
                spdlog::info("Switched to Center-weighted metering");
                break;
            case '3':
                params.mode = AutoExposureController::Mode::SPOT;
                spdlog::info("Switched to Spot metering");
                break;
            case '4':
                params.mode = AutoExposureController::Mode::MULTI_ZONE;
                spdlog::info("Switched to Multi-zone metering");
                break;
            case '5':
                params.mode = AutoExposureController::Mode::INTELLIGENT;
                spdlog::info("Switched to Intelligent metering");
                break;
            case 'l':
                params.lockExposure = !params.lockExposure;
                spdlog::info("Exposure lock: {}", params.lockExposure ? "ON" : "OFF");
                break;
            case 'f':
                params.enableFaceDetection = !params.enableFaceDetection;
                spdlog::info("Face detection: {}", params.enableFaceDetection ? "ON" : "OFF");
                break;
            case 's':
                params.enableSceneAnalysis = !params.enableSceneAnalysis;
                spdlog::info("Scene analysis: {}", params.enableSceneAnalysis ? "ON" : "OFF");
                break;
            case 'r':
                controller_->reset();
                spdlog::info("Auto exposure controller reset");
                break;
            case '+':
            case '=':
                params.exposureCompensation = std::min(2.0f, params.exposureCompensation + 0.1f);
                spdlog::info("Exposure compensation: {:.1f} EV", params.exposureCompensation);
                break;
            case '-':
                params.exposureCompensation = std::max(-2.0f, params.exposureCompensation - 0.1f);
                spdlog::info("Exposure compensation: {:.1f} EV", params.exposureCompensation);
                break;
            case 'h':
                printHelp();
                break;
        }
        
        controller_->setParameters(params);
    }
    
    void printHelp() {
        std::cout << "\n=== Auto Exposure Demo Controls ===" << std::endl;
        std::cout << "1-5: Switch metering modes" << std::endl;
        std::cout << "l: Toggle exposure lock" << std::endl;
        std::cout << "f: Toggle face detection" << std::endl;
        std::cout << "s: Toggle scene analysis" << std::endl;
        std::cout << "r: Reset controller" << std::endl;
        std::cout << "+/-: Adjust exposure compensation" << std::endl;
        std::cout << "h: Show this help" << std::endl;
        std::cout << "q/ESC: Quit" << std::endl;
        std::cout << "===================================" << std::endl;
    }
    
    std::string getModeString(AutoExposureController::Mode mode) {
        switch (mode) {
            case AutoExposureController::Mode::AVERAGE: return "Average";
            case AutoExposureController::Mode::CENTER_WEIGHTED: return "Center-weighted";
            case AutoExposureController::Mode::SPOT: return "Spot";
            case AutoExposureController::Mode::MULTI_ZONE: return "Multi-zone";
            case AutoExposureController::Mode::INTELLIGENT: return "Intelligent";
            default: return "Unknown";
        }
    }
    
    // Trackbar callbacks
    static void onTargetBrightnessChange(int value, void* userdata) {
        auto* demo = static_cast<AutoExposureDemo*>(userdata);
        auto params = demo->controller_->getParameters();
        params.targetBrightness = value / 100.0f;
        demo->controller_->setParameters(params);
    }
    
    static void onConvergenceSpeedChange(int value, void* userdata) {
        auto* demo = static_cast<AutoExposureDemo*>(userdata);
        auto params = demo->controller_->getParameters();
        params.convergenceSpeed = value / 100.0f;
        demo->controller_->setParameters(params);
    }
    
    static void onExposureCompensationChange(int value, void* userdata) {
        auto* demo = static_cast<AutoExposureDemo*>(userdata);
        auto params = demo->controller_->getParameters();
        params.exposureCompensation = (value / 50.0f) - 2.0f;
        demo->controller_->setParameters(params);
    }
    
    static void onMeteringModeChange(int value, void* userdata) {
        auto* demo = static_cast<AutoExposureDemo*>(userdata);
        auto params = demo->controller_->getParameters();
        params.mode = static_cast<AutoExposureController::Mode>(value);
        demo->controller_->setParameters(params);
    }

private:
    std::unique_ptr<AutoExposureController> controller_;
    std::shared_ptr<CameraDevice> camera_;
    std::atomic<bool> running_;
};

int main(int argc, char* argv[]) {
    // Set up logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S.%e] [%l] %v");
    
    std::string cameraId = "0";
    if (argc > 1) {
        cameraId = argv[1];
    }
    
    try {
        AutoExposureDemo demo;
        
        if (!demo.initialize(cameraId)) {
            spdlog::error("Failed to initialize demo");
            return -1;
        }
        
        // Print initial help
        std::cout << "Auto Exposure Demo starting..." << std::endl;
        std::cout << "Press 'h' for help, 'q' to quit" << std::endl;
        
        demo.run();
        
    } catch (const std::exception& e) {
        spdlog::error("Exception in demo: {}", e.what());
        return -1;
    }
    
    return 0;
}