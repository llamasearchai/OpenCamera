#include "opencam/camera.h"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

namespace opencam {

// Internal camera implementation that uses OpenCV's video capture
class OpenCVCameraDevice : public CameraDevice {
public:
    explicit OpenCVCameraDevice(const std::string& deviceId) 
        : deviceId_(deviceId), state_(CameraState::DISCONNECTED) {
        
        properties_.deviceId = deviceId;
        properties_.width = 1280;
        properties_.height = 720;
        properties_.fps = 30.0f;
        properties_.hasAutoExposure = true;
        properties_.hasAutoFocus = true;
        properties_.hasAutoWhiteBalance = true;
    }
    
    ~OpenCVCameraDevice() override {
        if (state_ != CameraState::DISCONNECTED) {
            disconnect();
        }
    }
    
    bool connect() override {
        if (state_ != CameraState::DISCONNECTED) {
            spdlog::warn("Camera already connected");
            return false;
        }
        
        try {
            // Convert deviceId to integer if it's numeric
            int id = -1;
            try {
                id = std::stoi(deviceId_);
            } catch (...) {
                // If not numeric, use as is
            }
            
            if (id >= 0) {
                capture_.open(id);
            } else {
                capture_.open(deviceId_);
            }
            
            if (!capture_.isOpened()) {
                spdlog::error("Failed to open camera device: {}", deviceId_);
                state_ = CameraState::ERROR;
                return false;
            }
            
            // Set default properties
            capture_.set(cv::CAP_PROP_FRAME_WIDTH, properties_.width);
            capture_.set(cv::CAP_PROP_FRAME_HEIGHT, properties_.height);
            capture_.set(cv::CAP_PROP_FPS, properties_.fps);
            
            // Read actual properties
            properties_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            properties_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            properties_.fps = capture_.get(cv::CAP_PROP_FPS);
            
            state_ = CameraState::CONNECTED;
            spdlog::info("Connected to camera: {}", deviceId_);
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Exception connecting to camera: {}", e.what());
            state_ = CameraState::ERROR;
            return false;
        }
    }
    
    bool disconnect() override {
        if (state_ == CameraState::STREAMING) {
            stopStreaming();
        }
        
        if (state_ == CameraState::DISCONNECTED) {
            return true;
        }
        
        capture_.release();
        state_ = CameraState::DISCONNECTED;
        spdlog::info("Disconnected from camera: {}", deviceId_);
        return true;
    }
    
    bool startStreaming() override {
        if (state_ != CameraState::CONNECTED) {
            spdlog::warn("Camera not in connected state. Current state: {}", 
                static_cast<int>(state_));
            return false;
        }
        
        state_ = CameraState::STREAMING;
        frameCount_ = 0;
        spdlog::info("Started streaming from camera: {}", deviceId_);
        return true;
    }
    
    bool stopStreaming() override {
        if (state_ != CameraState::STREAMING) {
            return false;
        }
        
        state_ = CameraState::CONNECTED;
        spdlog::info("Stopped streaming from camera: {}", deviceId_);
        return true;
    }
    
    CameraState getState() const override {
        return state_;
    }
    
    CameraProperties getProperties() const override {
        return properties_;
    }
    
    bool setExposure(float value) override {
        if (state_ == CameraState::DISCONNECTED || state_ == CameraState::ERROR) {
            return false;
        }
        
        return capture_.set(cv::CAP_PROP_EXPOSURE, value);
    }
    
    bool setFocus(float value) override {
        if (state_ == CameraState::DISCONNECTED || state_ == CameraState::ERROR) {
            return false;
        }
        
        return capture_.set(cv::CAP_PROP_FOCUS, value);
    }
    
    bool setWhiteBalance(float redGain, float greenGain, float blueGain) override {
        if (state_ == CameraState::DISCONNECTED || state_ == CameraState::ERROR) {
            return false;
        }
        
        // Not all cameras support separate RGB gains
        // This is a simplified implementation
        return capture_.set(cv::CAP_PROP_WHITE_BALANCE_BLUE_U, blueGain) &&
               capture_.set(cv::CAP_PROP_WHITE_BALANCE_RED_V, redGain);
    }
    
    std::shared_ptr<CameraFrame> getNextFrame() override {
        if (state_ != CameraState::STREAMING) {
            spdlog::warn("Camera not streaming. Current state: {}", 
                static_cast<int>(state_));
            return nullptr;
        }
        
        auto frame = std::make_shared<CameraFrame>();
        
        if (!capture_.read(frame->image)) {
            spdlog::error("Failed to read frame from camera: {}", deviceId_);
            return nullptr;
        }
        
        frame->frameNumber = frameCount_++;
        frame->timestamp = static_cast<int64_t>(1000.0 * capture_.get(cv::CAP_PROP_POS_MSEC));
        
        // Add metadata
        frame->metadata["exposure"] = static_cast<float>(capture_.get(cv::CAP_PROP_EXPOSURE));
        frame->metadata["brightness"] = static_cast<float>(capture_.get(cv::CAP_PROP_BRIGHTNESS));
        frame->metadata["fps"] = static_cast<float>(capture_.get(cv::CAP_PROP_FPS));
        
        return frame;
    }
    
private:
    std::string deviceId_;
    CameraState state_;
    CameraProperties properties_;
    cv::VideoCapture capture_;
    int frameCount_ = 0;
};

// Camera Manager implementation
CameraManager& CameraManager::getInstance() {
    static CameraManager instance;
    return instance;
}

std::vector<std::string> CameraManager::enumerateDevices() {
    std::vector<std::string> devices;
    
    // Try to find camera devices (this is platform-dependent)
    // In a real implementation, this would use platform-specific APIs
    for (int i = 0; i < 10; i++) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            devices.push_back(std::to_string(i));
            cap.release();
        }
    }
    
    spdlog::info("Found {} camera devices", devices.size());
    return devices;
}

std::shared_ptr<CameraDevice> CameraManager::getCamera(const std::string& deviceId) {
    return std::make_shared<OpenCVCameraDevice>(deviceId);
}

} // namespace opencam