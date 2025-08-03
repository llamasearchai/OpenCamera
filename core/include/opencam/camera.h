#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace opencam {

enum class CameraState { DISCONNECTED, CONNECTED, STREAMING, ERROR };

struct CameraProperties {
  int width = 0;
  int height = 0;
  float fps = 0.0f;
  bool hasAutoExposure = false;
  bool hasAutoFocus = false;
  bool hasAutoWhiteBalance = false;
  std::string deviceId;
};

struct CameraFrame {
  cv::Mat image{};
  int64_t timestamp = 0;
  int frameNumber = 0;
  std::unordered_map<std::string, float> metadata{};

  CameraFrame() = default;
};

class CameraDevice {
public:
  virtual ~CameraDevice() = default;

  virtual bool connect() = 0;
  virtual bool disconnect() = 0;
  virtual bool startStreaming() = 0;
  virtual bool stopStreaming() = 0;
  virtual CameraState getState() const = 0;
  virtual CameraProperties getProperties() const = 0;

  virtual bool setExposure(float value) = 0;
  virtual bool setFocus(float value) = 0;
  virtual bool setWhiteBalance(float redGain, float greenGain,
                               float blueGain) = 0;

  virtual std::shared_ptr<CameraFrame> getNextFrame() = 0;
};

class CameraManager {
public:
  static CameraManager &getInstance();

  std::vector<std::string> enumerateDevices();
  std::shared_ptr<CameraDevice> getCamera(const std::string &deviceId);

private:
  CameraManager() = default;
  ~CameraManager() = default;
  CameraManager(const CameraManager &) = delete;
  CameraManager &operator=(const CameraManager &) = delete;
};

} // namespace opencam