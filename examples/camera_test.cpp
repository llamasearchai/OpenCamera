#include "opencam/camera.h"
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <thread>

using namespace opencam;
using namespace testing;
using namespace std::chrono_literals;

class CameraDeviceTest : public ::testing::Test {
protected:
  void SetUp() override { manager_ = &CameraManager::getInstance(); }

  CameraManager *manager_;
};

TEST_F(CameraDeviceTest, EnumerateDevices) {
  auto devices = manager_->enumerateDevices();

  // We may have no devices in a CI environment
  EXPECT_GE(devices.size(), 0);

  if (!devices.empty()) {
    for (const auto &device : devices) {
      EXPECT_FALSE(device.empty());
    }
  }
}

TEST_F(CameraDeviceTest, GetCameraDevice) {
  auto camera = manager_->getCamera("0");
  EXPECT_NE(camera, nullptr);

  // Initial state should be disconnected
  EXPECT_EQ(camera->getState(), CameraState::DISCONNECTED);
}

TEST_F(CameraDeviceTest, ConnectDisconnect) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  // In CI or without a real camera, connect might fail
  bool connected = camera->connect();

  if (connected) {
    EXPECT_EQ(camera->getState(), CameraState::CONNECTED);

    // Check properties
    auto props = camera->getProperties();
    EXPECT_GT(props.width, 0);
    EXPECT_GT(props.height, 0);
    EXPECT_GT(props.fps, 0.0f);

    // Disconnect
    EXPECT_TRUE(camera->disconnect());
    EXPECT_EQ(camera->getState(), CameraState::DISCONNECTED);
  } else {
    // If no camera available, ensure we're in error state
    EXPECT_EQ(camera->getState(), CameraState::ERROR);
  }
}

TEST_F(CameraDeviceTest, StreamingLifecycle) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  // Can't start streaming when disconnected
  EXPECT_FALSE(camera->startStreaming());

  if (camera->connect()) {
    // Now we can start streaming
    EXPECT_TRUE(camera->startStreaming());
    EXPECT_EQ(camera->getState(), CameraState::STREAMING);

    // Get a few frames
    for (int i = 0; i < 3; i++) {
      auto frame = camera->getNextFrame();
      if (frame) {
        EXPECT_FALSE(frame->image.empty());
        EXPECT_EQ(frame->frameNumber, i);
        EXPECT_GT(frame->timestamp, 0);
      }
      std::this_thread::sleep_for(33ms); // ~30fps
    }

    // Stop streaming
    EXPECT_TRUE(camera->stopStreaming());
    EXPECT_EQ(camera->getState(), CameraState::CONNECTED);

    // Can't get frames when not streaming
    EXPECT_EQ(camera->getNextFrame(), nullptr);

    camera->disconnect();
  }
}

TEST_F(CameraDeviceTest, CameraControls) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  // Controls should fail when disconnected
  EXPECT_FALSE(camera->setExposure(0.5f));
  EXPECT_FALSE(camera->setFocus(0.5f));
  EXPECT_FALSE(camera->setWhiteBalance(1.0f, 1.0f, 1.0f));

  if (camera->connect()) {
    // These might fail depending on camera capabilities
    // but should not crash
    camera->setExposure(0.5f);
    camera->setFocus(0.5f);
    camera->setWhiteBalance(1.0f, 1.0f, 1.0f);

    camera->disconnect();
  }
}

TEST_F(CameraDeviceTest, InvalidDeviceId) {
  auto camera = manager_->getCamera("invalid_device_9999");
  ASSERT_NE(camera, nullptr);

  EXPECT_FALSE(camera->connect());
  EXPECT_EQ(camera->getState(), CameraState::ERROR);
}

TEST_F(CameraDeviceTest, MultipleConnectAttempts) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  if (camera->connect()) {
    // Second connect should fail
    EXPECT_FALSE(camera->connect());
    EXPECT_EQ(camera->getState(), CameraState::CONNECTED);

    camera->disconnect();
  }
}

TEST_F(CameraDeviceTest, FrameMetadata) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  if (camera->connect() && camera->startStreaming()) {
    auto frame = camera->getNextFrame();
    if (frame) {
      // Check metadata exists
      EXPECT_TRUE(frame->metadata.find("exposure") != frame->metadata.end());
      EXPECT_TRUE(frame->metadata.find("brightness") != frame->metadata.end());
      EXPECT_TRUE(frame->metadata.find("fps") != frame->metadata.end());
    }

    camera->stopStreaming();
    camera->disconnect();
  }
}

// Performance test
TEST_F(CameraDeviceTest, FrameRatePerformance) {
  auto camera = manager_->getCamera("0");
  ASSERT_NE(camera, nullptr);

  if (camera->connect() && camera->startStreaming()) {
    const int numFrames = 30;
    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < numFrames; i++) {
      auto frame = camera->getNextFrame();
      if (!frame)
        break;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);

    if (duration.count() > 0) {
      double fps = (numFrames * 1000.0) / duration.count();
      EXPECT_GT(fps, 10.0); // At least 10 FPS
    }

    camera->stopStreaming();
    camera->disconnect();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Set up any global test environment if needed
  spdlog::set_level(spdlog::level::warn);

  return RUN_ALL_TESTS();
}