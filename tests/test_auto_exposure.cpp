#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"
#include <opencv2/opencv.hpp>
#include <memory>

using namespace opencam;
using namespace opencam::algorithms;
using ::testing::_;
using ::testing::Return;
using ::testing::AtLeast;

// Mock camera device for testing
class MockCameraDevice : public CameraDevice {
public:
    MOCK_METHOD(bool, connect, (), (override));
    MOCK_METHOD(bool, disconnect, (), (override));
    MOCK_METHOD(bool, startStreaming, (), (override));
    MOCK_METHOD(bool, stopStreaming, (), (override));
    MOCK_METHOD(CameraState, getState, (), (const, override));
    MOCK_METHOD(CameraProperties, getProperties, (), (const, override));
    MOCK_METHOD(std::shared_ptr<CameraFrame>, getNextFrame, (), (override));
    MOCK_METHOD(bool, setExposure, (float value), (override));
    MOCK_METHOD(bool, setFocus, (float value), (override));
    MOCK_METHOD(bool, setWhiteBalance, (float redGain, float greenGain, float blueGain), (override));
};

class AutoExposureTest : public ::testing::Test {
protected:
    void SetUp() override {
        controller_ = std::make_unique<AutoExposureController>();
        mockCamera_ = std::make_shared<MockCameraDevice>();
        
        // Set up default parameters
        params_.mode = AutoExposureController::Mode::INTELLIGENT;
        params_.targetBrightness = 0.5f;
        params_.convergenceSpeed = 0.15f;
        params_.minExposure = 0.001f;
        params_.maxExposure = 1.0f;
        params_.lockExposure = false;
        
        controller_->setParameters(params_);
    }
    
    CameraFrame createTestFrame(int width = 640, int height = 480, float brightness = 0.5f) {
        CameraFrame frame;
        frame.image = cv::Mat::zeros(height, width, CV_8UC3);
        
        // Fill with specified brightness
        cv::Scalar color(brightness * 255, brightness * 255, brightness * 255);
        frame.image.setTo(color);
        
        frame.frameNumber = frameCounter_++;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        return frame;
    }
    
    CameraFrame createBacklitFrame(int width = 640, int height = 480) {
        CameraFrame frame;
        frame.image = cv::Mat::zeros(height, width, CV_8UC3);
        
        // Create backlit scene: bright background, dark subject in center
        frame.image.setTo(cv::Scalar(240, 240, 240)); // Bright background
        
        // Dark subject in center
        cv::Rect centerRect(width/4, height/4, width/2, height/2);
        frame.image(centerRect).setTo(cv::Scalar(50, 50, 50));
        
        frame.frameNumber = frameCounter_++;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        return frame;
    }
    
    CameraFrame createLowLightFrame(int width = 640, int height = 480) {
        CameraFrame frame;
        frame.image = cv::Mat::zeros(height, width, CV_8UC3);
        
        // Create low light scene with some noise
        cv::Mat noise;
        cv::randn(noise, cv::Scalar(30, 30, 30), cv::Scalar(10, 10, 10));
        frame.image = noise;
        
        frame.frameNumber = frameCounter_++;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        return frame;
    }
    
    std::unique_ptr<AutoExposureController> controller_;
    std::shared_ptr<MockCameraDevice> mockCamera_;
    AutoExposureController::Parameters params_;
    int frameCounter_ = 0;
};

TEST_F(AutoExposureTest, InitializationTest) {
    EXPECT_NO_THROW(AutoExposureController controller);
    
    auto retrievedParams = controller_->getParameters();
    EXPECT_EQ(retrievedParams.mode, AutoExposureController::Mode::INTELLIGENT);
    EXPECT_FLOAT_EQ(retrievedParams.targetBrightness, 0.5f);
    EXPECT_FLOAT_EQ(retrievedParams.convergenceSpeed, 0.15f);
}

TEST_F(AutoExposureTest, ParameterSetGetTest) {
    AutoExposureController::Parameters newParams;
    newParams.mode = AutoExposureController::Mode::CENTER_WEIGHTED;
    newParams.targetBrightness = 0.6f;
    newParams.convergenceSpeed = 0.2f;
    newParams.minExposure = 0.002f;
    newParams.maxExposure = 0.8f;
    newParams.lockExposure = true;
    
    controller_->setParameters(newParams);
    auto retrievedParams = controller_->getParameters();
    
    EXPECT_EQ(retrievedParams.mode, AutoExposureController::Mode::CENTER_WEIGHTED);
    EXPECT_FLOAT_EQ(retrievedParams.targetBrightness, 0.6f);
    EXPECT_FLOAT_EQ(retrievedParams.convergenceSpeed, 0.2f);
    EXPECT_FLOAT_EQ(retrievedParams.minExposure, 0.002f);
    EXPECT_FLOAT_EQ(retrievedParams.maxExposure, 0.8f);
    EXPECT_TRUE(retrievedParams.lockExposure);
}

TEST_F(AutoExposureTest, BasicExposureComputationTest) {
    // Test with normal brightness frame
    auto frame = createTestFrame(640, 480, 0.5f);
    float exposure = controller_->computeExposure(frame);
    
    EXPECT_GT(exposure, 0.0f);
    EXPECT_LT(exposure, 1.0f);
    EXPECT_GE(exposure, params_.minExposure);
    EXPECT_LE(exposure, params_.maxExposure);
}

TEST_F(AutoExposureTest, DarkFrameExposureTest) {
    // Test with dark frame - should increase exposure
    auto darkFrame = createTestFrame(640, 480, 0.2f);
    float initialExposure = controller_->computeExposure(createTestFrame(640, 480, 0.5f));
    float darkExposure = controller_->computeExposure(darkFrame);
    
    EXPECT_GT(darkExposure, initialExposure);
}

TEST_F(AutoExposureTest, BrightFrameExposureTest) {
    // Test with bright frame - should decrease exposure
    auto brightFrame = createTestFrame(640, 480, 0.8f);
    float initialExposure = controller_->computeExposure(createTestFrame(640, 480, 0.5f));
    float brightExposure = controller_->computeExposure(brightFrame);
    
    EXPECT_LT(brightExposure, initialExposure);
}

TEST_F(AutoExposureTest, BacklitSceneTest) {
    auto backlitFrame = createBacklitFrame();
    float exposure = controller_->computeExposure(backlitFrame);
    
    // Should handle backlit scene appropriately
    EXPECT_GT(exposure, 0.0f);
    EXPECT_LT(exposure, 1.0f);
    
    auto sceneAnalysis = controller_->getLastSceneAnalysis();
    // Note: This test might need adjustment based on actual ML model behavior
}

TEST_F(AutoExposureTest, LowLightSceneTest) {
    auto lowLightFrame = createLowLightFrame();
    float exposure = controller_->computeExposure(lowLightFrame);
    
    // Should increase exposure for low light
    EXPECT_GT(exposure, 0.3f); // Should be relatively high for low light
    
    auto sceneAnalysis = controller_->getLastSceneAnalysis();
    // Note: This test might need adjustment based on actual ML model behavior
}

TEST_F(AutoExposureTest, ExposureLockTest) {
    params_.lockExposure = true;
    controller_->setParameters(params_);
    
    auto frame1 = createTestFrame(640, 480, 0.3f);
    auto frame2 = createTestFrame(640, 480, 0.7f);
    
    float exposure1 = controller_->computeExposure(frame1);
    float exposure2 = controller_->computeExposure(frame2);
    
    // When locked, exposure should not change significantly
    EXPECT_NEAR(exposure1, exposure2, 0.01f);
}

TEST_F(AutoExposureTest, ConvergenceTest) {
    // Test convergence behavior
    auto frame = createTestFrame(640, 480, 0.4f);
    
    std::vector<float> exposures;
    for (int i = 0; i < 20; i++) {
        float exposure = controller_->computeExposure(frame);
        exposures.push_back(exposure);
    }
    
    // Should converge over time
    float initialVariance = 0.0f, finalVariance = 0.0f;
    
    // Calculate variance for first 5 frames
    float mean1 = std::accumulate(exposures.begin(), exposures.begin() + 5, 0.0f) / 5.0f;
    for (int i = 0; i < 5; i++) {
        float diff = exposures[i] - mean1;
        initialVariance += diff * diff;
    }
    initialVariance /= 5.0f;
    
    // Calculate variance for last 5 frames
    float mean2 = std::accumulate(exposures.end() - 5, exposures.end(), 0.0f) / 5.0f;
    for (int i = 15; i < 20; i++) {
        float diff = exposures[i] - mean2;
        finalVariance += diff * diff;
    }
    finalVariance /= 5.0f;
    
    // Final variance should be smaller (more converged)
    EXPECT_LT(finalVariance, initialVariance);
}

TEST_F(AutoExposureTest, MeteringModeTest) {
    auto frame = createTestFrame(640, 480, 0.5f);
    
    // Test different metering modes
    std::vector<AutoExposureController::Mode> modes = {
        AutoExposureController::Mode::AVERAGE,
        AutoExposureController::Mode::CENTER_WEIGHTED,
        AutoExposureController::Mode::SPOT,
        AutoExposureController::Mode::MULTI_ZONE,
        AutoExposureController::Mode::INTELLIGENT
    };
    
    std::vector<float> exposures;
    for (auto mode : modes) {
        params_.mode = mode;
        controller_->setParameters(params_);
        controller_->reset(); // Reset state for fair comparison
        
        float exposure = controller_->computeExposure(frame);
        exposures.push_back(exposure);
        
        EXPECT_GT(exposure, 0.0f);
        EXPECT_LT(exposure, 1.0f);
    }
    
    // Different modes should potentially give different results
    // (though with uniform test image, some might be similar)
}

TEST_F(AutoExposureTest, CameraApplicationTest) {
    EXPECT_CALL(*mockCamera_, setExposure(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(true));
    
    auto frame = createTestFrame();
    controller_->computeExposure(frame);
    
    bool result = controller_->applyToCamera(*mockCamera_);
    EXPECT_TRUE(result);
}

TEST_F(AutoExposureTest, CameraApplicationFailureTest) {
    EXPECT_CALL(*mockCamera_, setExposure(_))
        .Times(1)
        .WillOnce(Return(false));
    
    auto frame = createTestFrame();
    controller_->computeExposure(frame);
    
    bool result = controller_->applyToCamera(*mockCamera_);
    EXPECT_FALSE(result);
}

TEST_F(AutoExposureTest, EmptyFrameTest) {
    CameraFrame emptyFrame;
    // Don't set image, leaving it empty
    
    float exposure = controller_->computeExposure(emptyFrame);
    
    // Should return current exposure without crashing
    EXPECT_GT(exposure, 0.0f);
}

TEST_F(AutoExposureTest, StatisticsTest) {
    // Generate some frames to build statistics
    for (int i = 0; i < 10; i++) {
        auto frame = createTestFrame(640, 480, 0.4f + i * 0.02f);
        controller_->computeExposure(frame);
    }
    
    auto stats = controller_->getStatistics();
    
    EXPECT_GT(stats.frameCount, 0);
    EXPECT_GT(stats.averageExposure, 0.0f);
    EXPECT_GT(stats.averageBrightness, 0.0f);
    EXPECT_GE(stats.maxExposure, stats.minExposure);
    EXPECT_GE(stats.convergenceTimeMs, 0);
}

TEST_F(AutoExposureTest, ResetTest) {
    // Build up some state
    for (int i = 0; i < 5; i++) {
        auto frame = createTestFrame();
        controller_->computeExposure(frame);
    }
    
    auto statsBefore = controller_->getStatistics();
    EXPECT_GT(statsBefore.frameCount, 0);
    
    // Reset the controller
    controller_->reset();
    
    auto statsAfter = controller_->getStatistics();
    EXPECT_EQ(statsAfter.frameCount, 0);
    EXPECT_FALSE(statsAfter.isConverged);
}

TEST_F(AutoExposureTest, ExposureClampingTest) {
    // Test minimum exposure clamping
    params_.minExposure = 0.1f;
    params_.maxExposure = 0.9f;
    controller_->setParameters(params_);
    
    // Very dark frame should clamp to minimum
    auto veryDarkFrame = createTestFrame(640, 480, 0.01f);
    float exposure = controller_->computeExposure(veryDarkFrame);
    EXPECT_GE(exposure, params_.minExposure);
    
    // Very bright frame should clamp to maximum
    auto veryBrightFrame = createTestFrame(640, 480, 0.99f);
    exposure = controller_->computeExposure(veryBrightFrame);
    EXPECT_LE(exposure, params_.maxExposure);
}

TEST_F(AutoExposureTest, ConvergenceSpeedTest) {
    // Test different convergence speeds
    auto frame = createTestFrame(640, 480, 0.3f);
    
    // Fast convergence
    params_.convergenceSpeed = 0.8f;
    controller_->setParameters(params_);
    controller_->reset();
    
    float fastExposure1 = controller_->computeExposure(frame);
    float fastExposure2 = controller_->computeExposure(frame);
    float fastChange = std::abs(fastExposure2 - fastExposure1);
    
    // Slow convergence
    params_.convergenceSpeed = 0.1f;
    controller_->setParameters(params_);
    controller_->reset();
    
    float slowExposure1 = controller_->computeExposure(frame);
    float slowExposure2 = controller_->computeExposure(frame);
    float slowChange = std::abs(slowExposure2 - slowExposure1);
    
    // Fast convergence should have larger changes
    EXPECT_GE(fastChange, slowChange);
}

TEST_F(AutoExposureTest, SceneChangeDetectionTest) {
    // Start with one scene
    auto frame1 = createTestFrame(640, 480, 0.3f);
    controller_->computeExposure(frame1);
    controller_->computeExposure(frame1); // Stabilize
    
    float exposure1 = controller_->computeExposure(frame1);
    
    // Switch to very different scene
    auto frame2 = createTestFrame(640, 480, 0.8f);
    float exposure2 = controller_->computeExposure(frame2);
    
    // Should respond to scene change
    EXPECT_NE(exposure1, exposure2);
}

TEST_F(AutoExposureTest, ThreadSafetyTest) {
    // Test concurrent parameter access
    std::vector<std::thread> threads;
    std::atomic<bool> running{true};
    std::atomic<int> errorCount{0};
    
    // Thread that continuously updates parameters
    threads.emplace_back([&]() {
        AutoExposureController::Parameters testParams = params_;
        while (running) {
            testParams.targetBrightness = 0.3f + (rand() % 100) / 200.0f;
            controller_->setParameters(testParams);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Thread that continuously processes frames
    threads.emplace_back([&]() {
        while (running) {
            try {
                auto frame = createTestFrame();
                controller_->computeExposure(frame);
            } catch (...) {
                errorCount++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Thread that continuously reads parameters
    threads.emplace_back([&]() {
        while (running) {
            try {
                auto params = controller_->getParameters();
                auto stats = controller_->getStatistics();
            } catch (...) {
                errorCount++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(errorCount.load(), 0);
}

// Performance test
TEST_F(AutoExposureTest, PerformanceTest) {
    auto frame = createTestFrame(1920, 1080, 0.5f); // High resolution frame
    
    const int iterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        controller_->computeExposure(frame);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    double avgTimePerFrame = duration.count() / static_cast<double>(iterations);
    
    // Should process each frame in reasonable time (less than 10ms for 1080p)
    EXPECT_LT(avgTimePerFrame, 10000.0); // 10ms in microseconds
    
    std::cout << "Average processing time per frame: " << avgTimePerFrame << " microseconds" << std::endl;
}

// Integration test with different image types
TEST_F(AutoExposureTest, ImageTypeTest) {
    // Test with different image formats
    std::vector<int> imageTypes = {CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3};
    
    for (int imageType : imageTypes) {
        CameraFrame frame;
        frame.image = cv::Mat::zeros(480, 640, imageType);
        
        // Fill with test pattern
        if (frame.image.channels() == 1) {
            frame.image.setTo(cv::Scalar(128));
        } else {
            frame.image.setTo(cv::Scalar(128, 128, 128));
        }
        
        frame.frameNumber = frameCounter_++;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        EXPECT_NO_THROW({
            float exposure = controller_->computeExposure(frame);
            EXPECT_GT(exposure, 0.0f);
            EXPECT_LT(exposure, 1.0f);
        });
    }
}

// Test exposure compensation
TEST_F(AutoExposureTest, ExposureCompensationTest) {
    auto frame = createTestFrame(640, 480, 0.5f);
    
    // Test positive compensation
    params_.exposureCompensation = 1.0f; // +1 EV
    controller_->setParameters(params_);
    controller_->reset();
    float exposurePos = controller_->computeExposure(frame);
    
    // Test negative compensation
    params_.exposureCompensation = -1.0f; // -1 EV
    controller_->setParameters(params_);
    controller_->reset();
    float exposureNeg = controller_->computeExposure(frame);
    
    // Test no compensation
    params_.exposureCompensation = 0.0f;
    controller_->setParameters(params_);
    controller_->reset();
    float exposureNeutral = controller_->computeExposure(frame);
    
    // Positive compensation should result in higher exposure
    EXPECT_GT(exposurePos, exposureNeutral);
    // Negative compensation should result in lower exposure
    EXPECT_LT(exposureNeg, exposureNeutral);
}

class AutoExposureParameterizedTest : public AutoExposureTest,
                                     public ::testing::WithParamInterface<AutoExposureController::Mode> {
};

TEST_P(AutoExposureParameterizedTest, MeteringModeConsistencyTest) {
    AutoExposureController::Mode mode = GetParam();
    
    params_.mode = mode;
    controller_->setParameters(params_);
    
    auto frame = createTestFrame(640, 480, 0.5f);
    
    // Process same frame multiple times - should be consistent
    std::vector<float> exposures;
    for (int i = 0; i < 5; i++) {
        float exposure = controller_->computeExposure(frame);
        exposures.push_back(exposure);
    }
    
    // All exposures should be valid
    for (float exposure : exposures) {
        EXPECT_GT(exposure, 0.0f);
        EXPECT_LT(exposure, 1.0f);
        EXPECT_GE(exposure, params_.minExposure);
        EXPECT_LE(exposure, params_.maxExposure);
    }
    
    // Should converge to similar values
    float variance = 0.0f;
    float mean = std::accumulate(exposures.begin(), exposures.end(), 0.0f) / exposures.size();
    for (float exposure : exposures) {
        float diff = exposure - mean;
        variance += diff * diff;
    }
    variance /= exposures.size();
    
    EXPECT_LT(variance, 0.01f); // Should have low variance after convergence
}

INSTANTIATE_TEST_SUITE_P(
    AllMeteringModes,
    AutoExposureParameterizedTest,
    ::testing::Values(
        AutoExposureController::Mode::AVERAGE,
        AutoExposureController::Mode::CENTER_WEIGHTED,
        AutoExposureController::Mode::SPOT,
        AutoExposureController::Mode::MULTI_ZONE,
        AutoExposureController::Mode::INTELLIGENT
    )
);

// Benchmark test for different image sizes
class AutoExposureBenchmarkTest : public AutoExposureTest,
                                 public ::testing::WithParamInterface<std::pair<int, int>> {
};

TEST_P(AutoExposureBenchmarkTest, ImageSizePerformanceTest) {
    auto [width, height] = GetParam();
    auto frame = createTestFrame(width, height, 0.5f);
    
    const int iterations = 50;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        controller_->computeExposure(frame);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    double avgTimePerFrame = duration.count() / static_cast<double>(iterations);
    
    // Performance should scale reasonably with image size
    double pixelCount = width * height;
    double timePerMegapixel = avgTimePerFrame / (pixelCount / 1000000.0);
    
    // Should process roughly within reasonable time per megapixel
    EXPECT_LT(timePerMegapixel, 50000.0); // 50ms per megapixel
    
    std::cout << "Resolution: " << width << "x" << height 
              << ", Avg time: " << avgTimePerFrame << "μs"
              << ", Time per MP: " << timePerMegapixel << "μs" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    DifferentResolutions,
    AutoExposureBenchmarkTest,
    ::testing::Values(
        std::make_pair(320, 240),   // QVGA
        std::make_pair(640, 480),   // VGA
        std::make_pair(1280, 720),  // HD
        std::make_pair(1920, 1080), // Full HD
        std::make_pair(3840, 2160)  // 4K
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}