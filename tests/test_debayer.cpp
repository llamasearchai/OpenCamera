#include "algorithms/isp/debayer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <spdlog/spdlog.h>

using namespace opencam::algorithms;
using namespace testing;

class DebayerTest : public ::testing::Test {
protected:
  void SetUp() override { debayer_ = std::make_unique<Debayer>(); }

  cv::Mat createBayerPattern(int width, int height, BayerPattern pattern) {
    cv::Mat bayer(height, width, CV_8UC1);

    // Fill with a simple test pattern
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        uint8_t value = 0;

        // Create a gradient pattern based on position
        switch (pattern) {
        case BayerPattern::RGGB:
          if ((y % 2 == 0) && (x % 2 == 0))
            value = 200; // R
          else if ((y % 2 == 0) && (x % 2 == 1))
            value = 150; // G
          else if ((y % 2 == 1) && (x % 2 == 0))
            value = 150; // G
          else
            value = 100; // B
          break;
        case BayerPattern::BGGR:
          if ((y % 2 == 0) && (x % 2 == 0))
            value = 100; // B
          else if ((y % 2 == 0) && (x % 2 == 1))
            value = 150; // G
          else if ((y % 2 == 1) && (x % 2 == 0))
            value = 150; // G
          else
            value = 200; // R
          break;
        case BayerPattern::GRBG:
          if ((y % 2 == 0) && (x % 2 == 0))
            value = 150; // G
          else if ((y % 2 == 0) && (x % 2 == 1))
            value = 200; // R
          else if ((y % 2 == 1) && (x % 2 == 0))
            value = 100; // B
          else
            value = 150; // G
          break;
        case BayerPattern::GBRG:
          if ((y % 2 == 0) && (x % 2 == 0))
            value = 150; // G
          else if ((y % 2 == 0) && (x % 2 == 1))
            value = 100; // B
          else if ((y % 2 == 1) && (x % 2 == 0))
            value = 200; // R
          else
            value = 150; // G
          break;
        }

        bayer.at<uint8_t>(y, x) = value;
      }
    }

    return bayer;
  }

  std::unique_ptr<Debayer> debayer_;
};

TEST_F(DebayerTest, BasicFunctionality) {
  // Create a simple Bayer pattern image
  cv::Mat bayer = createBayerPattern(640, 480, BayerPattern::RGGB);

  // Process with default parameters
  cv::Mat rgb = debayer_->process(bayer);

  EXPECT_FALSE(rgb.empty());
  EXPECT_EQ(rgb.rows, bayer.rows);
  EXPECT_EQ(rgb.cols, bayer.cols);
  EXPECT_EQ(rgb.channels(), 3);
}

TEST_F(DebayerTest, EmptyImage) {
  cv::Mat empty;
  cv::Mat result = debayer_->process(empty);
  EXPECT_TRUE(result.empty());
}

TEST_F(DebayerTest, InvalidImageType) {
  // Create a 3-channel image (invalid for debayering)
  cv::Mat invalid(100, 100, CV_8UC3);
  cv::Mat result = debayer_->process(invalid);
  EXPECT_TRUE(result.empty());
}

TEST_F(DebayerTest, DifferentBayerPatterns) {
  const std::vector<BayerPattern> patterns = {
      BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG,
      BayerPattern::GBRG};

  for (auto pattern : patterns) {
    cv::Mat bayer = createBayerPattern(320, 240, pattern);

    Debayer::Parameters params;
    params.pattern = pattern;
    debayer_->setParameters(params);

    cv::Mat rgb = debayer_->process(bayer);

    EXPECT_FALSE(rgb.empty());
    EXPECT_EQ(rgb.channels(), 3);
  }
}

TEST_F(DebayerTest, DifferentMethods) {
  cv::Mat bayer = createBayerPattern(320, 240, BayerPattern::RGGB);

  const std::vector<Debayer::Method> methods = {
      Debayer::Method::NEAREST_NEIGHBOR, Debayer::Method::BILINEAR,
      Debayer::Method::MALVAR, Debayer::Method::ADAPTIVE};

  for (auto method : methods) {
    Debayer::Parameters params;
    params.method = method;
    debayer_->setParameters(params);

    cv::Mat rgb = debayer_->process(bayer);

    EXPECT_FALSE(rgb.empty());
    EXPECT_EQ(rgb.channels(), 3);
  }
}

TEST_F(DebayerTest, Bit16Support) {
  // Create 16-bit Bayer pattern
  cv::Mat bayer16(480, 640, CV_16UC1);
  cv::randu(bayer16, 0, 65535);

  cv::Mat rgb = debayer_->process(bayer16);

  EXPECT_FALSE(rgb.empty());
  EXPECT_EQ(rgb.channels(), 3);
}

TEST_F(DebayerTest, GetSetParameters) {
  Debayer::Parameters params;
  params.pattern = BayerPattern::GRBG;
  params.method = Debayer::Method::ADAPTIVE;

  debayer_->setParameters(params);

  auto retrievedParams = debayer_->getParameters();
  EXPECT_EQ(retrievedParams.pattern, params.pattern);
  EXPECT_EQ(retrievedParams.method, params.method);
}

TEST_F(DebayerTest, PerformanceCheck) {
  // Create a larger image for performance testing
  cv::Mat bayer = createBayerPattern(1920, 1080, BayerPattern::RGGB);

  auto start = std::chrono::high_resolution_clock::now();
  cv::Mat rgb = debayer_->process(bayer);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  EXPECT_FALSE(rgb.empty());
  // Should process Full HD in reasonable time (less than 100ms)
  EXPECT_LT(duration.count(), 100);
}

TEST_F(DebayerTest, ColorAccuracy) {
  // Create a Bayer pattern with known color values
  cv::Mat bayer(4, 4, CV_8UC1);

  // RGGB pattern with specific values
  bayer.at<uint8_t>(0, 0) = 255; // R
  bayer.at<uint8_t>(0, 1) = 128; // G
  bayer.at<uint8_t>(1, 0) = 128; // G
  bayer.at<uint8_t>(1, 1) = 0;   // B

  // Repeat pattern
  bayer.at<uint8_t>(0, 2) = 255; // R
  bayer.at<uint8_t>(0, 3) = 128; // G
  bayer.at<uint8_t>(1, 2) = 128; // G
  bayer.at<uint8_t>(1, 3) = 0;   // B

  bayer.at<uint8_t>(2, 0) = 255; // R
  bayer.at<uint8_t>(2, 1) = 128; // G
  bayer.at<uint8_t>(3, 0) = 128; // G
  bayer.at<uint8_t>(3, 1) = 0;   // B

  bayer.at<uint8_t>(2, 2) = 255; // R
  bayer.at<uint8_t>(2, 3) = 128; // G
  bayer.at<uint8_t>(3, 2) = 128; // G
  bayer.at<uint8_t>(3, 3) = 0;   // B

  Debayer::Parameters params;
  params.pattern = BayerPattern::RGGB;
  params.method = Debayer::Method::NEAREST_NEIGHBOR;
  debayer_->setParameters(params);

  cv::Mat rgb = debayer_->process(bayer);

  EXPECT_FALSE(rgb.empty());

  // Check that red channel is high where we expect
  cv::Vec3b pixel = rgb.at<cv::Vec3b>(0, 0);
  EXPECT_GT(pixel[2], 200); // Red channel should be high
}

TEST_F(DebayerTest, EdgePreservation) {
  // Create a Bayer pattern with sharp edge
  cv::Mat bayer(100, 100, CV_8UC1);

  // Left half dark, right half bright
  for (int y = 0; y < 100; y++) {
    for (int x = 0; x < 50; x++) {
      bayer.at<uint8_t>(y, x) = 50;
    }
    for (int x = 50; x < 100; x++) {
      bayer.at<uint8_t>(y, x) = 200;
    }
  }

  cv::Mat rgb = debayer_->process(bayer);

  EXPECT_FALSE(rgb.empty());

  // Check that edge is reasonably preserved
  cv::Mat gray;
  cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

  // Calculate gradient at the edge
  int leftValue = gray.at<uint8_t>(50, 48);
  int rightValue = gray.at<uint8_t>(50, 52);

  EXPECT_LT(leftValue, 100);
  EXPECT_GT(rightValue, 150);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Set up any global test environment if needed
  spdlog::set_level(spdlog::level::warn);

  return RUN_ALL_TESTS();
}