#include "debayer.h"
#include <spdlog/spdlog.h>

namespace opencam {
namespace algorithms {

Debayer::Debayer() {
  // Initialize with default parameters
}

void Debayer::setParameters(const Parameters &params) { params_ = params; }

Debayer::Parameters Debayer::getParameters() const { return params_; }

cv::Mat Debayer::process(const cv::Mat &bayerImage) {
  if (bayerImage.empty()) {
    spdlog::error("Empty image provided to debayer processor");
    return cv::Mat();
  }

  if (bayerImage.type() != CV_8UC1 && bayerImage.type() != CV_16UC1) {
    spdlog::error("Debayer requires 8-bit or 16-bit single channel image");
    return cv::Mat();
  }

  cv::Mat result;

  // Select debayering method
  switch (params_.method) {
  case Method::NEAREST_NEIGHBOR:
    result = processNearestNeighbor(bayerImage, params_.pattern);
    break;
  case Method::BILINEAR:
    result = processBilinear(bayerImage, params_.pattern);
    break;
  case Method::MALVAR:
    result = processMalvar(bayerImage, params_.pattern);
    break;
  case Method::ADAPTIVE:
    result = processAdaptive(bayerImage, params_.pattern);
    break;
  default:
    // Fallback to OpenCV's implementation if method not implemented
    int cvPattern;
    switch (params_.pattern) {
    case BayerPattern::RGGB:
      cvPattern = cv::COLOR_BayerBG2BGR;
      break;
    case BayerPattern::BGGR:
      cvPattern = cv::COLOR_BayerRG2BGR;
      break;
    case BayerPattern::GRBG:
      cvPattern = cv::COLOR_BayerGB2BGR;
      break;
    case BayerPattern::GBRG:
      cvPattern = cv::COLOR_BayerGR2BGR;
      break;
    default:
      cvPattern = cv::COLOR_BayerBG2BGR;
    }
    cv::cvtColor(bayerImage, result, cvPattern);
    break;
  }

  return result;
}

cv::Mat Debayer::processNearestNeighbor(const cv::Mat &bayerImage,
                                        BayerPattern pattern) {
  // For demonstration, we'll use OpenCV's implementation
  int cvPattern;
  switch (pattern) {
  case BayerPattern::RGGB:
    cvPattern = cv::COLOR_BayerBG2BGR;
    break;
  case BayerPattern::BGGR:
    cvPattern = cv::COLOR_BayerRG2BGR;
    break;
  case BayerPattern::GRBG:
    cvPattern = cv::COLOR_BayerGB2BGR;
    break;
  case BayerPattern::GBRG:
    cvPattern = cv::COLOR_BayerGR2BGR;
    break;
  default:
    cvPattern = cv::COLOR_BayerBG2BGR;
  }

  cv::Mat result;
  cv::cvtColor(bayerImage, result, cvPattern);
  return result;
}

cv::Mat Debayer::processBilinear(const cv::Mat &bayerImage,
                                 BayerPattern pattern) {
  // For demonstration, we'll use OpenCV's implementation with better quality
  int cvPattern;
  switch (pattern) {
  case BayerPattern::RGGB:
    cvPattern = cv::COLOR_BayerBG2BGR;
    break;
  case BayerPattern::BGGR:
    cvPattern = cv::COLOR_BayerRG2BGR;
    break;
  case BayerPattern::GRBG:
    cvPattern = cv::COLOR_BayerGB2BGR;
    break;
  case BayerPattern::GBRG:
    cvPattern = cv::COLOR_BayerGR2BGR;
    break;
  default:
    cvPattern = cv::COLOR_BayerBG2BGR;
  }

  cv::Mat result;
  cv::demosaicing(bayerImage, result, cvPattern);
  return result;
}

cv::Mat Debayer::processMalvar(const cv::Mat &bayerImage,
                               BayerPattern pattern) {
  // In a real implementation, this would be a custom implementation of
  // the Malvar-He-Cutler algorithm. For this demonstration, we'll use
  // OpenCV's demosaicing which uses a similar algorithm.
  int cvPattern;
  switch (pattern) {
  case BayerPattern::RGGB:
    cvPattern = cv::COLOR_BayerBG2BGR_EA;
    break;
  case BayerPattern::BGGR:
    cvPattern = cv::COLOR_BayerRG2BGR_EA;
    break;
  case BayerPattern::GRBG:
    cvPattern = cv::COLOR_BayerGB2BGR_EA;
    break;
  case BayerPattern::GBRG:
    cvPattern = cv::COLOR_BayerGR2BGR_EA;
    break;
  default:
    cvPattern = cv::COLOR_BayerBG2BGR_EA;
  }

  cv::Mat result;
  cv::demosaicing(bayerImage, result, cvPattern);
  return result;
}

cv::Mat Debayer::processAdaptive(const cv::Mat &bayerImage,
                                 BayerPattern pattern) {
  // This would be an implementation of an advanced adaptive algorithm
  // For this demonstration, we'll fall back to OpenCV's VNG implementation
  int cvPattern;
  switch (pattern) {
  case BayerPattern::RGGB:
    cvPattern = cv::COLOR_BayerBG2BGR_VNG;
    break;
  case BayerPattern::BGGR:
    cvPattern = cv::COLOR_BayerRG2BGR_VNG;
    break;
  case BayerPattern::GRBG:
    cvPattern = cv::COLOR_BayerGB2BGR_VNG;
    break;
  case BayerPattern::GBRG:
    cvPattern = cv::COLOR_BayerGR2BGR_VNG;
    break;
  default:
    cvPattern = cv::COLOR_BayerBG2BGR_VNG;
  }

  cv::Mat result;
  cv::demosaicing(bayerImage, result, cvPattern);
  return result;
}

} // namespace algorithms
} // namespace opencam