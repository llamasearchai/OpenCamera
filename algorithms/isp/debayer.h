#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace opencam {
namespace algorithms {

enum class BayerPattern { RGGB, BGGR, GRBG, GBRG };

class Debayer {
public:
  enum class Method { NEAREST_NEIGHBOR, BILINEAR, MALVAR, ADAPTIVE };

  struct Parameters {
    BayerPattern pattern = BayerPattern::RGGB;
    Method method = Method::BILINEAR;
  };

  Debayer();
  ~Debayer() = default;

  void setParameters(const Parameters &params);
  Parameters getParameters() const;

  // Process a Bayer pattern image and return a RGB image
  cv::Mat process(const cv::Mat &bayerImage);

private:
  Parameters params_;

  // Helper methods for different debayering algorithms
  cv::Mat processNearestNeighbor(const cv::Mat &bayerImage,
                                 BayerPattern pattern);
  cv::Mat processBilinear(const cv::Mat &bayerImage, BayerPattern pattern);
  cv::Mat processMalvar(const cv::Mat &bayerImage, BayerPattern pattern);
  cv::Mat processAdaptive(const cv::Mat &bayerImage, BayerPattern pattern);
};

} // namespace algorithms
} // namespace opencam