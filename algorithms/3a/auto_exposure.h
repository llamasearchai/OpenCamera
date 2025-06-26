#pragma once

#include "opencam/camera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <chrono>
#include <mutex>

namespace opencam {
namespace algorithms {

/**
 * @brief Advanced Auto Exposure Controller with ML-based scene analysis
 * 
 * This class implements state-of-the-art auto exposure algorithms including:
 * - Machine Learning based scene classification
 * - Intelligent face-priority metering
 * - Multi-zone adaptive metering
 * - Temporal smoothing and convergence optimization
 * - Scene-specific exposure compensation
 */
class AutoExposureController {
public:
    enum class Mode {
        DISABLED,           ///< Auto exposure disabled
        AVERAGE,           ///< Simple average metering
        CENTER_WEIGHTED,   ///< Center-weighted metering
        SPOT,              ///< Spot metering (center point)
        MULTI_ZONE,        ///< Multi-zone evaluative metering
        INTELLIGENT        ///< ML-based intelligent metering
    };
    
    /**
     * @brief Auto exposure parameters
     */
    struct Parameters {
        Mode mode = Mode::INTELLIGENT;          ///< Metering mode
        float targetBrightness = 0.5f;          ///< Target brightness (0.0-1.0)
        float convergenceSpeed = 0.15f;         ///< Convergence speed (0.0-1.0)
        float minExposure = 0.001f;             ///< Minimum exposure time
        float maxExposure = 1.0f;               ///< Maximum exposure time
        bool lockExposure = false;              ///< Lock current exposure
        bool enableFaceDetection = true;       ///< Enable face-priority metering
        bool enableSceneAnalysis = true;       ///< Enable ML scene analysis
        float exposureCompensation = 0.0f;     ///< Manual exposure compensation (-2.0 to +2.0 EV)
    };
    
    /**
     * @brief Scene analysis results from ML model
     */
    struct SceneAnalysis {
        std::string sceneType = "unknown";     ///< Scene classification
        float confidence = 0.0f;               ///< Classification confidence
        bool isLowLight = false;               ///< Low light condition detected
        bool isBacklit = false;                ///< Backlit condition detected
        bool isHighContrast = false;           ///< High contrast scene
        bool hasFaces = false;                 ///< Faces detected in scene
        std::vector<cv::Rect> faceRegions;     ///< Detected face regions
    };
    
    /**
     * @brief Convergence tracking point
     */
    struct ConvergencePoint {
        float exposure;
        float brightness;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    /**
     * @brief Auto exposure statistics
     */
    struct Statistics {
        float averageExposure = 0.0f;
        float averageBrightness = 0.0f;
        float minExposure = 0.0f;
        float maxExposure = 0.0f;
        bool isConverged = false;
        size_t frameCount = 0;
        int64_t convergenceTimeMs = 0;
    };
    
    /**
     * @brief Constructor
     */
    AutoExposureController();
    
    /**
     * @brief Destructor
     */
    ~AutoExposureController();
    
    /**
     * @brief Set auto exposure parameters
     * @param params New parameters
     */
    void setParameters(const Parameters& params);
    
    /**
     * @brief Get current parameters
     * @return Current parameters
     */
    Parameters getParameters() const;
    
    /**
     * @brief Process a frame and compute the new exposure value
     * @param frame Input camera frame
     * @return Computed exposure value
     */
    float computeExposure(const CameraFrame& frame);
    
    /**
     * @brief Apply the computed exposure to the camera
     * @param camera Camera device to apply exposure to
     * @return True if successful
     */
    bool applyToCamera(CameraDevice& camera);
    
    /**
     * @brief Check if auto exposure has converged
     * @return True if converged
     */
    bool isConverged() const;
    
    /**
     * @brief Get auto exposure statistics
     * @return Current statistics
     */
    Statistics getStatistics() const;
    
    /**
     * @brief Reset the auto exposure controller
     */
    void reset();
    
    /**
     * @brief Get the last scene analysis results
     * @return Scene analysis data
     */
    SceneAnalysis getLastSceneAnalysis() const { return lastSceneAnalysis_; }

private:
    // Forward declaration for histogram analyzer
    class HistogramAnalyzer;
    
    // Core parameters and state
    Parameters params_;
    mutable std::mutex paramsMutex_;
    float currentExposure_ = 0.5f;
    std::chrono::high_resolution_clock::time_point lastFrameTime_;
    
    // ML and scene analysis
    cv::dnn::Net mlModel_;
    bool mlModelEnabled_ = false;
    SceneAnalysis lastSceneAnalysis_;
    
    // Face detection
    cv::CascadeClassifier faceDetector_;
    bool faceDetectionEnabled_ = true;
    
    // Convergence tracking
    std::vector<ConvergencePoint> convergenceHistory_;
    bool sceneChangeDetected_ = true;
    cv::Mat previousFrame_;
    
    // Analysis tools
    std::unique_ptr<HistogramAnalyzer> histogramAnalyzer_;
    
    // Initialization methods
    void initializeMLModel();
    
    // Scene analysis methods
    SceneAnalysis analyzeScene(const cv::Mat& image);
    SceneAnalysis analyzeSceneML(const cv::Mat& image);
    SceneAnalysis analyzeSceneTraditional(const cv::Mat& image);
    
    // Brightness computation methods
    float computeBrightnessWithContext(const cv::Mat& gray, const SceneAnalysis& scene);
    float computeAverageBrightness(const cv::Mat& gray);
    float computeCenterWeightedBrightness(const cv::Mat& gray);
    float computeSpotBrightness(const cv::Mat& gray);
    float computeMultiZoneBrightness(const cv::Mat& gray);
    float computeIntelligentBrightness(const cv::Mat& gray, const SceneAnalysis& scene);
    float computeFacePriorityBrightness(const cv::Mat& gray, const std::vector<cv::Rect>& faces);
    float computeBacklitBrightness(const cv::Mat& gray);
    
    // Intelligent exposure computation
    float computeIntelligentExposure(float currentBrightness, const SceneAnalysis& scene);
    float getAdaptiveTarget(const SceneAnalysis& scene);
    float computeExposureCompensation(const SceneAnalysis& scene, float brightnessError);
    float computeDampingFactor();
    float applyNonLinearResponse(float adjustment, float error);
    float applySceneSpecificClamping(float exposure, const SceneAnalysis& scene);
    
    // Scene-specific adjustments
    float adjustForBacklight(const cv::Mat& gray, float brightness);
    float adjustForLowLight(const cv::Mat& gray, float brightness);
    
    // Temporal analysis
    void detectSceneChange(const cv::Mat& gray);
    void updateConvergenceTracking(float exposure, float brightness);
    float applyTemporalSmoothing(float newExposure);
};

} // namespace algorithms
} // namespace opencam