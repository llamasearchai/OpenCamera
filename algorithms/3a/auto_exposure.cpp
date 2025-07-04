#include "auto_exposure.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace opencam {
namespace algorithms {

AutoExposureController::AutoExposureController() {
    // Initialize ML model for scene analysis
    initializeMLModel();
    
    // Initialize histogram analyzer
    histogramAnalyzer_ = std::make_unique<HistogramAnalyzer>();
    
    // Initialize face detection for intelligent metering
    faceDetector_ = cv::CascadeClassifier();
    try {
        // The second argument ("required=false") prevents OpenCV from throwing if the file is not found
        std::string cascadePath = cv::samples::findFile("haarcascade_frontalface_alt.xml", false, false);

        if (!cascadePath.empty() && faceDetector_.load(cascadePath)) {
            spdlog::info("Face detection cascade loaded: {}", cascadePath);
        } else {
            spdlog::warn("Face detection cascade not found – face-aware metering disabled");
            faceDetectionEnabled_ = false;
        }
    } catch (const cv::Exception& e) {
        spdlog::warn("Exception while loading face detection cascade ({}). Disabling face detection.", e.what());
        faceDetectionEnabled_ = false;
    }
    
    spdlog::info("AutoExposureController initialized with ML-based scene analysis");
}

void AutoExposureController::setParameters(const Parameters& params) {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    params_ = params;
    
    // Reset convergence state when parameters change
    convergenceHistory_.clear();
    sceneChangeDetected_ = true;
    
    spdlog::debug("Auto exposure parameters updated - mode: {}, target: {:.3f}", 
                  static_cast<int>(params.mode), params.targetBrightness);
}

AutoExposureController::Parameters AutoExposureController::getParameters() const {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    return params_;
}

float AutoExposureController::computeExposure(const CameraFrame& frame) {
    if (frame.image.empty()) {
        spdlog::error("Empty frame provided to auto exposure controller");
        return currentExposure_;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Convert to grayscale for analysis
    cv::Mat gray;
    if (frame.image.channels() == 3) {
        cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.image.clone();
    }
    
    // Perform scene analysis using ML model
    SceneAnalysis sceneInfo = analyzeScene(frame.image);
    
    // Detect scene changes for adaptive behavior
    detectSceneChange(gray);
    
    // Compute brightness based on selected mode and scene context
    float currentBrightness = computeBrightnessWithContext(gray, sceneInfo);
    
    // Apply intelligent exposure adjustment
    float newExposure = computeIntelligentExposure(currentBrightness, sceneInfo);
    
    // Update convergence tracking
    updateConvergenceTracking(newExposure, currentBrightness);
    
    // Apply temporal smoothing
    newExposure = applyTemporalSmoothing(newExposure);
    
    // Clamp to valid range
    newExposure = std::clamp(newExposure, params_.minExposure, params_.maxExposure);
    
    currentExposure_ = newExposure;
    lastFrameTime_ = std::chrono::high_resolution_clock::now();
    
    // Performance logging
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    spdlog::debug("AE computed: exposure={:.4f}, brightness={:.3f}, scene={}, time={}μs",
                  newExposure, currentBrightness, sceneInfo.sceneType, duration.count());
    
    return newExposure;
}

bool AutoExposureController::applyToCamera(CameraDevice& camera) {
    if (params_.lockExposure) {
        return true; // Don't apply if exposure is locked
    }
    
    bool success = camera.setExposure(currentExposure_);
    if (!success) {
        spdlog::error("Failed to apply exposure value {:.4f} to camera", currentExposure_);
    }
    
    return success;
}

void AutoExposureController::initializeMLModel() {
    try {
        // Load pre-trained scene classification model
        // In production, this would be a custom trained model for camera scenes
        mlModel_ = cv::dnn::readNetFromONNX("models/scene_classifier.onnx");
        if (mlModel_.empty()) {
            spdlog::warn("Could not load ML scene classification model - using traditional methods");
            mlModelEnabled_ = false;
        } else {
            mlModelEnabled_ = true;
            spdlog::info("ML scene classification model loaded successfully");
        }
    } catch (const std::exception& e) {
        spdlog::warn("Exception loading ML model: {} - using traditional methods", e.what());
        mlModelEnabled_ = false;
    }
}

AutoExposureController::SceneAnalysis AutoExposureController::analyzeScene(const cv::Mat& image) {
    SceneAnalysis analysis;
    
    if (mlModelEnabled_ && !mlModel_.empty()) {
        analysis = analyzeSceneML(image);
    } else {
        analysis = analyzeSceneTraditional(image);
    }
    
    // Enhance with face detection
    if (faceDetectionEnabled_) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        std::vector<cv::Rect> faces;
        faceDetector_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        
        analysis.faceRegions = faces;
        analysis.hasFaces = !faces.empty();
        
        if (analysis.hasFaces) {
            analysis.sceneType = "portrait";
            analysis.confidence = std::max(analysis.confidence, 0.8f);
        }
    }
    
    return analysis;
}

AutoExposureController::SceneAnalysis AutoExposureController::analyzeSceneML(const cv::Mat& image) {
    SceneAnalysis analysis;
    
    try {
        // Prepare input blob
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(224, 224), cv::Scalar(0,0,0), true, false);
        
        mlModel_.setInput(blob);
        cv::Mat output = mlModel_.forward();
        
        // Process output to determine scene type and characteristics
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
        
        analysis.confidence = static_cast<float>(confidence);
        
        // Map class ID to scene type (this would be based on your training data)
        const std::vector<std::string> sceneTypes = {
            "indoor", "outdoor", "portrait", "landscape", "macro", "lowlight", "backlit", "highcontrast"
        };
        
        if (classIdPoint.x < sceneTypes.size()) {
            analysis.sceneType = sceneTypes[classIdPoint.x];
        } else {
            analysis.sceneType = "unknown";
        }
        
        // Extract additional scene characteristics from the output
        analysis.isBacklit = (analysis.sceneType == "backlit");
        analysis.isLowLight = (analysis.sceneType == "lowlight");
        analysis.isHighContrast = (analysis.sceneType == "highcontrast");
        
    } catch (const std::exception& e) {
        spdlog::error("ML scene analysis failed: {}", e.what());
        return analyzeSceneTraditional(image);
    }
    
    return analysis;
}

AutoExposureController::SceneAnalysis AutoExposureController::analyzeSceneTraditional(const cv::Mat& image) {
    SceneAnalysis analysis;
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Compute histogram
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Analyze histogram characteristics
    cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
    
    // Detect low light conditions
    float darkPixelRatio = 0.0f;
    for (int i = 0; i < 64; i++) {
        darkPixelRatio += hist.at<float>(i);
    }
    analysis.isLowLight = darkPixelRatio > 0.6f;
    
    // Detect backlit conditions
    float brightPixelRatio = 0.0f;
    for (int i = 192; i < 256; i++) {
        brightPixelRatio += hist.at<float>(i);
    }
    analysis.isBacklit = (brightPixelRatio > 0.3f && darkPixelRatio > 0.3f);
    
    // Detect high contrast
    float midtoneRatio = 0.0f;
    for (int i = 64; i < 192; i++) {
        midtoneRatio += hist.at<float>(i);
    }
    analysis.isHighContrast = midtoneRatio < 0.4f;
    
    // Determine scene type based on characteristics
    if (analysis.isLowLight) {
        analysis.sceneType = "lowlight";
    } else if (analysis.isBacklit) {
        analysis.sceneType = "backlit";
    } else if (analysis.isHighContrast) {
        analysis.sceneType = "highcontrast";
    } else {
        analysis.sceneType = "general";
    }
    
    analysis.confidence = 0.7f; // Traditional analysis confidence
    
    return analysis;
}

float AutoExposureController::computeBrightnessWithContext(const cv::Mat& gray, const SceneAnalysis& scene) {
    float brightness = 0.0f;
    
    switch (params_.mode) {
        case Mode::DISABLED:
            return 0.5f; // Return neutral value when disabled
            
        case Mode::AVERAGE:
            brightness = computeAverageBrightness(gray);
            break;
            
        case Mode::CENTER_WEIGHTED:
            brightness = computeCenterWeightedBrightness(gray);
            break;
            
        case Mode::SPOT:
            brightness = computeSpotBrightness(gray);
            break;
            
        case Mode::MULTI_ZONE:
            brightness = computeMultiZoneBrightness(gray);
            break;
            
        case Mode::INTELLIGENT:
            brightness = computeIntelligentBrightness(gray, scene);
            break;
    }
    
    // Apply scene-specific adjustments
    if (scene.isBacklit) {
        // For backlit scenes, weight shadows more heavily
        brightness = adjustForBacklight(gray, brightness);
    }
    
    if (scene.isLowLight) {
        // For low light, be more conservative with brightness estimation
        brightness = adjustForLowLight(gray, brightness);
    }
    
    return brightness;
}

float AutoExposureController::computeAverageBrightness(const cv::Mat& gray) {
    cv::Scalar meanValue = cv::mean(gray);
    return static_cast<float>(meanValue[0] / 255.0);
}

float AutoExposureController::computeCenterWeightedBrightness(const cv::Mat& gray) {
    int centerX = gray.cols / 2;
    int centerY = gray.rows / 2;
    int radius = std::min(gray.cols, gray.rows) / 3;
    
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(centerX, centerY), radius, cv::Scalar(255), -1);
    
    // Create weighted mask with Gaussian falloff
    cv::Mat weightMask;
    mask.convertTo(weightMask, CV_32F, 1.0/255.0);
    cv::GaussianBlur(weightMask, weightMask, cv::Size(radius/2*2+1, radius/2*2+1), radius/3.0);
    
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    
    cv::Mat weighted = grayFloat.mul(weightMask);
    cv::Scalar weightedSum = cv::sum(weighted);
    cv::Scalar totalWeight = cv::sum(weightMask);
    
    if (totalWeight[0] > 0) {
        return static_cast<float>(weightedSum[0] / totalWeight[0] / 255.0);
    }
    
    return computeAverageBrightness(gray);
}

float AutoExposureController::computeSpotBrightness(const cv::Mat& gray) {
    int centerX = gray.cols / 2;
    int centerY = gray.rows / 2;
    int spotSize = std::min(gray.cols, gray.rows) / 10;
    
    cv::Rect spotRect(centerX - spotSize/2, centerY - spotSize/2, spotSize, spotSize);
    spotRect &= cv::Rect(0, 0, gray.cols, gray.rows); // Ensure within bounds
    
    cv::Mat spotRegion = gray(spotRect);
    cv::Scalar meanValue = cv::mean(spotRegion);
    
    return static_cast<float>(meanValue[0] / 255.0);
}

float AutoExposureController::computeMultiZoneBrightness(const cv::Mat& gray) {
    const int zones = 9; // 3x3 grid
    const int rows = 3, cols = 3;
    
    int zoneWidth = gray.cols / cols;
    int zoneHeight = gray.rows / rows;
    
    std::vector<float> zoneBrightness;
    zoneBrightness.reserve(zones);
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            cv::Rect zoneRect(c * zoneWidth, r * zoneHeight, zoneWidth, zoneHeight);
            zoneRect &= cv::Rect(0, 0, gray.cols, gray.rows);
            
            cv::Mat zone = gray(zoneRect);
            cv::Scalar meanValue = cv::mean(zone);
            zoneBrightness.push_back(static_cast<float>(meanValue[0] / 255.0));
        }
    }
    
    // Weight center zones more heavily
    std::vector<float> weights = {
        0.5f, 1.0f, 0.5f,
        1.0f, 2.0f, 1.0f,
        0.5f, 1.0f, 0.5f
    };
    
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < zoneBrightness.size() && i < weights.size(); i++) {
        weightedSum += zoneBrightness[i] * weights[i];
        totalWeight += weights[i];
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0.5f;
}

float AutoExposureController::computeIntelligentBrightness(const cv::Mat& gray, const SceneAnalysis& scene) {
    float brightness = 0.0f;
    
    if (scene.hasFaces && !scene.faceRegions.empty()) {
        // Face-priority metering
        brightness = computeFacePriorityBrightness(gray, scene.faceRegions);
    } else if (scene.sceneType == "portrait") {
        // Use center-weighted for portraits
        brightness = computeCenterWeightedBrightness(gray);
    } else if (scene.sceneType == "landscape") {
        // Use multi-zone for landscapes
        brightness = computeMultiZoneBrightness(gray);
    } else if (scene.isBacklit) {
        // Special handling for backlit scenes
        brightness = computeBacklitBrightness(gray);
    } else {
        // Default to center-weighted
        brightness = computeCenterWeightedBrightness(gray);
    }
    
    return brightness;
}

float AutoExposureController::computeFacePriorityBrightness(const cv::Mat& gray, const std::vector<cv::Rect>& faces) {
    if (faces.empty()) {
        return computeCenterWeightedBrightness(gray);
    }
    
    float totalBrightness = 0.0f;
    float totalArea = 0.0f;
    
    for (const auto& face : faces) {
        cv::Rect safeFace = face & cv::Rect(0, 0, gray.cols, gray.rows);
        if (safeFace.area() > 0) {
            cv::Mat faceRegion = gray(safeFace);
            cv::Scalar meanValue = cv::mean(faceRegion);
            float faceBrightness = static_cast<float>(meanValue[0] / 255.0);
            
            totalBrightness += faceBrightness * safeFace.area();
            totalArea += safeFace.area();
        }
    }
    
    if (totalArea > 0) {
        return totalBrightness / totalArea;
    }
    
    return computeCenterWeightedBrightness(gray);
}

float AutoExposureController::computeBacklitBrightness(const cv::Mat& gray) {
    // For backlit scenes, focus on the subject (typically in the center/foreground)
    // and ignore the bright background
    
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Find the primary subject peak (usually in the darker regions for backlit)
    cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
    
    float subjectBrightness = 0.0f;
    float maxDensity = 0.0f;
    
    // Look for the peak in the lower 75% of the histogram
    for (int i = 0; i < 192; i++) {
        if (hist.at<float>(i) > maxDensity) {
            maxDensity = hist.at<float>(i);
            subjectBrightness = i / 255.0f;
        }
    }
    
    // Blend with center-weighted measurement
    float centerWeighted = computeCenterWeightedBrightness(gray);
    return 0.7f * subjectBrightness + 0.3f * centerWeighted;
}

float AutoExposureController::computeIntelligentExposure(float currentBrightness, const SceneAnalysis& scene) {
    if (params_.lockExposure) {
        return currentExposure_;
    }
    
    // Adjust target brightness based on scene type
    float adaptiveTarget = getAdaptiveTarget(scene);
    
    // Compute brightness error
    float brightnessError = adaptiveTarget - currentBrightness;
    
    // Apply scene-specific exposure compensation
    float exposureCompensation = computeExposureCompensation(scene, brightnessError);
    
    // Compute base exposure adjustment
    float exposureAdjustment = brightnessError * params_.convergenceSpeed;
    
    // Apply intelligent damping based on scene stability
    float dampingFactor = computeDampingFactor();
    exposureAdjustment *= dampingFactor;
    
    // Add exposure compensation
    exposureAdjustment += exposureCompensation;
    
    // Apply non-linear response for better convergence
    exposureAdjustment = applyNonLinearResponse(exposureAdjustment, brightnessError);
    
    float newExposure = currentExposure_ + exposureAdjustment;
    
    // Apply scene-specific clamping
    return applySceneSpecificClamping(newExposure, scene);
}

float AutoExposureController::getAdaptiveTarget(const SceneAnalysis& scene) {
    float baseTarget = params_.targetBrightness;
    
    // Adjust target based on scene characteristics
    if (scene.isLowLight) {
        // For low light, target slightly brighter to improve visibility
        baseTarget = std::min(1.0f, baseTarget * 1.2f);
    } else if (scene.isBacklit) {
        // For backlit scenes, target slightly darker to preserve highlights
        baseTarget *= 0.85f;
    } else if (scene.isHighContrast) {
        // For high contrast, be more conservative
        baseTarget *= 0.95f;
    }
    
    // Scene type specific adjustments
    if (scene.sceneType == "portrait" && scene.hasFaces) {
        // For portraits with faces, target optimal skin tone exposure
        baseTarget = 0.6f; // Slightly brighter for good skin tone
    } else if (scene.sceneType == "landscape") {
        // For landscapes, preserve highlight detail
        baseTarget *= 0.9f;
    }
    
    return std::clamp(baseTarget, 0.1f, 0.9f);
}

float AutoExposureController::computeExposureCompensation(const SceneAnalysis& scene, float brightnessError) {
    float compensation = 0.0f;
    
    if (scene.isBacklit && brightnessError > 0.1f) {
        // For backlit scenes with underexposure, apply positive compensation
        compensation = 0.3f * params_.convergenceSpeed;
    } else if (scene.isLowLight && brightnessError > 0.05f) {
        // For low light with slight underexposure, boost more aggressively
        compensation = 0.2f * params_.convergenceSpeed;
    } else if (scene.isHighContrast && std::abs(brightnessError) < 0.05f) {
        // For high contrast scenes, fine-tune more carefully
        compensation = brightnessError * 0.5f * params_.convergenceSpeed;
    }
    
    return compensation;
}

float AutoExposureController::computeDampingFactor() {
    float dampingFactor = 1.0f;
    
    // Reduce damping if scene change was detected
    if (sceneChangeDetected_) {
        dampingFactor = 1.5f; // Faster response to scene changes
        sceneChangeDetected_ = false;
    }
    
    // Apply temporal damping based on convergence history
    if (convergenceHistory_.size() >= 3) {
        float recentVariance = 0.0f;
        float mean = 0.0f;
        
        // Calculate variance of recent exposure values
        for (size_t i = convergenceHistory_.size() - 3; i < convergenceHistory_.size(); i++) {
            mean += convergenceHistory_[i].exposure;
        }
        mean /= 3.0f;
        
        for (size_t i = convergenceHistory_.size() - 3; i < convergenceHistory_.size(); i++) {
            float diff = convergenceHistory_[i].exposure - mean;
            recentVariance += diff * diff;
        }
        recentVariance /= 3.0f;
        
        // If exposure is oscillating, apply stronger damping
        if (recentVariance > 0.01f) {
            dampingFactor *= 0.5f;
        }
    }
    
    return std::clamp(dampingFactor, 0.1f, 2.0f);
}

float AutoExposureController::applyNonLinearResponse(float adjustment, float error) {
    // Apply sigmoid-like response for smoother convergence
    float absError = std::abs(error);
    
    if (absError > 0.3f) {
        // Large errors: faster response
        adjustment *= 1.5f;
    } else if (absError < 0.05f) {
        // Small errors: slower, more precise response
        adjustment *= 0.5f;
    }
    
    // Apply soft limiting to prevent overshooting
    float maxAdjustment = 0.2f * params_.convergenceSpeed;
    adjustment = std::clamp(adjustment, -maxAdjustment, maxAdjustment);
    
    return adjustment;
}

float AutoExposureController::applySceneSpecificClamping(float exposure, const SceneAnalysis& scene) {
    float minExp = params_.minExposure;
    float maxExp = params_.maxExposure;
    
    // Scene-specific exposure limits
    if (scene.isLowLight) {
        // Allow longer exposures for low light
        maxExp = std::min(maxExp * 2.0f, 2.0f);
    } else if (scene.sceneType == "outdoor") {
        // Limit exposure for outdoor scenes to prevent overexposure
        maxExp = std::min(maxExp, 0.5f);
    }
    
    return std::clamp(exposure, minExp, maxExp);
}

void AutoExposureController::detectSceneChange(const cv::Mat& gray) {
    if (previousFrame_.empty()) {
        previousFrame_ = gray.clone();
        return;
    }
    
    // Compute frame difference
    cv::Mat diff;
    cv::absdiff(gray, previousFrame_, diff);
    
    cv::Scalar meanDiff = cv::mean(diff);
    float changeAmount = static_cast<float>(meanDiff[0] / 255.0);
    
    // Detect significant scene changes
    const float sceneChangeThreshold = 0.1f;
    if (changeAmount > sceneChangeThreshold) {
        sceneChangeDetected_ = true;
        spdlog::debug("Scene change detected: change amount = {:.3f}", changeAmount);
    }
    
    // Update previous frame (with temporal filtering to reduce noise)
    cv::addWeighted(previousFrame_, 0.9, gray, 0.1, 0, previousFrame_);
}

void AutoExposureController::updateConvergenceTracking(float exposure, float brightness) {
    ConvergencePoint point;
    point.exposure = exposure;
    point.brightness = brightness;
    point.timestamp = std::chrono::high_resolution_clock::now();
    
    convergenceHistory_.push_back(point);
    
    // Keep only recent history
    const size_t maxHistorySize = 30;
    if (convergenceHistory_.size() > maxHistorySize) {
        convergenceHistory_.erase(convergenceHistory_.begin());
    }
}

float AutoExposureController::applyTemporalSmoothing(float newExposure) {
    if (convergenceHistory_.empty()) {
        return newExposure;
    }
    
    // Apply exponential moving average for smooth transitions
    float alpha = params_.convergenceSpeed;
    
    // Adjust smoothing based on scene stability
    if (sceneChangeDetected_) {
        alpha *= 1.5f; // Less smoothing for scene changes
    }
    
    alpha = std::clamp(alpha, 0.05f, 1.0f);
    
    float smoothedExposure = alpha * newExposure + (1.0f - alpha) * currentExposure_;
    
    return smoothedExposure;
}

float AutoExposureController::adjustForBacklight(const cv::Mat& gray, float brightness) {
    // Analyze the brightness distribution
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
    
    // Calculate the percentage of very bright pixels
    float brightPixelRatio = 0.0f;
    for (int i = 220; i < 256; i++) {
        brightPixelRatio += hist.at<float>(i);
    }
    
    // If there are many bright pixels (backlight), adjust the brightness calculation
    // to focus more on the subject (typically in mid-tones)
    if (brightPixelRatio > 0.2f) {
        float midtoneBrightness = 0.0f;
        float midtoneWeight = 0.0f;
        
        for (int i = 64; i < 192; i++) {
            midtoneBrightness += (i / 255.0f) * hist.at<float>(i);
            midtoneWeight += hist.at<float>(i);
        }
        
        if (midtoneWeight > 0) {
            float adjustedBrightness = midtoneBrightness / midtoneWeight;
            // Blend with original brightness
            brightness = 0.7f * adjustedBrightness + 0.3f * brightness;
        }
    }
    
    return brightness;
}

float AutoExposureController::adjustForLowLight(const cv::Mat& gray, float brightness) {
    // In low light, be more conservative with brightness estimation
    // to avoid overexposure of the few bright areas
    
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
    
    // Calculate noise-robust brightness using percentile-based approach
    std::vector<float> cumulativeHist(256, 0.0f);
    cumulativeHist[0] = hist.at<float>(0);
    
    for (int i = 1; i < 256; i++) {
        cumulativeHist[i] = cumulativeHist[i-1] + hist.at<float>(i);
    }
    
    float percentile25 = 0.0f, percentile75 = 0.0f;
    bool found25 = false, found75 = false;
    
    for (int i = 0; i < 256; i++) {
        if (!found25 && cumulativeHist[i] >= 0.25f) {
            percentile25 = i / 255.0f;
            found25 = true;
        }
        if (!found75 && cumulativeHist[i] >= 0.75f) {
            percentile75 = i / 255.0f;
            found75 = true;
            break;
        }
    }
    
    // Use robust brightness estimate in low light
    float robustBrightness = (percentile25 + percentile75) / 2.0f;
    
    // Blend with original brightness, favoring robust estimate in low light
    return 0.6f * robustBrightness + 0.4f * brightness;
}

bool AutoExposureController::isConverged() const {
    if (convergenceHistory_.size() < 5) {
        return false;
    }
    
    // Check if exposure has stabilized over recent frames
    float recentVariance = 0.0f;
    float mean = 0.0f;
    
    size_t startIdx = convergenceHistory_.size() - 5;
    for (size_t i = startIdx; i < convergenceHistory_.size(); i++) {
        mean += convergenceHistory_[i].exposure;
    }
    mean /= 5.0f;
    
    for (size_t i = startIdx; i < convergenceHistory_.size(); i++) {
        float diff = convergenceHistory_[i].exposure - mean;
        recentVariance += diff * diff;
    }
    recentVariance /= 5.0f;
    
    // Consider converged if variance is low
    const float convergenceThreshold = 0.001f;
    return recentVariance < convergenceThreshold;
}

AutoExposureController::Statistics AutoExposureController::getStatistics() const {
    Statistics stats;
    
    if (convergenceHistory_.empty()) {
        return stats;
    }
    
    // Calculate statistics from convergence history
    float sumExposure = 0.0f, sumBrightness = 0.0f;
    float minExposure = std::numeric_limits<float>::max();
    float maxExposure = std::numeric_limits<float>::lowest();
    
    for (const auto& point : convergenceHistory_) {
        sumExposure += point.exposure;
        sumBrightness += point.brightness;
        minExposure = std::min(minExposure, point.exposure);
        maxExposure = std::max(maxExposure, point.exposure);
    }
    
    size_t count = convergenceHistory_.size();
    stats.averageExposure = sumExposure / count;
    stats.averageBrightness = sumBrightness / count;
    stats.minExposure = minExposure;
    stats.maxExposure = maxExposure;
    stats.isConverged = isConverged();
    stats.frameCount = count;
    
    // Calculate convergence time
    if (count >= 2) {
        auto duration = convergenceHistory_.back().timestamp - convergenceHistory_.front().timestamp;
        stats.convergenceTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }
    
    return stats;
}

void AutoExposureController::reset() {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    
    convergenceHistory_.clear();
    previousFrame_ = cv::Mat();
    sceneChangeDetected_ = true;
    currentExposure_ = 0.5f; // Reset to middle value
    
    spdlog::info("Auto exposure controller reset");
}

// Histogram Analyzer Implementation
class AutoExposureController::HistogramAnalyzer {
public:
    struct HistogramStats {
        float mean;
        float median;
        float mode;
        float stddev;
        float skewness;
        float entropy;
        std::vector<float> percentiles;
    };
    
    HistogramStats analyze(const cv::Mat& gray) {
        HistogramStats stats;
        
        // Compute histogram
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
        
        // Normalize histogram
        cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
        
        // Calculate statistics
        stats.mean = calculateMean(hist);
        stats.median = calculateMedian(hist);
        stats.mode = calculateMode(hist);
        stats.stddev = calculateStdDev(hist, stats.mean);
        stats.skewness = calculateSkewness(hist, stats.mean, stats.stddev);
        stats.entropy = calculateEntropy(hist);
        stats.percentiles = calculatePercentiles(hist);
        
        return stats;
    }
    
private:
    float calculateMean(const cv::Mat& hist) {
        float mean = 0.0f;
        for (int i = 0; i < 256; i++) {
            mean += i * hist.at<float>(i);
        }
        return mean / 255.0f;
    }
    
    float calculateMedian(const cv::Mat& hist) {
        float cumulative = 0.0f;
        for (int i = 0; i < 256; i++) {
            cumulative += hist.at<float>(i);
            if (cumulative >= 0.5f) {
                return i / 255.0f;
            }
        }
        return 0.5f;
    }
    
    float calculateMode(const cv::Mat& hist) {
        float maxVal = 0.0f;
        int maxIdx = 0;
        for (int i = 0; i < 256; i++) {
            if (hist.at<float>(i) > maxVal) {
                maxVal = hist.at<float>(i);
                maxIdx = i;
            }
        }
        return maxIdx / 255.0f;
    }
    
    float calculateStdDev(const cv::Mat& hist, float mean) {
        float variance = 0.0f;
        float scaledMean = mean * 255.0f;
        
        for (int i = 0; i < 256; i++) {
            float diff = i - scaledMean;
            variance += diff * diff * hist.at<float>(i);
        }
        
        return std::sqrt(variance) / 255.0f;
    }
    
    float calculateSkewness(const cv::Mat& hist, float mean, float stddev) {
        if (stddev == 0.0f) return 0.0f;
        
        float skewness = 0.0f;
        float scaledMean = mean * 255.0f;
        float scaledStdDev = stddev * 255.0f;
        
        for (int i = 0; i < 256; i++) {
            float normalizedDiff = (i - scaledMean) / scaledStdDev;
            skewness += normalizedDiff * normalizedDiff * normalizedDiff * hist.at<float>(i);
        }
        
        return skewness;
    }
    
    float calculateEntropy(const cv::Mat& hist) {
        float entropy = 0.0f;
        for (int i = 0; i < 256; i++) {
            float p = hist.at<float>(i);
            if (p > 0) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    }
    
    std::vector<float> calculatePercentiles(const cv::Mat& hist) {
        std::vector<float> percentiles;
        std::vector<float> targets = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};
        
        float cumulative = 0.0f;
        size_t targetIdx = 0;
        
        for (int i = 0; i < 256 && targetIdx < targets.size(); i++) {
            cumulative += hist.at<float>(i);
            while (targetIdx < targets.size() && cumulative >= targets[targetIdx]) {
                percentiles.push_back(i / 255.0f);
                targetIdx++;
            }
        }
        
        // Fill remaining percentiles if needed
        while (percentiles.size() < targets.size()) {
            percentiles.push_back(1.0f);
        }
        
        return percentiles;
    }
};

// -----------------------------------------------------------------------------
// Destructor defined after HistogramAnalyzer to ensure complete type
// -----------------------------------------------------------------------------

AutoExposureController::~AutoExposureController() = default;

} // namespace algorithms
} // namespace opencam