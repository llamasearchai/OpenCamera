#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <opencv2/opencv.hpp>
#include "algorithms/3a/auto_exposure.h"
#include "opencam/camera.h"

namespace py = pybind11;
using namespace opencam;
using namespace opencam::algorithms;

// Helper function to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim == 2) {
        // Grayscale image
        cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC1, (unsigned char*)buf_info.ptr);
        return mat.clone();
    } else if (buf_info.ndim == 3 && buf_info.shape[2] == 3) {
        // Color image (BGR)
        cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
        return mat.clone();
    } else {
        throw std::runtime_error("Unsupported array dimensions");
    }
}

// Helper function to convert cv::Mat to numpy array
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.channels() == 1) {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols},
            {sizeof(uint8_t) * mat.cols, sizeof(uint8_t)},
            mat.data
        );
    } else if (mat.channels() == 3) {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols, mat.channels()},
            {sizeof(uint8_t) * mat.cols * mat.channels(), sizeof(uint8_t) * mat.channels(), sizeof(uint8_t)},
            mat.data
        );
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

PYBIND11_MODULE(opencam_auto_exposure, m) {
    m.doc() = "OpenCam Auto Exposure Algorithm Python Bindings";
    
    // CameraFrame binding
    py::class_<CameraFrame>(m, "CameraFrame")
        .def(py::init<>())
        .def_readwrite("frame_number", &CameraFrame::frameNumber)
        .def_readwrite("timestamp", &CameraFrame::timestamp)
        .def_readwrite("metadata", &CameraFrame::metadata)
        .def_property("image",
            [](const CameraFrame& frame) { return mat_to_numpy(frame.image); },
            [](CameraFrame& frame, py::array_t<uint8_t> img) { frame.image = numpy_to_mat(img); })
        .def("__repr__", [](const CameraFrame& frame) {
            return "<CameraFrame frame_number=" + std::to_string(frame.frameNumber) + 
                   " timestamp=" + std::to_string(frame.timestamp) + ">";
        });
    
    // AutoExposureController::Mode enum
    py::enum_<AutoExposureController::Mode>(m, "MeteringMode")
        .value("DISABLED", AutoExposureController::Mode::DISABLED)
        .value("AVERAGE", AutoExposureController::Mode::AVERAGE)
        .value("CENTER_WEIGHTED", AutoExposureController::Mode::CENTER_WEIGHTED)
        .value("SPOT", AutoExposureController::Mode::SPOT)
        .value("MULTI_ZONE", AutoExposureController::Mode::MULTI_ZONE)
        .value("INTELLIGENT", AutoExposureController::Mode::INTELLIGENT)
        .export_values();
    
    // AutoExposureController::Parameters binding
    py::class_<AutoExposureController::Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("mode", &AutoExposureController::Parameters::mode)
        .def_readwrite("target_brightness", &AutoExposureController::Parameters::targetBrightness)
        .def_readwrite("convergence_speed", &AutoExposureController::Parameters::convergenceSpeed)
        .def_readwrite("min_exposure", &AutoExposureController::Parameters::minExposure)
        .def_readwrite("max_exposure", &AutoExposureController::Parameters::maxExposure)
        .def_readwrite("lock_exposure", &AutoExposureController::Parameters::lockExposure)
        .def_readwrite("enable_face_detection", &AutoExposureController::Parameters::enableFaceDetection)
        .def_readwrite("enable_scene_analysis", &AutoExposureController::Parameters::enableSceneAnalysis)
        .def_readwrite("exposure_compensation", &AutoExposureController::Parameters::exposureCompensation)
        .def("__repr__", [](const AutoExposureController::Parameters& p) {
            return "<Parameters mode=" + std::to_string(static_cast<int>(p.mode)) + 
                   " target_brightness=" + std::to_string(p.targetBrightness) + ">";
        });
    
    // SceneAnalysis binding
    py::class_<AutoExposureController::SceneAnalysis>(m, "SceneAnalysis")
        .def(py::init<>())
        .def_readwrite("scene_type", &AutoExposureController::SceneAnalysis::sceneType)
        .def_readwrite("confidence", &AutoExposureController::SceneAnalysis::confidence)
        .def_readwrite("is_low_light", &AutoExposureController::SceneAnalysis::isLowLight)
        .def_readwrite("is_backlit", &AutoExposureController::SceneAnalysis::isBacklit)
        .def_readwrite("is_high_contrast", &AutoExposureController::SceneAnalysis::isHighContrast)
        .def_readwrite("has_faces", &AutoExposureController::SceneAnalysis::hasFaces)
        .def("__repr__", [](const AutoExposureController::SceneAnalysis& s) {
            return "<SceneAnalysis scene_type='" + s.sceneType + 
                   "' confidence=" + std::to_string(s.confidence) + ">";
        });
    
    // Statistics binding
    py::class_<AutoExposureController::Statistics>(m, "Statistics")
        .def(py::init<>())
        .def_readwrite("average_exposure", &AutoExposureController::Statistics::averageExposure)
        .def_readwrite("average_brightness", &AutoExposureController::Statistics::averageBrightness)
        .def_readwrite("min_exposure", &AutoExposureController::Statistics::minExposure)
        .def_readwrite("max_exposure", &AutoExposureController::Statistics::maxExposure)
        .def_readwrite("is_converged", &AutoExposureController::Statistics::isConverged)
        .def_readwrite("frame_count", &AutoExposureController::Statistics::frameCount)
        .def_readwrite("convergence_time_ms", &AutoExposureController::Statistics::convergenceTimeMs)
        .def("__repr__", [](const AutoExposureController::Statistics& s) {
            return "<Statistics avg_exposure=" + std::to_string(s.averageExposure) + 
                   " converged=" + (s.isConverged ? "True" : "False") + ">";
        });
    
    // AutoExposureController binding
    py::class_<AutoExposureController>(m, "AutoExposureController")
        .def(py::init<>())
        .def("set_parameters", &AutoExposureController::setParameters,
             "Set auto exposure parameters")
        .def("get_parameters", &AutoExposureController::getParameters,
             "Get current parameters")
        .def("compute_exposure", &AutoExposureController::computeExposure,
             "Compute exposure for a camera frame")
        .def("is_converged", &AutoExposureController::isConverged,
             "Check if auto exposure has converged")
        .def("get_statistics", &AutoExposureController::getStatistics,
             "Get auto exposure statistics")
        .def("get_last_scene_analysis", &AutoExposureController::getLastSceneAnalysis,
             "Get the last scene analysis results")
        .def("reset", &AutoExposureController::reset,
             "Reset the auto exposure controller")
        .def("__repr__", [](const AutoExposureController& controller) {
            auto stats = controller.getStatistics();
            return "<AutoExposureController frames=" + std::to_string(stats.frameCount) + 
                   " converged=" + (stats.isConverged ? "True" : "False") + ">";
        });
    
    // Utility functions
    m.def("create_test_frame", [](int width, int height, float brightness) {
        CameraFrame frame;
        frame.image = cv::Mat::zeros(height, width, CV_8UC3);
        cv::Scalar color(brightness * 255, brightness * 255, brightness * 255);
        frame.image.setTo(color);
        frame.frameNumber = 1;
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        return frame;
    }, "Create a test frame with specified dimensions and brightness",
    py::arg("width"), py::arg("height"), py::arg("brightness") = 0.5f);
    
    m.def("version", []() { return "1.0.0"; }, "Get version string");
    
    // Camera state and properties for completeness
    py::enum_<CameraState>(m, "CameraState")
        .value("DISCONNECTED", CameraState::DISCONNECTED)
        .value("CONNECTED", CameraState::CONNECTED)
        .value("STREAMING", CameraState::STREAMING)
        .value("ERROR", CameraState::ERROR)
        .export_values();
    
    py::class_<CameraProperties>(m, "CameraProperties")
        .def(py::init<>())
        .def_readwrite("width", &CameraProperties::width)
        .def_readwrite("height", &CameraProperties::height)
        .def_readwrite("fps", &CameraProperties::fps)
        .def_readwrite("has_auto_exposure", &CameraProperties::hasAutoExposure)
        .def_readwrite("has_auto_focus", &CameraProperties::hasAutoFocus)
        .def_readwrite("has_auto_white_balance", &CameraProperties::hasAutoWhiteBalance)
        .def_readwrite("device_id", &CameraProperties::deviceId);
} 