"""
FastAPI service for OpenCam Auto Exposure Algorithm
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import base64
from io import BytesIO

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Agent features disabled.")

# Try to import our package
try:
    from opencam import AutoExposure, Parameters, MeteringMode
    from opencam.utils import create_test_frame, compute_image_statistics
    from opencam.benchmark import AutoExposureBenchmark
    OPENCAM_AVAILABLE = True
except ImportError:
    OPENCAM_AVAILABLE = False
    logging.warning("OpenCam package not available. Using mock implementations.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="OpenCam Auto Exposure API",
    description="REST API for OpenCam Auto Exposure Algorithm with AI Agent Integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
if OPENCAM_AVAILABLE:
    global_controller = AutoExposure()
else:
    global_controller = None

if OPENAI_AVAILABLE:
    openai_client = OpenAI()
else:
    openai_client = None

# Pydantic models
class ParametersRequest(BaseModel):
    mode: str = Field(default="INTELLIGENT", description="Metering mode")
    target_brightness: float = Field(default=0.5, ge=0.0, le=1.0, description="Target brightness")
    convergence_speed: float = Field(default=0.15, ge=0.0, le=1.0, description="Convergence speed")
    min_exposure: float = Field(default=0.001, gt=0.0, description="Minimum exposure")
    max_exposure: float = Field(default=1.0, gt=0.0, description="Maximum exposure")
    lock_exposure: bool = Field(default=False, description="Lock exposure")
    enable_face_detection: bool = Field(default=True, description="Enable face detection")
    enable_scene_analysis: bool = Field(default=True, description="Enable scene analysis")
    exposure_compensation: float = Field(default=0.0, ge=-2.0, le=2.0, description="Exposure compensation")

class ExposureRequest(BaseModel):
    image_data: str = Field(description="Base64 encoded image data")
    frame_number: int = Field(default=0, description="Frame number")
    timestamp: Optional[int] = Field(default=None, description="Timestamp")
    parameters: Optional[ParametersRequest] = Field(default=None, description="Override parameters")

class ExposureResponse(BaseModel):
    exposure: float
    scene_analysis: Dict[str, Any]
    statistics: Dict[str, Any]
    processing_time_ms: float
    success: bool
    message: str

class BenchmarkRequest(BaseModel):
    iterations: int = Field(default=100, ge=1, le=10000, description="Number of iterations")
    resolutions: Optional[List[List[int]]] = Field(default=None, description="List of [width, height] pairs")
    modes: Optional[List[str]] = Field(default=None, description="List of metering modes")

class AgentRequest(BaseModel):
    query: str = Field(description="Natural language query for the AI agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class AgentResponse(BaseModel):
    response: str
    actions_taken: List[str]
    results: Optional[Dict[str, Any]]
    success: bool

# Utility functions
def decode_image(base64_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 image data"""
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', image_bgr)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

def create_mock_response(message: str) -> ExposureResponse:
    """Create mock response when OpenCam is not available"""
    return ExposureResponse(
        exposure=0.5,
        scene_analysis={"scene_type": "unknown", "confidence": 0.0},
        statistics={"frame_count": 0, "is_converged": False},
        processing_time_ms=1.0,
        success=False,
        message=message
    )

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenCam Auto Exposure API",
        "version": "1.0.0",
        "author": "Nik Jois <nikjois@llamasearch.ai>",
        "opencam_available": OPENCAM_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "parameters": "/parameters",
            "exposure": "/exposure",
            "benchmark": "/benchmark",
            "agent": "/agent",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "opencam_available": OPENCAM_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE
    }

@app.get("/parameters")
async def get_parameters():
    """Get current auto exposure parameters"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        params = global_controller.parameters
        return {
            "mode": params.mode.value,
            "target_brightness": params.target_brightness,
            "convergence_speed": params.convergence_speed,
            "min_exposure": params.min_exposure,
            "max_exposure": params.max_exposure,
            "lock_exposure": params.lock_exposure,
            "enable_face_detection": params.enable_face_detection,
            "enable_scene_analysis": params.enable_scene_analysis,
            "exposure_compensation": params.exposure_compensation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")

@app.post("/parameters")
async def set_parameters(params: ParametersRequest):
    """Set auto exposure parameters"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        # Convert to OpenCam parameters
        mode_map = {
            "DISABLED": MeteringMode.DISABLED,
            "AVERAGE": MeteringMode.AVERAGE,
            "CENTER_WEIGHTED": MeteringMode.CENTER_WEIGHTED,
            "SPOT": MeteringMode.SPOT,
            "MULTI_ZONE": MeteringMode.MULTI_ZONE,
            "INTELLIGENT": MeteringMode.INTELLIGENT,
        }
        
        if params.mode not in mode_map:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {params.mode}")
        
        new_params = Parameters(
            mode=mode_map[params.mode],
            target_brightness=params.target_brightness,
            convergence_speed=params.convergence_speed,
            min_exposure=params.min_exposure,
            max_exposure=params.max_exposure,
            lock_exposure=params.lock_exposure,
            enable_face_detection=params.enable_face_detection,
            enable_scene_analysis=params.enable_scene_analysis,
            exposure_compensation=params.exposure_compensation,
        )
        
        global_controller.parameters = new_params
        
        return {"success": True, "message": "Parameters updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set parameters: {str(e)}")

@app.post("/exposure", response_model=ExposureResponse)
async def compute_exposure(request: ExposureRequest):
    """Compute exposure for an image"""
    if not OPENCAM_AVAILABLE:
        return create_mock_response("OpenCam not available")
    
    start_time = time.perf_counter()
    
    try:
        # Decode image
        image = decode_image(request.image_data)
        
        # Set parameters if provided
        if request.parameters:
            await set_parameters(request.parameters)
        
        # Compute exposure
        exposure = global_controller.compute_exposure(
            image, 
            frame_number=request.frame_number,
            timestamp=request.timestamp
        )
        
        # Get analysis results
        scene_analysis = global_controller.get_scene_analysis()
        statistics = global_controller.get_statistics()
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ExposureResponse(
            exposure=exposure,
            scene_analysis={
                "scene_type": scene_analysis.scene_type,
                "confidence": scene_analysis.confidence,
                "is_low_light": scene_analysis.is_low_light,
                "is_backlit": scene_analysis.is_backlit,
                "is_high_contrast": scene_analysis.is_high_contrast,
                "has_faces": scene_analysis.has_faces,
            },
            statistics={
                "average_exposure": statistics.average_exposure,
                "average_brightness": statistics.average_brightness,
                "is_converged": statistics.is_converged,
                "frame_count": statistics.frame_count,
                "convergence_time_ms": statistics.convergence_time_ms,
            },
            processing_time_ms=processing_time,
            success=True,
            message="Exposure computed successfully"
        )
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.error(f"Exposure computation failed: {e}")
        
        return ExposureResponse(
            exposure=0.0,
            scene_analysis={},
            statistics={},
            processing_time_ms=processing_time,
            success=False,
            message=f"Failed to compute exposure: {str(e)}"
        )

@app.post("/exposure/file")
async def compute_exposure_from_file(file: UploadFile = File(...)):
    """Compute exposure for an uploaded image file"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        # Read file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode to base64 for processing
        image_base64 = encode_image(image)
        
        # Process using existing endpoint
        request = ExposureRequest(image_data=image_base64.split(',')[1])
        return await compute_exposure(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run performance benchmark"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        benchmark = AutoExposureBenchmark(global_controller)
        
        # Run benchmark in background
        def run_benchmark_task():
            result = benchmark.run_performance_benchmark(iterations=request.iterations)
            # Save results
            timestamp = int(time.time())
            benchmark.save_results(f"benchmark_results_{timestamp}.json")
            return result
        
        background_tasks.add_task(run_benchmark_task)
        
        return {
            "success": True,
            "message": f"Benchmark started with {request.iterations} iterations",
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start benchmark: {str(e)}")

@app.get("/benchmark/status")
async def get_benchmark_status():
    """Get benchmark status and results"""
    # This is a simplified implementation
    # In production, you'd track running benchmarks
    return {
        "status": "completed",
        "results_available": True,
        "message": "Check benchmark files for detailed results"
    }

@app.post("/agent", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    """Query the AI agent for auto exposure assistance"""
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")
    
    try:
        # System prompt for the auto exposure agent
        system_prompt = """
        You are an expert AI assistant for the OpenCam Auto Exposure system. 
        You help users with:
        - Configuring auto exposure parameters
        - Analyzing image exposure issues
        - Running benchmarks and interpreting results
        - Troubleshooting exposure problems
        
        You have access to the following API endpoints:
        - GET/POST /parameters - get/set exposure parameters
        - POST /exposure - compute exposure for images
        - POST /benchmark - run performance benchmarks
        
        Respond with specific, actionable advice and suggest API calls when appropriate.
        """
        
        # Add context if provided
        context_str = ""
        if request.context:
            context_str = f"\nCurrent context: {json.dumps(request.context, indent=2)}"
        
        # Query OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query + context_str}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Parse response for potential actions
        actions_taken = []
        results = {}
        
        # Simple action detection (in production, use function calling)
        if "benchmark" in request.query.lower():
            actions_taken.append("Suggested running benchmark")
        if "parameter" in request.query.lower():
            actions_taken.append("Suggested checking parameters")
        
        return AgentResponse(
            response=ai_response,
            actions_taken=actions_taken,
            results=results,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        return AgentResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            actions_taken=[],
            results={},
            success=False
        )

@app.get("/test/image")
async def generate_test_image():
    """Generate a test image for testing"""
    try:
        if OPENCAM_AVAILABLE:
            image = create_test_frame(640, 480, 0.5)
        else:
            # Create simple test image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        image_base64 = encode_image(image)
        
        return {
            "image_data": image_base64,
            "width": 640,
            "height": 480,
            "format": "RGB"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate test image: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get current auto exposure statistics"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        stats = global_controller.get_statistics()
        return {
            "average_exposure": stats.average_exposure,
            "average_brightness": stats.average_brightness,
            "min_exposure": stats.min_exposure,
            "max_exposure": stats.max_exposure,
            "is_converged": stats.is_converged,
            "frame_count": stats.frame_count,
            "convergence_time_ms": stats.convergence_time_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/reset")
async def reset_controller():
    """Reset the auto exposure controller"""
    if not OPENCAM_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenCam not available")
    
    try:
        global_controller.reset()
        return {"success": True, "message": "Controller reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset controller: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 