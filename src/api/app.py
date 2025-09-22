"""
FastAPI REST API for Multi-Model LLM System

This module provides a REST API interface for the multi-model LLM system.
It offers HTTP endpoints for query processing, health monitoring, and system
statistics. The API is designed for easy integration with other applications
and includes good error handling and validation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import sys
import os
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from routers.bge_router import BGERouter
from inference.enhanced_pipeline import EnhancedInferencePipeline
from app.config import settings

# Set up logging for API
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Model LLM System API",
    description="REST API for domain-specific question answering",
    version="1.0.0"
)

# Allow cross-origin requests (change for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for system components
router = None
inference_pipeline = None

# API models
class QueryRequest(BaseModel):
    question: str
    max_length: Optional[int] = settings.DEFAULT_MAX_LENGTH

class QueryResponse(BaseModel):
    answer: str
    domain: str
    confidence: float
    routing_method: str
    timestamp: float
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, str]

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global router, inference_pipeline
    
    try:
        logger.info("Initializing Multi-Model LLM System...")
        
        # Initialize BGE router for semantic similarity-based routing
        router = BGERouter(
            confidence_threshold=settings.ROUTER_CONFIDENCE_THRESHOLD,
            fallback_threshold=settings.ROUTER_FALLBACK_THRESHOLD
        )
        
        # Initialize inference pipeline with model paths
        model_paths = {
            "sleep": settings.SLEEP_MODEL_PATH,    # Sleep science specialized model
            "car": settings.CAR_MODEL_PATH         # Automotive history specialized model
        }
        inference_pipeline = EnhancedInferencePipeline(router, model_paths)
        
        logger.info("System initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        Dict[str, str]: Basic API information and available endpoints
    """
    return {
        "message": "Multi-Model LLM System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "query": "/query"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for system monitoring.
    
    Returns:
        HealthResponse: System health status and component information
    """
    try:
        # Check component status
        components = {
            "router": "healthy" if router is not None else "unhealthy",
            "inference_pipeline": "healthy" if inference_pipeline is not None else "unhealthy"
        }
        
        # Determine overall status
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            components={"error": str(e)}
        )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query processing endpoint.
    
    Processes user questions through the complete pipeline including
    routing, model selection, and response generation.
    
    Args:
        request (QueryRequest): Query request with question and parameters
        
    Returns:
        QueryResponse: Generated response with metadata
        
    Raises:
        HTTPException: If system is not initialized or processing fails
    """
    try:
        # Check if system is initialized
        if router is None or inference_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="System not initialized. Please try again later."
            )
        
        # Validate input
        if not request.question or len(request.question.strip()) < settings.MIN_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Question must be at least {settings.MIN_QUERY_LENGTH} characters long"
            )
        
        if len(request.question) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Question must be less than {settings.MAX_QUERY_LENGTH} characters"
            )
        
        # Process query through inference pipeline
        logger.info(f"Processing query: {request.question[:50]}...")
        start_time = time.time()
        
        result = inference_pipeline.inference(request.question, request.max_length)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.3f} seconds")
        
        # Return structured response
        return QueryResponse(
            answer=result["answer"],
            domain=result["domain"],
            confidence=result["confidence"],
            routing_method=result["method"],
            timestamp=result["timestamp"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """
    Get system performance statistics.
    
    Returns:
        Dict[str, Any]: Current system performance metrics
    """
    try:
        if inference_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="System not initialized"
            )
        
        stats = inference_pipeline.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

@app.get("/domains", response_model=Dict[str, Any])
async def get_domains():
    """
    Get information about supported domains.
    
    Returns:
        Dict[str, Any]: Information about supported domains and their capabilities
    """
    return {
        "supported_domains": ["sleep", "car"],
        "descriptions": {
            "sleep": "Sleep science and health-related questions",
            "car": "Automotive history and car-related questions"
        },
        "routing_method": "BGE semantic similarity",
        "confidence_thresholds": {
            "high_confidence": settings.ROUTER_CONFIDENCE_THRESHOLD,
            "medium_confidence": settings.ROUTER_FALLBACK_THRESHOLD
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)