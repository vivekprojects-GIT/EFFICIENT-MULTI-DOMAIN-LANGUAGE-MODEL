#!/usr/bin/env python3
"""
Multi-Model LLM System with Router

This system provides an interface for interacting with specialized language models
through intelligent query routing. It automatically determines which domain-specific
model should handle each query and provides REST API and evaluation interfaces.
"""

import argparse
import logging
import sys
import os
from typing import Dict, Any, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import system components
from routers.bge_router import BGERouter
from inference.enhanced_pipeline import EnhancedInferencePipeline
from evaluation.academic_evaluator import AcademicEvaluator
from app.config import settings

def setup_logging(level: str = None) -> None:
    """
    Set up logging for the entire system.
    
    This helps us see what's happening and debug any issues that come up.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR). If None, uses settings.LOG_LEVEL
    """
    if level is None:
        level = settings.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def initialize_system(confidence_threshold: float = None) -> Tuple[BGERouter, EnhancedInferencePipeline]:
    """
    Initialize the complete system with router and inference pipeline.
    
    Sets up all the components we need including the BGE router for query
    classification and the inference pipeline for model coordination.
    
    Args:
        confidence_threshold: Minimum confidence for reliable query routing. If None, uses settings.
        
    Returns:
        Tuple containing the initialized router and inference pipeline
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing Multi-Model LLM System...")
        
        model_paths = {
            "sleep": settings.SLEEP_MODEL_PATH,
            "car": settings.CAR_MODEL_PATH
        }
        
        if confidence_threshold is None:
            confidence_threshold = settings.ROUTER_CONFIDENCE_THRESHOLD
        
        router = BGERouter(
            confidence_threshold=confidence_threshold,
            fallback_threshold=settings.ROUTER_FALLBACK_THRESHOLD
        )
        
        pipeline = EnhancedInferencePipeline(router, model_paths)
        
        logger.info("System initialized successfully!")
        return router, pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def evaluate_system() -> None:
    """Run comprehensive academic evaluation."""
    print("Running Academic System Evaluation...")
    
    try:
        # Initialize system for evaluation
        router, pipeline = initialize_system()
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return
    
    # Enhanced test queries for comprehensive evaluation
    test_queries = [
        # Sleep domain
        "What is REM sleep?", "How many hours of sleep do I need?",
        "What causes insomnia?", "How does sleep affect memory?",
        "What are the different sleep stages?",
        
        # Car domain
        "Who invented the first car?", "When was the Ford Model T produced?",
        "What is the history of BMW?", "Who founded Mercedes-Benz?",
        "What is the history of the automobile industry?",
        
        # Medical domain
        "What are the main symptoms of diabetes?", "What is hypertension?",
        "What is the difference between a virus and bacteria?",
        
        # Legal domain
        "What is the difference between civil law and criminal law?",
        "What is the statute of limitations?",
        
        # Finance domain
        "What is compound interest?", "What is the difference between stocks and bonds?",
        
        # Technical domain
        "What is machine learning?", "What is the difference between HTTP and HTTPS?"
    ]
    
    true_labels = [
        "sleep", "sleep", "sleep", "sleep", "sleep",
        "car", "car", "car", "car", "car",
        "medical", "medical", "medical",
        "legal", "legal",
        "finance", "finance",
        "technical", "technical"
    ]
    
    # Run comprehensive academic evaluation
    evaluator = AcademicEvaluator(settings)
    results = evaluator.comprehensive_evaluation(
        test_queries, true_labels, router, pipeline, 
        baseline_methods=["keyword_routing", "rule_based_routing", "random_routing"],
        verbose=True
    )
    
    # Generate publication-ready figures
    evaluator.generate_publication_figures(results)
    
    print(f"\nAcademic evaluation complete!")
    print(f"Overall System Score: {results['overall_score']}/100")
    print(f"Routing Accuracy: {results['routing']['overall_accuracy']:.3f}")
    print(f"Semantic Similarity: {results['quality'].get('semantic_similarity', 0):.3f}")
    print(f"Error Rate: {results['errors']['error_rate']:.3f}")

def start_api_server(host: str = None, port: int = None) -> None:
    """Start the FastAPI web server."""
    import uvicorn
    from api.app import app
    
    if host is None:
        host = settings.API_HOST
    if port is None:
        port = settings.API_PORT
    
    print(f"Starting API server on {host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    
    # Start the FastAPI server with uvicorn
    uvicorn.run(app, host=host, port=port)

def main() -> None:
    """Main entry point."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Multi-Model LLM System with BGE Router")
    parser.add_argument("--mode", choices=["api", "evaluate"], default="api",
                       help="Mode: api (web server), evaluate (testing)")
    parser.add_argument("--host", default=settings.API_HOST, help="API host (for api mode)")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="API port (for api mode)")
    parser.add_argument("--log-level", default=settings.LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Set up logging with specified level
    setup_logging(args.log_level)
    
    # Launch appropriate mode based on arguments
    if args.mode == "api":
        start_api_server(args.host, args.port)
    elif args.mode == "evaluate":
        evaluate_system()

if __name__ == "__main__":
    main()