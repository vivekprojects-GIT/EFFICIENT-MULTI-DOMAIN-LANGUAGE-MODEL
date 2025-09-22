"""
Enhanced Inference Pipeline for Multi-Model LLM System

This module provides an inference pipeline that handles the complete flow from user
query input to domain-specific response generation. It coordinates query preprocessing,
intelligent routing, model selection, and response formatting to deliver accurate
and contextually appropriate answers.
"""

import logging
import time
import os
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class QueryPreprocessor:
    """
    Handles query cleaning, validation, and normalization.
    
    This component ensures that user queries are properly formatted and validated
    before being processed by the routing and inference systems. It handles
    text cleaning, length validation, and format standardization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean(self, query: str) -> str:
        """Clean and normalize input query."""
        if not query or not isinstance(query, str):
            return ""
        
        # Basic text cleaning
        query = query.strip()  # Remove leading/trailing whitespace
        query = " ".join(query.split())  # Normalize internal whitespace
        
        # Ensure proper punctuation for questions
        if query and not query.endswith(('?', '.', '!', ':')):
            query += "?"
        
        return query
    
    def validate(self, query: str) -> bool:
        """Check if query is valid for processing."""
        if not query or len(query.strip()) < 3:
            return False
        if len(query) > 1000:  # Reasonable length limit
            return False
        return True

class ResponseFormatter:
    """Formats model responses for optimal user experience."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format(self, response: str, domain: str, confidence: float, method: str) -> Dict[str, Any]:
        """Format the response with metadata and cleaning."""
        # Clean up the response text
        if response:
            response = response.strip()
            # Remove common response artifacts and repetitions
            response = self._clean_response(response)
        
        # Create structured response with metadata
        return {
            "answer": response,
            "domain": domain,
            "confidence": round(confidence, 3),
            "method": method,
            "routing_method": method,  # Keep both for compatibility
            "timestamp": time.time()
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean common response artifacts and repetitions."""
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Answer:", "Response:", "Here's the answer:",
            "Based on the information:", "According to:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove repetitive text patterns
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        return '\n'.join(cleaned_lines)

class ModelManager:
    """Manages specialized models for different domains."""
    
    def __init__(self, model_paths: Dict[str, str]):
        """Initialize with model paths for each domain."""
        self.model_paths = model_paths
        self.models = {}
        self.tokenizers = {}
        self.logger = logging.getLogger(__name__)
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all specialized models for the configured domains."""
        # Load base model once (shared by all LoRA adapters)
        base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            self.logger.info(f"Loading base model: {base_model_path}")
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            
            # Move to device
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.base_model = self.base_model.to(device)
            
            self.logger.info("Base model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise
        
        # Load LoRA adapters for each domain
        for domain, model_path in self.model_paths.items():
            try:
                self.logger.info(f"Loading {domain} LoRA adapter from {model_path}")
                
                # Try PyTorch format first, fallback to MLX format
                pytorch_path = f"{model_path}_pytorch"
                if os.path.exists(pytorch_path):
                    actual_path = pytorch_path
                    self.logger.info(f"Using PyTorch format: {actual_path}")
                else:
                    actual_path = model_path
                    self.logger.info(f"Using MLX format: {actual_path}")
                
                # Load tokenizer for the domain first
                tokenizer = AutoTokenizer.from_pretrained(actual_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Resize base model embeddings to match tokenizer vocabulary size
                # This is needed when special tokens were added during training
                if len(tokenizer) != self.base_model.get_input_embeddings().num_embeddings:
                    self.logger.info(f"Resizing model embeddings from {self.base_model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
                    self.base_model.resize_token_embeddings(len(tokenizer))
                
                # Load LoRA adapter
                model = PeftModel.from_pretrained(self.base_model, actual_path, local_files_only=True)
                
                # Store loaded model and tokenizer
                self.models[domain] = model
                self.tokenizers[domain] = tokenizer
                
                self.logger.info(f"Successfully loaded {domain} model")
                
            except Exception as e:
                self.logger.error(f"Failed to load {domain} model: {e}")
                raise
    
    def generate(self, domain: str, query: str, max_length: int = None) -> str:
        """Generate response using only specialized domain models."""
        if domain not in self.models:
            return f"Domain {domain} not available"
        
        if max_length is None:
            from app.config import settings
            max_length = settings.DEFAULT_MAX_LENGTH
        
        try:
            model = self.models[domain]
            tokenizer = self.tokenizers[domain]
            
            # Use domain-specific prompt format
            if domain == "sleep":
                prompt = f"Sleep Science Question: {query}\nAnswer:"
            elif domain == "car":
                prompt = f"Automotive History Question: {query}\nAnswer:"
            else:
                prompt = f"Q: {query}\nA:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up response
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            elif "A:" in response:
                response = response.split("A:")[-1].strip()
            
            # Validate response quality - if too short or generic, return unable
            response = response.strip()
            if self._is_poor_response(response, domain):
                return f"I'm unable to provide a specialized answer for this {domain} question. Please ask a more specific question within my domain expertise."
            
            return response
            
        except Exception as e:
            return f"I'm unable to process this {domain} question at the moment. Please try again."
    
    def _is_poor_response(self, response: str, domain: str) -> bool:
        """Check if response is too poor quality to be useful."""
        if not response or len(response.strip()) < 10:
            return True
        
        # Check for generic responses that indicate the model doesn't know
        poor_indicators = [
            "I don't know", "I can't answer", "I'm not sure",
            "I don't have information", "I cannot provide",
            "I'm unable to", "I don't have enough information",
            "I'm not trained on", "I don't have access to"
        ]
        
        response_lower = response.lower()
        for indicator in poor_indicators:
            if indicator in response_lower:
                return True
        
        # Check for very short responses that might be incomplete
        if len(response.split()) < 5:
            return True
            
        return False

class EnhancedInferencePipeline:
    """Main inference pipeline that coordinates query processing."""
    
    def __init__(self, router, model_paths: Dict[str, str]):
        """Initialize pipeline with router and model paths."""
        self.router = router
        self.preprocessor = QueryPreprocessor()
        self.formatter = ResponseFormatter()
        self.model_manager = ModelManager(model_paths)
        self.logger = logging.getLogger(__name__)
        
        # Simple stats
        self.stats = {
            "queries": 0,
            "errors": 0
        }
    
    def inference(self, query: str, max_length: int = None) -> Dict[str, Any]:
        """Main inference method."""
        if max_length is None:
            from app.config import settings
            max_length = settings.DEFAULT_MAX_LENGTH
            
        start_time = time.time()
        self.stats["queries"] += 1
        
        try:
            # Clean and validate query
            processed_query = self.preprocessor.clean(query)
            if not self.preprocessor.validate(processed_query):
                return self._handle_invalid_query(query)
            
            # Route to domain
            domain, confidence, method = self.router.route(processed_query)
            
            # Generate response
            response = self.model_manager.generate(domain, processed_query, max_length)
            
            # Format response
            result = self.formatter.format(response, domain, confidence, method)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            return self._handle_critical_error(e, query)
    
    def _handle_invalid_query(self, query: str) -> Dict[str, Any]:
        """Handle invalid or malformed queries."""
        return {
            "answer": "I didn't understand your question. Could you please rephrase it?",
            "domain": "error",
            "confidence": 0.0,
            "method": "invalid_query",
            "timestamp": time.time()
        }
    
    def _handle_critical_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """Handle critical errors during inference."""
        return {
            "answer": f"I apologize, but I encountered an error: {str(error)}",
            "domain": "error",
            "confidence": 0.0,
            "method": "error",
            "timestamp": time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.copy()