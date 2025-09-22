import time
import json
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sentence_transformers import SentenceTransformer
import torch

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SystemEvaluator:
    """Evaluation system for the multi-model LLM system."""
    
    def __init__(self, semantic_model_name: str = 'all-MiniLM-L6-v2'):
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.logger = logging.getLogger(__name__)
        
        # Store evaluation results
        self.results = {
            "routing_accuracy": [],
            "response_quality": [],
            "response_times": [],
            "confidence_scores": [],
            "routing_methods": [],
            "errors": []
        }
    
    def evaluate_routing_accuracy(self, test_queries: List[str], true_labels: List[str], 
                                 router, verbose: bool = True) -> Dict[str, float]:
        """Test routing accuracy on sample queries."""
        predictions = []
        confidences = []
        methods = []
        
        self.logger.info(f"Evaluating routing accuracy on {len(test_queries)} queries")
        
        for i, (query, true_label) in enumerate(zip(test_queries, true_labels)):
            try:
                domain, confidence, method = router.route(query)
                predictions.append(domain)
                confidences.append(confidence)
                methods.append(method)
                
                self.results["routing_accuracy"].append({
                    "query": query,
                    "true_label": true_label,
                    "predicted": domain,
                    "confidence": float(confidence),
                    "method": method,
                    "correct": domain == true_label
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating query {i}: {e}")
                predictions.append("error")
                confidences.append(0.0)
                methods.append("error")
                self.results["errors"].append({"query": query, "error": str(e)})
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        if verbose:
            print(f"Routing Accuracy: {accuracy:.3f}")
            print(f"Total Queries: {len(test_queries)}")
        
        return {"accuracy": accuracy, "total_queries": len(test_queries)}
    
    def evaluate_response_quality(self, test_queries: List[str], responses: List[str], 
                                 true_labels: List[str], verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate response quality using semantic similarity and other metrics.
        
        Args:
            test_queries: List of test questions
            responses: List of generated responses
            true_labels: List of true domain labels
            verbose: Whether to print detailed results
        """
        self.logger.info(f"Evaluating response quality on {len(test_queries)} responses")
        
        # Semantic similarity between query and response
        similarities = []
        for query, response in zip(test_queries, responses):
            try:
                query_emb = self.semantic_model.encode([query])
                response_emb = self.semantic_model.encode([response])
                similarity = np.dot(query_emb, response_emb.T)[0][0]
                similarities.append(similarity)
            except Exception as e:
                self.logger.error(f"Error calculating similarity: {e}")
                similarities.append(0.0)
        
        # Response length analysis
        response_lengths = [len(response.split()) for response in responses]
        
        # Domain consistency (check if response seems to match the expected domain)
        domain_consistency = self._evaluate_domain_consistency(responses, true_labels)
        
        results = {
            "avg_semantic_similarity": float(np.mean(similarities)),
            "avg_response_length": float(np.mean(response_lengths)),
            "domain_consistency": float(domain_consistency),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities))
        }
        
        # Store detailed results
        for i, (query, response, true_label) in enumerate(zip(test_queries, responses, true_labels)):
            self.results["response_quality"].append({
                "query": query,
                "response": response,
                "true_label": true_label,
                "similarity": float(similarities[i]),
                "length": response_lengths[i]
            })
        
        if verbose:
            self._print_quality_results(results)
        
        return results
    
    def _evaluate_domain_consistency(self, responses: List[str], true_labels: List[str]) -> float:
        """Evaluate if responses are consistent with their expected domains."""
        sleep_keywords = ["sleep", "dream", "rest", "insomnia", "REM", "bedtime", "sleeping"]
        car_keywords = ["car", "automobile", "vehicle", "engine", "driving", "automotive", "manufacturer"]
        
        consistent_count = 0
        
        for response, label in zip(responses, true_labels):
            response_lower = response.lower()
            
            if label == "sleep":
                # Check if response contains sleep-related keywords
                if any(keyword in response_lower for keyword in sleep_keywords):
                    consistent_count += 1
            elif label == "car":
                # Check if response contains car-related keywords
                if any(keyword in response_lower for keyword in car_keywords):
                    consistent_count += 1
        
        return consistent_count / len(responses) if responses else 0.0
    
    def evaluate_response_times(self, inference_pipeline, test_queries: List[str], 
                               verbose: bool = True) -> Dict[str, float]:
        """Evaluate response times for the inference pipeline."""
        self.logger.info(f"Evaluating response times on {len(test_queries)} queries")
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            try:
                result = inference_pipeline.inference(query)
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                self.results["response_times"].append({
                    "query": query,
                    "response_time": float(response_time),
                    "domain": result.get("domain", "unknown"),
                    "confidence": float(result.get("confidence", 0.0))
                })
                
            except Exception as e:
                self.logger.error(f"Error timing query: {e}")
                response_times.append(float('inf'))
        
        # Filter out infinite times (errors)
        valid_times = [t for t in response_times if t != float('inf')]
        
        results = {
            "avg_response_time": float(np.mean(valid_times)) if valid_times else 0.0,
            "min_response_time": float(np.min(valid_times)) if valid_times else 0.0,
            "max_response_time": float(np.max(valid_times)) if valid_times else 0.0,
            "std_response_time": float(np.std(valid_times)) if valid_times else 0.0,
            "total_queries": len(test_queries),
            "successful_queries": len(valid_times),
            "error_rate": float((len(test_queries) - len(valid_times)) / len(test_queries)) if test_queries else 0.0
        }
        
        if verbose:
            self._print_timing_results(results)
        
        return results
    
    def comprehensive_evaluation(self, test_queries: List[str], true_labels: List[str], 
                               router, inference_pipeline, verbose: bool = True) -> Dict[str, Any]:
        """Run complete evaluation of the entire system."""
        self.logger.info("Starting complete system evaluation")
        
        # 1. Routing accuracy
        routing_results = self.evaluate_routing_accuracy(test_queries, true_labels, router, verbose)
        
        # 2. Response times
        timing_results = self.evaluate_response_times(inference_pipeline, test_queries, verbose)
        
        # 3. Generate responses for quality evaluation
        responses = []
        for query in test_queries:
            try:
                result = inference_pipeline.inference(query)
                responses.append(result.get("answer", ""))
            except Exception as e:
                self.logger.error(f"Error generating response for quality evaluation: {e}")
                responses.append("")
        
        # 4. Response quality
        quality_results = self.evaluate_response_quality(test_queries, responses, true_labels, verbose)
        
        # 5. Overall system score
        overall_score = self._calculate_overall_score(routing_results, quality_results, timing_results)
        
        complete_results = {
            "routing": routing_results,
            "quality": quality_results,
            "timing": timing_results,
            "overall_score": overall_score,
            "evaluation_timestamp": time.time()
        }
        
        if verbose:
            self._print_complete_results(complete_results)
        
        return complete_results
    
    def _calculate_overall_score(self, routing_results: Dict, quality_results: Dict, 
                                timing_results: Dict) -> float:
        """Calculate overall system score (0-100)."""
        # Weighted combination of different metrics
        routing_score = routing_results.get("overall_accuracy", 0) * 100
        quality_score = quality_results.get("avg_semantic_similarity", 0) * 100
        timing_score = max(0, 100 - timing_results.get("avg_response_time", 10) * 10)  # Penalize slow responses
        
        # Weighted average
        overall_score = (routing_score * 0.4 + quality_score * 0.4 + timing_score * 0.2)
        return round(overall_score, 2)
    
    def _print_quality_results(self, results: Dict):
        """Print response quality results."""
        print("\n" + "="*50)
        print("RESPONSE QUALITY RESULTS")
        print("="*50)
        print(f"Average Semantic Similarity: {results['avg_semantic_similarity']:.3f}")
        print(f"Average Response Length: {results['avg_response_length']:.1f} words")
        print(f"Domain Consistency: {results['domain_consistency']:.3f}")
        print(f"Similarity Range: {results['min_similarity']:.3f} - {results['max_similarity']:.3f}")
        print(f"Similarity Std Dev: {results['std_similarity']:.3f}")
    
    def _print_timing_results(self, results: Dict):
        """Print response timing results."""
        print("\n" + "="*50)
        print("RESPONSE TIMING RESULTS")
        print("="*50)
        print(f"Average Response Time: {results['avg_response_time']:.3f}s")
        print(f"Min Response Time: {results['min_response_time']:.3f}s")
        print(f"Max Response Time: {results['max_response_time']:.3f}s")
        print(f"Std Dev: {results['std_response_time']:.3f}s")
        print(f"Success Rate: {(1 - results['error_rate']):.3f}")
    
    def _print_complete_results(self, results: Dict):
        """Print complete evaluation results."""
        print("\n" + "="*60)
        print("COMPLETE SYSTEM EVALUATION")
        print("="*60)
        print(f"Overall System Score: {results['overall_score']}/100")
        print(f"Evaluation Timestamp: {time.ctime(results['evaluation_timestamp'])}")
    
    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"Evaluation results saved to {filepath}")
    
    def load_test_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Load test data from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        queries = [item['question'] for item in data]
        labels = [item['domain'] for item in data]
        
        return queries, labels