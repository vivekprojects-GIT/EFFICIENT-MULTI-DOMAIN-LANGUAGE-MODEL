"""
Academic-grade evaluation framework for multi-domain LLM systems.
Implements comprehensive evaluation metrics, baseline comparisons, and statistical analysis.
"""

import time
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# Import evaluation libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    EVALUATION_LIBS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some evaluation libraries not available: {e}")
    EVALUATION_LIBS_AVAILABLE = False

@dataclass
class EvaluationResult:
    """Structured evaluation result."""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DomainResult:
    """Results for a specific domain."""
    domain: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    error_cases: Optional[List[Dict[str, Any]]] = None

class AcademicEvaluator:
    """
    Comprehensive evaluation framework for academic research.
    Implements state-of-the-art evaluation metrics and statistical analysis.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        # Initialize evaluation models
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if EVALUATION_LIBS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Set random seed for reproducibility
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)
        
        # Create output directories
        Path(config.EVAL_OUTPUT_DIR).mkdir(exist_ok=True)
    
    def comprehensive_evaluation(
        self, 
        test_queries: List[str], 
        true_labels: List[str],
        router,
        inference_pipeline,
        baseline_methods: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including routing, quality, and baseline comparisons.
        """
        if baseline_methods is None:
            baseline_methods = self.config.BASELINE_METHODS
        
        self.logger.info("Starting comprehensive academic evaluation...")
        start_time = time.time()
        
        # 1. Routing Evaluation
        routing_results = self.evaluate_routing_accuracy(test_queries, true_labels, router)
        
        # 2. Response Quality Evaluation
        quality_results = self.evaluate_response_quality(
            test_queries, true_labels, inference_pipeline
        )
        
        # 3. Timing Evaluation
        timing_results = self.evaluate_timing(test_queries, inference_pipeline)
        
        # 4. Baseline Comparisons
        baseline_results = self.evaluate_baselines(
            test_queries, true_labels, baseline_methods
        )
        
        # 5. Statistical Analysis
        statistical_results = self.perform_statistical_analysis(
            quality_results, baseline_results
        )
        
        # 6. Error Analysis
        error_results = self.perform_error_analysis(
            test_queries, true_labels, inference_pipeline
        )
        
        # 7. Domain-specific Analysis
        domain_results = self.analyze_domain_performance(
            test_queries, true_labels, inference_pipeline
        )
        
        # Combine all results
        complete_results = {
            "routing": routing_results,
            "quality": quality_results,
            "timing": timing_results,
            "baselines": baseline_results,
            "statistical": statistical_results,
            "errors": error_results,
            "domains": domain_results,
            "overall_score": self._calculate_overall_score(
                routing_results, quality_results, timing_results
            ),
            "evaluation_metadata": {
                "timestamp": time.time(),
                "config": self.config.dict(),
                "total_queries": len(test_queries),
                "domains": list(set(true_labels)),
                "evaluation_time": time.time() - start_time
            }
        }
        
        if verbose:
            self._print_comprehensive_results(complete_results)
        
        # Save results
        if self.config.EVALUATION_RESULTS:
            self._save_results(complete_results)
        
        return complete_results
    
    def evaluate_routing_accuracy(
        self, 
        test_queries: List[str], 
        true_labels: List[str], 
        router
    ) -> Dict[str, Any]:
        """Evaluate routing accuracy with comprehensive metrics."""
        self.logger.info("Evaluating routing accuracy...")
        
        predictions = []
        confidences = []
        routing_times = []
        
        for query in test_queries:
            start_time = time.time()
            domain, confidence, method = router.route(query)
            routing_time = time.time() - start_time
            
            predictions.append(domain)
            confidences.append(confidence)
            routing_times.append(routing_time)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-domain metrics
        domain_metrics = {}
        unique_labels = list(set(true_labels))
        for domain in unique_labels:
            domain_precision, domain_recall, domain_f1, domain_support = precision_recall_fscore_support(
                true_labels, predictions, labels=[domain], average='binary'
            )
            domain_metrics[domain] = {
                "precision": domain_precision[0] if len(domain_precision) > 0 else 0,
                "recall": domain_recall[0] if len(domain_recall) > 0 else 0,
                "f1": domain_f1[0] if len(domain_f1) > 0 else 0,
                "support": domain_support[0] if len(domain_support) > 0 else 0
            }
        
        # Confidence analysis
        confidence_stats = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "median": np.median(confidences)
        }
        
        # Timing analysis
        timing_stats = {
            "mean": np.mean(routing_times),
            "std": np.std(routing_times),
            "min": np.min(routing_times),
            "max": np.max(routing_times),
            "median": np.median(routing_times)
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        results = {
            "overall_accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
            "domain_metrics": domain_metrics,
            "confidence_stats": confidence_stats,
            "timing_stats": timing_stats,
            "confusion_matrix": cm.tolist(),
            "unique_labels": unique_labels,
            "predictions": predictions,
            "confidences": confidences
        }
        
        return results
    
    def evaluate_response_quality(
        self, 
        test_queries: List[str], 
        true_labels: List[str],
        inference_pipeline
    ) -> Dict[str, Any]:
        """Evaluate response quality using multiple metrics."""
        self.logger.info("Evaluating response quality...")
        
        responses = []
        response_times = []
        
        # Generate responses
        for query in test_queries:
            start_time = time.time()
            result = inference_pipeline.inference(query)
            response_time = time.time() - start_time
            
            responses.append(result["answer"])
            response_times.append(response_time)
        
        # Calculate various quality metrics
        metrics = {}
        
        # 1. BLEU Scores (if available)
        if EVALUATION_LIBS_AVAILABLE:
            metrics.update(self._calculate_bleu_scores(responses, test_queries))
            metrics.update(self._calculate_rouge_scores(responses, test_queries))
            metrics.update(self._calculate_meteor_scores(responses, test_queries))
        
        # 2. Semantic Similarity
        metrics.update(self._calculate_semantic_similarity(responses, test_queries))
        
        # 3. Perplexity
        metrics.update(self._calculate_perplexity(responses))
        
        # 4. Diversity Metrics
        metrics.update(self._calculate_diversity_metrics(responses))
        
        # 5. Domain Consistency
        metrics.update(self._calculate_domain_consistency(responses, true_labels))
        
        # 6. Response Length Analysis
        response_lengths = [len(response.split()) for response in responses]
        metrics.update({
            "avg_response_length": np.mean(response_lengths),
            "std_response_length": np.std(response_lengths),
            "min_response_length": np.min(response_lengths),
            "max_response_length": np.max(response_lengths)
        })
        
        # 7. Response Time Analysis
        metrics.update({
            "avg_response_time": np.mean(response_times),
            "std_response_time": np.std(response_times),
            "min_response_time": np.min(response_times),
            "max_response_time": np.max(response_times)
        })
        
        return metrics
    
    def _calculate_bleu_scores(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores."""
        smoothing = SmoothingFunction().method1
        bleu_scores = []
        
        for response, reference in zip(responses, references):
            # Tokenize
            response_tokens = response.split()
            reference_tokens = reference.split()
            
            # Calculate BLEU
            bleu = sentence_bleu([reference_tokens], response_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        return {
            "bleu_1": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "bleu_min": np.min(bleu_scores),
            "bleu_max": np.max(bleu_scores)
        }
    
    def _calculate_rouge_scores(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for response, reference in zip(responses, references):
            scores = self.rouge_scorer.score(reference, response)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        return {
            "rouge1_f1": np.mean(rouge_scores["rouge1"]),
            "rouge2_f1": np.mean(rouge_scores["rouge2"]),
            "rougeL_f1": np.mean(rouge_scores["rougeL"])
        }
    
    def _calculate_meteor_scores(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate METEOR scores."""
        meteor_scores = []
        
        for response, reference in zip(responses, references):
            score = meteor_score([reference], response)
            meteor_scores.append(score)
        
        return {
            "meteor": np.mean(meteor_scores),
            "meteor_std": np.std(meteor_scores)
        }
    
    def _calculate_semantic_similarity(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity using sentence transformers."""
        similarities = []
        
        for response, reference in zip(responses, references):
            # Get embeddings
            response_embedding = self.semantic_model.encode([response])
            reference_embedding = self.semantic_model.encode([reference])
            
            # Calculate cosine similarity
            similarity = np.dot(response_embedding[0], reference_embedding[0]) / (
                np.linalg.norm(response_embedding[0]) * np.linalg.norm(reference_embedding[0])
            )
            similarities.append(similarity)
        
        return {
            "semantic_similarity": np.mean(similarities),
            "semantic_similarity_std": np.std(similarities),
            "semantic_similarity_min": np.min(similarities),
            "semantic_similarity_max": np.max(similarities)
        }
    
    def _calculate_perplexity(self, responses: List[str]) -> Dict[str, float]:
        """Calculate perplexity (simplified version)."""
        # This is a simplified perplexity calculation
        # In practice, you'd need access to the model's probability distributions
        total_tokens = sum(len(response.split()) for response in responses)
        total_chars = sum(len(response) for response in responses)
        
        # Rough estimate based on character-to-token ratio
        estimated_perplexity = total_chars / total_tokens if total_tokens > 0 else 0
        
        return {
            "estimated_perplexity": estimated_perplexity,
            "avg_tokens_per_response": total_tokens / len(responses) if responses else 0
        }
    
    def _calculate_diversity_metrics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics."""
        all_tokens = []
        all_bigrams = []
        
        for response in responses:
            tokens = response.split()
            all_tokens.extend(tokens)
            
            # Create bigrams
            for i in range(len(tokens) - 1):
                all_bigrams.append((tokens[i], tokens[i + 1]))
        
        # Calculate distinct-n metrics
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "vocabulary_size": len(set(all_tokens)),
            "avg_unique_tokens_per_response": np.mean([len(set(response.split())) for response in responses])
        }
    
    def _calculate_domain_consistency(self, responses: List[str], true_labels: List[str]) -> Dict[str, float]:
        """Calculate domain consistency metrics."""
        domain_keywords = {
            "sleep": ["sleep", "dream", "rest", "insomnia", "REM", "bedtime", "sleeping", "awake"],
            "car": ["car", "automobile", "vehicle", "engine", "driving", "automotive", "manufacturer", "vehicle"],
            "medical": ["medical", "health", "doctor", "patient", "disease", "treatment", "medicine", "hospital"],
            "legal": ["legal", "law", "court", "contract", "rights", "justice", "attorney", "legal"],
            "finance": ["finance", "money", "bank", "investment", "stock", "market", "financial", "economy"],
            "technical": ["technical", "technology", "software", "programming", "computer", "code", "system", "technical"]
        }
        
        consistent_count = 0
        total_count = len(responses)
        
        for response, true_label in zip(responses, true_labels):
            response_lower = response.lower()
            
            if true_label in domain_keywords:
                keywords = domain_keywords[true_label]
                # Check if response contains domain-relevant keywords
                if any(keyword in response_lower for keyword in keywords):
                    consistent_count += 1
        
        return {
            "domain_consistency": consistent_count / total_count if total_count > 0 else 0,
            "consistent_responses": consistent_count,
            "total_responses": total_count
        }
    
    def evaluate_timing(self, test_queries: List[str], inference_pipeline) -> Dict[str, Any]:
        """Evaluate system timing performance."""
        self.logger.info("Evaluating timing performance...")
        
        times = []
        
        for query in test_queries:
            start_time = time.time()
            result = inference_pipeline.inference(query)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            "avg_response_time": np.mean(times),
            "std_response_time": np.std(times),
            "min_response_time": np.min(times),
            "max_response_time": np.max(times),
            "median_response_time": np.median(times),
            "p95_response_time": np.percentile(times, 95),
            "p99_response_time": np.percentile(times, 99)
        }
    
    def evaluate_baselines(
        self, 
        test_queries: List[str], 
        true_labels: List[str],
        baseline_methods: List[str]
    ) -> Dict[str, Any]:
        """Evaluate baseline methods for comparison."""
        self.logger.info("Evaluating baseline methods...")
        
        baseline_results = {}
        
        for method in baseline_methods:
            try:
                if method == "keyword_routing":
                    results = self._evaluate_keyword_routing_baseline(test_queries, true_labels)
                elif method == "rule_based_routing":
                    results = self._evaluate_rule_based_routing_baseline(test_queries, true_labels)
                elif method == "random_routing":
                    results = self._evaluate_random_routing_baseline(test_queries, true_labels)
                else:
                    self.logger.warning(f"Baseline method {method} not implemented, skipping...")
                    continue
                
                baseline_results[method] = results
                
            except Exception as e:
                self.logger.error(f"Error evaluating baseline {method}: {e}")
                baseline_results[method] = {"error": str(e)}
        
        return baseline_results
    
    def _evaluate_keyword_routing_baseline(self, test_queries: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Evaluate keyword-based routing baseline."""
        domain_keywords = {
            "sleep": ["sleep", "dream", "rest", "insomnia", "REM", "bedtime"],
            "car": ["car", "automobile", "vehicle", "engine", "driving"],
            "medical": ["medical", "health", "doctor", "patient", "disease"],
            "legal": ["legal", "law", "court", "contract", "rights"],
            "finance": ["finance", "money", "bank", "investment", "stock"],
            "technical": ["technical", "technology", "software", "programming"]
        }
        
        predictions = []
        for query in test_queries:
            query_lower = query.lower()
            scores = {}
            
            for domain, keywords in domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                scores[domain] = score
            
            # Predict domain with highest keyword match
            if scores:
                predicted_domain = max(scores, key=scores.get)
            else:
                predicted_domain = "sleep"  # Default fallback
            
            predictions.append(predicted_domain)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "method": "keyword_routing"
        }
    
    def _evaluate_rule_based_routing_baseline(self, test_queries: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Evaluate rule-based routing baseline."""
        # Simple rule-based routing based on question words
        predictions = []
        
        for query in test_queries:
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
                # Default to sleep domain for general questions
                predicted_domain = "sleep"
            elif any(word in query_lower for word in ["who", "which", "whom"]):
                # Default to car domain for specific entity questions
                predicted_domain = "car"
            else:
                # Default fallback
                predicted_domain = "sleep"
            
            predictions.append(predicted_domain)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "method": "rule_based_routing"
        }
    
    def _evaluate_random_routing_baseline(self, test_queries: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Evaluate random routing baseline."""
        unique_labels = list(set(true_labels))
        predictions = np.random.choice(unique_labels, size=len(test_queries)).tolist()
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "method": "random_routing"
        }
    
    def perform_statistical_analysis(
        self, 
        quality_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        self.logger.info("Performing statistical analysis...")
        
        statistical_results = {}
        
        # Extract main metrics for comparison
        main_metrics = ["overall_accuracy", "weighted_f1", "semantic_similarity"]
        
        for metric in main_metrics:
            if metric in quality_results:
                main_value = quality_results[metric]
                baseline_values = []
                baseline_names = []
                
                for baseline_name, baseline_result in baseline_results.items():
                    if isinstance(baseline_result, dict) and metric in baseline_result:
                        baseline_values.append(baseline_result[metric])
                        baseline_names.append(baseline_name)
                
                if baseline_values:
                    statistical_results[metric] = {
                        "main_value": main_value,
                        "baseline_comparisons": {}
                    }
                    
                    for baseline_name, baseline_value in zip(baseline_names, baseline_values):
                        # Effect size (Cohen's d approximation)
                        effect_size = (main_value - baseline_value) / np.sqrt(
                            (np.var([main_value]) + np.var([baseline_value])) / 2
                        )
                        
                        statistical_results[metric]["baseline_comparisons"][baseline_name] = {
                            "baseline_value": baseline_value,
                            "effect_size": effect_size,
                            "improvement": main_value - baseline_value,
                            "improvement_percentage": ((main_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                        }
        
        return statistical_results
    
    def perform_error_analysis(
        self, 
        test_queries: List[str], 
        true_labels: List[str],
        inference_pipeline
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        self.logger.info("Performing error analysis...")
        
        error_cases = []
        domain_errors = {}
        
        for i, (query, true_label) in enumerate(zip(test_queries, true_labels)):
            try:
                result = inference_pipeline.inference(query)
                predicted_domain = result["domain"]
                confidence = result["confidence"]
                response = result["answer"]
                
                # Check for errors
                is_routing_error = predicted_domain != true_label
                is_quality_error = self._is_poor_quality_response(response)
                is_low_confidence = confidence < self.config.ROUTER_CONFIDENCE_THRESHOLD
                
                if is_routing_error or is_quality_error or is_low_confidence:
                    error_case = {
                        "query_id": i,
                        "query": query,
                        "true_label": true_label,
                        "predicted_label": predicted_domain,
                        "confidence": confidence,
                        "response": response,
                        "error_types": {
                            "routing_error": is_routing_error,
                            "quality_error": is_quality_error,
                            "low_confidence": is_low_confidence
                        }
                    }
                    error_cases.append(error_case)
                    
                    # Track domain-specific errors
                    if true_label not in domain_errors:
                        domain_errors[true_label] = []
                    domain_errors[true_label].append(error_case)
                    
            except Exception as e:
                error_cases.append({
                    "query_id": i,
                    "query": query,
                    "true_label": true_label,
                    "error": str(e),
                    "error_types": {"system_error": True}
                })
        
        return {
            "total_errors": len(error_cases),
            "error_rate": len(error_cases) / len(test_queries) if test_queries else 0,
            "error_cases": error_cases,
            "domain_errors": domain_errors,
            "error_distribution": {
                "routing_errors": sum(1 for case in error_cases if case.get("error_types", {}).get("routing_error", False)),
                "quality_errors": sum(1 for case in error_cases if case.get("error_types", {}).get("quality_error", False)),
                "low_confidence": sum(1 for case in error_cases if case.get("error_types", {}).get("low_confidence", False)),
                "system_errors": sum(1 for case in error_cases if case.get("error_types", {}).get("system_error", False))
            }
        }
    
    def _is_poor_quality_response(self, response: str) -> bool:
        """Check if response is poor quality."""
        if not response or len(response.strip()) < 10:
            return True
        
        poor_indicators = [
            "I don't know", "I can't answer", "I'm not sure",
            "I don't have information", "I cannot provide",
            "I'm unable to", "I don't have enough information"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in poor_indicators)
    
    def analyze_domain_performance(
        self, 
        test_queries: List[str], 
        true_labels: List[str],
        inference_pipeline
    ) -> Dict[str, Any]:
        """Analyze performance across different domains."""
        self.logger.info("Analyzing domain-specific performance...")
        
        domain_results = {}
        unique_domains = list(set(true_labels))
        
        for domain in unique_domains:
            domain_queries = [query for query, label in zip(test_queries, true_labels) if label == domain]
            domain_responses = []
            
            for query in domain_queries:
                result = inference_pipeline.inference(query)
                domain_responses.append(result["answer"])
            
            # Calculate domain-specific metrics
            domain_metrics = {
                "num_queries": len(domain_queries),
                "avg_response_length": np.mean([len(response.split()) for response in domain_responses]),
                "avg_semantic_similarity": self._calculate_semantic_similarity(domain_responses, domain_queries)["semantic_similarity"],
                "domain_consistency": self._calculate_domain_consistency(domain_responses, [domain] * len(domain_responses))["domain_consistency"]
            }
            
            domain_results[domain] = domain_metrics
        
        return domain_results
    
    def _calculate_overall_score(
        self, 
        routing_results: Dict[str, Any], 
        quality_results: Dict[str, Any], 
        timing_results: Dict[str, Any]
    ) -> float:
        """Calculate overall system score."""
        # Weighted combination of different metrics
        routing_score = routing_results.get("overall_accuracy", 0) * 100
        quality_score = quality_results.get("semantic_similarity", 0) * 100
        timing_score = max(0, 100 - timing_results.get("avg_response_time", 10) * 10)  # Penalize slow responses
        
        # Weighted average
        overall_score = (routing_score * 0.4 + quality_score * 0.4 + timing_score * 0.2)
        return round(overall_score, 2)
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive evaluation results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ACADEMIC EVALUATION RESULTS")
        print("="*80)
        
        print(f"Overall System Score: {results['overall_score']}/100")
        print(f"Evaluation Time: {results['evaluation_metadata']['evaluation_time']:.2f} seconds")
        print(f"Total Queries: {results['evaluation_metadata']['total_queries']}")
        
        # Routing Results
        routing = results['routing']
        print(f"\nRouting Accuracy: {routing['overall_accuracy']:.3f}")
        print(f"Routing F1 Score: {routing['weighted_f1']:.3f}")
        print(f"Avg Routing Time: {routing['timing_stats']['mean']:.3f}s")
        
        # Quality Results
        quality = results['quality']
        print(f"\nSemantic Similarity: {quality.get('semantic_similarity', 0):.3f}")
        print(f"Avg Response Length: {quality.get('avg_response_length', 0):.1f} words")
        print(f"Avg Response Time: {quality.get('avg_response_time', 0):.2f}s")
        
        # Error Analysis
        errors = results['errors']
        print(f"\nError Rate: {errors['error_rate']:.3f}")
        print(f"Total Errors: {errors['total_errors']}")
        
        # Baseline Comparisons
        if results['baselines']:
            print(f"\nBaseline Comparisons:")
            for baseline_name, baseline_result in results['baselines'].items():
                if 'accuracy' in baseline_result:
                    print(f"  {baseline_name}: {baseline_result['accuracy']:.3f}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        output_path = Path(self.config.EVAL_OUTPUT_DIR) / f"academic_evaluation_results_{int(time.time())}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to: {output_path}")
    
    def generate_publication_figures(self, results: Dict[str, Any]):
        """Generate publication-ready figures."""
        self.logger.info("Generating publication figures...")
        
        # Create figures directory
        figures_dir = Path(self.config.EVAL_OUTPUT_DIR) / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Routing Accuracy Comparison
        self._plot_routing_accuracy_comparison(results, figures_dir)
        
        # 2. Domain Performance Heatmap
        self._plot_domain_performance_heatmap(results, figures_dir)
        
        # 3. Response Quality Distribution
        self._plot_response_quality_distribution(results, figures_dir)
        
        # 4. Timing Performance
        self._plot_timing_performance(results, figures_dir)
        
        self.logger.info(f"Figures saved to: {figures_dir}")
    
    def _plot_routing_accuracy_comparison(self, results: Dict[str, Any], figures_dir: Path):
        """Plot routing accuracy comparison with baselines."""
        plt.figure(figsize=(10, 6))
        
        # Main system accuracy
        main_accuracy = results['routing']['overall_accuracy']
        
        # Baseline accuracies
        baseline_names = []
        baseline_accuracies = []
        
        for baseline_name, baseline_result in results['baselines'].items():
            if 'accuracy' in baseline_result:
                baseline_names.append(baseline_name.replace('_', ' ').title())
                baseline_accuracies.append(baseline_result['accuracy'])
        
        # Plot
        x_pos = range(len(baseline_names) + 1)
        accuracies = [main_accuracy] + baseline_accuracies
        labels = ['Our Method'] + baseline_names
        
        bars = plt.bar(x_pos, accuracies, color=['#2E86AB'] + ['#A23B72'] * len(baseline_names))
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Method')
        plt.ylabel('Routing Accuracy')
        plt.title('Routing Accuracy Comparison')
        plt.xticks(x_pos, labels, rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'routing_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_performance_heatmap(self, results: Dict[str, Any], figures_dir: Path):
        """Plot domain performance heatmap."""
        domain_results = results['domains']
        
        if not domain_results:
            return
        
        # Extract metrics for heatmap
        domains = list(domain_results.keys())
        metrics = ['num_queries', 'avg_response_length', 'avg_semantic_similarity', 'domain_consistency']
        
        # Create data matrix
        data_matrix = []
        for domain in domains:
            row = []
            for metric in metrics:
                value = domain_results[domain].get(metric, 0)
                row.append(value)
            data_matrix.append(row)
        
        # Normalize data for better visualization
        data_matrix = np.array(data_matrix)
        data_matrix_norm = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_matrix_norm, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=domains,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd')
        
        plt.title('Domain Performance Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Domains')
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'domain_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_response_quality_distribution(self, results: Dict[str, Any], figures_dir: Path):
        """Plot response quality distribution."""
        quality = results['quality']
        
        # Extract quality metrics
        metrics = ['semantic_similarity', 'avg_response_length', 'distinct_1', 'distinct_2']
        values = [quality.get(metric, 0) for metric in metrics]
        labels = [metric.replace('_', ' ').title() for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.ylabel('Score')
        plt.title('Response Quality Metrics')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'response_quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timing_performance(self, results: Dict[str, Any], figures_dir: Path):
        """Plot timing performance."""
        timing = results['timing']
        
        # Extract timing metrics
        metrics = ['avg_response_time', 'min_response_time', 'max_response_time', 'median_response_time']
        values = [timing.get(metric, 0) for metric in metrics]
        labels = [metric.replace('_', ' ').title() for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}s', ha='center', va='bottom')
        
        plt.ylabel('Time (seconds)')
        plt.title('Timing Performance Metrics')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'timing_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
