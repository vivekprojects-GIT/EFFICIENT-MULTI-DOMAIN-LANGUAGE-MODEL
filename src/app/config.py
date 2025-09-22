from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Dict, Any, Optional
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )

    # ==================== DOMAIN CONFIGURATION ====================
    SUPPORTED_DOMAINS: List[str] = Field(
        default=["sleep", "car", "medical", "legal", "finance", "technical"],
        description="List of supported domains for evaluation"
    )
    
    # DATA PATHS
    SLEEP_DATA_PATH: str = Field(
        default="data/training_qna_sleep.json",
        description="Path to sleep domain training data"
    )
    CAR_DATA_PATH: str = Field(
        default="data/training_qna_car.json", 
        description="Path to car domain training data"
    )
    MEDICAL_DATA_PATH: str = Field(
        default="data/training_qna_medical.json",
        description="Path to medical domain training data"
    )
    LEGAL_DATA_PATH: str = Field(
        default="data/training_qna_legal.json",
        description="Path to legal domain training data"
    )
    FINANCE_DATA_PATH: str = Field(
        default="data/training_qna_finance.json",
        description="Path to finance domain training data"
    )
    TECHNICAL_DATA_PATH: str = Field(
        default="data/training_qna_technical.json",
        description="Path to technical domain training data"
    )

    # ==================== MODEL CONFIGURATION ====================
    GEN_BASE_MODEL: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Base model for fine-tuning"
    )
    
    # LoRA Configuration
    LORA_RANK: int = Field(default=16, description="LoRA rank parameter")
    LORA_ALPHA: int = Field(default=32, description="LoRA alpha scaling")
    LORA_DROPOUT: float = Field(default=0.1, description="LoRA dropout rate")
    LORA_TARGET_MODULES: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA adaptation"
    )
    
    # ==================== EVALUATION CONFIGURATION ====================
    # Standard Benchmarks
    BENCHMARK_DATASETS: List[str] = Field(
        default=["glue", "superglue", "domain_specific"],
        description="Benchmark datasets to evaluate on"
    )
    
    # Evaluation Metrics
    AUTOMATIC_METRICS: List[str] = Field(
        default=["bleu", "rouge", "bert_score", "meteor", "chr_f", "perplexity"],
        description="Automatic evaluation metrics"
    )
    
    SEMANTIC_METRICS: List[str] = Field(
        default=["cosine_similarity", "semantic_search_accuracy", "domain_consistency"],
        description="Semantic evaluation metrics"
    )
    
    HUMAN_EVALUATION_ENABLED: bool = Field(default=False, description="Enable human evaluation")
    HUMAN_ANNOTATORS: int = Field(default=10, description="Number of human annotators")
    
    # Statistical Analysis
    STATISTICAL_TESTS: List[str] = Field(
        default=["paired_t_test", "wilcoxon_signed_rank", "anova"],
        description="Statistical tests to perform"
    )
    
    CONFIDENCE_LEVEL: float = Field(default=0.95, description="Confidence level for statistical tests")
    MULTIPLE_COMPARISON_CORRECTION: str = Field(default="bonferroni", description="Multiple comparison correction method")

    # ==================== BASELINE CONFIGURATION ====================
    BASELINE_METHODS: List[str] = Field(
        default=[
            "single_model", "multi_task_learning", "mixture_of_experts",
            "domain_adaptation", "retrieval_augmented", "keyword_routing",
            "rule_based_routing", "bert_classification"
        ],
        description="Baseline methods to compare against"
    )
    
    EXTERNAL_APIS: Dict[str, str] = Field(
        default={
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-2",
            "huggingface": "meta-llama/Llama-2-7b-chat-hf"
        },
        description="External API models for comparison"
    )

    # ==================== EXPERIMENTAL DESIGN ====================
    # Hardware Configurations
    HARDWARE_CONFIGS: List[str] = Field(
        default=["consumer", "enterprise"],
        description="Hardware configurations to test"
    )
    
    # Hyperparameter Grid
    HYPERPARAMETER_GRID: Dict[str, List[Any]] = Field(
        default={
            "lora_rank": [4, 8, 16, 32, 64],
            "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
            "batch_size": [1, 2, 4, 8],
            "epochs": [1, 3, 5, 10],
            "confidence_threshold": [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        description="Hyperparameter grid for experimentation"
    )
    
    # Cross-Validation
    CROSS_VALIDATION_FOLDS: int = Field(default=5, description="Number of CV folds")
    TRAIN_SPLIT: float = Field(default=0.8, description="Training data split")
    VAL_SPLIT: float = Field(default=0.1, description="Validation data split")
    TEST_SPLIT: float = Field(default=0.1, description="Test data split")

    # ==================== REPRODUCIBILITY ====================
    RANDOM_SEED: int = Field(default=42, description="Random seed for reproducibility")
    DETERMINISTIC_TRAINING: bool = Field(default=True, description="Enable deterministic training")
    MODEL_CHECKPOINTS: bool = Field(default=True, description="Save model checkpoints")
    EVALUATION_RESULTS: bool = Field(default=True, description="Save evaluation results")

    # ==================== API CONFIGURATION ====================
    API_HOST: str = Field(default="0.0.0.0", description="API host address")
    API_PORT: int = Field(default=8080, description="API port number")
    API_WORKERS: int = Field(default=4, description="Number of API workers")
    
    # LOGGING CONFIGURATION
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # SECURITY CONFIGURATION
    SECRET_KEY: str = Field(
        default="academic-research-key-change-in-production", 
        description="Secret key for API security"
    )
    ALLOWED_ORIGINS: str = Field(
        default="*", 
        description="Comma-separated list of allowed CORS origins"
    )
    
    # ROUTER CONFIGURATION
    ROUTER_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum confidence for high-confidence routing"
    )
    ROUTER_FALLBACK_THRESHOLD: float = Field(
        default=0.5,
        description="Minimum confidence for medium-confidence routing"
    )
    
    # INFERENCE CONFIGURATION
    DEFAULT_MAX_LENGTH: int = Field(
        default=256,
        description="Default maximum response length in tokens"
    )
    MAX_QUERY_LENGTH: int = Field(
        default=1000,
        description="Maximum allowed query length"
    )
    MIN_QUERY_LENGTH: int = Field(
        default=3,
        description="Minimum allowed query length"
    )
    
    # MODEL PATHS
    SLEEP_MODEL_PATH: str = Field(
        default="models/sleep_model",
        description="Path to sleep domain model"
    )
    CAR_MODEL_PATH: str = Field(
        default="models/car_model",
        description="Path to car domain model"
    )
    
    # TRAINING CONFIGURATION
    TRAINING_EPOCHS: int = Field(
        default=3,
        description="Number of training epochs"
    )
    TRAINING_LEARNING_RATE: float = Field(
        default=5e-5,
        description="Learning rate for training"
    )
    TRAINING_BATCH_SIZE: int = Field(
        default=2,
        description="Training batch size"
    )
    TRAINING_MAX_LENGTH: int = Field(
        default=128,
        description="Maximum sequence length for training"
    )
    TRAINING_LOGGING_STEPS: int = Field(
        default=10,
        description="Steps between logging during training"
    )
    TRAINING_SAVE_STEPS: int = Field(
        default=100,
        description="Steps between saving checkpoints"
    )
    TRAINING_EVAL_STEPS: int = Field(
        default=100,
        description="Steps between evaluation during training"
    )
    TRAINING_SAVE_TOTAL_LIMIT: int = Field(
        default=3,
        description="Maximum number of checkpoints to keep"
    )
    
    # EVALUATION CONFIGURATION
    EVAL_OUTPUT_DIR: str = Field(
        default="evaluation_results",
        description="Directory for evaluation results"
    )
    EVAL_RESULTS_FILE: str = Field(
        default="evaluation_results.json",
        description="Filename for evaluation results"
    )
    
    def get_allowed_origins(self) -> list:
        """Parse ALLOWED_ORIGINS string into a list."""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # ==================== VALIDATION ====================
    @validator('TEST_SPLIT', 'TRAIN_SPLIT', 'VAL_SPLIT')
    def validate_splits(cls, v, values):
        if 'TRAIN_SPLIT' in values and 'VAL_SPLIT' in values:
            total = values['TRAIN_SPLIT'] + values['VAL_SPLIT'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Splits must sum to 1.0, got {total}")
        return v

    @validator('SUPPORTED_DOMAINS')
    def validate_domains(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 domains required for multi-domain evaluation")
        return v

    # ==================== UTILITY METHODS ====================
    def get_domain_data_path(self, domain: str) -> str:
        """Get data path for a specific domain."""
        domain_paths = {
            "sleep": self.SLEEP_DATA_PATH,
            "car": self.CAR_DATA_PATH,
            "medical": self.MEDICAL_DATA_PATH,
            "legal": self.LEGAL_DATA_PATH,
            "finance": self.FINANCE_DATA_PATH,
            "technical": self.TECHNICAL_DATA_PATH
        }
        
        if domain.lower() not in domain_paths:
            raise ValueError(f"Domain {domain} not supported. Available: {list(domain_paths.keys())}")
        return domain_paths[domain.lower()]
    
    def get_domain_model_path(self, domain: str) -> str:
        """Get model path for a specific domain."""
        return f"models/{domain.lower()}_model"
    
    def get_allowed_origins_list(self) -> List[str]:
        """Get allowed CORS origins as a list."""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    def is_domain_supported(self, domain: str) -> bool:
        """Check if domain is supported."""
        return domain.lower() in [d.lower() for d in self.SUPPORTED_DOMAINS]

# Create global settings instance
settings = Settings()