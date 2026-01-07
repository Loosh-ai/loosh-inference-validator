from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

from validator.utils.device import get_optimal_device
from validator.scripts.download_models import download_models

class ModelManager:
    """Manages the loading and caching of  models."""
    #TODO: eventually we'll have models in huggingface for EEG analysis. We'll use this class to manage them
    def __init__(self, device: Optional[str] = None):
        self.device = get_optimal_device(device)
        self.models: Dict[str, Any] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Define model paths
        self.model_paths:dict[str,Path] = {
             "brainwave": self.data_dir / "brainwave-detection.pt"
        }
        
        # Check if models exist, download if missing
        self._ensure_models_exist()
    
    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items() 
            if not path.exists()
        ]
        
        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()
    
    def load_model(self, model_name: str):
        """
        Load a model by name, using cache if available.
        
        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
            
        Returns:
            MODEL: The loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_paths[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please ensure all required models are downloaded."
            )
        
        logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
        model = None
        #MODEL(str(model_path)).to(device=self.device)
        self.models[model_name] = model
        return model
    
    def load_all_models(self) -> None:
        """Load all models into cache."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name)
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name)
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.models.clear() 