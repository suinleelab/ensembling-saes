"""
Base class for ensembling methods.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from sae_lens import SAE
from sae_lens.config import LanguageModelSAERunnerConfig
from torch import nn
from transformer_lens import HookedTransformer


class BaseEnsembling(nn.Module, ABC):
    """
    Base class for ensembling methods.
    """
    def __init__(self, 
        logger: logging.Logger,
        lm_cfg: LanguageModelSAERunnerConfig,
        model: HookedTransformer,
        test_saes: List[SAE] = None,
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.logger = logger
        self.test_saes = test_saes
        self.device = device
        self.model = model
        self.lm_cfg = lm_cfg

    @abstractmethod
    def fit(self, num_train_saes: int=None, train_saes: List[SAE]=None) -> None:
        """
        Fit the ensembling method to the training SAEs.
        """
        pass
    
    @abstractmethod
    def eval_stability(self, num_train_saes: int, train_seeds: int) -> float:
        """
        Calculate stability for the ensembling method.
        """
        pass

    @abstractmethod
    def load(self, num_train_saes: int=None, train_saes: List[SAE]=None, train_seed: int=None) -> None:
        """
        Load the ensembling method with the training SAEs.
        """
        pass
    
    @abstractmethod
    def eval_recall(self, alpha: float) -> float:
        """
        Calculate recall for the ensembling method.
        """
        pass

    @abstractmethod
    def eval_recon(self, total_test_tokens: int) -> Dict[str, float]:
        """
        Evaluate the ensembling method for l0, mse, and explained variance
        """
        pass
    
    @abstractmethod
    def eval_connectivity(self, total_test_tokens: int=100_000) -> float:
        """
        Evaluate the connectivity of the activations in the ensembling method.
        """
        pass

    @abstractmethod
    def eval_diversity(self, threshold: float = 0.7) -> float:
        """
        Evaluate the number of features which are diverse across the ensemble in terms of the cosine similarity
        """
        pass
