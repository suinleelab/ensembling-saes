import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate.utils import set_seed
from sae_lens import SAE
from sae_lens.config import LanguageModelSAERunnerConfig
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore
from ensembling_saes.ensembling.base import BaseEnsembling
from ensembling_saes.evals import (
    connectivity,
    diversity,
    get_sparsity_and_recons_metrics,
    recall,
    stability_relaxed,
)
from ensembling_saes.paths import get_checkpoint_path, get_run_name
from ensembling_saes.sae_training_runner import (
    BoostedSAETrainingRunner,
)


class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result
    
def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    if activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            return torch.tanh(input)

        return tanh_relu
    if activation_fn == "topk":
        if "k" not in kwargs:
            raise ValueError("TopK activation function requires a k value.")
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided

        return TopK(k, postact_fn)
    raise ValueError(f"Unknown activation function: {activation_fn}")


class Boosting(BaseEnsembling):
    """
    A PyTorch module for ensembling multiple SAE models by boosting.
    """
    def __init__(self,
        logger: logging.Logger,
        lm_cfg: LanguageModelSAERunnerConfig,
        model: HookedTransformer,
        test_saes: List[SAE] = None,
        device: str = "cuda:0"
    ) -> None:
        super().__init__(logger=logger,
            lm_cfg=lm_cfg,
            model=model,
            test_saes=test_saes,
            device=device
        )
        self.ensembled_saes = None
    
    def fit(self, num_train_saes: int = None, train_saes: List[SAE] = None) -> None:
        if num_train_saes is None:
            raise ValueError("Please pass in the number of train SAEs.")
        
        self.ensembled_saes = []
        orig_run_name = self.lm_cfg.run_name
        set_seed(self.lm_cfg.seed)

        for i in range(num_train_saes):
            self.lm_cfg.run_name = f"{orig_run_name}_ENSEMBLE-boosting_INDEX-{i}"
            self.lm_cfg.checkpoint_path = get_checkpoint_path(self.lm_cfg.wandb_project, self.lm_cfg.run_name)

            # If the checkpoint already exists, load the checkpoint
            if os.path.exists(self.lm_cfg.checkpoint_path):
                checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(self.lm_cfg.checkpoint_path)))[0]
                loading_path = os.path.join(self.lm_cfg.checkpoint_path, checkpoint_dir_name)
                child_sae = SAE.load_from_pretrained(loading_path, device=self.device)
            # if the checkpoint does not exist, train a new boosted SAE
            else:
                training_runner = BoostedSAETrainingRunner(cfg=self.lm_cfg, split='train', parent_saes=self.ensembled_saes)
                child_sae = training_runner.run()

            self.ensembled_saes.append(child_sae)
    
    def load(
        self, 
        num_train_saes: int = None, 
        train_saes: List[SAE] = None, 
        train_seed: int = 42
    ) -> None:
        self.ensembled_saes : List[SAE] = []
        if self.lm_cfg.architecture == 'topk':
            k = self.lm_cfg.activation_fn_kwargs['k']
            l1_coefficient = None
        else:
            l1_coefficient = self.lm_cfg.l1_coefficient
            k = None
            
        orig_run_name = get_run_name(
                        seed=train_seed, 
                        expansion_factor=self.lm_cfg.expansion_factor, 
                        lr=self.lm_cfg.lr, 
                        l1_coefficient=l1_coefficient, 
                        training_tokens=self.lm_cfg.training_tokens,
                        k=k
                    )
        
        for i in range(num_train_saes):
            run_name = f"{orig_run_name}_ENSEMBLE-boosting_INDEX-{i}"
            checkpoint_path = get_checkpoint_path(self.lm_cfg.wandb_project, run_name)

            if os.path.exists(checkpoint_path):
                checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(checkpoint_path)))[0]
                loading_path = os.path.join(checkpoint_path, checkpoint_dir_name)

                trained_sae = SAE.load_from_pretrained(loading_path, device=self.device)
                self.ensembled_saes.append(trained_sae)
            else:
                raise ValueError(f"Checkpoint path {checkpoint_path} does not exist. Please check the path.")
        self.num_train_saes = num_train_saes
        self.W_dec = torch.cat([sae.W_dec for sae in self.ensembled_saes], dim=0)
        self.dtype = self.ensembled_saes[0].dtype
    
    def encode(self, x):
        hidden_acts_list = []
        sae_out = torch.zeros_like(x, device=self.device)
        for sae in self.ensembled_saes:
            hidden_acts = sae.encode(x - sae_out)
            hidden_acts_list.append(hidden_acts)
            sae_out += sae.decode(hidden_acts)
        return torch.cat(hidden_acts_list, dim=-1)

    def decode(self, feature_acts):
        feature_acts_per_sae = torch.chunk(
            feature_acts,
            self.num_train_saes,
            dim=-1,
        )
        out = self.ensembled_saes[0].decode(feature_acts_per_sae[0])
        for i in range(1, self.num_train_saes):
            out += self.ensembled_saes[i].decode(feature_acts_per_sae[i])
        return out

    def forward(self, x: torch.tensor) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        l0_per_sae = []
        with torch.no_grad():
            x = x.to(self.device)
            sae_out = torch.zeros_like(x, device=self.device)
            for sae in self.ensembled_saes:
                hidden_acts = sae.encode(x - sae_out)
                sae_out += sae.decode(hidden_acts)

                l0 = (hidden_acts > 0).sum(dim=-1).float()
                l0_per_sae.append(l0)
        
        return l0_per_sae, sae_out
    
    def eval_stability(self, num_train_saes: int, train_seeds: int) -> float:
        self.logger.info("Loading ensembled SAE for each seed")
        ensemble_W_decs = []

        for train_seed in tqdm(train_seeds):
            all_W_decs = []
            for boosted_index in range(num_train_saes):
                l1_coefficient = None
                k = None
                
                if self.lm_cfg.architecture == 'topk':
                    k = self.lm_cfg.activation_fn_kwargs['k']
                else:
                    l1_coefficient = self.lm_cfg.l1_coefficient

                run_name = get_run_name(
                                seed=train_seed, 
                                expansion_factor=self.lm_cfg.expansion_factor, 
                                lr=self.lm_cfg.lr, 
                                l1_coefficient=l1_coefficient, 
                                training_tokens=self.lm_cfg.training_tokens,
                                k=k
                            )

                run_name += f"_ENSEMBLE-boosting_INDEX-{boosted_index}"
                checkpoint_path =  get_checkpoint_path(self.lm_cfg.wandb_project, run_name)

                if os.path.exists(checkpoint_path):
                    checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(checkpoint_path)))[0]
                    loading_path = os.path.join(checkpoint_path, checkpoint_dir_name)

                    sae = SAE.load_from_pretrained(loading_path, device=self.device)
                    W_dec = sae.W_dec
                        
                    all_W_decs.append(W_dec.detach().cpu())
                    del sae
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"Checkpoint path {checkpoint_path} does not exist. Please check the path.")
            
            ensemble_W_decs.append(torch.cat(all_W_decs, dim=0))
        
        self.logger.info("Calculating stability")

        return stability_relaxed(ensemble_W_decs)

    def eval_recall(self,
        alpha: float = 0.6
    ) -> float:
        if self.ensembled_saes is None:
            raise ValueError("Boosted SAEs have not been set. Please set boosted SAEs using the fit method.")

        if self.test_saes is None:
            raise ValueError("Test SAEs have not been set. Please set test SAEs using the constructor.")
        
        return recall(alpha, self.sae_type, self.test_saes, self.ensembled_saes)
    

    def eval_recon(self, total_test_tokens: int) -> Dict[str, float]:
        """
        Calculate L0, MSE, and explained variance metrics
        """
        # Setup activation store
        activations_store = EnsemblingSAEsActivationsStore(self.model, self.lm_cfg, override_dataset=None)

        # Get metrics
        return get_sparsity_and_recons_metrics(activations_store, total_test_tokens, self.forward)
    
    def eval_diversity(self, threshold:float = 0.7) -> float:
        all_W_decs = []
        with torch.no_grad():
            for sae in self.ensembled_saes:
                # sae.W_dec.data /= (torch.norm(sae.W_dec.data, dim=1, keepdim=True) + 1e-6)
                all_W_decs.append(sae.W_dec.detach())
            
            all_W_decs = torch.cat(all_W_decs, dim=0)

        return diversity(all_W_decs, threshold=threshold)
    
        
    def eval_connectivity(self, total_test_tokens: int=7_000_000) -> float:
        """
        Calculate the connectivity of the ensemble
        """
        # Setup activation store
        activations_store = EnsemblingSAEsActivationsStore(self.model, self.lm_cfg, override_dataset=None)

        # Get metrics
        return connectivity(activations_store, total_test_tokens, self.encode)
