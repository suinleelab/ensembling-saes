import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from sae_lens import SAE
from sae_lens.config import LanguageModelSAERunnerConfig
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


class Bagging(BaseEnsembling):
    """
    A PyTorch module for concatenating multiple SAE models and calculating feature overlap.
    This class does not require training and is used as a naive baseline for evaluating feature overlap.
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
        if train_saes is None:
            raise ValueError("Please pass in the train SAEs.")
        
        # Naive concatentation baseline does not require training
        self.ensembled_saes = train_saes
        self.num_train_saes = num_train_saes
        self.W_dec = torch.cat([sae.W_dec for sae in train_saes], dim=0)
        self.dtype = train_saes[0].dtype
    
    def load(self, num_train_saes: int = None, train_saes: List[SAE] = None, train_seed: int = 42) -> None:
        if train_saes is None:
            raise ValueError("Please pass in the train SAEs.")
        
        # Naive concatentation baseline does not require training
        self.ensembled_saes = train_saes
        self.num_train_saes = num_train_saes
        self.W_dec = torch.cat([sae.W_dec for sae in train_saes], dim=0)
        self.dtype = train_saes[0].dtype

    def encode(self, x):
        hidden_acts = []
        for sae in self.ensembled_saes:
            hidden_acts.append(sae.encode(x))
        return torch.cat(hidden_acts, dim=-1) / self.num_train_saes
    
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
        sae_out_per_sae = []

        # Loop over all train SAEs
        for sae in self.ensembled_saes:
            # Calculate metrics
            with torch.no_grad():
                x = x.to(self.device)
                hidden_acts = sae.encode(x)
                sae_out = sae.decode(hidden_acts)

                l0 = (hidden_acts > 0).sum(dim=-1).float()
                l0_per_sae.append(l0)
                sae_out_per_sae.append(sae_out)
        
        return l0_per_sae, sae_out_per_sae
    
    def eval_stability(self, num_train_saes: int, train_seeds: int) -> float:
        self.logger.info("Loading ensembled SAE for each seed")
        ensemble_W_decs = []
        start_train_seed = 1
        curr_train_ensemble = 0

        while curr_train_ensemble < len(train_seeds):
            all_W_decs = []
            for index in range(start_train_seed, start_train_seed + num_train_saes):
                l1_coefficient = None
                k = None
                
                if self.lm_cfg.architecture == 'topk':
                    k = self.lm_cfg.activation_fn_kwargs['k']
                else:
                    l1_coefficient = self.lm_cfg.l1_coefficient

                run_name = get_run_name(
                                seed=index, 
                                expansion_factor=self.lm_cfg.expansion_factor, 
                                lr=self.lm_cfg.lr, 
                                l1_coefficient=l1_coefficient, 
                                training_tokens=self.lm_cfg.training_tokens,
                                k=k
                            )
                checkpoint_path =  get_checkpoint_path(self.lm_cfg.wandb_project, run_name)

                if os.path.exists(checkpoint_path):
                    checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(checkpoint_path)))[0]
                    loading_path = os.path.join(checkpoint_path, checkpoint_dir_name)
                    
                    sae = SAE.load_from_pretrained(loading_path, device=self.device)
                    W_dec = F.normalize(sae.W_dec, dim=1)
                        
                    all_W_decs.append(W_dec.detach().cpu())
                    del sae
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"Checkpoint path {checkpoint_path} does not exist. Please check the path.")
            
            start_train_seed += num_train_saes
            curr_train_ensemble += 1
            
            ensemble_W_decs.append(torch.cat(all_W_decs, dim=0))
    
        self.logger.info("Calculating stability")

        return stability_relaxed(ensemble_W_decs)

    def eval_recall(self,
        alpha: float = 0.6
    ) -> int:
        if self.ensembled_saes is None:
            raise ValueError("Train SAEs have not been set. Please set train SAEs using the fit method.")

        if self.test_saes is None:
            raise ValueError("Test SAEs have not been set. Please set test SAEs using the constructor.")
        
        return recall(alpha, self.test_saes, self.ensembled_saes)
    

    def eval_recon(self, total_test_tokens: int) -> Dict[str, float]:
        """
        Calculate L0 and MSE metrics for the bagging baseline
        """
        if self.ensembled_saes is None:
            raise ValueError("Train SAEs have not been set. Please set train SAEs using the fit method.")

        # Setup activation store
        activations_store = EnsemblingSAEsActivationsStore(self.model, self.lm_cfg, override_dataset=None)

        return get_sparsity_and_recons_metrics(activations_store, total_test_tokens, self.forward)
    
    def eval_connectivity(self, total_test_tokens: int=7_000_000) -> float:
        """
        Calculate L0 and MSE metrics for the bagging baseline
        """
        if self.ensembled_saes is None:
            raise ValueError("Train SAEs have not been set. Please set train SAEs using the fit method.")

        # Setup activation store
        activations_store = EnsemblingSAEsActivationsStore(self.model, self.lm_cfg, override_dataset=None)

        return connectivity(activations_store, total_test_tokens, self.encode)
    
    def get_mse_loss(self, total_test_tokens: int, mask: torch.Tensor) -> Dict[str, float]:
        """
        Calculate L0 and MSE metrics for the bagging baseline
        """
        if self.ensembled_saes is None:
            raise ValueError("Train SAEs have not been set. Please set train SAEs using the fit method.")

        # Setup activation store
        activations_store = EnsemblingSAEsActivationsStore(self.model, self.lm_cfg, override_dataset=None)
        tokens = 0
        mse_loss = None

        while tokens < total_test_tokens:
            model_acts = activations_store.next_batch()[:, 0, :].to(activations_store.device)
            l0, sae_out = self.forward(model_acts, mask=mask)

            sae_out = torch.stack(sae_out).mean(dim=0)

            per_item_mse_loss = torch.nn.functional.mse_loss(sae_out, model_acts, reduction="none")
            if mse_loss is None:
                mse_loss = per_item_mse_loss.sum(dim=-1)
            else:
                mse_loss = torch.cat([mse_loss, per_item_mse_loss.sum(dim=-1)], dim=0)
            
            tokens += activations_store.train_batch_size_tokens

        return mse_loss.mean()
    
    def eval_diversity(self, threshold: float = 0.7) -> float:
        all_W_decs = []

        with torch.no_grad():
            for sae in self.ensembled_saes:
                sae.W_dec.data /= (torch.norm(sae.W_dec.data, dim=1, keepdim=True) + 1e-6)
                all_W_decs.append(sae.W_dec.detach())
            
            all_W_decs = torch.cat(all_W_decs, dim=0)

        return diversity(all_W_decs, threshold=threshold)
