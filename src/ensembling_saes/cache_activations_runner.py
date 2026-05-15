from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig
from transformer_lens.hook_points import HookedRootModule

from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore


def _mk_activations_store_with_split(
    model: HookedRootModule,
    cfg: CacheActivationsRunnerConfig,
    split: str = "train"
) -> EnsemblingSAEsActivationsStore:
    """
    Internal method used in CacheActivationsRunnerWithSplit. Used to create a cached dataset
    from an EnsemblingSAEsActivationsStore with the functionality to support splits.
    """
    return EnsemblingSAEsActivationsStore(
        model=model, 
        cfg=cfg,
        split=split
    )

class CacheActivationsRunnerWithSplit(CacheActivationsRunner):
    """"
    Custom activation runner class to support different splits (train/val/test)
    """
    def __init__(
        self,
        cfg: CacheActivationsRunnerConfig,
        split: str = "train"
    ):

        super().__init__(cfg=cfg)

        self.activations_store = _mk_activations_store_with_split(
            self.model,
            self.cfg,
            split=split
        )
