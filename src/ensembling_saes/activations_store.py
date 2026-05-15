import warnings
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.training.activations_store import ActivationsStore
from torch.utils.data import DataLoader
from transformer_lens.hook_points import HookedRootModule


class EnsemblingSAEsActivationsStore(ActivationsStore):
    """"
    Custom activation store class for loading buffer without shuffling and supporting splits
    """
    def __init__(
        self, 
        model: HookedRootModule, 
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig, 
        override_dataset: HfDataset | None=None,
        split: str='train'
    ):
        # Different variable values need to be set based on the config type
        if isinstance(cfg, LanguageModelSAERunnerConfig):
            cached_activations_path = cfg.cached_activations_path

            # set cached_activations_path to None if we're not using cached activations
            if not cfg.use_cached_activations:
                cached_activations_path = None
            
            device = torch.device(cfg.act_store_device)
            hook_head_index = cfg.hook_head_index
            store_batch_size_prompts = cfg.store_batch_size_prompts
            train_batch_size_tokens = cfg.train_batch_size_tokens
            normalize_activations = cfg.normalize_activations 

        if isinstance(cfg, CacheActivationsRunnerConfig):
            device = torch.device("cpu")
            hook_head_index = None
            store_batch_size_prompts = cfg.model_batch_size
            train_batch_size_tokens = -1
            normalize_activations = "none"
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )
        
        super().__init__(model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=normalize_activations,
            device=device,
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice)

        dataset = cfg.dataset_path
        if isinstance(dataset, str):
            # if the split is val or test, we need to specify the corresponding file name to download
            data_files = None
            if split != 'train':
                data_files = f"{split}.jsonl.zst"

            # split argument is only used when split is train
            self.dataset = load_dataset(
                        dataset,
                        split='train',
                        streaming=cfg.streaming,
                        data_files=data_files,
                        trust_remote_code=cfg.dataset_trust_remote_code,  # type: ignore
                    )
        else:
            self.dataset = dataset


    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader (without shuffling) which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.train_batch_size_tokens

        try:
            new_samples = self.get_buffer(self.half_buffer_size, raise_on_epoch_end=True)
        except StopIteration:
            warnings.warn(
                "All samples in the training dataset have been exhausted, we are now beginning a new epoch with the same samples."
            )
            self._storage_buffer = (
                None  # dump the current buffer so samples do not leak between epochs
            )
            try:
                new_samples = self.get_buffer(self.half_buffer_size)
            except StopIteration:
                raise ValueError(
                    "We were unable to fill up the buffer directly after starting a new epoch. \
                    This could indicate that there are less samples in the dataset than are required to fill up the buffer. \
                    Consider reducing batch_size or n_batches_in_buffer. "
                )

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [new_samples, self.storage_buffer],
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        return iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=False,
            )
        )
