import pickle
from pathlib import Path
from typing import Optional, Protocol

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule


class SaveCheckpointFn(Protocol):
    def __call__(
        self,
        trainer: "SAETrainer",
        checkpoint_name: str,
        wandb_aliases: Optional[list[str]] = None,
    ) -> None: ...

class EnsemblingSAETrainer(SAETrainer):
    def __init__(
        self,
        model: HookedRootModule,
        sae: TrainingSAE,
        activation_store: ActivationsStore,
        save_checkpoint_fn: SaveCheckpointFn,
        cfg: LanguageModelSAERunnerConfig,
    ) -> None:
        super().__init__(model=model,
            sae=sae,
            activation_store=activation_store,
            save_checkpoint_fn=save_checkpoint_fn,
            cfg=cfg)
        
    
    def fit(self) -> TrainingSAE:
        pbar = tqdm(total=(self.cfg.total_training_tokens), desc="Training SAE")

        self.activations_store.set_norm_scaling_factor_if_needed()

        # Calculate number of tokens for which each feature was fired
        all_firing_freq = []

        # Train loop
        while self.n_training_tokens < (self.cfg.total_training_tokens):
            # Do a training step.
            layer_acts = self.activations_store.next_batch()[:, 0, :].to(
                self.sae.device
            )
                
            self.n_training_tokens += self.cfg.train_batch_size_tokens
            step_output = self._train_step(sae=self.sae, sae_in=layer_acts)
            firing_freq = (step_output.feature_acts > self.cfg.dead_feature_threshold).sum(dim=0)

            if len(all_firing_freq) == 0:
                all_firing_freq = firing_freq
            else:
                all_firing_freq += firing_freq

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            self._begin_finetuning_if_needed()

        # fold the estimated norm scaling factor into the sae weights
        if self.activations_store.estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activations_store.estimated_norm_scaling_factor
            )
            self.activations_store.estimated_norm_scaling_factor = None

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        ) 

        base_path = Path(self.cfg.checkpoint_path) / f"final_{self.n_training_tokens}"

        with open(str(base_path / "firing_freq.pkl"), "wb") as f:
            pickle.dump(all_firing_freq, f)

        pbar.close()
        return self.sae
