from typing import Any, List, cast

from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig
from transformer_lens.hook_points import HookedRootModule

import wandb
from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore
from ensembling_saes.training.sae_trainer import EnsemblingSAETrainer
from ensembling_saes.training.training_boosted_sae import TrainingBoostedSAE


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()

class EnsemblingSAETrainingRunner(SAETrainingRunner):
    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE | None = None,
        split: str = 'train'
    ) -> None:
        super().__init__(cfg=cfg,
            override_dataset=override_dataset,
            override_model=override_model,
            override_sae=override_sae)

        self.activations_store = EnsemblingSAEsActivationsStore(self.model, 
                                    self.cfg, 
                                    override_dataset=None,
                                    split=split
                                )

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = EnsemblingSAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

class BoostedSAETrainingRunner(EnsemblingSAETrainingRunner):
    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE | None = None,
        split: str = 'train',
        parent_saes: List[TrainingBoostedSAE] | None = None,
    ) -> None:
        super().__init__(cfg=cfg,
            override_dataset=override_dataset,
            override_model=override_model,
            override_sae=override_sae,
            split=split
        )
    
        if override_sae is None:
            if self.cfg.from_pretrained_path is not None:
                self.sae = TrainingBoostedSAE.load_from_pretrained(
                    self.cfg.from_pretrained_path, self.cfg.device
                )
            else:
                self.sae = TrainingBoostedSAE(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict()
                    ),
                    parent_saes=parent_saes
                )
                self._init_sae_group_b_decs()
        else:
            self.sae = override_sae
