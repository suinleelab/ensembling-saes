import logging

from accelerate.utils import set_seed
from sae_lens import LanguageModelSAERunnerConfig

from ensembling_saes.paths import get_checkpoint_path
from ensembling_saes.sae_training_runner import EnsemblingSAETrainingRunner
from ensembling_saes.utils import get_lm_sae_runner_cfg, parse_lm_sae_runner_config_args

logger = logging.getLogger(__name__)

def main():
    args = parse_lm_sae_runner_config_args()
    cfg = get_lm_sae_runner_cfg(args)
    if args.log_file_path is not None:
        logging.basicConfig(level=logging.INFO, filename=args.log_file_path)
    else:
        logging.basicConfig(level=logging.INFO)

    # Assigning beta1 due to some error during training with default values where it is not detected as a float
    cfg['adam_beta1'] = 0.0
    
    cfg = LanguageModelSAERunnerConfig(**cfg)

    if 'gemma' in cfg.model_name:
        cfg.n_batches_in_buffer = 2
    
    logger.info(f"Run name updated to: {cfg.run_name}")
    cfg.checkpoint_path = get_checkpoint_path(
        wandb_project=cfg.wandb_project,
        run_name = cfg.run_name
    )
    logger.info(f"Checkpoint path updated to: {cfg.checkpoint_path}")

    set_seed(cfg.seed)
    training_runner = EnsemblingSAETrainingRunner(cfg=cfg, split=args.split)
    training_runner.run()
    
    # TODO: Modify the runner class to have:
    # 1. More frequent progress bar update.
    # 2. Final step evaluation. Save the metrics at the final step for easier analysis.
    logger.info("Done!")


if __name__ == "__main__":
    main()
