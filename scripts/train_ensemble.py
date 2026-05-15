import logging

from sae_lens import LanguageModelSAERunnerConfig
from transformer_lens import HookedTransformer

from ensembling_saes.ensembling import Bagging, Boosting
from ensembling_saes.paths import get_checkpoint_path
from ensembling_saes.utils import (
    get_lm_sae_runner_cfg,
    load_all_saes,
    parse_lm_sae_runner_config_args,
)

logger = logging.getLogger(__name__)

def main():
    args = parse_lm_sae_runner_config_args()
    cfg = get_lm_sae_runner_cfg(args)
    if args.log_file_path is not None:
        logging.basicConfig(level=logging.INFO, filename=args.log_file_path)
    else:
        logging.basicConfig(level=logging.INFO)

    lm_cfg = LanguageModelSAERunnerConfig(**cfg)
    if 'gemma' in lm_cfg.model_name:
        lm_cfg.n_batches_in_buffer = 2
        
    lm_cfg.checkpoint_path = get_checkpoint_path(
        wandb_project=lm_cfg.wandb_project,
        run_name = lm_cfg.run_name
    )

    all_train_saes = None
    if args.ensembling_method == 'bagging':
        try:
            all_train_saes = load_all_saes(list(range(args.seed, args.seed + args.num_train_saes + 1)), cfg, device=cfg['device'])
        except(FileNotFoundError):
            logger.error("Bagging requires that the SAE models are already trained. Please train the SAEs first.")
            logger.error(f"Seeds to be trained: {list(range(args.seed, args.seed + args.num_train_saes + 1))}")
            return

    # Assigning beta1 due to some error during training with default values where it is not detected as a float
    cfg['adam_beta1'] = 0.0
    ensembling_args = {
        'logger': logger, 
        'lm_cfg': lm_cfg, 
        'model': HookedTransformer.from_pretrained(cfg['model_name'], device=cfg['device']),
        'device': cfg['device']
    }

    if args.ensembling_method == 'model_soup':
        ensemble = ModelSoup(**ensembling_args)
    elif args.ensembling_method == 'boosting':
        ensemble = Boosting(**ensembling_args)
    elif args.ensembling_method == 'bagging':
        ensemble = Bagging(**ensembling_args)
    else:
        raise ValueError(f"Unknown ensembling method: {args.ensembling_method}")

    ensemble.fit(num_train_saes=args.num_train_saes, train_saes=all_train_saes)
    
    # TODO: Modify the runner class to have:
    # 1. More frequent progress bar update.
    # 2. Final step evaluation. Save the metrics at the final step for easier analysis.
    logger.info("Done!")


if __name__ == "__main__":
    main()
