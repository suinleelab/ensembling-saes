import logging
import os

import pandas as pd
import yaml
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from transformer_lens import HookedTransformer

from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore
from ensembling_saes.evals import get_sparsity_and_recons_metrics
from ensembling_saes.paths import get_checkpoint_path
from ensembling_saes.utils import get_lm_sae_runner_cfg, parse_lm_sae_runner_config_args

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # Load the config
    args = parse_lm_sae_runner_config_args()
    cfg = get_lm_sae_runner_cfg(args)
    lm_cfg = LanguageModelSAERunnerConfig(**cfg)

    if 'gemma' in lm_cfg.model_name:
        lm_cfg.n_batches_in_buffer = 2

    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)
    
    device = f"cuda:{cfg_params['cuda_available_devices'][0]}"

    # Load model
    model = HookedTransformer.from_pretrained_no_processing(cfg['model_name'], device=device)
    
    metric_dict = {}
    metric_dict['expansion_factor'] = []
    metric_dict['l0'] = []
    metric_dict['mse'] = []
    metric_dict['lr'] = []

    sparsity_key = 'k' if cfg['architecture'] == 'topk' else 'l1_coefficient'
    
    metric_dict[sparsity_key] = []
    metric_dict['explained_var'] = []

    # Loop over different learning rates
    for lr in cfg_params['params']['lr']:
        # Loop over different expansion factors
        for expansion_factor in cfg_params['params']['expansion_factor']:
            # Loop over sparsity penalties
            for sparsity_key_value in cfg_params['params'][sparsity_key]:
                run_name_args = {
                    "seed": cfg["seed"],
                    "expansion_factor": expansion_factor,
                    "lr": lr,
                    "training_tokens": cfg["training_tokens"],
                    "l1_coefficient": None
                }

                arg_name = 'l1_coefficient' if sparsity_key == 'l1_coefficient' else 'k'

                run_name_args[arg_name] = sparsity_key_value
                # cfg["run_name"] = get_run_name(**run_name_args)

                cfg["run_name"] = "SEED-43_R-32_LR-0.0003_L1-0.75_TOKS-800000000_SHUFFLE-False_SINGLE-bagging-GREEDY-MASKING"
                
                checkpoint_path = get_checkpoint_path(
                    wandb_project=cfg["wandb_project"],
                    run_name=cfg["run_name"]
                )
                logger.info(f"Checkpoint path: {checkpoint_path}")

                if os.path.isdir(checkpoint_path):
                    # Load trained SAE
                    checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(checkpoint_path)))[0]
                    sae = SAE.load_from_pretrained(os.path.join(checkpoint_path, checkpoint_dir_name), device=device)
                    sae.eval()
                    activations_store = EnsemblingSAEsActivationsStore(model, lm_cfg, override_dataset=None)

                    metrics = get_sparsity_and_recons_metrics(activations_store, cfg_params['total_test_tokens'], forward_func=None, sae=sae)

                    # Populate metric dict
                    metric_dict['l0'].append(metrics['l0'])
                    metric_dict['mse'].append(metrics['mse'])
                    metric_dict['explained_var'].append(metrics['explained_variance'])
                    metric_dict['lr'].append(lr)
                    metric_dict[sparsity_key].append(sparsity_key_value)
                    metric_dict['expansion_factor'].append(expansion_factor)
            
    # Set up directory to save metrics.
    cfg_basename = os.path.basename(args.config).split(".")[0]
    results_dir = os.path.join("experiments", cfg_basename, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save the sparsity metrics
    results_file_name = f"seed_{cfg['seed']}_lr_{cfg['lr']}_tokens_{cfg['training_tokens']}_metrics.csv"
    pd.DataFrame.from_dict(metric_dict).to_csv(os.path.join(results_dir, results_file_name))

if __name__ == "__main__":
    main()
