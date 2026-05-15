import logging
import os

import pandas as pd
import yaml
from sae_lens.config import LanguageModelSAERunnerConfig
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ensembling_saes.ensembling import Bagging, Boosting
from ensembling_saes.utils import (
    get_lm_sae_runner_cfg,
    parse_lm_sae_runner_config_args,
)

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # Load the config
    args = parse_lm_sae_runner_config_args(verbose=False)
    cfg = get_lm_sae_runner_cfg(args)

    lm_cfg = LanguageModelSAERunnerConfig(**cfg)

    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)
    
    # Set up directory to save metrics.
    cfg_basename = os.path.basename(args.config).split(".")[0]
    results_dir = os.path.join("experiments", cfg_basename, "results")
    os.makedirs(results_dir, exist_ok=True)

    ensembling_args = {
        'logger': logger, 
        'lm_cfg': lm_cfg, 
        'model': HookedTransformer.from_pretrained(cfg['model_name'], device=cfg['device']),
        'device': cfg['device']
    }

    ensemble_method = args.ensembling_method

    # Set up the baseline class
    if ensemble_method == 'bagging':
        ensemble = Bagging(**ensembling_args)
    elif ensemble_method == 'boosting':
        ensemble = Boosting(**ensembling_args)

    max_saes_in_ensemble = cfg_params['max_saes_in_ensemble']
    num_seeds = args.num_seeds
    num_sae_list = []
    sae_indices = []
    stability_all = []

    for curr_num_saes in tqdm(range(1, max_saes_in_ensemble + 1)):
        all_train_seeds = cfg_params['train_seeds']
        logger.info(f"Evaluating {ensemble_method} Baseline")

        logger.info("-" * 10)
        logger.info(f"Num SAES in ensemble: {curr_num_saes}")
        logger.info("-" * 10)

        # Loading the set of train SAEs
        logger.info("Loading the current number of train SAEs")
    
        stability = ensemble.eval_stability(curr_num_saes, all_train_seeds[:num_seeds])
        stability_all += stability
        num_sae_list += [curr_num_saes] * num_seeds
        sae_indices += list(range(num_seeds))

        logger.info(f"Num SAEs: {curr_num_saes}, Stability: {stability}")

    cfg_basename = os.path.basename(args.config).split(".")[0]
    results_dir = os.path.join("experiments", cfg_basename, "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame({
                        "Num SAEs": num_sae_list,
                        "Stability": stability_all,
                        "SAE Index": sae_indices
                    })
    
    # Save the overlap results
    results_file_name = f"{ensemble_method}_stability_metric_with_ci.csv"
    df.to_csv(os.path.join(results_dir, results_file_name))

if __name__ == "__main__":
    main()
