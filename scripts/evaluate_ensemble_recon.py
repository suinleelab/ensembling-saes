import logging
import os

import pandas as pd
import torch
import yaml
from sae_lens.config import LanguageModelSAERunnerConfig
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ensembling_saes.ensembling import Bagging, Boosting
from ensembling_saes.utils import (
    get_lm_sae_runner_cfg,
    load_all_saes,
    parse_lm_sae_runner_config_args,
)

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # Load the config
    args = parse_lm_sae_runner_config_args(verbose=False)
    cfg = get_lm_sae_runner_cfg(args)

    lm_cfg = LanguageModelSAERunnerConfig(**cfg)
    
    if 'gemma' in lm_cfg.model_name:
        lm_cfg.n_batches_in_buffer = 2

    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)
    
    device = cfg['device']
    
    # Set up directory to save metrics.
    cfg_basename = os.path.basename(args.config).split(".")[0]
    results_dir = os.path.join("experiments", cfg_basename, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading all test SAEs")
    all_test_saes = None
    
    if 'recall_auc' in cfg_params['metrics_to_evaluate']:
        all_test_saes = load_all_saes(cfg_params['test_seeds'], cfg, device)

    ensembling_args = {
        'logger': logger, 
        'lm_cfg': lm_cfg, 
        'model': HookedTransformer.from_pretrained(cfg['model_name'], device=cfg['device']),
        'test_saes': all_test_saes,
        'device': cfg['device']
    }

    ensemble_method = args.ensembling_method

    # Set up the baseline class
    if ensemble_method == 'bagging':
        ensemble = Bagging(**ensembling_args)
    elif ensemble_method == 'boosting':
        ensemble = Boosting(**ensembling_args)

    max_saes_in_ensemble = cfg_params['max_saes_in_ensemble']
    logger.info(f"Evaluating {ensemble_method} Baseline")
    pbar = tqdm(total=max_saes_in_ensemble * len(cfg_params['train_seeds']), desc=f"Evaluating {ensemble_method} Baseline")

    # Set up dictionary to store metrics
    metric_dict = {eval_metric: [] for eval_metric in cfg_params['metrics_to_evaluate']}
    metric_dict['seed'] = []

    # Looping through all possible training seeds
    for seed_idx, seed in enumerate(cfg_params['train_seeds']):
        num_saes_in_ensemble = 1

        while num_saes_in_ensemble <= max_saes_in_ensemble:
            logger.info("-" * 10)
            logger.info(f"Num Train SAES: {num_saes_in_ensemble}")
            logger.info("-" * 10)
            metric_dict['seed'].append(seed)


            # Loading the set of train SAEs
            logger.info("Loading the current number of train SAEs")
            all_train_saes = None 
            
            if args.ensembling_method == 'bagging':
                seed_start = (num_saes_in_ensemble * seed_idx) + 1
                all_train_saes = load_all_saes(
                                    list(range(seed_start, seed_start + num_saes_in_ensemble)), 
                                    cfg, 
                                    cfg['device']
                                )

            # Load the ensemble with the current set of train SAEs
            ensemble.load(num_train_saes=num_saes_in_ensemble, train_saes=all_train_saes, train_seed=seed)
            sparsity_metric_dict = None
            pbar_desc = f"Num SAEs: {num_saes_in_ensemble}"

            for eval_metric in cfg_params['metrics_to_evaluate']:
                if eval_metric == 'recall_auc':
                    # Looping over different thresholds for calculating recall
                    all_recalls = []
                    logger.info("Calculating recall across all thresholds")
                    for alpha in cfg_params['alphas']:
                        recall = ensemble.get_recall(alpha)
                        logger.info(f"Recall at alpha={alpha}: {recall: .3f}")
                        all_recalls.append(recall)
                    
                    # Calculate AUC between recalls and alphas
                    auc = torch.trapz(torch.tensor(all_recalls), torch.tensor(cfg_params['alphas']))
                    metric_dict[eval_metric].append(auc.item())
                    pbar_desc += f", Recall AUC: {auc.item(): .3f}"
                elif eval_metric == 'l0' or eval_metric == 'mse' or eval_metric == 'explained_variance':
                    # Only calculate the sparsity metrics once
                    if not sparsity_metric_dict:
                        sparsity_metric_dict = ensemble.eval_recon(total_test_tokens=cfg_params['total_test_tokens'])

                    metric_dict[eval_metric].append(sparsity_metric_dict[eval_metric])
                    pbar_desc += f", {eval_metric}: {sparsity_metric_dict[eval_metric]:.3f}"
                elif eval_metric == 'diversity':
                    diversity = ensemble.eval_diversity()
                    metric_dict[eval_metric].append(diversity)
                    pbar_desc += f", Diversity: {diversity}"
                elif eval_metric == 'connectivity':
                    connectivity = ensemble.eval_connectivity(total_test_tokens=cfg_params['total_test_tokens'])
                    metric_dict[eval_metric].append(connectivity)
                    pbar_desc += f", Connectivity: {connectivity}"
                else:
                    raise ValueError(f"Unknown metric: {eval_metric}")

            pbar.set_description(pbar_desc)
            num_saes_in_ensemble += 1
            pbar.update(1)

    pbar.close()

    # Save the metrics
    results_file_name = f"{ensemble_method}_evaluation_metrics_with_seeds_8_saes.csv"
    metric_dict['num_saes'] = list(range(1, max_saes_in_ensemble + 1))  * len(cfg_params['train_seeds'])
    pd.DataFrame.from_dict(metric_dict).to_csv(os.path.join(results_dir, results_file_name))


if __name__ == "__main__":
    main()
