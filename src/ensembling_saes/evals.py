import logging
import os
import random
from itertools import combinations
from typing import Any, List

import einops
import numpy as np
import torch
from sae_lens.evals import EvalConfig, get_featurewise_weight_based_metrics
from sae_lens.sae import SAE
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore
from ensembling_saes.paths import ACTS_DIR


@torch.no_grad()
def run_evals(
    sae: SAE,
    embeddings_store,
    eval_config: EvalConfig = EvalConfig(),
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_metrics = {
        "model_behavior_preservation": {},
        "model_performance_preservation": {},
        "reconstruction_quality": {},
        "shrinkage": {},
        "sparsity": {},
        "token_stats": {},
    }

    if (
        eval_config.compute_l2_norms
        or eval_config.compute_sparsity_metrics
        or eval_config.compute_variance_metrics
    ):
        sparsity_variance_metrics, feature_metrics = get_sparsity_and_variance_metrics(
            sae=sae,
            embeddings_store=embeddings_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            compute_featurewise_density_statistics=eval_config.compute_featurewise_density_statistics,
            verbose=verbose,
        )

        if eval_config.compute_l2_norms:
            all_metrics["shrinkage"].update(
                {
                    "l2_norm_in": sparsity_variance_metrics["l2_norm_in"],
                    "l2_norm_out": sparsity_variance_metrics["l2_norm_out"],
                    "l2_ratio": sparsity_variance_metrics["l2_ratio"],
                    "relative_reconstruction_bias": sparsity_variance_metrics[
                        "relative_reconstruction_bias"
                    ],
                }
            )

        if eval_config.compute_sparsity_metrics:
            all_metrics["sparsity"].update(
                {
                    "l0": sparsity_variance_metrics["l0"],
                    "l1": sparsity_variance_metrics["l1"],
                }
            )

        if eval_config.compute_variance_metrics:
            all_metrics["reconstruction_quality"].update(
                {
                    "explained_variance": sparsity_variance_metrics[
                        "explained_variance"
                    ],
                    "mse": sparsity_variance_metrics["mse"],
                    "cossim": sparsity_variance_metrics["cossim"],
                }
            )
    else:
        feature_metrics = {}

    if eval_config.compute_featurewise_weight_based_metrics:
        feature_metrics |= get_featurewise_weight_based_metrics(sae)

    if len(all_metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    total_tokens_evaluated_eval_reconstruction = (
        embeddings_store.batch_size
        * eval_config.n_eval_reconstruction_batches
    )

    total_tokens_evaluated_eval_sparsity_variance = (
        embeddings_store.batch_size
        * eval_config.n_eval_sparsity_variance_batches
    )

    all_metrics["token_stats"] = {
        "total_tokens_eval_reconstruction": total_tokens_evaluated_eval_reconstruction,
        "total_tokens_eval_sparsity_variance": total_tokens_evaluated_eval_sparsity_variance,
    }

    # Remove empty metric groups
    all_metrics = {k: v for k, v in all_metrics.items() if v}

    return all_metrics, feature_metrics


def get_sparsity_and_variance_metrics(
    sae: SAE,
    embeddings_store,
    compute_l2_norms: bool,
    compute_sparsity_metrics: bool,
    compute_variance_metrics: bool,
    compute_featurewise_density_statistics: bool,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_dict = {}
    feature_metric_dict = {}

    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
        metric_dict["relative_reconstruction_bias"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []
        metric_dict["cossim"] = []
    if compute_featurewise_density_statistics:
        feature_metric_dict["feature_density"] = []
        feature_metric_dict["consistent_activation_heuristic"] = []

    total_feature_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_feature_prompts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_tokens = 0

    for embeddings, _ in tqdm(embeddings_store.eval_dataloader, desc="Sparsity and Variance Batches", disable = not verbose):
        sae_feature_activations = sae.encode(embeddings.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(embeddings.device)

        flattened_sae_input = einops.rearrange(embeddings, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

            # Equation 10 from https://arxiv.org/abs/2404.16014
            # https://github.com/saprmarks/dictionary_learning/blob/main/evaluation.py
            x_hat_norm_squared = torch.norm(flattened_sae_out, dim=-1) ** 2
            x_dot_x_hat = (flattened_sae_input * flattened_sae_out).sum(dim=-1)
            relative_reconstruction_bias = (
                x_hat_norm_squared.mean() / x_dot_x_hat.mean()
            ).unsqueeze(0)

            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)
            metric_dict["relative_reconstruction_bias"].append(
                relative_reconstruction_bias
            )

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            )
            total_sum_of_squares = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
            )

            mse = resid_sum_of_squares / flattened_sae_input.size(-1)
            explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares

            x_normed = flattened_sae_input / torch.norm(
                flattened_sae_input, dim=-1, keepdim=True
            )
            x_hat_normed = flattened_sae_out / torch.norm(
                flattened_sae_out, dim=-1, keepdim=True
            )
            cossim = (x_normed * x_hat_normed).sum(dim=-1)

            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(mse)
            metric_dict["cossim"].append(cossim)

        if compute_featurewise_density_statistics:
            sae_feature_activations_bool = (sae_feature_activations > 0).float()
            total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
            total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(
                dim=0
            )
            total_tokens += flattened_sae_feature_acts.size(0)

    # Aggregate scalar metrics
    metrics: dict[str, float] = {}
    for metric_name, metric_values in metric_dict.items():
        metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

    # Aggregate feature-wise metrics
    feature_metrics: dict[str, list[float]] = {}
    feature_metrics["feature_density"] = (total_feature_acts / total_tokens).tolist()
    feature_metrics["consistent_activation_heuristic"] = (
        total_feature_acts / total_feature_prompts
    ).tolist()

    return metrics, feature_metrics

def get_no_overlap(
    all_W_decs: list[torch.nn.Parameter],
    similarity_threshold: float = 0.6,
    logger: logging.Logger = None,
) -> tuple[list[int], list[int]]:
    """
    Get the overlap of all possible combinations of the trained SAEs
    """
    number_of_saes = len(all_W_decs)
    no_overlap_all = []
    avg_max_sim_all = []

    # Loop over the total number of SAEs
    for num_comb in range(1, number_of_saes):
        no_overlap_all_base_saes = 0
        avg_max_sim_all_base_saes = 0
        subset_W_decs = all_W_decs[0:num_comb + 1]
        
        base_sae_indices = range(len(subset_W_decs))

        if len(subset_W_decs) > 10:
            base_sae_indices = random.sample(base_sae_indices, 10)

        # Loop over all possible SAEs as the base SAE
        for base_sae_index in base_sae_indices:

            # Select one SAE as the base SAE
            base_SAE_W_dec = subset_W_decs[base_sae_index]
            
            # Rest of the SAEs will be used for getting overlap
            rest_W_decs = [W_dec for i, W_dec in enumerate(subset_W_decs) if i != base_sae_index]

            # Get all possible combinations for the current number of combinations
            all_combs = list(combinations(rest_W_decs, num_comb))
            if len(all_combs) > 10:
                all_combs = random.sample(all_combs, 10)
            # all_combs = get_subset_of_combinations(rest_W_decs, num_comb, 10)

            no_overlap = 0
            avg_max_sim = 0

            # Loop over each combination
            for comb in all_combs:
                rows_with_no_overlap = np.ones(base_SAE_W_dec.shape[0])
                all_cosine_sims = []

                # Loop over each decoder matrix in the combination 
                for W_dec in comb:

                    # Get the cosine similarity with the base SAE
                    cosine_sims = base_SAE_W_dec @ W_dec.T 
                    
                    rows_with_no_overlap -= torch.any(cosine_sims > similarity_threshold, dim=1).cpu().numpy()
                    rows_with_no_overlap[rows_with_no_overlap < 0] = 0

                    if len(all_cosine_sims) == 0 :
                        all_cosine_sims = cosine_sims
                    else:
                        all_cosine_sims = torch.cat((all_cosine_sims, cosine_sims), dim=1)


                no_overlap += rows_with_no_overlap.sum() / base_SAE_W_dec.shape[0]

                # Get the average of the vector with the highest cosine similarities between the candidate SAEs and the base SAE
                avg_max_sim += torch.mean(torch.max(all_cosine_sims, dim=1)[0])

            no_overlap_all_base_saes += no_overlap/len(all_combs)
            avg_max_sim_all_base_saes += avg_max_sim/len(all_combs)

        logger.info("-"*8)
        logger.info(f"Number of SAEs: {num_comb + 1}")
        logger.info("-"*8)
        no_overlap_all_base_saes /= len(base_sae_indices)
        avg_max_sim_all_base_saes /= len(base_sae_indices)

        logger.info(f"Proportion of features in only one SAE: {no_overlap_all_base_saes}")
        logger.info(f"Average of max cosine similarities: {avg_max_sim_all_base_saes}")

        no_overlap_all.append(no_overlap_all_base_saes)
        avg_max_sim_all.append(avg_max_sim_all_base_saes)
    
    return no_overlap_all, avg_max_sim_all


def stability_exact(
    all_W_decs: list[torch.nn.Parameter]
) -> float:
    """
    Calculate exact stability between the SAEs by running the hungarian algorithm to match decoder directions
    before getting cosine similarity. Adopted from https://arxiv.org/abs/2502.12892
    """
    stability_score_all_saes = 0

    for base_sae_index in range(len(all_W_decs)):
        # Select one SAE as the base SAE
        base_SAE_W_dec = all_W_decs[base_sae_index]
        
        # Rest of the SAEs will be used for getting overlap
        rest_W_decs = [W_dec for i, W_dec in enumerate(all_W_decs) if i != base_sae_index]
        stability_score_per_sae = 0

        for W_dec in rest_W_decs:
            n = W_dec.shape[0]

            # Get hungarian matching
            cost_matrix = torch.cdist(base_SAE_W_dec, W_dec, p=2).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Build optimal permutation matrix 
            pi = torch.zeros(n, n, device=base_SAE_W_dec.device)
            pi[row_ind, col_ind] = 1.0

            stability_score = torch.trace(base_SAE_W_dec.T @ pi @ W_dec) / n
            stability_score_per_sae += stability_score
        
        stability_score_all_saes += stability_score_per_sae / len(rest_W_decs)

    return stability_score_all_saes / len(all_W_decs)

def coherence_max(all_W_decs):
    """
    Computes the maximum absolute cosine similarity between all distinct rows of D.

    Args:
        D (torch.Tensor): shape (N, d)
        batch_size (int): rows per chunk

    Returns:
        float: max |cosine similarity| (excluding diagonal)
    """
    if len(all_W_decs) == 1:
        all_W_decs = all_W_decs[0]
        diag = torch.eye(all_W_decs.shape[0], device=all_W_decs.device)
        sim_abs = (all_W_decs @ all_W_decs.T).abs()
        sim_abs = sim_abs.masked_fill(diag.bool(), 0.0)
        return (sim_abs).max().item()

    max_cosine_sim_all_saes = 0

    for base_sae_index in tqdm(range(len(all_W_decs))):
        # Select one SAE as the base SAE
        base_SAE_W_dec = all_W_decs[base_sae_index]
        # Rest of the SAEs will be used for getting overlap
        rest_W_decs = [W_dec for i, W_dec in enumerate(all_W_decs) if i != base_sae_index]
        max_cosine_sims = torch.zeros(base_SAE_W_dec.shape[0], device=base_SAE_W_dec.device)

        for W_dec in tqdm(rest_W_decs):
            # Get the cosine similarity with the base SAE
            cosine_sims = (base_SAE_W_dec @ W_dec.T).abs()
            max_cosine_sims = torch.max(max_cosine_sims, torch.max(cosine_sims,dim=1)[0])
        
        max_cosine_sim_all_saes += max_cosine_sims.sum() / base_SAE_W_dec.shape[0]

    return (max_cosine_sim_all_saes / len(all_W_decs)).item()

def diversity(D, batch_size: int = 1024, threshold: float = 0.7) -> float:
    N = D.shape[0]
    sim_sum = 0

    # Loop over all the test SAEs
    for i in range(0, N, batch_size):
        D_i = D[i:i+batch_size] 

        cosine_sims = D_i @ D.T
        cosine_sims[:, i:i+batch_size] = 0.0
        sim_sum += torch.all(cosine_sims < threshold, dim=1).sum()

        del cosine_sims
        torch.cuda.empty_cache()
        
    return sim_sum.item()

def stability_relaxed(
    all_W_decs: list[torch.nn.Parameter]    
) -> float:
    """
    Calculate a relaxed version of exact stability between the SAEs by getting the maximum cosine similarirty
    between the SAEs without running hungarian algorithm. Adopted from https://arxiv.org/abs/2502.12892
    """
    max_cosine_sim_all_saes = []

    for base_sae_index in tqdm(range(len(all_W_decs))):
        # Select one SAE as the base SAE
        base_SAE_W_dec = all_W_decs[base_sae_index]
        # Rest of the SAEs will be used for getting overlap
        rest_W_decs = [W_dec for i, W_dec in enumerate(all_W_decs) if i != base_sae_index]
        max_cosine_sims = torch.zeros(base_SAE_W_dec.shape[0], device=base_SAE_W_dec.device)

        for W_dec in tqdm(rest_W_decs):
            # Get the cosine similarity with the base SAE
            cosine_sims = base_SAE_W_dec @ W_dec.T
            max_cosine_sims = torch.max(max_cosine_sims, torch.max(cosine_sims,dim=1)[0])
        
        max_cosine_sim_all_saes.append((max_cosine_sims.sum() / base_SAE_W_dec.shape[0]).item())

    return max_cosine_sim_all_saes

def overlap(
    all_W_decs: list[torch.nn.Parameter],
    similarity_threshold: float = 0.6
) -> float:
    """
    Get the overlap of all possible combinations of the decoder directions of the trained SAEs.
    The overlap is calculated as the proportion of features in the base SAE that are found in at least one other SAE.
    """
    overlap_score_all_saes = 0

    for base_sae_index in tqdm(range(len(all_W_decs))):
        # Select one SAE as the base SAE
        base_SAE_W_dec = all_W_decs[base_sae_index]
        rows_with_overlap = np.zeros(base_SAE_W_dec.shape[0])

        # Rest of the SAEs will be used for getting overlap
        rest_W_decs = [W_dec for i, W_dec in enumerate(all_W_decs) if i != base_sae_index]

        for W_dec in tqdm(rest_W_decs):
            # Get the cosine similarity with the base SAE
            cosine_sims = base_SAE_W_dec @ W_dec.T 
            rows_with_overlap += torch.any(cosine_sims > similarity_threshold, dim=1).cpu().numpy()
            rows_with_overlap[rows_with_overlap > 0] = 1
        
        overlap_score_all_saes += rows_with_overlap.sum() / base_SAE_W_dec.shape[0]

    return overlap_score_all_saes / len(all_W_decs)

def get_sparsity_and_recons_metrics(
    activations_store: EnsemblingSAEsActivationsStore,
    total_test_tokens: int,
    forward_func: callable = None,
    sae: SAE = None
):
    """
    Get the mse, l0, and explained variance for the given SAE.
    """
    l0_list = []
    mse_list = []
    metric_dict = {}
    sum_of_residual_error_across_batch = 0
    total_sum_of_squares = 0

    pbar = tqdm(total=total_test_tokens, desc="Getting sparsity and reconstruction metrics")
    tokens = 0
    load_file_name = os.path.join(ACTS_DIR, sae.cfg.model_name, sae.cfg.hook_name, "all_test_activations.pt")
    mean_activation = torch.load(load_file_name).mean(dim=0).to(sae.device)

    # Calculate metrics
    while tokens < total_test_tokens:
        with torch.no_grad():
            model_acts = activations_store.next_batch()[:, 0, :].to(activations_store.device)

            with torch.no_grad():
                if forward_func:
                    l0, sae_out = forward_func(model_acts)

                    # For bagging ensembling, we need to get the average reconstruction 
                    if isinstance(sae_out, list):
                        sae_out = torch.stack(sae_out).mean(dim=0)
                    
                    # For bagging and boosting ensembling, we need to get the combined l0
                    if isinstance(l0, list):
                        l0 = torch.stack(l0).sum(dim=0)
                else:
                    hidden_acts = sae.encode(model_acts)
                    sae_out = sae.decode(hidden_acts)

                    l0 = (hidden_acts > 0).sum(dim=-1).float()
                
                l0_list.append(l0)

            # sum of residual errors is used to calculate MSE
            squared_error = (sae_out - model_acts).pow(2)
            sum_of_residual_error = squared_error.sum(dim=-1)

            # sum of residual errors across the batch is used to calculate variance
            sum_of_residual_error_across_batch += squared_error.sum(dim=0)
            total_sum_of_squares += (model_acts - mean_activation).pow(2).sum(dim=0)
            
            # Get the L2 norm by dividing residual error sum with the total number of tokens in the batch
            # mse = sum_of_residual_error / model_acts.shape[0]
            mse_list.append(sum_of_residual_error)

            pbar.update(activations_store.train_batch_size_tokens)
            tokens += activations_store.train_batch_size_tokens
    
    # Calculate explained variance
    explained_var_per_dimension = 1 - (sum_of_residual_error_across_batch / total_sum_of_squares)

    # Populate metric dict
    metric_dict['l0'] = torch.cat(l0_list).mean().item()
    metric_dict['mse'] = torch.cat(mse_list).mean().item()
    metric_dict['explained_variance'] = explained_var_per_dimension.mean().item()

    return metric_dict

def connectivity(
    activations_store: EnsemblingSAEsActivationsStore,
    total_test_tokens: int,
    encode_func: callable = None,
    sae: SAE = None
):
    """
    Get the mse, l0, and explained variance for the given SAE.
    """
    all_paired_similarities = None
    pbar = tqdm(total=total_test_tokens, desc="Getting connectivity metric")
    tokens = 0

    # Calculate metrics
    while tokens < total_test_tokens:
        with torch.no_grad():
            model_acts = activations_store.next_batch()[:, 0, :].to(activations_store.device)

            if encode_func:
                hidden_acts = encode_func(model_acts)
            else:
                hidden_acts = sae.encode(model_acts)

            paired_similarity = hidden_acts.T @ hidden_acts
            if all_paired_similarities is None:
                all_paired_similarities = paired_similarity
            else:
                all_paired_similarities += paired_similarity
            
            del paired_similarity
            torch.cuda.empty_cache()

            pbar.update(activations_store.train_batch_size_tokens)
            tokens += activations_store.train_batch_size_tokens

    m = all_paired_similarities.shape[1]
    return 1 - (torch.norm(all_paired_similarities, p=0).item() / (m * m))

def recall(
        alpha: float,
        test_saes: List[SAE], 
        trained_sae: List[SAE] | SAE
) -> float:
        all_overlap = 0

        # Loop over all the test SAEs
        for test_sae in tqdm(test_saes):
            test_sae.eval()
            test_sae_W_dec = test_sae.W_dec

            # Create a mask to track overlapping features
            overlap_features_mask = torch.zeros(test_sae_W_dec.shape[0], dtype=torch.bool, device=test_sae.device)

            # If the trained sae is in a list form (for ensembling methods like boosting), consider all W_decs in the list
            if isinstance(trained_sae, list):
                for train_sae in trained_sae:
                    train_sae.eval()
                    train_W_dec = train_sae.W_dec

                    cosine_sims = test_sae_W_dec @ train_W_dec.T

                    # Update the mask with the overlapping features
                    overlap_features_mask = torch.logical_or(overlap_features_mask, torch.any(cosine_sims >= alpha, dim=1))
            else:
                trained_sae.eval()
                # Calculate the overlap between the test SAE and all train SAEs
                train_W_dec = trained_sae.W_dec
                    
                cosine_sims = test_sae_W_dec @ train_W_dec.T

                # Update the mask with the overlapping features
                overlap_features_mask = torch.logical_or(overlap_features_mask, torch.any(cosine_sims >= alpha, dim=1))

            # Calculate the recall for the test SAE
            all_overlap += overlap_features_mask.sum().item() / test_sae_W_dec.shape[0]
        
        # Return the average recall across all test SAEs
        return all_overlap / len(test_saes)
