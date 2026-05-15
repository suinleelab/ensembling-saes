import argparse
import heapq
import logging
import os
import sys
from typing import Dict, List

import torch
import yaml
from accelerate.utils import set_seed
from sae_lens.sae import SAE
from tqdm import tqdm

from ensembling_saes.paths import (
    get_checkpoint_path,
    get_run_name,
    get_wandb_project,
)

logger = logging.getLogger(__name__)

def parse_lm_sae_runner_config_args(
    verbose: bool=True
) -> argparse.Namespace:
    """
    Argument parser for configuring the LanguageModelSAERunner
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml file containing SAE training configurations",
    )
    parser.add_argument(
        "--config-params",
        type=str,
        help="path to yaml file containing SAE parameter values to sweep",
        dest="config_params"
    )
    parser.add_argument(
        "--config-cache",
        type=str,
        help="path to cache activations config yaml file",
        dest="config_cache"
    )
    parser.add_argument(
        "--split",
        type=str,
        help="data split to use",
        choices=['train', 'val', 'test'],
        default='train'
    )
    # For the following parameters: if not None, then override the corresponding values
    # set in config.
    parser.add_argument(
        "--log-file-path",
        type=str,
        help="path to the file to stream logger output",
        default=None,
        dest="log_file_path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility",
        default=None,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="SAE architecture",
        default=None,
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        help="expansion factor for the SAE latent dimension",
        default=None,
        dest="expansion_factor",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=None,
    )
    parser.add_argument(
        "--l1-coefficient",
        type=float,
        help="coefficient for L1 regularization",
        default=None,
        dest="l1_coefficient",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="number of epochs for training with cached activations",
        default=1,
        dest="num_epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="GPU device to use",
        default="cuda:0"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="name of Wandb project",
        default=None,
        dest="wandb_project",
    )
    parser.add_argument(
        "--num-train-saes",
        type=int,
        help="number of training SAEs to use for ensembling",
        default=None,
        dest="num_train_saes",
    )
    parser.add_argument(
        "--ensembling-method",
        type=str,
        help="Type of ensembling method to use",
        choices=['boosting', 'bagging'],
        default='boosting',
        dest="ensembling_method"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        help="Number of seeds to use for calculating stability",
        default=5,
        dest="num_seeds"
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="k value for the topk activation function",
        default=None,
        dest="topk"
    )
    
    args = parser.parse_args()
    
    if verbose:
        logger.info(f"Running {sys.argv[0]} with arguments")
        for arg in vars(args):
            logger.info(f"\t{arg}={getattr(args, arg)}")

    return args

def get_lm_sae_runner_cfg(
    args: argparse.Namespace
) -> Dict[str, int | str]:
    """
    Create the config from the parsed args
    """
    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.architecture is not None:
        cfg["architecture"] = args.architecture
    if args.expansion_factor is not None:
        cfg["expansion_factor"] = args.expansion_factor
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.l1_coefficient is not None:
        cfg["l1_coefficient"] = args.l1_coefficient
    if args.num_epochs is not None:
        cfg["training_tokens"] = cfg["training_tokens"]
    if args.device is not None:
        cfg["device"] = args.device
    if args.topk is not None:
        cfg["activation_fn_kwargs"]["k"] = args.topk
    
    if cfg['architecture'] == 'topk':
        k = cfg['activation_fn_kwargs']['k']
        l1_coefficient = None
    else:
        l1_coefficient = cfg['l1_coefficient']
        k = None

    # Set up wandb logging info.
    cfg["run_name"] = get_run_name(
        seed=cfg["seed"],
        expansion_factor=cfg["expansion_factor"],
        lr=cfg["lr"],
        l1_coefficient=l1_coefficient,
        training_tokens=cfg["training_tokens"],
        k=k,
    )
    cfg["wandb_project"] = get_wandb_project(
        model_name=cfg["model_name"],
        hook_name=cfg["hook_name"],
    )
    if args.wandb_project is not None:
        # Override the Wandb project. This is useful when debugging.
        cfg["wandb_project"] = args.wandb_project

    # Assigning beta1 due to some error during training with default values where it is not detected as a float
    cfg['adam_beta1'] = 0.0

    return cfg

def load_sae_with_seed(
    seed: int,
    cfg: Dict[str, int | str],
    device: str = "cuda:0"
) -> SAE:
    """
    Load an SAE with the provided seed
    """
    set_seed(seed)

    l1_coefficient = None
    k = None
    if cfg['architecture'] == 'topk':
        k = cfg['activation_fn_kwargs']['k']
    else:
        l1_coefficient = cfg['l1_coefficient']

    cfg["run_name"] = get_run_name(
        seed=seed,
        expansion_factor=cfg['expansion_factor'],
        lr=cfg['lr'],
        l1_coefficient=l1_coefficient,
        k=k,
        training_tokens=cfg["training_tokens"],
    )
    
    checkpoint_path = get_checkpoint_path(
        wandb_project=cfg["wandb_project"],
        run_name=cfg["run_name"]
    )
    
    checkpoint_dir_name = list(filter(lambda dir: 'final' in dir, os.listdir(checkpoint_path)))[0]

    return SAE.load_from_pretrained(os.path.join(checkpoint_path, checkpoint_dir_name), device=device)

def load_all_saes(
    seeds: List[int],
    cfg: Dict[str, int | str],
    device: str = "cuda:0"
) -> List[SAE]:
    """
    Loads all the SAEs corresponding to the provided seeds
    """
    all_saes =[]
    
    for seed in tqdm(seeds):
        sae = load_sae_with_seed(seed, cfg, device)
        all_saes.append(sae)

    return all_saes

class OnlineKMeans:
    def __init__(self, 
        n_clusters: int, 
        dim: int, 
        lr: float = 0.1,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.float32
    ) -> None:
        """
        Online K-Means with streaming data and K-Means++ initialization.
        """
        self.n_clusters = n_clusters
        self.dim = dim
        self.lr = lr
        self.centroids = None
        self.device = device
        self.dtype = dtype

    def kmeans_plus_plus_init(self, X: torch.Tensor) -> torch.Tensor:
        """K-Means++ Initialization (optimized for PyTorch GPU)."""
        n_samples = X.shape[0]
        centroids = torch.empty((self.n_clusters, X.shape[1]), dtype=self.dtype, device=self.device)

        # Step 1: Pick the first centroid randomly
        centroids[0] = X[torch.randint(n_samples, (1,))]

        # Step 2: Pick remaining k-1 centroids based on distance probabilities
        for i in tqdm(range(1, self.n_clusters), desc="Running K-Means++ Initialization"):
            distances = torch.min(torch.cdist(X, centroids[:i]), dim=1).values
            probabilities = distances ** 2
            probabilities /= probabilities.sum()  # Normalize
            centroids[i] = X[torch.multinomial(probabilities, 1)]

        return centroids

    def predict(self, X: torch.tensor) -> int:
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        distances = torch.cdist(X, self.centroids)

        # Find the assigned cluster index
        return torch.argmin(distances, dim=1)

    def fit(self, X: torch.tensor) -> None:
        """
        Incrementally updates centroids based on streaming batch data.
        """
        if self.centroids is None:
            raise ValueError("Centroids not initialized. Please initialize with a batch of data.")

        # Compute distances between all points in X and centroids 
        distances = torch.cdist(X, self.centroids)

        # Find the nearest centroid for each data point
        closest = torch.argmin(distances, dim=1)

        # Update centroids
        self.centroids[closest] = (1 - self.lr) * self.centroids[closest] + self.lr * X

def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    threshold: float = 0.5,
    topk: int = None
) -> torch.Tensor:
    """
    Gumbel Sigmoid function. Adapted from: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/functional.py#L2146
    """
    gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )    
    
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()
    
    if hard:
        if topk is not None:
            # Get the top-k indices
            _, indices = torch.topk(y_soft, topk, dim=-1)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, indices, 1.0)
        else:
            y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft

    return y_soft

def coherence_sum(
    D: torch.Tensor
) -> torch.FloatType:

    N = D.size(0)
    batch_size = 4096
    total_sum = None
    total_pairs = 0

    # Chunking the computation to avoid OOM errors
    for i in range(0, N, batch_size):
        D_i = D[i:i+batch_size]

        for j in range(0, i+1, batch_size):
            D_j = D[j:j+batch_size]

            sim = D_i @ D_j.T
            sim_abs = sim.abs()

            if i == j:
                triu_mask = torch.triu(torch.ones_like(sim), diagonal=0)
                sim_abs = sim_abs.masked_fill(triu_mask.bool(), 0.0)
                sim_abs = sim_abs - torch.diag_embed(torch.diagonal(sim_abs))
                num_pairs = (D_i.size(0) * (D_i.size(0) - 1)) // 2
            else:
                num_pairs = D_i.size(0) * D_j.size(0)

            if total_sum is None:
                total_sum = sim_abs.sum()
            else:
                total_sum += sim_abs.sum()
            total_pairs += num_pairs

    return total_sum / total_pairs


def greedy_pruning(
    D: torch.Tensor, 
    similarity_threshold: float = 0.95, 
    device: str='cuda'
) -> torch.BoolTensor:
    D = D.to(device)
    N = D.size(0)

    # Step 1: Precompute full cosine similarity matrix
    sim_matrix = D @ D.T  # [N, N]
    sim_matrix.fill_diagonal_(-float('inf'))  # ignore self-similarity

    # Step 2: Greedy loop using similarity matrix
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    selected = []

    for i in tqdm(range(N), desc="Running greedy pruning"):
        if not selected:
            mask[i] = True
            selected.append(i)
            continue

        sims = sim_matrix[i, selected]  # similarity to all selected so far
        if torch.all(sims < similarity_threshold):
            mask[i] = True
            selected.append(i)

    logger.info(f"Num features retained: {mask.sum().item()}")

    return mask

def greedy_pruning_topk(
    D: torch.Tensor, 
    k: int, 
    device: str ='cuda:0'
) -> torch.BoolTensor:
    D = D.to(device)
    N = D.size(0)
    assert k < N, "k must be less than the number of input vectors"

    block_size = 10000
    removed = torch.zeros(N, dtype=torch.bool, device=device)
    remaining = N

    # Priority queue to store (-similarity, i, j)
    heap = []

    for i in tqdm(range(0, N, block_size)):
        D_i = D[i:i + block_size]
        for j in tqdm(range(i + 1, N, block_size)):
            D_j = D[j:j + block_size]
            sims = torch.matmul(D_i, D_j.T)  # [B_i, B_j]

            # Vectorized flatten and sort
            flat_sims = sims.flatten()
            sorted_indices = torch.argsort(flat_sims, descending=True)

            B_j = D_j.size(0)
            for idx in sorted_indices.tolist():
                bi = idx // B_j
                bj = idx % B_j
                idx_i = i + bi
                idx_j = j + bj
                sim = flat_sims[idx].item()
                heapq.heappush(heap, (-sim, idx_i, idx_j))

    # Greedy removal of most similar vectors
    while remaining > k and heap:
        _, i_idx, j_idx = heapq.heappop(heap)
        if removed[i_idx] or removed[j_idx]:
            continue
        removed[j_idx] = True  # arbitrary removal
        remaining -= 1

    return ~removed
