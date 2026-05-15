"""Constants and utilities for handling paths."""
import os

# Parent directory for activation datasets.
ACTS_DIR = "<PATH_TO_ACTIVATIONS>"

# Parent directory for saving SAE checkpoints.
SAE_DIR = "<PATH_TO_SAVE_CHECKPOINTS>"

def get_activation_dataset_path(
    model_name: str,
    hook_name: str,
    dataset_path: str,
    context_size: int,
    training_tokens: int,
    split: str = 'train'
) -> str:
    """Get the path for caching an activation dataset on disk."""
    source = os.path.basename(dataset_path)
    source += f"_CS-{context_size}"
    source += f"_TOKS-{training_tokens}"
    source += f"_SPLIT-{split}"
    return os.path.join(ACTS_DIR, model_name, hook_name, source)

def get_run_name(
    seed: int,
    expansion_factor: int,
    lr: float,
    l1_coefficient: float,
    training_tokens: int,
    k: int = None
) -> str:
    """Get Wandb run name."""
    run_name = f"SEED-{seed}"
    run_name += f"_R-{expansion_factor}"
    run_name += f"_LR-{lr}"

    if k is not None:
        run_name += f"_TOPK-{k}"
    else:
        run_name += f"_L1-{l1_coefficient}"
        
    run_name += f"_TOKS-{training_tokens}"
    return run_name + "_SHUFFLE-False"

def get_wandb_project(model_name: str, hook_name: str) -> str:
    """Get Wandb project name."""
    return f"{model_name}--{hook_name}"

def get_checkpoint_path(wandb_project: str, run_name: str) -> str:
    return os.path.join(SAE_DIR, wandb_project, run_name)
