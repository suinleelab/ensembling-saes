import argparse
import logging
import sys

import yaml
from sae_lens import CacheActivationsRunnerConfig

from ensembling_saes.cache_activations_runner import CacheActivationsRunnerWithSplit
from ensembling_saes.paths import get_activation_dataset_path

logger = logging.getLogger(__name__)

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to config yaml file",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="GPU device to use",
        default="cuda:0"
    )
    args = parser.parse_args()
    logger.info(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        logger.info(f"\t{arg}={getattr(args, arg)}")
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_arguments()
    with open(args.config) as file:
        cfg = yaml.safe_load(file)
    
    for first_key, first_val in cfg.items():
        logger.info(f"{first_key} config")
        for second_key, second_val in first_val.items():
            logger.info(f"\t{second_key}: {second_val}")
    
    # Pass in configs related to the language model.
    model_name = cfg["model"]["model_name"]
    hook_name = cfg["model"]["hook_name"]
    hook_layer = cfg["model"]["hook_layer"]
    d_in = cfg["model"]["d_in"]
    model_batch_size = cfg["model"]["model_batch_size"]

    # Pass in configs related to the input dataset for generating activations.
    dataset_path = cfg["dataset"]["dataset_path"]
    context_size = cfg["dataset"]["context_size"]
    training_tokens = cfg["dataset"]["training_tokens"]
    split = cfg['dataset']['split']

    new_cached_activations_path = get_activation_dataset_path(
        model_name=model_name,
        hook_name=hook_name,
        dataset_path=dataset_path,
        context_size=context_size,
        training_tokens=training_tokens,
        split=split
    )
    runner_cfg = CacheActivationsRunnerConfig(
        model_name=model_name,
        hook_name=hook_name,
        hook_layer=hook_layer,
        model_batch_size=model_batch_size,
        d_in=d_in,
        dataset_path=dataset_path,
        new_cached_activations_path=new_cached_activations_path,
        training_tokens=training_tokens,
        dataset_trust_remote_code=True,
        context_size=context_size,
        shuffle=False,
        device=args.device,
    )
    _ = CacheActivationsRunnerWithSplit(runner_cfg, split=split).run()
    logger.info(f"Activation data saved to {new_cached_activations_path}")


if __name__ == "__main__":
    main()
