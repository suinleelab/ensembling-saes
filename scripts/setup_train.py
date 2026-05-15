import argparse
import itertools
import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)

def _parse_arguments():
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
    )
    
    args = parser.parse_args()

    logger.info(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        logger.info(f"\t{arg}={getattr(args, arg)}")
    return args

def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_arguments()

    # Config file for parameters to sweep
    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)

    # Set up directory for files to run the experiment.
    cfg_basename = os.path.basename(args.config).split(".")[0]
    os.makedirs(os.path.join("experiments", cfg_basename), exist_ok=True)
    args_dict = vars(args)

    keys, values = zip(*cfg_params['params'].items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    with open(os.path.join("experiments", cfg_basename, "all_train_cmds.txt"), "w") as f:
        for combination in combinations:
            cmd_to_write = "python scripts/train.py"

            for key, value in combination.items():
                cmd_to_write += f" --{key.replace('_', '-')} {value}"

            # Add config path
            cmd_to_write += f" --config {args_dict['config']}"
            f.write(cmd_to_write + "\n")

if __name__ == "__main__":
    main()
