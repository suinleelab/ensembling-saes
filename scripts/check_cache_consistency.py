"""
Check whether two cached activation datasets are identical.
This is helpful for checking whether the pipeline for generating activations is reproducible.
"""
import argparse
import logging
import sys

import torch
from datasets import load_from_disk
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds1",
        type=str,
        help="path to the first activation dataset",
    )
    parser.add_argument(
        "--ds2",
        type=str,
        help="path to the second activation dataset",
    )
    args = parser.parse_args()
    logger.info(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        logger.info(f"\t{arg}={getattr(args, arg)}")
    return args

def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_arguments()

    ds1= load_from_disk(args.ds1)
    ds1.set_format(type="torch")

    ds2 = load_from_disk(args.ds2)
    ds2.set_format(type="torch")

    logger.info("Checking number of rows...")
    assert ds1.num_rows == ds2.num_rows

    logger.info("Checking features...")
    assert ds1.features == ds2.features

    logger.info("Checking feature values in each row...")
    features = [key for key in ds1.features]
    for i in tqdm(range(ds1.num_rows)):
        for feature in features:
            ds1_feat = ds1[i][feature]
            ds2_feat = ds2[i][feature]
            assert torch.equal(ds1_feat, ds2_feat)
    
    logger.info("All checks passed!")

if __name__ == "__main__":
    main()
    
