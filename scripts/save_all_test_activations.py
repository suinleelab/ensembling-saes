import logging
import os

import torch
import yaml
from sae_lens.config import LanguageModelSAERunnerConfig
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ensembling_saes.activations_store import EnsemblingSAEsActivationsStore
from ensembling_saes.paths import ACTS_DIR
from ensembling_saes.utils import get_lm_sae_runner_cfg, parse_lm_sae_runner_config_args

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # Load the config
    args = parse_lm_sae_runner_config_args()
    cfg = get_lm_sae_runner_cfg(args)
    lm_cfg = LanguageModelSAERunnerConfig(**cfg)

    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)

    # Load model
    model = HookedTransformer.from_pretrained(cfg['model_name'])

    pbar = tqdm(total=cfg_params['total_test_tokens'],
        desc="Saving test activations")
    activations_store = EnsemblingSAEsActivationsStore(model, lm_cfg, override_dataset=None)
    all_model_acts = []
    activations = 0

    # Save activations
    while activations < cfg_params['total_test_tokens']:
        model_acts = activations_store.next_batch()[:, 0, :].to(activations_store.device)
        all_model_acts.append(model_acts)

        pbar.update(activations_store.train_batch_size_tokens)
        activations += activations_store.train_batch_size_tokens
    
    save_file_name = os.path.join(ACTS_DIR, cfg['model_name'], cfg['hook_name'], "all_test_activations.pt")
    torch.save(torch.cat(all_model_acts), save_file_name)

if __name__ == "__main__":
    main()
