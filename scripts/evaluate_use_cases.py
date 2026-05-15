import logging
import os

import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import yaml
from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_lens import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE

from ensembling_saes.ensembling.bagging import Bagging
from ensembling_saes.ensembling.boosting import Boosting
from ensembling_saes.paths import USE_CASE_DIR, get_checkpoint_path
from ensembling_saes.utils import (
    get_lm_sae_runner_cfg,
    load_sae_with_seed,
    parse_lm_sae_runner_config_args,
)

logger = logging.getLogger(__name__)

def main():
    args = parse_lm_sae_runner_config_args()
    cfg = get_lm_sae_runner_cfg(args)
    if args.log_file_path is not None:
        logging.basicConfig(level=logging.INFO, filename=args.log_file_path)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Assigning beta1 due to some error during training with default values where it is not detected as a float
    cfg["adam_beta1"] = 0.0

    lm_cfg = LanguageModelSAERunnerConfig(**cfg)

    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)

    checkpoint_path = get_checkpoint_path(
        wandb_project=lm_cfg.wandb_project,
        run_name = lm_cfg.run_name
    )

    # Single SAE.
    final_checkpoint = [path for path in os.listdir(checkpoint_path) if "final" in path]
    final_checkpoint = os.path.join(checkpoint_path, final_checkpoint[0])
    sae = SAE.load_from_pretrained(final_checkpoint)
    sae.eval()
    sae_id = f"single_SEED={lm_cfg.seed}"

    # Concatenated ensemble.
    concat_train_saes = [load_sae_with_seed(seed, cfg, "cuda").eval() for seed in cfg_params["train_seeds"]]
    concat_sae = Bagging(
        logger=logger,
        cfg_params=cfg_params,
        lm_cfg=lm_cfg,
        model=None,
        device="cuda",
    )
    concat_sae.load(num_train_saes=len(concat_train_saes), train_saes=concat_train_saes)
    concat_sae.cfg = CustomSAEConfig(
        model_name=lm_cfg.model_name,
        d_in=lm_cfg.d_in,
        d_sae=concat_sae.W_dec.size(0),
        hook_layer=lm_cfg.hook_layer,
        hook_name=lm_cfg.hook_name,
        dtype=lm_cfg.dtype,
        device="cuda",
    )
    concat_sae_first_seed = cfg_params["train_seeds"][0]
    concat_sae_id = f"concat_SEED={concat_sae_first_seed}"

    # Boosted ensemble.
    num_train_saes = len(concat_train_saes)
    boost_sae = Boosting(
        logger=logger,
        lm_cfg=lm_cfg,
        model=None,
        device="cuda",
    )
    boost_sae.load(num_train_saes=num_train_saes)
    boost_sae.cfg = CustomSAEConfig(
        model_name=lm_cfg.model_name,
        d_in=lm_cfg.d_in,
        d_sae=boost_sae.W_dec.size(0),
        hook_layer=lm_cfg.hook_layer,
        hook_name=lm_cfg.hook_name,
        dtype=lm_cfg.dtype,
        device="cuda",
    )
    boost_sae_id = f"boost_SEED={lm_cfg.seed}"

    selected_saes=[
        (boost_sae_id, boost_sae),
        (concat_sae_id, concat_sae),
        (sae_id, sae),
    ]
    output_dir = os.path.join(USE_CASE_DIR, lm_cfg.wandb_project)
    _ = scr_and_tpp.run_eval(
        config=ScrAndTppEvalConfig(
            model_name=lm_cfg.model_name,
            llm_batch_size=32,
            llm_dtype=lm_cfg.dtype,
            sae_batch_size=32,
            dataset_names=["LabHC/bias_in_bios_class_set1"],
            perform_scr=True,
        ),
        selected_saes=selected_saes,
        device="cuda",
        output_path=os.path.join(output_dir, "scr"),
    )
    _ = sparse_probing.run_eval(
        config=SparseProbingEvalConfig(
            model_name=lm_cfg.model_name,
            llm_batch_size=32,
            llm_dtype=lm_cfg.dtype,
            sae_batch_size=32,
        ),
        selected_saes=selected_saes,
        device="cuda",
        output_path=os.path.join(output_dir, "sparse_probing"),
    )


if __name__ == "__main__":
    main()
