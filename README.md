# [ICML 2026] Ensembling Sparse Autoencoders
This repository provides the code to run the framework presented in the paper on [Ensembling Sparse Autoencoders](https://arxiv.org/pdf/2505.16077). We show that ensembling sparse autoencoders improves the activation reconstruction while promoting feature stability, and downstream performance. 

![Concept Figure](./figures/concept_fig.pdf)

## Dataset
The Pile dataset used for training the SAEs can be obtained from [here](https://huggingface.co/datasets/monology/pile-uncopyrighted). To cache the activations, run the following command:

```
python scripts/cache_activations --config <path_to_config_yaml_file> --device <gpu_device_to_use>
```

The `cache_activation_configs` directory contains the configuration for caching activations from different language models.

## Environment Setup
1. Git clone this repository
2. `cd ensembling-saes`
3. Create and activate the specified conda environment by running
    ```
    conda env create -f environment.yml
    conda activate ensembling-saes-env
    ```
4. Install the `ensembling-saes` package and necessary dependencies for
development by running `pip install -e ".[dev]"`
5. Git pre-commit hooks (https://pre-commit.com/) are used to automatically
check and fix formatting errors before a Git commit happens. Run
`pre-commit install` to install all the hooks.

## Training
The `train_config` directory contains the training configurations used for the different language models. Take a look at the `run_train_parallel.sh` script to train multiple SAEs in parallel across multiple GPUs.

### Training single SAE
To train a single SAE, run the following command:
```
python scripts/train.py --config <path_to_config_yaml_file> --device <gpu_device_to_use>
```

### Training ensembling methods
To train an ensembled SAE, run the following command:
```
python scripts/train_ensemble.py \
        --config <path_to_config_yaml_file> \
        --ensembling-method <boosting or bagging> \
        --num-train-saes <num_saes_in_the_ensemble> \
        --device <gpu_device_to_use>
```

## Evaluation
The `test_configs` directory contains the evaluation configurations used for the different language models. The `ensemble-eval-param-list.yaml` file contains parameter configurations shared by all language models.

### Evaluating metrics other than stability
To evaluate metrics like mse, l0, explained variance, connectivity, and diversity, run the following command:
```
python scripts/evaluate_ensemble_recon.py \
        --ensembling-method <boosting or bagging> \
        --config <path_to_test_config_yaml_file> \ 
        --config-params test_configs/ensemble-eval-param-list.yaml \
        --device <gpu_device_to_use>
```

### Evaluating stability
To evaluate stability, run the following command:
```
python scripts/evaluate_ensemble_stability.py \
        --ensembling-method <boosting or bagging> \
        --config <path_to_test_config_yaml_file> \ 
        --config-params test_configs/ensemble-eval-param-list.yaml \
        --num_seeds <num_of_seeds_to_use> \
        --device <gpu_device_to_use>
```
