from typing import List, Optional

import numpy as np
import torch
from sae_lens.sae import SAE
from sae_lens.training.training_sae import (
    Step,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepOutput,
)


class TrainingBoostedSAE(TrainingSAE):
    """
    Class used for A-SAE training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """ 
    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False, parent_saes: List[SAE] = None):
        super().__init__(cfg, use_error_term)
        self.cfg = cfg  # type: ignore

        if cfg.architecture == "standard" or cfg.architecture == "topk":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre
        elif cfg.architecture == "gated":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre_gated
        elif cfg.architecture == "jumprelu":
            self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre_jumprelu
            self.bandwidth = cfg.jumprelu_bandwidth
            self.log_threshold.data = torch.ones(
                self.cfg.d_sae, dtype=self.dtype, device=self.device
            ) * np.log(cfg.jumprelu_init_threshold)

        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")

        self.check_cfg_compatibility()

        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.turn_off_forward_pass_hook_z_reshaping()
        self.parent_saes = parent_saes

        self.mse_loss_fn = self._get_mse_loss_fn()

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        
        # Calculate recontructions
        with torch.no_grad():
            recon = torch.zeros_like(sae_in)
            if len(self.parent_saes) > 0:
                for parent_sae in self.parent_saes:
                    recon += parent_sae.forward(sae_in - recon)

        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        residual = sae_in - recon
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(residual)
        sae_out = self.decode(feature_acts)
        
        # Residual Loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, residual)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        losses = {}

        if self.cfg.architecture == "gated":
            # Gated SAE Loss Calculation

            # Shared variables
            sae_in_centered = (
                self.reshape_fn_in(sae_in) - self.b_dec * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                current_l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()
            loss = mse_loss + l1_loss + aux_reconstruction_loss
            losses["auxiliary_reconstruction_loss"] = aux_reconstruction_loss
            losses["l1_loss"] = l1_loss
        elif self.cfg.architecture == "jumprelu":
            threshold = torch.exp(self.log_threshold)
            l0 = torch.sum(Step.apply(hidden_pre, threshold, self.bandwidth), dim=-1)  # type: ignore
            l0_loss = (current_l1_coefficient * l0).mean()
            loss = mse_loss + l0_loss
            losses["l0_loss"] = l0_loss
        elif self.cfg.architecture == "topk":
            topk_loss = self.calculate_topk_aux_loss(
                sae_in=sae_in,
                sae_out=sae_out,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            losses["auxiliary_reconstruction_loss"] = topk_loss
            loss = mse_loss + topk_loss
        else:
            # default SAE sparsity loss
            weighted_feature_acts = feature_acts
            if self.cfg.scale_sparsity_penalty_by_decoder_norm:
                weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension

            l1_loss = (current_l1_coefficient * sparsity).mean()
            loss = mse_loss + l1_loss
            if (
                self.cfg.use_ghost_grads
                and self.training
                and dead_neuron_mask is not None
            ):
                ghost_grad_loss = self.calculate_ghost_grad_loss(
                    x=sae_in,
                    sae_out=sae_out,
                    per_item_mse_loss=per_item_mse_loss,
                    hidden_pre=hidden_pre,
                    dead_neuron_mask=dead_neuron_mask,
                )
                losses["ghost_grad_loss"] = ghost_grad_loss
                loss = loss + ghost_grad_loss
            losses["l1_loss"] = l1_loss

        losses["mse_loss"] = mse_loss

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )
