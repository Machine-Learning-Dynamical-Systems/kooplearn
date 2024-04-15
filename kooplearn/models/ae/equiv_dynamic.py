import logging
import math
from functools import wraps
from typing import Optional, Union

import escnn.nn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType
from morpho_symm.utils.abstract_harmonics_analysis import isotypic_basis

from kooplearn._src.linalg import full_rank_equivariant_lstsq
from kooplearn.data import TensorContextDataset
from kooplearn.models import DynamicAE
from kooplearn.models.ae.utils import flatten_context_data, unflatten_context_data

logger = logging.getLogger(__name__)


class EquivDynamicAE(DynamicAE):

    def __init__(self,
                 encoder: type[escnn.nn.EquivariantModule],
                 decoder: type[escnn.nn.EquivariantModule],
                 encoder_kwargs: dict[str, any],
                 decoder_kwargs: dict[str, any],
                 loss_weights: Optional[dict] = None,
                 use_lstsq_for_evolution: bool = False,
                 evolution_op_bias: bool = False,
                 evolution_op_init_mode: str = "stable"
                 ):

        # TODO: Dunno why not pass the instance of module instead of encoder(**encoder_kwargs).
        self.state_type = encoder_kwargs['in_type']
        self.latent_state_type = decoder_kwargs['in_type']
        self.use_lstsq_for_evolution = use_lstsq_for_evolution

        super(EquivDynamicAE, self).__init__(encoder,
                                             decoder,
                                             encoder_kwargs,
                                             decoder_kwargs,
                                             latent_dim=self.latent_state_type.size,
                                             loss_weights=loss_weights,
                                             evolution_op_bias=evolution_op_bias,
                                             evolution_op_init_mode=evolution_op_init_mode)

    @wraps(DynamicAE.encode_contexts)  # Copies docstring from parent implementation
    def encode_contexts(self, state: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        # Since encoder receives as input a escnn.nn.GeometricTensor, we need to mildly modify encode_contexts
        encoder = self.encoder
        # From (batch, context_length, *features_shape) to (batch * context_length, *features_shape)
        flat_encoded_contexts = encoder(self.state_type(flatten_context_data(state)))
        # From (batch * context_length, latent_dim) to (batch, context_length, latent_dim)
        latent_obs_contexts = unflatten_context_data(flat_encoded_contexts.tensor,
                                                     batch_size=len(state),
                                                     features_shape=(self.latent_dim,))
        return latent_obs_contexts

    @wraps(DynamicAE.decode_contexts)  # Copies docstring from parent implementation
    def decode_contexts(self, latent_obs: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        # Since decoder receives as input a escnn.nn.GeometricTensor, we need to mildly modify decode_contexts
        decoder = self.decoder
        # From (batch, context_length, latent_dim) to (batch * context_length, latent_dim)
        flat_decoded_contexts = decoder(self.latent_state_type(flatten_context_data(latent_obs)))
        # From (batch * context_length, *features_shape) to (batch, context_length, *features_shape)
        decoded_contexts = unflatten_context_data(flat_decoded_contexts.tensor,
                                                  batch_size=len(latent_obs),
                                                  features_shape=self.state_features_shape)
        return decoded_contexts

    @wraps(DynamicAE.get_linear_dynamics_model)  # Copies docstring from parent implementation
    def get_linear_dynamics_model(self, enforce_constant_fn: bool = False) -> torch.nn.Module:
        if enforce_constant_fn:
            raise NotImplementedError("Need to handle bias term in evolution and eigdecomposition")
        return escnn.nn.Linear(self.latent_state_type, self.latent_state_type, bias=enforce_constant_fn)

    @wraps(DynamicAE.fit_linear_decoder)  # Copies docstring from parent implementation
    def fit_linear_decoder(self, latent_states: torch.Tensor, states: torch.Tensor):
        lin_decoder, _ = full_rank_equivariant_lstsq(X=latent_states,
                                                     Y=states,
                                                     rep_X=self.latent_state_type.representation,
                                                     rep_Y=self.state_type.representation,
                                                     bias=False)
        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert lin_decoder.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {lin_decoder.shape}"
        return lin_decoder

    @wraps(DynamicAE.initialize_evolution_operator)  # Copies docstring from parent implementation
    def initialize_evolution_operator(self, init_mode: str):

        from escnn.nn.modules.basismanager import BasisManager
        basis_expansion: BasisManager = self.linear_dynamics.basisexpansion
        identity_coefficients = torch.zeros((basis_expansion.dimension(),))

        if self.evolution_op_bias:
            self.linear_dynamics.bias.data = torch.zeros_like(self.transfer_op.bias.data)

        if init_mode == "stable":
            # Beware: Incredibly shady hack in order to get the identity matrix as the initial evolution operator.
            for io_pair in basis_expansion._representations_pairs:
                # retrieve the basis
                block_expansion = getattr(basis_expansion, f"block_expansion_{basis_expansion._escape_pair(io_pair)}")
                # retrieve the indices
                start_coeff = basis_expansion._weights_ranges[io_pair][0]
                end_coeff = basis_expansion._weights_ranges[io_pair][1]
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter

                # Basis Matrices spawing the space of equivariant linear maps of this block
                basis_set_linear_map = block_expansion.sampled_basis.detach().cpu().numpy()[:, :, :, 0]
                # We want to find the coefficients of this basis responsible for the identity matrix. These are the
                # elements of the basis having no effect on off-diagonal elements of the block.
                basis_dimension = basis_set_linear_map.shape[0]
                singlar_value_dimensions = []
                for element_num in range(basis_dimension):
                    # Get the basis matrix corresponding to this element
                    basis_matrix = basis_set_linear_map[element_num]
                    # Assert that all elements off-diagonal are zero
                    is_singular_value = np.allclose(basis_matrix, np.diag(np.diag(basis_matrix)), rtol=1e-4, atol=1e-4)
                    if is_singular_value:
                        singlar_value_dimensions.append(element_num)
                coefficients = torch.zeros((basis_dimension,))
                coefficients[singlar_value_dimensions] = 1

                # retrieve the linear coefficients for the basis expansion
                identity_coefficients[start_coeff:end_coeff] = coefficients

            self.linear_dynamics.weights.data = identity_coefficients
            matrix, _ = self.linear_dynamics.expand_parameters()
            eigvals = torch.linalg.eigvals(matrix)
            eigvals_real = eigvals.real.detach().cpu().numpy()
            eigvals_imag = eigvals.imag.detach().cpu().numpy()
            assert np.allclose(np.abs(eigvals_real), np.ones_like(eigvals_real), rtol=1e-4, atol=1e-4), \
                f"Eigenvalues with real part different from 1: {eigvals_real}"
            assert np.allclose(eigvals_imag, np.zeros_like(eigvals_imag), rtol=1e-4, atol=1e-4), \
                f"Eigenvalues with imaginary part: {eigvals_imag}"

        else:
            logger.warning(f"Evolution operator init mode {init_mode} not implemented")
            return
        logger.info(f"Trainable evolution operator initialized with mode {init_mode}")

    @property
    def evolution_operator(self):
        if self.use_lstsq_for_evolution:
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
        else:
            # Equivariant linear maps are constrained in parameter space. The expand_parameters method returns the
            # resulting evolution operator, generated by the free degrees of freedom of the linear dynamics.
            matrix, bias = self.linear_dynamics.expand_parameters()
            if bias is not None:
                return matrix, bias
            return matrix
