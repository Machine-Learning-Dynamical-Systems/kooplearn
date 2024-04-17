import logging
import math
from functools import wraps
from typing import Optional, Union

import escnn.nn
import numpy as np
import scipy.sparse
import torch
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
                 latent_dim: int,
                 loss_weights: Optional[dict] = None,
                 use_lstsq_for_evolution: bool = False,
                 evolution_op_bias: bool = False,
                 evolution_op_init_mode: str = "stable"
                 ):
        assert 'out_type' not in encoder_kwargs.keys() and 'in_type' not in decoder_kwargs.keys(), \
            f"Encoder `out_type` (and decoder `in_type`) is automatically defined by {self.__class__.__name__}."
        # TODO: Dunno why not pass the instance of module instead of encoder(**encoder_kwargs).
        self.state_type: FieldType = encoder_kwargs['in_type']

        # Define the group representation of the latent observable space, as multiple copies of the group representation
        # in the original state space. This latent group rep is defined in the `isotypic basis`.
        multiplicity = math.ceil(latent_dim / self.state_type.size)
        # Define the observation space representation in the isotypic basis. This function returns two `OrderedDict`
        # mapping iso-space ids (str) to `escnn.group.Representation` and dimensions (Slice) in the latent space.
        self.latent_iso_reps, self.latent_iso_dims = isotypic_basis(representation=self.state_type.representation,
                                                                    multiplicity=multiplicity,
                                                                    prefix='LatentState')
        # Thus, if you want the observables of the `isoX` latent subspace (isoX in self.latent_iso_reps.keys(), do:
        # z_isoX = z[..., self.latent_iso_dims[isoX]]  : where z is a vector of shape (..., latent_dim)
        # Similarly to apply the symmetry transformation to this vector-value observable field, do
        # g ▹ z_isoX = rep_IsoX(g) @ z_isoX     :    rep_IsoX = self.latent_iso_reps[isoX]
        # Define the latent group representation as a direct sum of the representations of each isotypic subspace.
        self.latent_state_type = FieldType(self.state_type.gspace,
                                           [rep_iso for rep_iso in self.latent_iso_reps.values()])

        encoder_kwargs['out_type'] = self.latent_state_type
        decoder_kwargs['in_type'] = self.latent_state_type

        super(EquivDynamicAE, self).__init__(encoder,
                                             decoder,
                                             encoder_kwargs,
                                             decoder_kwargs,
                                             latent_dim=self.latent_state_type.size,
                                             loss_weights=loss_weights,
                                             evolution_op_bias=evolution_op_bias,
                                             evolution_op_init_mode=evolution_op_init_mode,
                                             use_lstsq_for_evolution=use_lstsq_for_evolution)

        logger.debug(f"Initialized Equivariant Dynamic Autoencoder with encoder {encoder} and decoder {decoder}.")

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
    def decode_contexts(
            self, latent_obs: TensorContextDataset, decoder: torch.nn.Module, **kwargs
            ) -> Union[dict, TensorContextDataset]:
        # Since an Equivariant decoder receives as input a escnn.nn.GeometricTensor, we need to:
        if isinstance(decoder, escnn.nn.EquivariantModule):
            # From (batch, context_length, latent_dim) to GeometricTensor(batch * context_length, latent_dim)
            flat_decoded_contexts = decoder(self.latent_state_type(flatten_context_data(latent_obs)))
            # From  GeometricTensor(batch * context_length, *features_shape) to (batch, context_length, *features_shape)
            decoded_contexts = unflatten_context_data(flat_decoded_contexts.tensor,
                                                      batch_size=len(latent_obs),
                                                      features_shape=self.state_features_shape)
        else:
            decoded_contexts = super(EquivDynamicAE, self).decode_contexts(latent_obs, decoder, **kwargs)

        return decoded_contexts

    @wraps(DynamicAE.eig)
    def eig(self,
            eval_left_on: Optional[TensorContextDataset] = None,
            eval_right_on: Optional[TensorContextDataset] = None,
            ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Considering the block-diagonal structure of the evolution operator, we can compute the eigendecomposition
        # by block, and concatenate the eigenvalues and eigenvectors of each isotypic subspace.
        if not hasattr(self, "_eigvals"):
            # Check the latent space group representation is defined in the isotypic basis._____________________________
            from scipy.sparse import coo_matrix
            Q: coo_matrix = self.latent_state_type.change_of_basis
            # Check that all entries are on the diagonal and close to 1
            on_diagonal = Q.row == Q.col
            diagonal_values = np.allclose(Q.data[on_diagonal], 1, rtol=1e-5, atol=1e-5)
            # Ensure there are no off-diagonal entries
            no_off_diagonal_entries = np.allclose(Q.data[~on_diagonal], 0, rtol=1e-5, atol=1e-5)
            is_close_to_identity = diagonal_values and no_off_diagonal_entries
            assert is_close_to_identity, \
                f"We assume the latent group representation is defined on the isotypic basis. {Q.toarray()}"  # ________

            # T is a square real-valued matrix of shape (latent_dim, latent_dim) representing the evolution operator.
            T = self.evolution_operator
            T_np = T.detach().cpu().numpy()
            # Get evolution operators per isotypic subspace T_iso_k : Z_k -> Z_k
            T_iso_spaces = {iso_id: T_np[np.ix_(idx, idx)] for iso_id, idx in self.latent_iso_dims.items()}

            # Compute eigendecomposition of each isotypic subspace
            eivals_iso, eigvecs_l_iso, eigvecs_r_iso = [], [], []
            for iso_id, T_iso in T_iso_spaces.items():
                iso_eigvals, iso_eigvecs_l, iso_eigvecs_r = scipy.linalg.eig(T_iso, left=True, right=True)
                eivals_iso.append(iso_eigvals)
                eigvecs_l_iso.append(iso_eigvecs_l)
                eigvecs_r_iso.append(iso_eigvecs_r)

            # Concatenate the eigenvalues and eigenvectors of each isotypic subspace
            eigvals = np.concatenate(eivals_iso)
            eigvecs_r = scipy.linalg.block_diag(*eigvecs_r_iso)
            eigvecs_r_inv = np.linalg.inv(eigvecs_r)
            eigvecs_l = scipy.linalg.block_diag(*eigvecs_l_iso)
            eigvecs_l_inv = np.linalg.inv(eigvecs_l)

            # Check the eigendecomposition is correct. Fails for non-diagonalizable matrices.
            # T_rec = eigvecs_r @ np.diag(eigvals) @ np.linalg.inv(eigvecs_r)
            # assert np.allclose(T_np, T_rec, rtol=1e-5, atol=1e-5)

            self._eigvals = torch.nn.Parameter(torch.tensor(eigvals, device=T.device), requires_grad=False)
            self._eigvecs_l = torch.nn.Parameter(torch.tensor(eigvecs_l, device=T.device), requires_grad=False)
            self._eigvecs_l_inv = torch.nn.Parameter(torch.tensor(eigvecs_l_inv, device=T.device), requires_grad=False)
            self._eigvecs_r = torch.nn.Parameter(torch.tensor(eigvecs_r, device=T.device), requires_grad=False)
            self._eigvecs_r_inv = torch.nn.Parameter(torch.tensor(eigvecs_r_inv, device=T.device), requires_grad=False)

        # Having computed the eigendecomposition using the block-diagonal structure, default to parent implementation
        return super(EquivDynamicAE, self).eig(eval_left_on, eval_right_on)

    @wraps(DynamicAE.get_linear_dynamics_model)  # Copies docstring from parent implementation
    def get_linear_dynamics_model(self, enforce_constant_fn: bool = False) -> torch.nn.Module:
        if enforce_constant_fn:
            raise NotImplementedError("Need to handle bias term in evolution and eigdecomposition")
        return escnn.nn.Linear(self.latent_state_type, self.latent_state_type, bias=enforce_constant_fn)

    @wraps(DynamicAE.fit_linear_decoder)  # Copies docstring from parent implementation
    def fit_linear_decoder(self, latent_states: torch.Tensor, states: torch.Tensor):
        use_bias = False  # TODO: Unsure if to enable. This can be another hyperparameter, or set to true by default.

        D, bias = full_rank_equivariant_lstsq(X=latent_states,
                                              Y=states,
                                              # rep_X=self.latent_state_type.representation, # TODO: Fix this.
                                              # rep_Y=self.state_type.representation,
                                              bias=use_bias)

        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert D.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {D.shape}"

        # Create a non-trainable linear layer to store the linear decoder matrix and bias term
        # TODO: project learn matrix to the basis of equivariant linear maps and instanciate an escnn.nn.Linear module
        lin_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                      out_features=np.prod(self.state_features_shape),
                                      bias=use_bias,
                                      dtype=self.evolution_operator.dtype)
        lin_decoder.weight.data = torch.tensor(D, dtype=torch.float32)
        lin_decoder.weight.requires_grad = False

        if use_bias:
            lin_decoder.bias.data = torch.tensor(bias, dtype=torch.float32).T
            lin_decoder.bias.requires_grad = False

        return lin_decoder.to(device=self.evolution_operator.device)

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

                # Basis Matrices spawning the space of equivariant linear maps of this block
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
                # Add small perturbation to the off-diagonal elements
                coefficients = 0.001 * torch.randn((basis_dimension,))
                # coefficients = torch.zeros((basis_dimension,))
                coefficients[singlar_value_dimensions] = 1

                # retrieve the linear coefficients for the basis expansion
                identity_coefficients[start_coeff:end_coeff] = coefficients

            self.linear_dynamics.weights.data = identity_coefficients
            matrix, _ = self.linear_dynamics.expand_parameters()
            eigvals = torch.linalg.eigvals(matrix)
            eigvals_real = eigvals.real.detach().cpu().numpy()
            eigvals_imag = eigvals.imag.detach().cpu().numpy()
            assert np.allclose(np.abs(eigvals_real), np.ones_like(eigvals_real), rtol=1e-2, atol=1e-2), \
                f"Eigenvalues with real part different from 1: {eigvals_real}"
            assert np.allclose(eigvals_imag, np.zeros_like(eigvals_imag), rtol=1e-2, atol=1e-2), \
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
