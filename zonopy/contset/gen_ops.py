# TODO DOCUMENT

from __future__ import annotations
from .polynomial_zonotope.utils import mergeExpMatrix
import torch
import numpy as np
from typing import TYPE_CHECKING
import zonopy.internal as zpi

if TYPE_CHECKING:
    from typing import Union, Tuple
    from .polynomial_zonotope.poly_zono import polyZonotope as PZType
    from .polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from .polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from .polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType
    from .zonotope.zono import zonotope as ZonoType
    from .zonotope.batch_zono import batchZonotope as BZonoType
    from .zonotope.mat_zono import matZonotope as MZonoType
    from .zonotope.batch_mat_zono import batchMatZonotope as BMZonoType

def _expand_to_batch_right(x: torch.Tensor, target_batch: tuple, feature_ndim: int) -> torch.Tensor:
    """
    Make `x` broadcastable to `target_batch + x.shape[-feature_ndim:]` by
    inserting 1s just before the feature dims (right-padding the batch).
    """
    b = x.shape[:-feature_ndim]
    need = len(target_batch) - len(b)
    if need > 0:
        x = x.reshape(b + (1,) * need + x.shape[-feature_ndim:])
    return x.expand(target_batch + x.shape[-feature_ndim:])

def _add_genzono_impl(z1, z2, batch_shape: tuple = ()):
    assert z1.dimension == z2.dimension

    # Decide the target batch shape (right-padded)
    b1 = z1.center.shape[:-1]
    b2 = z2.center.shape[:-1]
    nd = max(len(b1), len(b2))
    pad_right = lambda b: b + (1,) * (nd - len(b))
    b1p, b2p = pad_right(b1), pad_right(b2)

    # Validate broadcastability & compute target
    for a, b in zip(b1p, b2p):
        if not (a == b or a == 1 or b == 1):
            raise ValueError(f"Batch shapes {b1} and {b2} cannot be broadcast")
    target_batch = tuple(max(a, b) for a, b in zip(b1p, b2p))

    # Centers: feature_ndim = 1
    c1 = _expand_to_batch_right(z1.center, target_batch, feature_ndim=1)
    c2 = _expand_to_batch_right(z2.center, target_batch, feature_ndim=1)
    c_sum = c1 + c2  # [..., d]

    # Generators: feature_ndim = 2  (generator_count, d)
    g1 = _expand_to_batch_right(z1.generators, target_batch, feature_ndim=2)
    g2 = _expand_to_batch_right(z2.generators, target_batch, feature_ndim=2)

    Z = torch.cat((c_sum.unsqueeze(-2), g1, g2), dim=-2)  # [..., 1+g1+g2, d]
    return Z


def _add_genzono_num_impl(
        zono: Union[ZonoType, BZonoType, PZType, BPZType],
        num: Union[torch.Tensor, float, int]
        ) -> torch.Tensor:
    if zpi.__debug_extra__:
        assert isinstance(num, (float,int)) or len(num.shape) == 0 \
            or num.shape[-1] == zono.dimension or num.shape[-1] == 1, \
            f'dimension does not match: should be {zono.dimension} or 1, not {num.shape[-1]}.'
    Z = torch.clone(zono.Z)
    Z[...,0,:] += num
    return Z


def _mul_genzono_num_impl(
        zono: Union[ZonoType, BZonoType, PZType, BPZType],
        num: Union[torch.Tensor, float, int],
        batch_shape: Tuple = None
        ) -> torch.Tensor:
    if zpi.__debug_extra__:
        assert isinstance(num, (float,int)) or len(num.shape) == 0 \
            or zono.dimension == num.shape[0] or zono.dimension == 1 \
            or len(num.shape) == 1, 'Invalid dimension.'
    if batch_shape is not None \
            and isinstance(num, torch.Tensor) \
            and num.shape[:len(batch_shape)] == batch_shape:
        for _ in range(len(zono.Z.shape) - len(num.shape[1:]) - len(batch_shape)):
            num = num[..., None]
    Z = zono.Z * num
    return Z


def _matmul_genmzono_impl(
        mzono1: Union[MZonoType, BMZonoType],
        mzono2: Union[MZonoType, BMZonoType]
        ) -> torch.Tensor:
    assert mzono1.n_cols == mzono2.n_rows, 'Incompatible matrix dimensions!'

    # Generate new Z matrix
    Z = mzono1.Z.unsqueeze(-3)@mzono2.Z.unsqueeze(-4)
    return Z.flatten(-4,-3)

# exact Plus
def _add_genpz_impl(
        pz1: Union[PZType, BPZType],
        pz2: Union[PZType, BPZType],
        batch_shape: Tuple = ()
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    id, expMat1, expMat2 = mergeExpMatrix(pz1.id, pz2.id, pz1.expMat, pz2.expMat)
    expMat = torch.vstack((expMat1,expMat2))
    n_dep_gens = pz1.n_dep_gens + pz2.n_dep_gens

    expand_shape = batch_shape+(-1, -1)
    Zlist = (
        (pz1.c+pz2.c).unsqueeze(-2),
        pz1.G.expand(expand_shape),
        pz2.G.expand(expand_shape),
        pz1.Grest.expand(expand_shape),
        pz2.Grest.expand(expand_shape)
        )
    Z = torch.cat(Zlist, dim=-2)
    return Z, n_dep_gens, expMat, id


def _add_genpz_zono_impl(
        pz: Union[PZType, BPZType],
        zono: Union[ZonoType, BZonoType],
        batch_shape: Tuple = ()
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    expand_shape = batch_shape+(-1, -1)
    Zlist = (
        (pz.c+zono.center).unsqueeze(-2),
        pz.G.expand(expand_shape),
        pz.Grest.expand(expand_shape),
        zono.generators.expand(expand_shape)
        )
    Z = torch.cat(Zlist, dim=-2)
    return Z, pz.n_dep_gens, pz.expMat, pz.id


@torch.jit.script
def __mul_Z_tensormerge(Z1: torch.Tensor, Z2: torch.Tensor, z1_ndep: int, z2_ndep: int) -> torch.Tensor:
    # _Z = Z1.unsqueeze(-2)*Z2.unsqueeze(-3)
    _Z = torch.einsum("...id, ...jd->...ijd",Z1,Z2)
    z1 = _Z[..., :z1_ndep+1, 0, :]
    z2 = _Z[..., :z1_ndep+1, 1:z2_ndep+1, :].flatten(-3,-2) # COPIES
    z3 = _Z[..., z1_ndep+1:, :, :].flatten(-3,-2) # COPIES
    z4 = _Z[..., :z1_ndep+1, z2_ndep+1:, :].flatten(-3,-2) # COPIES
    # One way to improve this is to create the output tensor and use views to save to these components directly
    # In that case, the very slight torchscript benefit here probably wouldn't matter
    Z = torch.cat((z1,z2,z3,z4),dim=-2)
    return Z

import time

def _mul_genpz_impl(
        pz1: Union[PZType, BPZType],
        pz2: Union[PZType, BPZType]
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    assert (pz1.dimension == pz2.dimension) or (pz1.dimension == 1) or (pz2.dimension == 1), \
        "polyZonotope dims must match, unless one is 1 (scalar)."

    # Generate the expMat for the overlapping parts
    id, expMat1, expMat2 = mergeExpMatrix(pz1.id, pz2.id, pz1.expMat, pz2.expMat)
    first = expMat2.expand((pz1.n_dep_gens,-1,-1)).reshape(pz1.n_dep_gens*expMat2.shape[0],expMat2.shape[1])
    second = expMat1.expand((pz2.n_dep_gens,-1,-1)).transpose(0,1).reshape(pz2.n_dep_gens*expMat1.shape[0],expMat1.shape[1])
    expMat = torch.vstack((expMat1,expMat2,first + second))
    n_dep_gens = (pz1.n_dep_gens+1) * (pz2.n_dep_gens+1)-1 
    
    # ---- Broadcast Z tensors over batch dims and spatial dim ----
    # Shapes: Z1: [B..., R1, d1], Z2: [B..., R2, d2]
    Z1, Z2 = pz1.Z, pz2.Z
    d1, d2 = Z1.shape[-1], Z2.shape[-1]
    R1, R2 = Z1.shape[-2], Z2.shape[-2]

    # Common batch shape
    bshape1 = Z1.shape[:-2]
    bshape2 = Z2.shape[:-2]
    target_bshape = torch.broadcast_shapes(bshape1, bshape2)


    def _expand_to(t: torch.Tensor, bshape: Tuple[int, ...], r: int, d: int,
                   target_bshape: Tuple[int, ...], target_d: int) -> torch.Tensor:
        # Expand batch dims
        if bshape != target_bshape:
            t = t.expand(*target_bshape, r, d)
        # Expand spatial dim if scalar
        if d == 1 and target_d > 1:
            t = t.expand(*t.shape[:-1], target_d)
        return t

    d_out = d1 if d2 == 1 else d2 if d1 == 1 else d1  # if both >1, they are equal by assert

    Z1b = _expand_to(Z1, bshape1, R1, d1, target_bshape, d_out)
    Z2b = _expand_to(Z2, bshape2, R2, d2, target_bshape, d_out)

    # ---- Fuse numeric blocks (reuses your existing kernel) ----
    # This expects matched batch and spatial dims now.
    Z = __mul_Z_tensormerge(Z1b, Z2b, pz1.n_dep_gens, pz2.n_dep_gens)

    return Z, n_dep_gens, expMat, id


# NOTE: this is 'OVERAPPROXIMATED' multiplication for keeping 'fully-k-sliceables'
# The actual multiplication should take
# dep. gnes.: C_G, G_c, G_G, Grest_Grest, G_Grest, Grest_G
# indep. gens.: C_Grest, Grest_c
#
# But, the sliceable multiplication takes
# dep. gnes.: C_G, G_c, G_G (fully-k-sliceable)
# indep. gnes.: C_Grest, Grest_c, Grest_Grest
#               G_Grest, Grest_G (partially-k-sliceable)


@torch.jit.script
def __matmul_Z_tensormerge(Z1: torch.Tensor, Z2: torch.Tensor, z1_ndep: int, z2_ndep: int) -> torch.Tensor:
    _Z = Z1.unsqueeze(-3) @ Z2.unsqueeze(-4)
    z1 = _Z[..., :z1_ndep+1, 0, :, :]
    z2 = _Z[..., :z1_ndep+1, 1:z2_ndep+1, :, :].flatten(-4,-3) # COPIES
    z3 = _Z[..., z1_ndep+1:, :, :, :].flatten(-4,-3) # COPIES
    z4 = _Z[..., :z1_ndep+1, z2_ndep+1:, :, :].flatten(-4,-3) # COPIES
    # One way to improve this is to create the output tensor and use views to save to these components directly
    # In that case, the very slight torchscript benefit here probably wouldn't matter
    Z = torch.cat((z1,z2,z3,z4),dim=-3)
    return Z

# NOTE: DUE TO THE OVERAPPROXIMATED NATURE, BOTH INPUTS MUST BE COMPRESSED ALREADY
def _matmul_genmpz_impl(
        mpz1: Union[MPZType, BMPZType],
        mpz2: Union[MPZType, BMPZType]
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    assert mpz1.n_cols == mpz2.n_rows, 'Incompatible matrix dimensions!'

    # Generate the expMat for the overlapping parts
    id, expMat1, expMat2 = mergeExpMatrix(mpz1.id,mpz2.id,mpz1.expMat,mpz2.expMat)
    first = expMat2.expand((mpz1.n_dep_gens,)+expMat2.shape).reshape(mpz1.n_dep_gens*expMat2.shape[0],expMat2.shape[1])
    second = expMat1.expand((mpz2.n_dep_gens,)+expMat1.shape).transpose(0,1).reshape(mpz2.n_dep_gens*expMat1.shape[0],expMat1.shape[1])
    expMat = torch.vstack((expMat1,expMat2,first + second))
    n_dep_gens = (mpz1.n_dep_gens+1) * (mpz2.n_dep_gens+1)-1 

    # Generate new Z matrix
    Z = __matmul_Z_tensormerge(mpz1.Z, mpz2.Z, mpz1.n_dep_gens, mpz2.n_dep_gens)

    # Generate the expMat for the overlapping parts
    return Z, n_dep_gens, expMat, id