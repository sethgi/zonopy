"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA, Patrick Holme's implementation
"""
from typing import Iterable
from zonopy.contset.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix, pz_repr
import zonopy as zp
import torch
from ..gen_ops import (
    _add_genpz_impl,
    _add_genzono_num_impl,
    _add_genpz_zono_impl,
    _mul_genpz_impl,
    _mul_genzono_num_impl,
)
import zonopy.internal as zpi

import warnings
warnings.simplefilter("once", UserWarning)


class polyZonotope:
    r""" 1D Polynomial Zonotopes

    The Polynomial Zonotope (similar to Talyor model) is a non-convex set representation of a summation over multivariate polynomials multiplied with intervals.

    It is defined as a set of the following form:

    .. math::
        \mathcal{PZ} := \left\{
            c + \sum_{i=1}^{N} \left( \prod_{k=1}^{p}\alpha_{k}^{E_{(k,i)}}\right) G_{(\cdot,i)}+\sum_{j=1}^{M}\beta_{j}G_{rest(\cdot,j)}
            \; \middle\vert \;
            \begin{array}{l}
                \alpha_k, \beta_j \in [-1,1] \\
                \forall k = 1,...,p \\
                \forall j=1,...,M
            \end{array}
            \right\}

    where

    * :math:`c\in\mathbb{R}^d` is the center vector,
    * :math:`G\in\mathbb{R}^{d\times N}` is the dependent generator matrix,
    * :math:`G_{rest}\in\mathbb{R}^{d\times M}` is the independent generator matrix,
    * :math:`E\in\mathbb{R}^{p\times N}` is the exponent matrix,
    * :math:`N` is the number of dependent generators,
    * :math:`M` is the number of independent generators, and
    * :math:`p` is the number of indeterminants.
    """

    def __init__(self, Z, n_dep_gens=0, expMat=None, ids=None, copy_Z=True, dtype=None, device=None):
        r''' Initialize the polynomial zonotope

        Args:
            Z (torch.Tensor): The center and generator matrix of the polynomial zonotope :math:`\mathbf{Z} = [c, G, G_{rest}]^T`
            n_dep_gens (int, optional): The number of dependent generators. Default: 0
            expMat (torch.Tensor, optional): The exponent matrix of the dependent generators. If ``None``, it will be the identity matrix. Default: None
            id (torch.Tensor, optional): The integer identifiers for the dependent generators. If ``None``, it will be the range of the number of dependent generators. Default: None
            copy_Z (bool, optional): If ``True``, it will copy the input ``Z`` value. Default: ``True``
            dtype (torch.dtype, optional): The data type of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``
            device (torch.device, optional): The device of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``

        Raises:
            AssertionError: If the exponent matrix does not seem to be valid for the given dependent generators or ids.
            AssertionError: If the number of dependent generators does not match the number of ids.
            AssertionError: If the exponent matrix is not a non-negative integer matrix.
        '''
        # If compress=2, it will always copy.

        # Make sure Z is a tensor
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        
        self._id = ids

        # Make an expMat and id if not given
        if expMat is None and ids is None:
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)Z
            self.id = torch.arange(self.expMat.shape[1],dtype=torch.int64, device=Z.device)
            
        # Otherwise make sure expMat is right
        elif expMat is not None:
            expMat = torch.as_tensor(expMat,dtype=torch.long,device=Z.device)
            assert expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            if zpi.__debug_extra__: assert torch.all(expMat >= 0), 'Invalid exponent matrix.' 
            
            self.expMat = expMat
            # Make sure ID is right
            if ids is not None:
                self.id = torch.as_tensor(ids, dtype=torch.int64, device=device).flatten()
            else:
                self.id = torch.arange(self.expMat.shape[1],dtype=torch.int64, device=Z.device)
        
        # Otherwise ID is given, but not the expMat, so make identity
        else:
            if isinstance(ids, torch.Tensor):
                self.id = ids.clone().detach()
            else:
                self.id = torch.tensor(ids, dtype=torch.int).flatten()
            
            assert len(self.id) == n_dep_gens, 'Number of dependent generators must match number of id\'s!'
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device)


        # Copy the Z if requested
        if copy_Z:
            self.Z = torch.clone(Z)
        # Or save it itself
        else:
            self.Z = Z
        self.n_dep_gens = int(n_dep_gens)
        
        self._validate()
        
        assert self.id.device == self.Z.device
        
    @property
    def id(self) -> torch.Tensor:
        return self._id
    
    @id.setter
    def id(self, val: torch.Tensor):
        self._id = val
        
    def _validate(self):
        assert self.expMat.shape[0] == self.n_dep_gens, 'Invalid exponent matrix.'
        if zpi.__debug_extra__:
            assert torch.all(self.expMat >= 0), 'Invalid exponent matrix.'


    @property
    def expMat(self) -> torch.Tensor:
        return self._expMat
    
    @expMat.setter
    def expMat(self, new_val):
        self._expMat = new_val
        # self._compute_id_from_expmat()
        
    def norm(self, ord=2, dims=None, inv=False):
        """
        Sound polyZonotope enclosure of ||x||_p for x in Z (or its reciprocal).
        Keeps dependent structure via a linear (supporting) term and adds one
        independent "radius" generator so that the interval endpoints match the
        sharp triangle-inequality bounds.

        Args:
            ord: p in ||.||_p (float/int as in torch.linalg.vector_norm).
            dims: int or iterable of indices to project before taking the norm.
            inv: if True, return 1/||.||_p as a 1D zonotope interval (requires
                the lower bound > 0).

        Returns:
            If inv is False:
                polyZonotope of shape (1 + n_dep_gens' + n_indep_gens' + 1, 1)
            If inv is True:
                zonotope (1D interval) encoded by a 2x1 tensor [center; radius].
        """
        import torch
        import zonopy as zp

        Z = self.project(dims) if dims is not None else self

        # Shapes (non-batch):
        # c:            (d,)
        # Gd = Z.G:     (N_dep, d)   (may be empty)
        # Gi = Z.Grest: (N_ind, d)   (may be empty)
        c  = Z.c
        Gd = Z.G
        Gi = Z.Grest

        p = ord
        eps = torch.finfo(c.dtype).eps

        # Center p-norm and triangle-inequality radius r = sum ||gens||_p
        c_norm = torch.linalg.vector_norm(c, ord=p, dim=-1)                          # ()
        r_dep  = torch.linalg.vector_norm(Gd, ord=p, dim=-1).sum() if Gd.numel() else torch.zeros((), dtype=c.dtype, device=c.device)
        r_ind  = torch.linalg.vector_norm(Gi, ord=p, dim=-1).sum() if Gi.numel() else torch.zeros((), dtype=c.dtype, device=c.device)
        r = r_dep + r_ind

        # Build (sub)gradient at c to preserve dependent structure
        if p == 2:
            denom = torch.clamp(c_norm, min=eps)
            grad = c / denom                                                        # (d,)
        elif isinstance(p, (int, float)) and p > 1:
            denom = torch.clamp(c_norm, min=eps) ** (p - 1)
            grad = torch.sign(c) * (torch.abs(c) ** (p - 1)) / denom                # (d,)
        elif p == 1:
            grad = torch.sign(c)                                                    # (d,)
        else:
            # Fallback for unsupported/other ord (incl. inf):
            # Return tight interval as a scalar PZ with one independent gen.
            new_c = c_norm                                                          # ()
            extra = r                                                               # ()
            total_gens = 1 + 0 + 1                                                  # center + (no dep) + 1 indep
            Zs = torch.zeros((total_gens, 1), dtype=c.dtype, device=c.device)
            Zs[0, 0] = new_c
            Zs[1, 0] = extra
            out = type(self)(
                Zs,
                n_dep_gens=0,
                expMat=torch.zeros((0, 0), dtype=torch.long, device=c.device),
                id=torch.empty(0, dtype=torch.int64, device=c.device),
                copy_Z=False,
            )
            if inv:
                l = torch.clamp(new_c - extra, min=0.0)
                u = new_c + extra
                if (l <= 0).item():
                    raise ValueError("Reciprocal norm undefined: interval touches zero.")
                inv_low  = 1.0 / u
                inv_high = 1.0 / l
                center = 0.5 * (inv_low + inv_high)
                rad    = 0.5 * (inv_high - inv_low)
                Z1 = torch.stack((center, rad)).reshape(2, 1)                       # (2,1)
                return zp.zonotope(Z1)
            return out.compress(1)

        # Near-zero center: gradient becomes uninformative
        near_zero = (c_norm <= eps)

        # Dependent and independent scalar generators via the supporting plane
        if Gd.numel():
            # s_dep_i = grad · g_i
            s_dep = (Gd @ grad)                                                     # (N_dep,)
        else:
            s_dep = torch.zeros((0,), dtype=c.dtype, device=c.device)

        if Gi.numel():
            # t_ind_j = grad · grest_j
            t_ind = (Gi @ grad)                                                     # (N_ind,)
        else:
            t_ind = torch.zeros((0,), dtype=c.dtype, device=c.device)

        # Add one extra independent "radius" generator to make endpoints tight
        sums  = s_dep.abs().sum() + t_ind.abs().sum()                               # ()
        extra = torch.clamp(r - sums, min=0.0)                                      # ()

        # If center is (near) zero, drop linear structure and just use interval radius r
        if near_zero:
            s_dep = torch.zeros_like(s_dep)
            t_ind = torch.zeros_like(t_ind)
            extra = r

        # Assemble scalar PZ:
        # rows: [center; s_dep (N_dep); t_ind (N_ind); extra]
        n_dep = Z.n_dep_gens
        n_ind = Z.n_indep_gens
        total_gens = 1 + n_dep + n_ind + 1
        Zs = torch.zeros((total_gens, 1), dtype=c.dtype, device=c.device)

        # center
        Zs[0, 0] = c_norm
        # keep SAME exponent structure/id mapping for dependent part
        if n_dep:
            Zs[1:1+n_dep, 0] = s_dep
        if n_ind:
            Zs[1+n_dep:1+n_dep+n_ind, 0] = t_ind
        # extra independent radius generator
        Zs[1+n_dep+n_ind, 0] = extra

        out = type(self)(
            Zs,
            n_dep_gens=n_dep,
            expMat=Z.expMat,
            id=Z.id,
            copy_Z=False
        ).compress(1)

        if inv:
            # tight scalar interval from triangle inequality
            l = torch.clamp(c_norm - r, min=0.0)
            u = c_norm + r
            if (l <= 0).item():
                raise ValueError("Reciprocal norm undefined: interval touches zero.")
            inv_low  = 1.0 / u
            inv_high = 1.0 / l
            center = 0.5 * (inv_low + inv_high)
            rad    = 0.5 * (inv_high - inv_low)
            Z1 = torch.stack((center, rad)).reshape(2, 1)                           # (2,1)
            return zp.zonotope(Z1)


    def compress(self, compression_level):
        # Remove zero generators
        if compression_level == 1:
            nonzero_g = torch.sum(self.G != 0, -1) != 0  # non-zero generator index
            G = self.G[nonzero_g]
            expMat = self.expMat[nonzero_g]

        # Remove generators related to redundant exponents
        elif compression_level == 2:
            expMat, G = removeRedundantExponents(self.expMat, self.G)

        else:
            raise ValueError("Can only compress to 1 or 2!")

        # Update self
        self.Z = torch.vstack((self.Z[0], G, self.Z[1 + self.n_dep_gens:]))
        self.expMat = expMat
        self.n_dep_gens = G.shape[0]

        # For chaining
        return self
    

    @property
    def itype(self):
        '''
        The data type of a polynomial zonotope exponent matrix
        return torch.short, torch.int64, torch.long
        '''
        return self.expMat.dtype

    @property
    def dtype(self):
        '''
        The data type of vector elements (ex. center) of a polynomial zonotope
        return torch.float or torch.double
        '''
        return self.Z.dtype

    @property
    def device(self):
        '''
        The device of a polynomial zonotope properties
        return 'cpu', 'cuda:0', or ...
        '''
        return self.Z.device

    @property
    def c(self):
        '''
        The center of a polynimal zonotope
        return <torch.Tensor>
        , shape [nx]
        '''
        return self.Z[0]

    @property
    def G(self):
        '''
        Dependent generators of a polynimal zonotope
        return <torch.Tensor>
        , shape [N, nx]
        '''
        return self.Z[1:self.n_dep_gens + 1]

    @property
    def Grest(self):
        '''
        Independent generators of a polynimal zonotope
        return <torch.Tensor>
        , shape [M, nx]
        '''
        return self.Z[self.n_dep_gens + 1:]

    @property
    def n_generators(self):
        return len(self.Z) - 1

    @property
    def n_indep_gens(self):
        return len(self.Z) - 1 - self.n_dep_gens

    @property
    def dimension(self):
        return self.Z.shape[1]

    @property
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        order = torch.argsort(self.id)
        expMat_sorted = self.expMat[:, order]
        return self.Z, self.n_dep_gens, expMat_sorted, self.id[order]

    def to(self, dtype=None, itype=None, device=None):
        Z = self.Z.to(dtype=dtype, device=device, non_blocking=True)
        expMat = self.expMat.to(dtype=itype, device=device, non_blocking=True)
        return polyZonotope(Z, self.n_dep_gens, expMat, self.id, copy_Z=False, device=device)

    def clone(self):
        return polyZonotope(
            Z=torch.clone(self.Z),
            n_dep_gens=self.n_dep_gens,
            expMat=self.expMat.clone(),
            ids=torch.clone(self.id),
            copy_Z=False  # already cloned above
        )
        
    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        id = self.id.cpu()
        return polyZonotope(Z, self.n_dep_gens, expMat, id, copy_Z=False)

    def __str__(self):
        if self.expMat.numel() == 0:
            expMat_print = torch.tensor([])
        else:
            expMat_print = self.expMat[:, torch.argsort(self.id)]

        pz_str = f"""center: \n{self.c.to(dtype=torch.float)} \n\nnumber of dependent generators: {self.G.shape[-1]} 
            \ndependent generators: \n{self.G.to(dtype=torch.float)}  \n\nexponent matrix: \n {expMat_print.to(dtype=torch.long)}
            \nnumber of independent generators: {self.Grest.shape[-1]} \n\nindependent generators: \n {self.Grest.to(dtype=torch.float)}
            \ndimension: {self.dimension} \ndtype: {self.dtype}\nitype: {self.itype}\ndtype: {self.device}"""

        del_dict = {'tensor': ' ', '    ': ' ', '(': '', ')': ''}
        for del_el in del_dict.keys():
            pz_str = pz_str.replace(del_el, del_dict[del_el])
        return pz_str

    def __repr__(self):
        return pz_repr(self)

    def __add__(self, other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # if other is a polynomial zonotope
        if isinstance(other, polyZonotope):  # exact Plus
            args = _add_genpz_impl(self, other)
            return polyZonotope(*args).compress(2)

        # if other is a vector
        elif isinstance(other, (torch.Tensor, float, int)):
            Z = _add_genzono_num_impl(self, other)
            return polyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False)

        # if other is a zonotope
        elif isinstance(other, zp.zonotope):
            args = _add_genpz_zono_impl(self, other)
            return polyZonotope(*args, copy_Z=False)

        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return self.__add__(-other)

    def __rsub__(self, other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return -self.__sub__(other)

    def __pos__(self):
        return self

    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <polyZonotope>
        return <polyZonotope>
        '''
        return polyZonotope(torch.vstack((-self.Z[:1 + self.n_dep_gens], self.Grest)), self.n_dep_gens, self.expMat, self.id, copy_Z=False)

    def __mul__(self, other):
        # if other is a vector
        if isinstance(other, (torch.Tensor, int, float)):
            Z = _mul_genzono_num_impl(self, other)
            return polyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False).compress(1)

        # if other is a polynomial zonotope
        elif isinstance(other, polyZonotope):
            args = _mul_genpz_impl(self, other)
            return polyZonotope(*args).compress(2)

        else:
            return NotImplemented

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''

        # if other is a matrix
        if isinstance(other, torch.Tensor):
            Z = self.Z @ other.T
            return polyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False).compress(1)
        else:
            return NotImplemented

    def reduce(self, order, option='girard'):
        # extract dimensions
        N = self.dimension
        P = self.n_dep_gens
        Q = self.n_indep_gens

        # number of gens kept (N gens will be added back after reudction)
        K = int(N * order - N)
        # check if the order need to be reduced
        if P + Q > N * order and K >= 0:
            G = self.Z[1:]
            # half the generators length for exponents that are all even
            temp = torch.any(self.expMat % 2, 1)
            ind = (~temp).nonzero().squeeze()
            G[ind] *= 0.5
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2, 1)
            # determine the smallest gens to remove
            ind = torch.argsort(len, descending=True)
            ind_rem, ind_red = ind[:K], ind[K:]
            # split the indices into the ones for dependent and independent
            indDep_red = ind_red[ind_red < P]
            ind_RED = torch.hstack((indDep_red, ind_red[ind_red >= P]))

            indDep_rem = ind_rem[ind_rem < P]
            ind_REM = torch.hstack((indDep_rem, ind_rem[ind_rem >= P]))
            # construct a zonotope from the gens that are removed
            n_dg_red = indDep_red.shape[0]
            Ered = self.expMat[indDep_red]
            Ztemp = torch.vstack((torch.zeros(N, dtype=self.dtype, device=self.device), G[ind_RED]))
            pZtemp = polyZonotope(Ztemp, n_dg_red, Ered, None).compress(1)
            zono = pZtemp.to_zonotope()  # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRem = zono.reduce(1, option)

            # remove the gens that got reduce from the gen matrices
            expMatRem = self.expMat[indDep_rem]
            n_dg_rem = indDep_rem.shape[0]
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.c + zonoRem.center, G[ind_REM], zonoRem.generators))
        else:
            ZRed = self.Z
            n_dg_rem = self.n_dep_gens
            expMatRem = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = (torch.sum(expMatRem, 0) > 0)
        # ind = temp.nonzero().reshape(-1)
        expMatRem = expMatRem[:, ind]
        idRem = self.id[ind]
        if self.dimension == 1:
            ZRed = torch.vstack((ZRed[0], ZRed[1:n_dg_red + 1].sum(0), ZRed[n_dg_red + 1:]))
            n_dg_rem = 1
        return polyZonotope(ZRed, n_dg_rem, expMatRem, idRem, copy_Z=False).compress(1)

    def reduce_indep(self, order, option='girard'):
        if order is None:
            return self
        
        # extract dimensions
        N = self.dimension
        Q = self.n_indep_gens

        # number of gens kept (N gens will be added back after reudction)
        K = int(N * order - N)
        # check if the order need to be reduced
        if Q > N * order and K >= 0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2, 1)
            # determine the smallest gens to remove
            ind = torch.argsort(len, descending=True)
            ind_rem, ind_red = ind[:K], ind[K:]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G[ind_red]), 0)
            Gbox = torch.diag(d)
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.c, self.G, G[ind_rem], Gbox))
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.dimension == 1 and n_dg_red != 1:
            ZRed = torch.vstack((ZRed[0], ZRed[1:n_dg_red + 1].sum(0), ZRed[n_dg_red + 1:]))
            n_dg_red = 1
        return polyZonotope(ZRed, n_dg_red, self.expMat, self.id, copy_Z=False).compress(1)

    def exactCartProd(self, other, merge_ids = True):
        '''
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''
        if isinstance(other, polyZonotope):
            c = torch.hstack((self.c, other.c))
            if merge_ids:
                id, expMat1, expMat2 = mergeExpMatrix(self.id, other.id, self.expMat, other.expMat)
                expMat = torch.vstack((expMat1, expMat2))

            else:
                id = torch.concatenate((self.id, other.id + self.id.max() + 1))
                expMat1 = self.expMat
                expMat2 = other.expMat
                expMat = torch.block_diag(expMat1, expMat2)

            G = torch.block_diag(self.G, other.G)
            Grest = torch.block_diag(self.Grest, other.Grest)
        Z = torch.vstack((c, G, Grest))
        n_dep_gens = self.n_dep_gens + other.n_dep_gens
        return polyZonotope(Z, n_dep_gens, expMat, id).compress(2)

    def lift_to_ndim(self, ndim: int) -> "polyZonotope":

        assert self.dimension == 1, "Can only lift a 1D polyZonotope"
        assert ndim >= 1, "ndim must be >= 1"

        Z_blocks = []
        expMat_blocks = []

        for i in range(ndim):
            Z_i = torch.zeros((self.Z.shape[0], ndim), dtype=self.dtype, device=self.device)
            Z_i[:, i] = self.Z[:, 0]
            Z_blocks.append(Z_i)

            expMat_blocks.append(self.expMat.clone())

        Z_lifted = torch.cat(Z_blocks, dim=0)
        expMat_lifted = torch.cat(expMat_blocks, dim=0)

        return polyZonotope(Z=Z_lifted, n_dep_gens=expMat_lifted.shape[0], expMat=expMat_lifted).compress(2)

    def to_zonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat % 2, 1)
            Gquad = self.G[~ind]
            c = self.c + 0.5 * torch.sum(Gquad, 0)
            Z = torch.vstack((c, self.G[ind], 0.5 * Gquad, self.Grest))
        else:
            Z = self.Z
        return zp.zonotope(Z)

    def to_interval(self, method='interval'):
        if method == 'interval':
            return self.to_zonotope().to_interval()
        else:
            assert False, 'Not implemented'


    def center_slice_all_dep(self, val_slc):
        # Ensure dtype/device
        if not isinstance(val_slc, torch.Tensor):
            val_slc = torch.as_tensor(val_slc, dtype=self.dtype, device=self.device)
        else:
            val_slc = val_slc.to(dtype=self.dtype, device=self.device)

        device = self.device
        exp_dtype = self.expMat.dtype

        tgt_ids = torch.as_tensor(self.id, device=device, dtype=torch.long)  # (n_ids,)
        n_ids = tgt_ids.numel()
        src_width = val_slc.shape[-1]

        # Compile-safe alignment:
        # If already compact/aligned (K == n_ids): identity gather.
        # Otherwise treat val_slc as dense-by-ID and gather by raw IDs with masking.
        if src_width == n_ids:
            gather_idx = torch.arange(n_ids, device=device)
            gather_idx = gather_idx.expand(*val_slc.shape[:-1], n_ids)
            vals = torch.gather(val_slc, dim=-1, index=gather_idx)  # (..., n_ids)
        else:
            idx = tgt_ids
            mask = (idx >= 0) & (idx < src_width)
            safe_idx = idx.clamp_min(0).clamp_max(src_width - 1)
            gather_idx = safe_idx.expand(*val_slc.shape[:-1], n_ids)
            vals = torch.gather(val_slc, dim=-1, index=gather_idx) * mask.to(val_slc.dtype)

        # [..., 1, n_ids]
        vals = vals[..., None, :].to(exp_dtype)

        # alpha: (..., n_dep_gens)
        alpha_coeffs = torch.prod(vals ** self.expMat, dim=-1)

        # offset: (..., dim)
        offset = torch.einsum("...g,...gd->...d", alpha_coeffs, self.G)
        return self.c + offset


    def grad_center_slice_all_dep(self, val_slc):
        # Ensure dtype/device
        if not isinstance(val_slc, torch.Tensor):
            val_slc = torch.as_tensor(val_slc, dtype=self.dtype, device=self.device)
        else:
            val_slc = val_slc.to(dtype=self.dtype, device=self.device)

        device = self.device
        exp_dtype = self.expMat.dtype

        tgt_ids = torch.as_tensor(self.id, device=device, dtype=torch.long)  # (n_ids,)
        n_ids = tgt_ids.numel()
        src_width = val_slc.shape[-1]

        # Align inputs to target id order
        if src_width == n_ids:
            gather_idx = torch.arange(n_ids, device=device)
            gather_idx = gather_idx.expand(*val_slc.shape[:-1], n_ids)
            vals = torch.gather(val_slc, dim=-1, index=gather_idx)  # (..., n_ids)
        else:
            idx = tgt_ids
            mask = (idx >= 0) & (idx < src_width)
            safe_idx = idx.clamp_min(0).clamp_max(src_width - 1)
            gather_idx = safe_idx.expand(*val_slc.shape[:-1], n_ids)
            vals = torch.gather(val_slc, dim=-1, index=gather_idx) * mask.to(val_slc.dtype)

        vals = vals.to(exp_dtype)

        # expMat_red: (n_ids, n_dep_gens, n_ids)
        eye = torch.eye(n_ids, dtype=exp_dtype, device=device)
        expMat_red = self.expMat.unsqueeze(0) - eye.unsqueeze(1)

        # (..., 1, 1, n_ids)
        v_pow = vals[..., None, None, :]

        # alpha per id: (..., n_ids, n_dep_gens)
        alpha_coeffs = torch.prod((v_pow ** expMat_red).nan_to_num(), dim=-1)
        alpha_coeffs = alpha_coeffs * self.expMat.transpose(-1, -2)

        # grad in target-id space: (..., n_ids, dim) -> (..., dim, n_ids)
        grad_tgt = torch.matmul(alpha_coeffs, self.G).transpose(-1, -2)

        # If source width equals n_ids, we are done; else place into dense ID domain
        if src_width == n_ids:
            return grad_tgt
        else:
            # Build placement matrix P: (n_ids, src_width) with one-hot rows at raw IDs
            safe_idx = tgt_ids.clamp_min(0).clamp_max(src_width - 1)
            P = torch.nn.functional.one_hot(safe_idx, num_classes=src_width).to(grad_tgt.dtype)  # (n_ids, src_width)
            mask = ((tgt_ids >= 0) & (tgt_ids < src_width)).to(grad_tgt.dtype).unsqueeze(-1)     # (n_ids, 1)
            P = P * mask
            # Place: (..., dim, n_ids) @ (n_ids, src_width) -> (..., dim, src_width)
            grad_full = torch.matmul(grad_tgt, P)
            return grad_full
        
    # TODO Unverified since update
    def hess_center_slice_all_dep(self, val_slc):
        n_ids = self.id.shape[0]
        val_slc = val_slc[:n_ids]
        expMat = self.expMat[:, torch.argsort(self.id)]
        # a tensor of reduced order expMat for each column
        expMat_red = expMat.unsqueeze(0).repeat(n_ids, 1, 1) - torch.eye(n_ids, dtype=int).unsqueeze(-2)
        expMat_twice_red = expMat.reshape((1, 1) + expMat.shape).repeat(n_ids, n_ids, 1, 1) - torch.eye(
            n_ids, dtype=int).unsqueeze(-2) - torch.eye(n_ids, dtype=int).reshape(n_ids, 1, 1, n_ids)
        expMat_first = expMat.T.unsqueeze(1).repeat(1, n_ids, 1)
        return (self.G * (expMat_first * expMat_red.transpose(-1, -2) * torch.prod(val_slc**expMat_twice_red, dim=-1).nan_to_num()).unsqueeze(-1)).sum(-2).squeeze(-1).transpose(0, -1)

    def slice_all_dep(self, val_slc):
        '''
        Slice polynomial zonotpe in all depdent generators


        id_slc: id to slice
        val_slc: indeterminant to slice
        return,
        c: <torch.Tensor>, shape [nx]
        grad_c: <torch.Tensor>, shape [n_ids,nx]

        '''

        ##################################
        centers = self.center_slice_all_dep(val_slc)
        if len(centers.shape) > 1:
            Z = torch.cat((centers.unsqueeze(-2), self.Grest.repeat(*centers.shape[:-1], 1, 1)))
            return zp.batchZonotope(Z)
        return zp.zonotope(torch.vstack((self.center_slice_all_dep(val_slc), self.Grest)))

    def slice_dep(self, slice_ids, val_slc):
        slice_ids: torch.Tensor = torch.atleast_1d(torch.as_tensor(slice_ids))
        val_slc: torch.Tensor = torch.atleast_1d(torch.as_tensor(val_slc))

        slice_ids = slice_ids.to(dtype=torch.long, device=self.device).flatten()
        val_slc = val_slc.to(dtype=self.dtype, device=self.device).flatten()

        # Build mask for which IDs to slice
        # This will give us a mask of shape [n_ids] indicating which expMat columns to slice
        is_slice = (self.id[None, :] == slice_ids[:, None]).any(dim=0)

        if not torch.any(is_slice):
            return self  # Nothing to slice

        # Get values aligned to expMat columns to be sliced
        idx_cols = torch.nonzero(is_slice, as_tuple=False).squeeze(1)
        matched_ids = matched_ids = self.id[idx_cols]
        val_slc_aligned = torch.zeros_like(matched_ids, dtype=self.dtype)
        for i, slice_id in enumerate(matched_ids):
            val_slc_aligned[i] = val_slc[(slice_ids == slice_id).nonzero(as_tuple=False)[0, 0]]

        # Compute new center offset by evaluating monomials
        exponents = self.expMat[:, is_slice]
        coeffs = torch.prod(val_slc_aligned.unsqueeze(0) ** exponents, dim=1)
        offset = coeffs @ self.G

        new_c = self.c.clone() + offset

        # Remove sliced columns
        expMat_unsliced = self.expMat[:, ~is_slice]

        # Remove collapsed generators (i.e. all exponents zero)
        keep = torch.any(expMat_unsliced != 0, dim=1)
        new_G = self.G[keep]
        new_expMat = expMat_unsliced[keep]
        if new_expMat.shape[1] == 0:
            new_expMat = torch.eye(0).to(new_expMat)

        new_Z = torch.vstack((new_c, new_G, self.Grest))
        return polyZonotope(new_Z, new_G.shape[0], new_expMat).compress(2)
    
    def deleteZerosGenerators(self, eps=0):
        expMat, G = removeRedundantExponents(self.expMat, self.G)
        ind = torch.sum(expMat, 1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind], 0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c

        id = self.id
        ind = torch.sum(expMat, 0) == 0
        if torch.any(ind):
            expMat = expMat[:, ~ind]
            id = id[~ind]
        return polyZonotope(torch.vstack((c, G, self.Grest)), G.shape[0], expMat, id, copy_Z=False)

    def project(self, dim=[0, 1]):
        if isinstance(dim, int):
            dim = [dim]
        return polyZonotope(self.Z[:, dim], self.n_dep_gens, self.expMat, self.id, copy_Z=False).compress(1)
    '''
    def plot(self,dim=[0,1]):
        pz = self.project(dim)
    '''

    def split_dep_indep(self, center_on_dep=True):
        Z_dep = torch.clone(self.Z[:self.n_dep_gens + 1])
        Z_indep = torch.clone(self.Z[-(self.n_indep_gens + 1):])
        Z_indep[0] *= 0
        if not center_on_dep:
            Z_indep[0] += Z_dep[0]
            Z_dep[0] += 0
        deps = polyZonotope(Z_dep, self.n_dep_gens, self.expMat, self.id, copy_Z=False)
        indeps = zp.zonotope(Z_indep)
        return deps, indeps

    @staticmethod
    def zeros(dims, dtype=None, device=None):
        Z = torch.zeros((1, dims), dtype=dtype, device=device)
        expMat = torch.empty((0, 0), dtype=torch.int64, device=device)
        id = torch.empty(0, dtype=torch.int64, device=Z.device)
        return zp.polyZonotope(Z, 0, expMat=expMat, ids=id, copy_Z=False)

    @staticmethod
    def ones(dims, dtype=None, device=None):
        Z = torch.ones((1, dims), dtype=dtype, device=device)
        expMat = torch.empty((0, 0), dtype=torch.int64, device=device)
        id = torch.empty(0, dtype=torch.int64, device=Z.device)
        return zp.polyZonotope(Z, 0, expMat=expMat, ids=id, copy_Z=False)

    def cross(
        self,
        other: "polyZonotope",
        reduce_order: int = None,
        reduce_option: str = "girard",
        reduce_indep_order: int = None
    ) -> "polyZonotope":
        """
        Compute the cross product of two 3D polyZonotopes, with optional reduction.

        Args:
            other (polyZonotope): Another 3D polyZonotope.
            reduce_order (float, optional): Order for dependent generator reduction.
            reduce_option (str, optional): Reduction method, e.g., 'girard'.
            reduce_indep_order (float, optional): Order for independent generator reduction.

        Returns:
            polyZonotope: Resulting 3D polyZonotope.
        """
        def _validate_and_split(arg):
            if isinstance(arg, polyZonotope):
                assert arg.dimension == 3
                return arg.project(0), arg.project(1), arg.project(2)
            elif isinstance(arg, torch.Tensor):
                assert arg.numel() == 3
                return arg.flatten()
        
        a0, a1, a2 = _validate_and_split(self)
        b0, b1, b2 = _validate_and_split(other)
        
        # Compute vector components of the cross product
        s0 = a1 * b2 - a2 * b1
        s1 = a2 * b0 - a0 * b2
        s2 = a0 * b1 - a1 * b0

        # Stack into full 3D vector via cartesian product
        pz_out = polyZonotope.direct_product(s0, s1, s2)

        # Optional generator reduction
        if reduce_order is not None:
            pz_out = pz_out.reduce(order=reduce_order, option=reduce_option)

        if reduce_indep_order is not None:
            pz_out = pz_out.reduce_indep(order=reduce_indep_order, option=reduce_option)

        return pz_out
    
    @staticmethod
    def direct_product(*pzs: "zp.polyZonotope") -> "polyZonotope":
        output = pzs[0]
        for pz in pzs[1:]:
            output = output.exactCartProd(pz, False)
        return output