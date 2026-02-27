"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA, Patrick Holme's implementation
"""
from typing import List, Union
from zonopy.contset.polynomial_zonotope.utils import removeRedundantExponentsBatch, mergeExpMatrix
from zonopy.contset.polynomial_zonotope.poly_zono import polyZonotope
import zonopy as zp
import torch
from ..gen_ops import (
    _add_genpz_impl,
    _add_genzono_num_impl,
    _add_genpz_zono_impl,
    _mul_genpz_impl,
    _mul_genzono_num_impl,
    _matmul_genmpz_impl,
    )
import zonopy.internal as zpi

class batchPolyZonotope:
    r''' Batched 1D polynomial zonotope

    Batched form of the :class:`polyZonotope` class.
    This class is used to represent a batch of polynomial zonotopes over the same domain
    with arbitrary batch dimensions.
    It follows a similar formulation from the :class:`polyZonotope` class as the
    :class:`batchZonotope` class did from :class:`zonotope`.

    This results in a :math:`\mathbf{Z} \in \mathbb{R}^{B_1 \times B_2 \times \cdots \times B_b \times (N+M+1) \times d}` tensor

    Refer to the :class:`polyZonotope` class for more information polynomial zonotops.
    '''
    # NOTE: property for mat pz
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,copy_Z=True, dtype=None, device=None):
        r''' Constructor for the batchPolyZonotope class
        
        Args:
            Z (torch.Tensor): The center and generator matrix of the polynomial zonotope.
                The shape of Z should be :math:`(B_1, B_2, \cdots, B_b, N+M+1, d)` where :math:`B_1, B_2, \cdots, B_b` are the batch dimensions,
                :math:`N` is the number of dependent generators, :math:`M` is the number of independent generators, and :math:`d` is the dimension of the zonotope.
            n_dep_gens (int, optional): The number of dependent generators in the polynomial zonotope. Default is 0.
            expMat (torch.Tensor, optional): The exponent matrix of the dependent generators. If ``None``, it will be the identity matrix. Default: None
            id (torch.Tensor, optional): The integer identifiers for the dependent generators. If ``None``, it will be the range of the number of dependent generators. Default: None
            copy_Z (bool, optional): If ``True``, it will copy the input ``Z`` value. Default: ``True``
            dtype (torch.dtype, optional): The data type of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``
            device (torch.device, optional): The device of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``

        Raises:
            AssertionError: If the dimension of Z input is not 3 or more.
            AssertionError: If the exponent matrix does not seem to be valid for the given dependent generators or ids.
            AssertionError: If the number of dependent generators does not match the number of ids.
            AssertionError: If the exponent matrix is not a non-negative integer matrix.
        '''
        # If compress=2, it will always copy.

        # Make sure Z is a tensor and shaped right
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        assert len(Z.shape) > 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'

        self.batch_dim = len(Z.shape) - 2
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])

        # Make an expMat and id if not given
        if expMat is None and id is None:
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.id = torch.arange(self.expMat.shape[1],dtype=torch.int64, device=Z.device)
            
        # Otherwise make sure expMat is right
        elif expMat is not None:
            #check correctness of user input
            expMat = torch.as_tensor(expMat,dtype=torch.long,device=Z.device)
            assert expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.'
            if zpi.__debug_extra__: assert torch.all(expMat >= 0), 'Invalid exponent matrix.' 
            
            self.expMat = expMat
            
            # Make sure ID is right
            if id is not None:
                self.id = torch.as_tensor(id, dtype=torch.int64, device=Z.device).flatten()
            else:
                self.id = torch.arange(self.expMat.shape[1],dtype=torch.int64, device=Z.device)
                
        # Otherwise ID is given, but not the expMat, so make identity
        else:
            self.id = torch.as_tensor(id, dtype=torch.int64, device=Z.device).flatten()
            assert len(self.id) == n_dep_gens, 'Number of dependent generators must match number of id\'s!'
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device)

        # Copy the Z if requested
        if copy_Z:
            self.Z = torch.clone(Z)
        # Or save it itself
        else:
            self.Z = Z
        self.n_dep_gens = n_dep_gens
        assert self.id.device == self.Z.device

    def compress(self, compression_level):
        # Remove zero generators
        if compression_level == 1:
            nonzero_g = torch.sum(self.G!=0,tuple(range(self.batch_dim))+(-1,))!=0 # non-zero generator index
            G = self.G[...,nonzero_g,:]
            expMat = self.expMat[nonzero_g]

        # Remove generators related to redundant exponents
        elif compression_level == 2: 
            expMat, G = removeRedundantExponentsBatch(self.expMat, self.G, [])

        else:
            raise ValueError("Can only compress to 1 or 2!")

        # Update self
        self.Z = torch.cat((self.c.unsqueeze(-2), G, self.Grest), dim=-2)
        self.expMat = expMat
        self.n_dep_gens = G.shape[-2]

        # For chaining
        return self

    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 2:
            return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        else:
            return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        
    @property 
    def batch_shape(self):
        return self.Z.shape[:-2]
    @property
    def itype(self):
        return self.expMat.dtype
    @property 
    def dtype(self):
        return self.Z.dtype 
    @property
    def device(self):
        return self.Z.device
    @property 
    def c(self):
        return self.Z[self.batch_idx_all+(0,)]
    @property 
    def G(self):
        return self.Z[self.batch_idx_all+(slice(1,self.n_dep_gens+1),)]
    @property 
    def Grest(self):
        return self.Z[self.batch_idx_all+(slice(self.n_dep_gens+1,None),)]
    @property
    def n_generators(self):
        return self.Z.shape[-2]-1
    @property
    def n_indep_gens(self):
        return self.Z.shape[-2]-1-self.n_dep_gens
    @property 
    def dimension(self):
        return self.Z.shape[-1]
    @property 
    def shape(self):
        return self.Z.shape[-1:]
    @property 
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        expMat_sorted = self.expMat[:,order] 
        return self.Z, self.n_dep_gens, expMat_sorted, id_sorted

    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device, non_blocking=True)
        expMat = self.expMat.to(dtype=itype,device=device, non_blocking=True)
        # id = self.id.to(device=device)
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)

    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        # id = self.id.cpu()
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)

    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # if other is a vector
        if  isinstance(other, (torch.Tensor, float, int)):
            Z = _add_genzono_num_impl(self, other)
            return batchPolyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False)

        # if other is a polynomial zonotope
        elif isinstance(other, (polyZonotope, batchPolyZonotope)): # exact Plus
            args = _add_genpz_impl(self, other, batch_shape=self.batch_shape)
            return batchPolyZonotope(*args).compress(2)
        
        # if other is a zonotope
        elif isinstance(other, (zp.zonotope, zp.batchZonotope)):
            args = _add_genpz_zono_impl(self, other)
            return batchPolyZonotope(*args, copy_Z=False)

        else:
            return NotImplemented
        
    __radd__ = __add__

    def __sub__(self,other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return self.__add__(-other)
    
    def __rsub__(self,other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return -self.__sub__(other)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        # center + dependent
        head = -self.Z[..., :1 + self.n_dep_gens, :]
        # independent generators
        tail = self.Z[..., 1 + self.n_dep_gens:, :]
        Zneg = torch.cat((head, tail), dim=-2)
        return batchPolyZonotope(
            Zneg,
            n_dep_gens=self.n_dep_gens,
            expMat=self.expMat,
            id=self.id,
            copy_Z=False,
            dtype=Zneg.dtype,
            device=Zneg.device,
        )
        
    def __mul__(self,other):
        # if other is a vector
        if isinstance(other,(torch.Tensor,int,float)):
            Z = _mul_genzono_num_impl(self, other, batch_shape=self.batch_shape)
            return batchPolyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False)

        # if other is a polynomial zonotope or batch polynomial zonotope
        elif isinstance(other,(polyZonotope,batchPolyZonotope)):
            args = _mul_genpz_impl(self, other)
            return batchPolyZonotope(*args).compress(2)
        
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        
        # if other is a matrix
        if isinstance(other, torch.Tensor):            
            Z = self.Z@other.transpose(-2,-1)
            return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1) # TODO IS THIS RIGHT?
        
        if isinstance(other, zp.matPolyZonotope):
            # Shim self to batchMatPolyZono and return that matmul
            shim_self = zp.batchMatPolyZonotope(self.Z.unsqueeze(-1),self.n_dep_gens,self.expMat,self.id,copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(other, shim_self)
            return zp.batchPolyZonotope(Z.squeeze(-1), n_dep_gens, expMat, id).compress(2)
        
        else:
            return NotImplemented
    
    # def __len__(self):
    #     return self.Z.shape[0]

    # NOTE - this is a shim for reducing each individual pz in the batch
    def reduce(self, order, option='girard'):
        batch_shape = tuple(self.batch_shape)
        len_ents = int(torch.tensor(batch_shape).prod().item())
        idx_flat = torch.arange(len_ents)
        idx_tuple = torch.unravel_index(idx_flat, batch_shape)
        pzlist = [None] * len_ents
        for out_i, idxs in enumerate(zip(*(t.tolist() for t in idx_tuple))):
            pzlist[out_i] = self[idxs].reduce(order, option=option)
        return zp.batchPolyZonotope.from_pzlist(pzlist, batch_shape=self.batch_shape)
    
    # TODO Inspect for speedup?
    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,-1) # NOTE -1
            # determine the smallest gens to remove            
            ind = torch.argsort(len,dim=-1,descending=True).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            ind_rem, ind_red = ind[self.batch_idx_all+(slice(K),)], ind[self.batch_idx_all+(slice(K,None),)]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G.gather(-2,ind_red)),-2)
            Gbox = torch.diag_embed(d)
            # add the reduced gens as new indep gens
            ZRed = torch.cat((self.c.unsqueeze(-2),self.G,G.gather(-2,ind_rem),Gbox),dim=-2)
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.dimension == 1 and n_dg_red != 1:            
            ZRed = torch.cat((ZRed[self.batch_idx_all+(0,)],ZRed[self.batch_idx_all+(slice(1,n_dg_red+1),)].sum(-2).unsqueeze(-2),ZRed[self.batch_idx_all+(slice(n_dg_red+1,None),)]),dim=-2)
            n_dg_red = 1
        return batchPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,copy_Z=False)

    def exactCartProd(self, other, merge_ids: bool = True):
        '''
        self: <batchPolyZonotope>
        other: <polyZonotope> | <batchPolyZonotope>
        return <batchPolyZonotope>
        '''

        # ----- Centers -----
        if isinstance(other, polyZonotope):
            # other.c has no batch; expand to our batch and hstack
            c = torch.hstack([self.c, other.c.expand(self.c.shape[:-1] + (other.dimension,))]).unsqueeze(-2)
        elif isinstance(other, batchPolyZonotope):
            c = torch.hstack((self.c, other.c)).unsqueeze(-2)
        else:
            raise TypeError(f"exactCartProd expects polyZonotope or batchPolyZonotope, got {type(other)}")

        # ----- expMat / id -----
        if merge_ids:
            # Align shared ids, stack exponent rows
            id_, expMat1, expMat2 = mergeExpMatrix(self.id, other.id, self.expMat, other.expMat)
            expMat = torch.vstack((expMat1, expMat2))
        else:
            # Keep ids distinct by offsetting other's ids; build block-diagonal expMat
            # Note: self.id/other.id are 1D (no batch); expMat is 2D (no batch)
            offset = self.id.max().item() + 1 if len(self.id.flatten()) > 0 else 0
            id_ = torch.concatenate((self.id, other.id + offset))
            expMat = torch.block_diag(self.expMat, other.expMat)

        # ----- Dependent generators -----
        n_dep_gens = self.n_dep_gens + other.n_dep_gens

        # ----- G (dependent) -----
        new_G_shape = self.G.shape[:-2] + (
            self.G.shape[-2] + other.G.shape[-2],
            self.G.shape[-1] + other.G.shape[-1],
        )
        G = torch.zeros(new_G_shape, dtype=self.dtype, device=self.device)

        g1_slice = self.batch_idx_all + (slice(None, self.G.shape[-2]), slice(None, self.G.shape[-1]))
        g2_slice = self.batch_idx_all + (slice(self.G.shape[-2], None), slice(self.G.shape[-1], None))
        G[g1_slice] = self.G
        G[g2_slice] = other.G

        # ----- Grest (independent) -----
        new_Grest_shape = self.Grest.shape[:-2] + (
            self.Grest.shape[-2] + other.Grest.shape[-2],
            self.Grest.shape[-1] + other.Grest.shape[-1],
        )
        Grest = torch.zeros(new_Grest_shape, dtype=self.dtype, device=self.device)

        g1_slice = self.batch_idx_all + (slice(None, self.Grest.shape[-2]), slice(None, self.Grest.shape[-1]))
        g2_slice = self.batch_idx_all + (slice(self.Grest.shape[-2], None), slice(self.Grest.shape[-1], None))
        Grest[g1_slice] = self.Grest
        Grest[g2_slice] = other.Grest

        # ----- Pack & return -----
        Z = torch.cat((c, G, Grest), dim=-2)
        return batchPolyZonotope(Z, n_dep_gens, expMat, id_).compress(2)

    def to_batchZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[self.batch_idx_all+(~ind,)]
            c = self.c + 0.5*torch.sum(Gquad,-2)
            Z = torch.cat((c.unsqueeze(-2), self.G[self.batch_idx_all+(ind,)],0.5*Gquad,self.Grest),-2)
        else: 
            Z = self.Z
        return zp.batchZonotope(Z)

    def to_interval(self,method='interval'):
        if method == 'interval':
            return self.to_batchZonotope().to_interval()
        else:
            assert False, 'Not implemented'

    # TODO Inspect for speedup?
    def slice_dep(self,id_slc,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to dlice
        val_slc: indeterminant to slice
        '''
        if isinstance(id_slc,(int,list)):
            if isinstance(id_slc,int):
                id_slc = [id_slc]
            id_slc = torch.tensor(id_slc)
        if isinstance(val_slc,(int,float,list)):
            if isinstance(val_slc,(int,float)):
                val_slc = [val_slc]
            val_slc = torch.tensor(val_slc,dtype=self.dtype,device=self.device)
        
        if any(abs(val_slc)>1):
            import pdb; pdb.set_trace()
        #assert all(val_slc<=1) and all(val_slc>=-1), 'Indereminant should be in [-1,1].'
        
        id_slc, val_slc = id_slc.reshape(-1,1), val_slc.reshape(1,-1)
        order = torch.argsort(id_slc.reshape(-1))
        id_slc = id_slc.flatten()[order]
        val_slc = val_slc[:, order]

        # boolean matches
        ind = torch.any(self.id == id_slc, dim=0)  # corresponding id for self.id  
        ind2 = torch.any(self.id == id_slc, dim=1) # corresponding id for id_slc
        #assert ind.numel()==len(id_slc), 'Some specidied IDs do not exist!'
        if ind.shape[0] != 0:
            G = self.G*torch.prod(val_slc[:,ind2]**self.expMat[:,ind],dim=1)
            expMat = self.expMat[:,~ind]
            id = self.id[:,~ind]
        else:
            G = self.G
            expMat = self.expMat
            id = self.id

        #expMat, G = removeRedundantExponents(expMat,G)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        '''
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        '''
        
        if G.shape[0] == 0 and self.Grest.shape[0] == 0:
            return polyZonotope(c,0,expMat,id).compress(2)
        else:
            return polyZonotope(torch.vstack((c,G,self.Grest)), G.shape[0],expMat,id).compress(2)
    
    @staticmethod
    def _align_vals_to_ids(val_slc: torch.Tensor,
                           val_slc_ids: torch.Tensor,
                           target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder the last-dim of val_slc (indexed by val_slc_ids) to match target_ids.
        Returns (val_slc_reordered, index_map) where index_map maps target_ids -> positions in val_slc.
        All ops are tensor-only and torch.compile-friendly.
        """
        device = val_slc.device
        tid = target_ids.to(device=device)
        vid = val_slc_ids.to(device=device)

        # Sort val_slc_ids once, then searchsorted for target_ids
        sorted_vals, sorted_idx = torch.sort(vid)                      # (n_ids,)
        pos_in_sorted = torch.searchsorted(sorted_vals, tid)           # (n_ids,)
        # Map back to original positions of val_slc
        index_map = sorted_idx.gather(0, pos_in_sorted)                # (n_ids,)

        # Optional safety mask (no control-flow): mismatches zero out contribution
        match_mask = (sorted_vals.gather(0, pos_in_sorted) == tid)     # (n_ids,)
        # Reorder values; mask mismatches to 0 to avoid OOB semantics
        vals = torch.index_select(val_slc, dim=-1, index=index_map)    # (..., n_ids)
        vals = vals * match_mask.to(vals.dtype)                        # (..., n_ids)
        return vals, index_map

    def center_slice_all_dep(self, val_slc: torch.Tensor) -> torch.Tensor:
        device = self.device

        tgt_ids = torch.as_tensor(self.id, device=device, dtype=torch.long)
        n_ids = tgt_ids.numel()
        src_width = val_slc.shape[-1]

        # Choose gather indices without changing the signature
        # Case 1: already compact/ordered like self.id -> identity indices
        # Case 2: dense-by-ID layout -> gather by raw IDs (with masking)
        if src_width == n_ids:
            idx = torch.arange(n_ids, device=device)
            vals = torch.gather(val_slc, -1, idx.expand(*val_slc.shape[:-1], n_ids))
        else:
            idx = tgt_ids
            mask = (idx >= 0) & (idx < src_width)
            safe_idx = idx.clamp_min(0).clamp_max(src_width - 1)
            vals = torch.gather(val_slc, -1, safe_idx.expand(*val_slc.shape[:-1], n_ids))
            vals = vals * mask.to(vals.dtype)

        v = vals[..., None, :]
        alpha = torch.prod(v ** self.expMat, dim=-1)
        offset = torch.einsum("...g,...gd->...d", alpha.float(), self.G.float())
        return self.c + offset.squeeze(-2)

    def grad_center_slice_all_dep(self, val_slc: torch.Tensor) -> torch.Tensor:
        device = self.device

        tgt_ids = torch.as_tensor(self.id, device=device, dtype=torch.long)
        n_ids = tgt_ids.numel()
        src_width = val_slc.shape[-1]

        if src_width == n_ids:
            idx = torch.arange(n_ids, device=device)
            vals = torch.gather(val_slc, -1, idx.expand(*val_slc.shape[:-1], n_ids))
        else:
            idx = tgt_ids
            mask = (idx >= 0) & (idx < src_width)
            safe_idx = idx.clamp_min(0).clamp_max(src_width - 1)
            vals = torch.gather(val_slc, -1, safe_idx.expand(*val_slc.shape[:-1], n_ids))
            vals = vals * mask.to(vals.dtype)

        eye = torch.eye(n_ids, device=device)
        exp_red = self.expMat.unsqueeze(0) - eye.unsqueeze(1)  # (n_ids, n_dep_gens, n_ids)

        v_pow = vals[..., None, None, :]
        alpha = torch.prod((v_pow ** exp_red).nan_to_num(), dim=-1)  # (..., n_ids, n_dep_gens)
        alpha = alpha * self.expMat.transpose(-1, -2)                # (..., n_ids, n_dep_gens)

        grad_id_dim = torch.matmul(alpha, self.G)                    # (..., n_ids, dim)
        grad = grad_id_dim.transpose(-1, -2)                         # (..., dim, n_ids)
        return grad
    # TODO Unverified since update
    def hess_center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        n_vals = val_slc.shape[-1]
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].reshape(self.batch_shape+(1,1,1,n_ids)) # b1, b2,..., 1, 1, 1, n_ids
        expMat = self.expMat[:, torch.argsort(self.id)]
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) # a tensor of reduced order expMat for each column
        expMat_twice_red = expMat.reshape((1,1)+expMat.shape).repeat(n_ids,n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) - torch.eye(n_ids,dtype=int).reshape(n_ids,1,1,n_ids)
        expMat_first = expMat.T.unsqueeze(1).repeat(1,n_ids,1)
        hess = torch.zeros(self.batch_shape+(self.dimension,n_vals,n_vals),dtype=self.dtype,device=self.device)
        hess[self.batch_idx_all+(slice(None),slice(n_ids),slice(n_ids))] = ((expMat_first*expMat_red.transpose(-1,-2)*torch.prod(val_slc**expMat_twice_red,dim=-1).nan_to_num())@self.G.unsqueeze(-3)).transpose(-3,-1)
        return hess


    def slice_all_dep(self,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to slice
        val_slc: indeterminant to slice
        c: <torch.Tensor>, shape [b1,b2,...,nx]
        grad_c: <torch.Tensor>, d c/d val_slc
        shape [b1,b2,...,nx,n_ids]
        '''

        ##################################
        return zp.batchZonotope(torch.cat((self.center_slice_all_dep(val_slc).unsqueeze(-2),self.Grest), -2))


    def deleteZerosGenerators(self,eps=0):
        expMat, G = removeRedundantExponentsBatch(self.expMat,self.G,self.batch_idx_all)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        return polyZonotope(torch.vstack((c,G,self.Grest)),G.shape[0],expMat,id,copy_Z=False).compress(2)

    def select_batch_entry(self, batch_dim, value, keepdim=False):
        if isinstance(value, int):
            new_Z = torch.select(self.Z, batch_dim, value)
            if keepdim:
                new_Z = new_Z.unsqueeze(batch_dim)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            start, end = value
            length = end - start
            new_Z = torch.narrow(self.Z, batch_dim, start, length)
        elif isinstance(value, slice):
            new_Z = self.Z[(slice(None),) * batch_dim + (value,)]
        else:
            raise TypeError("value must be int, (start, end), or slice")

        return zp.batchPolyZonotope(
            new_Z,
            self.n_dep_gens,
            self.expMat,
            self.id,
            copy_Z=False
        )

    def project(self,dim=[0,1]):
        if isinstance(dim, int):
            dim = [dim]
        Z = self.Z[self.batch_idx_all+(slice(None),dim)]
        return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
    
    def split_dep_indep(self, center_on_dep=True):
        Z_dep = torch.clone(self.Z[...,:self.n_dep_gens+1,:])
        Z_indep = torch.clone(self.Z[...,-(self.n_indep_gens+1):,:])
        Z_indep[...,0,:] *= 0
        if not center_on_dep:
            Z_indep[...,0,:] += Z_dep[...,0,:]
            Z_dep[...,0,:] += 0
        deps = batchPolyZonotope(Z_dep,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        indeps = zp.batchZonotope(Z_indep)
        return deps, indeps

    @staticmethod
    def from_pzlist(pzlist: List["zp.polyZonotope"], batch_shape=None):
        assert len(pzlist) > 0, "Expected at least 1 element input!"
        # Check type
        assert all([isinstance(pz, polyZonotope) for pz in pzlist]), "Expected all elements to be of type polyZonotope"
        # Validate dimensions match
        n_pz = len(pzlist)
        dim = pzlist[0].dimension
        dtype = pzlist[0].dtype
        device = pzlist[0].device
        [pz.dimension for pz in pzlist].count(dim) == n_pz, "Expected all elements to have the same dimensions!"

        # First loop to extract key parts
        all_ids = [None]*n_pz
        dep_gens = [None]*n_pz
        all_c = [None]*n_pz
        n_grest = [None]*n_pz
        for i, pz in enumerate(pzlist):
            all_ids[i] = pz.id
            dep_gens[i] = pz.n_dep_gens
            all_c[i] = pz.c.unsqueeze(0)
            n_grest[i] = pz.n_indep_gens
        
        all_ids = torch.unique(torch.cat(all_ids, dim=0))
        all_dep_gens = torch.sum(torch.tensor(dep_gens))
        dep_gens_idxs = torch.cumsum(torch.tensor([0] + dep_gens), dim=0)
        n_grest = torch.max(torch.tensor(n_grest))
        all_c = torch.stack(all_c)

        # Preallocate
        all_G = torch.zeros((n_pz, all_dep_gens, dim), dtype=dtype, device=device)
        all_grest = torch.zeros((n_pz, n_grest, dim), dtype=dtype, device=device)
        all_expMat = torch.zeros((all_dep_gens, len(all_ids)), dtype=torch.int64, device=device)
        last_expMat_idx = 0

        # expand remaining values
        for pzid in range(n_pz):
            # Expand ExpMat (replace any with nonzero to fix order bug!)
            matches = torch.nonzero(
                pzlist[pzid].id.unsqueeze(1) == all_ids, as_tuple=False
            )[:, 1]            
            end_idx = last_expMat_idx + pzlist[pzid].expMat.shape[0]
            all_expMat[last_expMat_idx:end_idx,matches] = pzlist[pzid].expMat
            last_expMat_idx = end_idx
        
            # expand out all G matrices
            all_G[pzid,dep_gens_idxs[pzid]:dep_gens_idxs[pzid+1]] = pzlist[pzid].G

            # Expand out all grest
            grest = pzlist[pzid].Grest
            all_grest[pzid,:grest.shape[0]] = grest
        
        # Combine, reduce, output.
        Z = torch.concat((all_c, all_G, all_grest), dim=-2)
        if batch_shape is not None:
            Z = Z.reshape(batch_shape + Z.shape[-2:])
        out = zp.batchPolyZonotope(Z, all_dep_gens, all_expMat, all_ids, copy_Z=False).compress(2)
        return out
    
    @staticmethod
    def combine_bpz(bpzlist, idxs):
        # Takes a list of bpz and respective idxs for them and combines them appropriately
        size = torch.cat(idxs).max().item() + 1
        out_list = [None] * size
        for i, locations in enumerate(idxs):
            out_list[locations] = [bpzlist[i][j] for j in range(len(locations))]
        return zp.batchPolyZonotope.from_pzlist(out_list)

    @staticmethod
    def zeros(batch_size, dims, dtype=None, device=None):
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)
        Z = torch.zeros((1, dims), dtype=dtype, device=device).expand(*batch_size, -1, -1)
        expMat = torch.empty((0, 0), dtype=torch.int64, device=device)
        id = torch.empty(0, dtype=torch.int64, device=device)
        return zp.batchPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)

    @staticmethod
    def ones(batch_size, dims, dtype=None, device=None):
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)
        Z = torch.ones((1, dims), dtype=dtype, device=device).expand(*batch_size, -1, -1)
        expMat = torch.empty((0, 0), dtype=torch.int64, device=device)
        id = torch.empty(0, dtype=torch.int64, device=device)
        return zp.batchPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    
    @staticmethod
    def direct_product(*pzs: "batchPolyZonotope", merge_ids=False) -> "batchPolyZonotope":
        """
        Batched Cartesian product of multiple batchPolyZonotopes.
        Like polyZonotope.direct_product, but supports batch dims.

        Args:
            *pzs: sequence of batchPolyZonotope objects with broadcastable batch shapes.

        Returns:
            batchPolyZonotope: combined product zonotope.
        """
        assert len(pzs) >= 1, "Need at least one zonotope"
        output = pzs[0]
        for pz in pzs[1:]:
            assert isinstance(pz, batchPolyZonotope)
            output = output.exactCartProd(pz, merge_ids=merge_ids)
        return output
    
    def cross(
            self,
            other: Union["batchPolyZonotope", torch.Tensor],
            reduce_order: int = None,
            reduce_option: str = "girard",
            reduce_indep_order: int = None,
        ) -> "batchPolyZonotope":
        """
        Cross product of two 3D batchPolyZonotopes or a batchPolyZonotope with a 3D torch.Tensor.
        Naive stacking: uses direct_product to block-diagonalize components.

        Args:
            other: batchPolyZonotope or torch.Tensor of shape (3,) or broadcastable to batch.

        Returns:
            batchPolyZonotope: 3D vector zonotope result.
        """

        def _validate_and_split(arg):
            if isinstance(arg, batchPolyZonotope):
                assert arg.dimension == 3, "cross is defined for 3D vectors"
                return arg.project([0]), arg.project([1]), arg.project([2])
            elif isinstance(arg, torch.Tensor):
                arg = arg.to(self.Z.device, dtype=self.Z.dtype)
                assert arg.shape[-1] == 3 and arg.ndim <= self.Z.ndim, \
                    "tensor must be a 3D vector, possibly broadcastable over batch"
                comps = []
                for i in range(3):
                    # shape: [..., 1] -> make Z with center and no generators
                    ci = arg[..., i].unsqueeze(-1)
                    comps.append(ci)
                return tuple(comps)
            else:
                raise TypeError("cross expects batchPolyZonotope or 3D torch.Tensor")

        a0, a1, a2 = _validate_and_split(self)
        b0, b1, b2 = _validate_and_split(other)

        # Scalar components (batchPolyZonotope each of dimension 1)
        s0 = a1 * b2 - a2 * b1
        s1 = a2 * b0 - a0 * b2
        s2 = a0 * b1 - a1 * b0

        # Naive 3D stack
        pz_out = batchPolyZonotope.direct_product(s0, s1, s2, merge_ids=True)

        if reduce_order is not None:
            pz_out = pz_out.reduce(order=reduce_order, option=reduce_option)
        if reduce_indep_order is not None:
            pz_out = pz_out.reduce_indep(order=reduce_indep_order, option=reduce_option)

        return pz_out

    def norm(self, ord=2, dims=None, inv=False):
        """
        Sound batchPolyZonotope enclosure of ||x||_p for x in Z (or its reciprocal).
        Keeps dependent structure via a linear (supporting) term and adds one
        independent "radius" generator so that the interval endpoints match the
        sharp triangle-inequality bounds.

        Args:
            ord: p in ||.||_p (float/int as in torch.linalg.vector_norm).
            dims: int or iterable of indices to project before taking the norm.
            inv: if True, return 1/||.||_p as a batchZonotope interval (requires
                the lower bound > 0).

        Returns:
            If inv is False:
                batchPolyZonotope of shape batch_shape + (n_dep_gens' + n_indep_gens' + 1, 1)
            If inv is True:
                batchZonotope of shape batch_shape + (2, 1) encoding the interval.
        """
        import torch
        import zonopy as zp

        Z = self.project(dims) if dims is not None else self

        c = Z.c                           # [..., d]
        Gd = Z.G                          # [..., N_dep, d]
        Gi = Z.Grest                      # [..., N_indep, d]
        batch_idx = Z.batch_idx_all

        # Center norm and per-generator p-norms for triangle bound
        p = ord
        # torch.linalg.vector_norm handles p as float/int; keep dim=-1
        c_norm = torch.linalg.vector_norm(c, ord=p, dim=-1)                  # [...]
        r_dep = torch.linalg.vector_norm(Gd, ord=p, dim=-1).sum(dim=-1) if Gd.numel() else torch.zeros_like(c_norm)
        r_ind = torch.linalg.vector_norm(Gi, ord=p, dim=-1).sum(dim=-1) if Gi.numel() else torch.zeros_like(c_norm)
        r = r_dep + r_ind                                                     # [...]

        # Build a (sub)gradient at c to carry dependent structure.
        # p == 2: grad = c / ||c||
        # general p > 1: grad_i = sign(c_i) |c_i|^{p-1} / ||c||_p^{p-1}
        # p == 1: use a valid subgradient: sign(c) (0 where c==0)
        eps = torch.finfo(Z.dtype).eps
        if p == 2:
            denom = torch.clamp(c_norm, min=eps)[..., None]
            grad = c / denom                                                # [..., d]
        elif isinstance(p, (int, float)) and p > 1:
            denom = torch.clamp(c_norm, min=eps)[..., None] ** (p - 1)
            grad = torch.sign(c) * (torch.abs(c) ** (p - 1)) / denom
        elif p == 1:
            grad = torch.sign(c)
        else:
            # Fallback: no gradient structure for other/unsupported ord (incl. inf);
            # just return the sharp interval as a 1D PZ with one independent gen.
            new_c = c_norm
            extra = r
            # Construct scalar PZ with no dep structure
            Zout = torch.zeros(Z.batch_shape + (1 + 0 + 1, 1), dtype=Z.dtype, device=Z.device)
            Zout[batch_idx + (0, 0)] = new_c
            if extra.numel():
                Zout[batch_idx + (1, 0)] = extra
            out = type(self)(Zout, n_dep_gens=0,
                            expMat=torch.zeros((0, 0), dtype=torch.long, device=Z.device),
                            id=torch.empty(0, dtype=torch.int64), copy_Z=False)
            if inv:
                l = torch.clamp(new_c - extra, min=0.0)
                u = new_c + extra
                if torch.any(l <= 0):
                    raise ValueError("Reciprocal norm undefined: interval touches zero.")
                inv_low = 1.0 / u
                inv_high = 1.0 / l
                # batchZonotope interval [inv_low, inv_high]
                center = 0.5 * (inv_low + inv_high)
                rad = 0.5 * (inv_high - inv_low)
                Z1 = torch.stack((center, rad), dim=-2)  # [..., 2, 1]
                return zp.batchZonotope(Z1)
            return out

        # If center is near zero, the gradient is uninformative; fall back to pure interval.
        near_zero = (c_norm <= eps)
        # Dependent scalar generators via supporting plane: s_i = grad · g_i
        if Gd.numel():
            s_dep = torch.einsum('...d,...nd->...n', grad, Gd)  # [..., N_dep]
        else:
            s_dep = torch.zeros(Z.batch_shape + (0,), dtype=Z.dtype, device=Z.device)

        # Independent scalar generators from the same plane: t_j = grad · grest_j
        if Gi.numel():
            t_ind = torch.einsum('...d,...md->...m', grad, Gi)  # [..., N_indep]
        else:
            t_ind = torch.zeros(Z.batch_shape + (0,), dtype=Z.dtype, device=Z.device)

        # One extra independent "radius" generator so the overall range matches [c_norm - r, c_norm + r].
        # This guarantees a sound enclosure and makes the interval endpoints tight,
        # while preserving dependent structure inside.
        sums = s_dep.abs().sum(dim=-1) + t_ind.abs().sum(dim=-1)             # [...]
        extra = (r - sums).clamp_min(0.0)                                    # [...]

        # If center is (near) zero, drop linear structure and just use the interval radius r
        if torch.any(near_zero):
            # Blend per batch element
            s_dep = torch.where(near_zero[..., None], torch.zeros_like(s_dep), s_dep)
            t_ind = torch.where(near_zero[..., None], torch.zeros_like(t_ind), t_ind)
            extra = torch.where(near_zero, r, extra)

        # Assemble scalar PZ: center + sum alpha^e * s_dep_i  +  sum beta_j * t_ind_j  +  beta_extra * extra
        # Build Z tensor [..., (1 + N_dep + N_indep + 1), 1]
        n_dep = Z.n_dep_gens
        n_ind = Z.n_indep_gens
        total_gens = 1 + n_dep + n_ind + 1
        Zs = torch.zeros(Z.batch_shape + (total_gens, 1), dtype=Z.dtype, device=Z.device)

        # Center
        Zs[batch_idx + (0, 0)] = c_norm
        # Dependent scalar gens (keep the SAME expMat/id mapping)
        if n_dep:
            Zs[batch_idx + (slice(1, 1 + n_dep), 0)] = s_dep
        # Independent scalar gens from Grest
        if n_ind:
            Zs[batch_idx + (slice(1 + n_dep, 1 + n_dep + n_ind), 0)] = t_ind
        # Extra independent radius gen
        Zs[batch_idx + (slice(1 + n_dep + n_ind, 1 + n_dep + n_ind + 1), 0)] = extra.unsqueeze(-1)

        out = type(self)(
            Zs,
            n_dep_gens=n_dep,
            expMat=Z.expMat,  # carry over the same exponent structure for the dep part
            id=Z.id,
            copy_Z=False
        ).compress(1)  # strip any zeros if possible

        if inv:
            # interval [l, u] from sharp triangle bound
            l = torch.clamp(c_norm - r, min=0.0)
            u = c_norm + r
            if torch.any(l <= 0):
                raise ValueError("Reciprocal norm undefined: interval touches zero.")
            inv_low = 1.0 / u
            inv_high = 1.0 / l
            center = 0.5 * (inv_low + inv_high)
            rad = 0.5 * (inv_high - inv_low)
            Z1 = torch.stack((center, rad), dim=-2)  # [..., 2, 1]
            import zonopy as zp
            return zp.batchZonotope(Z1)

        return out

    
    def to_spheres(self, buffer_radius=0):
        centers = self.c
        centered = self - centers
        radii = centered.norm(2).to_interval().sup
        return centers, radii