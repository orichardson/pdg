import pandas as pd
import numpy as np
import networkx as nx

from pgmpy.inference.ExactInference import BeliefPropagation

from abc import ABC
from typing import FrozenSet, List, Type, TypeVar, Mapping
import collections

from functools import reduce
from operator import and_, mul

from . import utils
from . import rv
Var = rv.Variable

import warnings
import itertools
import re
	
from .alg.bp import avg_init_pgmpy_BP_calibrate

try:
	from pgmpy.factors.discrete import TabularCPD
except ImportError:
	warnings.warn("pgmpy not loaded")


# recipe from https://docs.python.org/2.7/library/itertools.html#recipes
def powerset(iterable, reverse=False):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return itertools.chain.from_iterable(itertools.combinations(s, r) 
		for r in ( reversed(range(len(s)+1)) if reverse else range(len(s)+1))
	)


def z_mult(joint, masked):
	""" multiply assuming zeros override nans; keep mask otherwise"""
	return np.ma.where(joint == 0, 0, joint * masked)

def zz1_div(top, bot):
	""" divide assuming 0/0 = 1. """
	rslt = np.ma.divide(top,bot)
	rslt = np.ma.where( np.logical_and(top == 0, bot == 0), 1, rslt)
	return rslt

def D_KL(d1,d2):
	return z_mult(d1, np.ma.log(zz1_div(d1,d2))).sum()

try:
	import torch

	def D_KL_torch(t1, t2, LOGZERO=1E12):
		where = torch.where
		return where( t1 == 0., 0., 
			t1*(torch.log( where(t1==0., LOGZERO, t1) - torch.log(where(t2==0, LOGZERO, t2))))).sum()
except ImportError:
	print("No torch; only numpy backend")




def _idxs(varlist, *varis, multi=False):
	""" Given a list `varlist` of atomic Variables, returns a list of indices
		reflecting all components of the arguments that follow. """
	idxs = []
	for V in varis:
		##old version: split doesn't work well.
		# if V in self.varlist and (multi or V not in idxs):
		# 	idxs.append(self.varlist.index(V))
		# elif '×' in V.name:
		# 	idxs.extend([v for v in self._idxs(*V.split()) if (multi or v not in idxs)])
		## new version with atomic
		for a in V.atoms:
			try:
				i = varlist.index(a)
			except ValueError:
				# print("varlist: ", varlist)
				raise
			if multi or (i not in idxs):
				idxs.append(i)
		##older version
		#     for v in V.name.split('×'):
		#         idxs.append([v])

	return idxs

def broadcast(cpt, varlist, 
		vfrom: Var = None,
		vto: Var =None ) -> np.array:
	""" returns its argument, but shaped
	so that it broadcasts properly (e.g., for taking expectations) in this
	distribution. For example, if the var list is [A, B, C, D], the cpt
	B -> D would be broadcast to shape [1, 2, 1, 3] if |B| = 2 and |D| =3.

	Parameters
	----
	> cpt: the argument to be broadcast; might be a dataframe, a CPT, or a np.matrix
	> vfrom,vto: the attached variables (supply only if cpt does not have this data)
	"""
	if vfrom is None: vfrom = cpt.nfrom
	if vto is None: vto = cpt.nto

	IDX = _idxs(varlist, vfrom,vto,multi=True)
	UIDX = np.unique(IDX).tolist() # sorted also

	init_shape = [1] * (len(varlist)+len(IDX)-len(UIDX))

	for j,i in enumerate(IDX):
		init_shape[j] = len(varlist[i])

	cpt_mat = cpt.to_numpy() if isinstance(cpt, pd.DataFrame) else cpt

	# if idxt < idxf:
	#     cpt_mat = cpt_mat.T

	cpt_mat = cpt_mat.reshape(*init_shape)
	cpt_mat = np.einsum(cpt_mat, [*IDX,...], [*UIDX, ...])

	# clones = [i != j for i,j in enumerate(types)]

	clones = [varlist.index(v) != i for i,v in enumerate(varlist)]

	cpt_mat = np.moveaxis(cpt_mat, np.arange(len(UIDX)), UIDX)

	# re-expanding in case varlist has duplicates. 
	if any(clones):
		# uvarlist = []
		# for v in varlist:
		# 	if v not in uvarlist: 
		# 		uvarlist.append(v)
		
		# counting_types = [uvarlist.index(v) for v in varlist]
		idx_types = [varlist.index(v) for v in varlist]

		# print('idx_types', idx_types)
		# print('counting_types', counting_types)

		# print(tuple((len(V) if types[i] in IDX else 1) for i,V in enumerate(varlist)))
		# print('outputshape', tuple((len(V) if varlist.index(V) in IDX else 1) for V in varlist))

		output = np.zeros(tuple((len(V) if idx_types[i] in IDX else 1) for i,V in enumerate(varlist)))
		# output = np.zeros( tuple((len(V) if varlist.index(V) in IDX else 1) for V in varlist))
		# print('output shape', output.shape)
		# print('cptmat.shape ', cpt_mat.shape)
		# print('einsum target: ', np.einsum(output, idx_types, UIDX + list(set(idx_types) - set(UIDX)) ).shape)
		# print('clones', clones)
		# print('IDX: ', IDX)
		# print("UIDX: ", UIDX)
		# print('Candidate 1: ', tuple((len(v) if i in IDX else 1 ) for i,v in enumerate(varlist) if not clones[i]))
		# print("Candidate 2: ", [ d for i,d in enumerate(cpt_mat.shape) if not clones[UIDX[i]]])

		cpt_mat_tailored = cpt_mat.reshape( tuple((len(v) if i in IDX else 1 ) for i,v in enumerate(varlist) if not clones[i]))
		# cpt_mat_tailored = cpt_mat.reshape([ d for d,c in zip(cpt_matbshape, clones) if not c])
		np.einsum(output, idx_types, UIDX + list(set(idx_types) - set(UIDX)) )[:] = cpt_mat_tailored
	else: 
		output = cpt_mat

	# cpt_mat = np.moveaxis(cpt_mat, np.arange(len(UIDX)), UIDX)
	# return cpt_mat
	return output


def _process_vars(varlist, vars, given=None):
	if vars is ...:
		vars = varlist

	if isinstance(vars, rv.Variable) \
		or isinstance(vars, rv.ConditionRequest) or vars is ...:
			vars = [vars]
	
	if isinstance(vars, str):
		if '|' in vars:
			t, g = vars.split("|")
			vars = [*re.split(r'[\s,]', t), '|', *re.split(r'[\s,]', g)]
		else:
			vars = re.split(r'[\s,]', vars)

	targetvars = []
	conditionvars = list(given) if given else []

	mode = "join"

	for var in vars:
		if isinstance(var, rv.ConditionRequest) or var == '|':
			if mode == "condition":
				raise ValueError("Only one bar is allowed to condition")
			
			mode = "condition"

			if isinstance(var, rv.ConditionRequest):
				targetvars.append(var.target)
				conditionvars.append(var.given)
		else:
			l = (conditionvars if mode == "condition" else targetvars)
			if isinstance(var, rv.Variable):
				l.append(var)
			elif isinstance(var, str):
				if len(var) == 0 : continue
				try:
					l.append(next(v for v in varlist if v.name == var))
				except StopIteration:
					raise ValueError("No variable named \"%s\" in dist"%var)
				# if mode == "condition":
				#     conditionvars.append(var)
				# elif mode == "join":
				#     targetvars.append(var)
			elif var is ...:
				l.extend(v for v in varlist if v not in l)
			else:
				raise ValueError("Could not interpret ",var," as a variable")

	return targetvars, conditionvars



class CDist(ABC): pass
class Dist(CDist): pass

SubCPT = TypeVar('SubCPT' , bound='CPT')


class CPT(CDist, pd.DataFrame, metaclass=utils.CopiedABC):
	PARAMS = {"nfrom", "nto"}
	_internal_names = pd.DataFrame._internal_names + ["nfrom", "nto"]
	_internal_names_set = set(_internal_names)

	def __init__(self,*args,**kwargs):
		# print("CPT constructor")
		# self.style.background_gradient(cmap=greens, axis=None)
		pass

	# def __call__(self, pmf):
	#     pass
	# def __matmul__(self, other) :
	#     """ Overriding matmul.... """
	#     pass

	def flattened_idx(self):
		cols = self.columns.to_flat_index().map(lambda s: s[0])
		rows = self.index.to_flat_index().map(lambda s: s[0])

		return pd.DataFrame(self.to_numpy(), columns = cols, index=rows)

	@classmethod
	def _from_matrix_inner(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
		def makeidx( vari ):
			if multi and False:
				names=vari.name.split("×")

				# maxdepth = utils.tuple_depth(vari.ordered[0])
				# depth = maxdepth - len(names) if flatten else 0

				# print("levels", depth)
				# print([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
				# else (v,) ) for v in vari.ordered ])

				# print(depth)
				print('v', vari.ordered[0])
				print(np.array([ str(v) for v in vari.ordered ]).shape)
				print(names)

				# print([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
				# else (v,) ) for v in [vari.ordered[0]] ])
				# print(np.array([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
				# else (v,) ) for v in vari.ordered ],).shape)
				# print(names)

				return pd.MultiIndex.from_tuples(
					[ str(v) for v in vari.ordered ],
					names=names)

				# return pd.MultiIndex.from_tuples(
				#     [ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
				#     else (v,) ) for v in vari.ordered ],
				#     names=names)
			else:
				return vari.ordered

		return cls(matrix, index=makeidx(nfrom), columns=makeidx(nto), nto=nto,nfrom=nfrom)

	@classmethod
	def from_matrix(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
		return cls._from_matrix_inner(nfrom,nto,matrix,multi,flatten).check_normalized()

	@classmethod
	def make_stoch(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
		return cls._from_matrix_inner(nfrom,nto,matrix,multi,flatten).renormalized()

	@classmethod
	def from_ddict(cls: Type[SubCPT], nfrom, nto, data) -> SubCPT:
		for a in nfrom:
			row = data[a]
			if not isinstance(row, Mapping):
				try:
					iter(row)
				except:
					data[a] = { nto.default_value : row }
				else:
					data[a] = { b : v for (b,v) in zip(nto,row)}

			total = sum(v for b,v in data[a].items())
			remainder = nto - set(data[a].keys())
			if len(remainder) == 1:
				data[a][next(iter(remainder))] = 1 - total
			elif total == 1:
				for b in remainder:
					data[a][b] = 0

		matrix = pd.DataFrame.from_dict(data , orient='index')
		return cls(matrix, index=nfrom.ordered, columns=nto.ordered, nto=nto,nfrom=nfrom).check_normalized()

	@classmethod
	def make_random(cls : Type[SubCPT], vfrom, vto):
		mat = np.random.rand(len(vfrom), len(vto))
		# mat /= mat.sum(axis=1, keepdims=True)
		return cls._from_matrix_inner(vfrom,vto,mat).renormalized()

	@classmethod
	def det(cls: Type[SubCPT], vfrom, vto, mapping, **kwargs) -> SubCPT:
		mat = np.zeros((len(vfrom), len(vto)))
		for i, fi in enumerate(vfrom.ordered):
			# for j, tj in enumerate(vto.ordered):
			mapfi = mapping[fi] if isinstance(mapping,dict) else mapping(fi)
			mat[i, vto.ordered.index(mapfi)] = 1

		return cls.from_matrix(vfrom,vto,mat, **kwargs)
		# return cls.from_matrix(, index=vfrom.ordered, columns=vto.ordered, nto=vto, nfrom= vfrom)


	@classmethod
	def from_pgmpy(cls: Type[SubCPT], tcpd : TabularCPD, **kwargs):
		tgt = rv.Variable(tcpd.state_names[tcpd.variable], 
			name=tcpd.variable)
		srcnames = tcpd.variables[1:] ## !!! note that get_cardinality() reverses ...
			# the order of the vaiables (so they don't line up with values) for no reason!
		if len(srcnames) > 0 :
			src = reduce(and_, [rv.Variable(tcpd.state_names[l], name=l) 
				for l in srcnames if l != tgt.name])
		else: src = rv.Unit

		return cls.from_matrix(src, tgt, 
			np.moveaxis(tcpd.values,0,-1).reshape(-1, len(tgt)), **kwargs)
	
	################################

	def copy(self, deep=True):
		return CPT(self, nto=self.nto,nfrom=self.nfrom)

	def to_pgmpy(self):
		return TabularCPD(self.nto.name, len(self.nto), 
			values = self.to_numpy().reshape(-1, len(self.nto)).T, 
			evidence = [v.name for v in self.nfrom.atoms], 
			evidence_card = [len(v) for v in self.nfrom.atoms],
			state_names = {v.name : v.ordered for v in [self.nto, *self.nfrom.atoms]}
			)

	def broadcast_to(self, varlist) -> np.ndarray:
		return broadcast(self, varlist, self.nfrom, self.nto)


	def check_normalized(self) -> bool:
		amt = np.where(np.all(np.isfinite(self),axis=1), (np.sum(self, axis=1)-1)**2 ,0).sum()
		if amt > 1E-5:
			warnings.warn("%.4f-Unnormalized CPT"%amt)

		return self

	def renormalized(self):
		self /= np.sum(self, axis=1).to_numpy()[:, None]
		return self
		
	def sample(self, xval):
		u = np.random.rand()
		return (u < self.loc[xval].cumsum()).idxmax()
		
	####### CONVERSION TO PGMPY ########
	def as_(self, targetclass):
		"""
		Supported targetclasses:
			pgmpy.factors.discrete.TabularCPD
			pgmpy.factors.discrete.DiscreteFactor
			numpy.array
			
		"""
		pass
		

## useless helper methods to either use dict values or list.
def _definitely_a_list( somedata ):
	if type(somedata) is dict:
		return list(somedata.values())
	return list(somedata)




# define an event to be a value of a random variable.
class RawJointDist(Dist):
	def __init__(self, data, varlist, use_torch=False):
		if use_torch and not torch.is_tensor(data):
			data = torch.tensor(data)
		elif not use_torch and not isinstance(data, np.ndarray):
			try: 
				if torch.is_tensor(data): 
					data = data.detach().numpy()
			except NameError: pass
			
		self._torch = use_torch        
		self.data = data.reshape(*(len(X) for X in varlist))
		self.varlist = varlist

		# if rv.Unit not in varlist:
		#     self.varlist = [rv.Unit] + self.varlist
		#     self.data = self.data.reshape(1, *self.data.shape)

		self._query_mode = "dataframe" # query mode can either be
			# dataframe or ndarray
			
	def clone(self):
		return RawJointDist(
			self.data.clone() if self._torch else self.data.copy(),
			self.varlist, self._torch)
		
	############# CONVERSIONS #################
	def npify(self, inplace=False):
		data = self.data.detach().numpy() if self._torch else self.data
		if inplace:
			self.data = data
			self._torch = False
		else:
			return RawJointDist(data, self.varlist, False)    

	def torchify(self, requires_grad=True):
		# """ if requires_grad is different, and already has a torch tensor
		# as a back-end, do this in in place. """
		if self._torch:
			if self.data.requires_grad == requires_grad:
				return self
				
			data = self.data.detach()
			data.requires_grad = requires_grad
			
		elif not self._torch:
			data = torch.tensor(self.data, requires_grad=requires_grad)
			
		return RawJointDist(data, self.varlist, True)    
	
	def to_dit(self):
		from dit import Distribution
		import itertools as itt

		data = self.data.detach().numpy() if self._torch else self.data

		return Distribution([tuple(str(v) for v in p) for p in itt.product(*self.varlist)],
					  		data.flat)
	
	def require_grad(self):
		""" an in-place method to re-enable gradients. """
		if self._torch:
			# self.data.grad = None
			if self.data.requires_grad:
				# reset instead
				self.data.grad = None
				self.data = self.data.detach()
				self.data.requires_grad = True
			else:
				self.data.requires_grad = True
			return self
			
		raise ValueError("This RJD does not have a torch back-end. First torchify.")

	def  to_pgmpy_discrete_factor(self):
		from pgmpy.factors.discrete import DiscreteFactor
		return DiscreteFactor( 
			[v.name for v in self.varlist], 
			[len(v) for v in self.varlist], 
			self.data)
	
	# Both __mul__ and __rmul__ reqiured to do things like multiply by constants...
	def __mul__(self,other):
		return RawJointDist(self.data * other, self.varlist)
	def __rmul__(self,other):
		return RawJointDist(self.data * other, self.varlist)

	def __pos__(self):
		return self.renormalized()
	
	def __floordiv__(self,other):
		if self._torch and other._torch:
			return D_KL_torch(self.data, other.data)
		
		narr = utils.nparray_of
		return D_KL(narr(self.data), narr(other.data))

	def __len__(self):
		if self._torch:
			return self.data.numel()

		return self.data.size

	def __contains__(self, var):
		if isinstance(var, Var):
			return var in self.varlist or all(
				a in self.varlist for a in var.atoms)
		
		elif isinstance(var, str):
			return all( any(a.strip() == v.name for v in self.varlist) for a in var.split(","))

	def __repr__(self):
		# varstrs = [v.name+"%d"%len(v) for v in self.varlist]
		varstrs = [v.name for v in self.varlist]
		# return f"RJD Δ[{';'.join(varstrs)}]--{np.prod(self.shape)} params"
		# for python 3.5, with no string interpolation
		if self._torch:
			return "RJD Δ("+('; '.join(varstrs))+") as tensor⟨"+','.join(map(str, self.data.shape))+"⟩"
			 # + str(self.data.numel())+" params"
		else:
			# return "(np) RJD Δ[" + ('; '.join(varstrs)) + " ndarray]" 
			return "RJD Δ("+('; '.join(varstrs))+") as ndarray⟨"+','.join(map(str,self.data.shape))+"⟩"
			# + repr(np.prod(self.shape)) + " params"


	@property
	def shape(self):
		return self.data.shape

	def _process_vars(self, vars, given=None):
		return _process_vars(self.varlist, vars, given)

	def _idx(self, var):
		try:
			return self.varlist.index(var)
		except ValueError:
			raise ValueError("The queried varable", var, " is not part of this joint distribution")

	def _idxs(self, *varis, multi=False):
		return _idxs(self.varlist, *varis, multi=multi)

	def broadcast(self, cpt : CPT, vfrom=None, vto=None) -> np.array:
		return broadcast(cpt, self.varlist, vfrom, vto)
	

	####################### OPERATIONS #######################

	def subdist_expand(self):
		ar = np.zeros(tuple(d+1 for d in self.data.shape))
		ar[tuple(-1 for d in self.data.shape)] = 1 - self.data.sum()
		ar[tuple(slice(0,d) for d in self.data.shape)] = self.data

		nullvarlist = [
			Var(V | {'∅'}, name=V.name, default_value='∅')
				for V in self.varlist
		]

		return RawJointDist(ar, nullvarlist)


	def renormalized(self):
		self.data /= self.data.sum()
		return self
	
	################### MARGINALIZATION, INFERENCE QUERIES ##################

	def conditional_marginal(self, vars, query_mode=None):
		if query_mode is None: query_mode = self._query_mode
		# if coordinate_mode is "joint": query_mode = "ndarray"

		# print(type(vars), vars, isinstance(vars, rv.Variable))
		targetvars, conditionvars = self._process_vars(vars)

		idxt = self._idxs(*targetvars, multi=True)
		idxc = self._idxs(*conditionvars, multi=True)
		IDX = idxt + idxc
		# UIDX = list(dict.fromkeys(IDX))
		neitheridx =  [i for i in range(len(self.varlist)) if i not in IDX ]

		
		if self._torch:
			# not really expanded, but so that it's the same variable 
			# this is really stupid, because of the numpy incompatibility
			if not len(neitheridx):
				joint_expanded = self.data
			else:
				joint_expanded = self.data.sum(dim=neitheridx, keepdim=True)

			## TODO actually make this expanded.
		else:
			# sum across anything not in the index
			joint = self.data.sum(axis=tuple(neitheridx) )
			
			# duplicate dimensions that occur multiple times by
			# an einsum diagonalization... (only works in numpy)        
			joint_expanded = np.zeros([self.data.shape[i] for i in IDX])
			np.einsum(joint_expanded, IDX, np.unique(IDX).tolist())[...] = joint


		if len(idxc) > 0:
			if self._torch:
				normalizer = joint_expanded.sum(dim=idxt, keepdim=True)
				matrix = (joint_expanded / normalizer).permute(IDX+neitheridx).squeeze()
				# The torch version still has to reorder the columns...
			else:            
				# if idxt is first...
				normalizer = joint_expanded.sum(axis=tuple(i for i in range(len(idxt))), keepdims=True)
				#if idxt is last...
				# normalizer = joint_expanded.sum(axis=tuple(-i-1 for i in range(len(idxt))), keepdims=True)
				with np.errstate(divide='ignore',invalid='ignore'):
					matrix = joint_expanded / normalizer

			if query_mode == "ndarray":
				return matrix
			elif query_mode == "dataframe":
				vfrom = reduce(and_,conditionvars)
				vto = reduce(and_,targetvars)
				if self._torch: matrix = matrix.detach().numpy()
				mat2 = matrix.reshape(len(vto),len(vfrom)).T

				return CPT.from_matrix(vfrom,vto, mat2,multi=False)
		else:
			# matrix = joint_expanded.permute(UIDX+neitheridx).squeeze() if self._torch else joint_expanded
			matrix = joint_expanded.permute(IDX+neitheridx).squeeze() if self._torch else joint_expanded
			
			# return joint_expanded
			if query_mode == "ndarray":
				return matrix
			elif query_mode == "dataframe":
				if self._torch: matrix = matrix.detach().numpy()
				mat1 = matrix.reshape(-1,1).T;
				return CPT.from_matrix(rv.Unit, reduce(and_,targetvars), mat1,multi=False)

	# returns the marginal on a variable
	def __getitem__(self, vars):
		return self.conditional_marginal(vars, self._query_mode)


	def prob_matrix(self, *vars, given=None):
		""" A global, less user-friendly version of
		conditional_marginal(), which keeps indices for broadcasting.
		TODO: Does not handle duplicate dimensions yet! """
		tarvars, cndvars = self._process_vars(vars, given=given)
		# print([t.name for t in tarvars], "|", [c.name for c in cndvars])
		idxt = self._idxs(*tarvars)
		idxc = self._idxs(*cndvars)
		# print("idxt: ", idxt, " \tidxc", idxc)
		IDX = idxt + idxc

		N = len(self.varlist)
		dim_nocond = tuple(i for i in range(N) if i not in idxc )
		dim_neither = tuple(i for i in range(N) if i not in IDX ) 
		# want tosum across anything not in the index
		
		if self._torch: # wow, torch's nonparamatricity of sum for dim=[] is crazy
			collapsed = self.data.sum(dim=dim_neither,keepdim=True) if  len(dim_neither) \
				else self.data

		else: collapsed = self.data.sum(axis=dim_neither, keepdims=True)

		if len(cndvars) > 0:
			if self._torch: # nans are correct, but destroy the gradient. So we set them equal to zero.
				denom = collapsed.sum(dim=dim_nocond, keepdim=True)
				collapsed = torch.divide(collapsed, torch.where(denom==0, 1., denom))
				# if denominator is zero, so is numerator, so at least this is a valid answer
			else:
				collapsed = np.ma.divide(collapsed, collapsed.sum(axis=dim_nocond, keepdims=True))

		return collapsed
	

	####################### INFORMATION QUERIES #####################

	def H(self, *vars, base=2, given=None):
		""" Computes the entropy, or conditional
		entropy of the list of variables, given all those
		that occur after a ConditionRequest. """
		P = self.prob_matrix(*vars, given=given)
		d = self.data
		if self._torch:
			return - (torch.log( torch.where(P==0, 1., P)) * d).sum() / np.log(base)
		else:
			return - (np.ma.log( P ) * d).sum() / np.log(base)

		## The expanded version looks like this, but is
		## a bit slower and not really simpler.
		# collapsed = self.prob_matrix(vars)
		# surprise = - np.ma.log( collapsed ) / np.log(base)		raise NotImplemented

	def I(self, *vars, given=None):
		tarvars, cndvars = self._process_vars(vars, given)

		tot = 0
		# n = len(tarvars)

		for s in powerset(tarvars):
			# print(s, (-1)**(n-len(s)), self.H(*s, given=cndvars))
			tot += (-1)**(len(s)+1) * self.H(*s, given=cndvars) # sum += (-1)**(n-len(s)+1) * self.H(*s, given=cndvars)
		return tot

	# def _info_in(self, vars_in, vars_fixed):
		# return self.H(vars_in | vars_fixed)
	#
	def iprofile(self) :
		"""
		Returns a tensor of shape 2*2*2*...*2, one dimension for each
		variable. For example,
			00000 is going to always have zero.
			01000 is the information H(X1 | X0, X2, ... Xn)
			11000 is the conditional mutual information I(X1; X2 | ...)

		"""
		for S in powerset(self.varlist):
			pass


	def info_diagram(self, X, Y, Z=None):
		# import matplotlib.pyplot as plt
		from matplotlib_venn import venn3

		# H = self.H
		I = self.I

		infos = [I(X|Y,Z), I(Y|X,Z), I(X,Y|Z), I(Z|X,Y), I(X,Z|Y), I(Y,Z|X), I(X,Y,Z) ]
		infos = [round(i, 3) for i in infos]
		# infos = [int(round(i * 100)) for i in infos]
		# Make the diagram
		v = venn3(subsets = infos, set_labels=[X.name,Y.name,Z.name])
		return v

	#################### CONSTRUCTION ######################

	@staticmethod
	def unif( vars) -> 'RawJointDist':
		varlist = _definitely_a_list(vars)
		data = np.ones( tuple(len(X) for X in varlist) )
		return RawJointDist(data / data.size, varlist)

	@staticmethod
	def random( vars) -> 'RawJointDist':
		varlist = _definitely_a_list(vars)
		data = np.random.exponential(1, [len(X) for X in varlist] )
		return RawJointDist(data / np.sum(data), varlist)


# def _key(rjd : RawJointDist) -> FrozenSet:
# 	"""turns RawJointDist's variable list into a hashable key""" 
# 	return frozenset(rjd.varlist)
class RawSubDist(Dist):
	pass

class CliqueForest(Dist):
	def __init__(self, rjds : List[RawJointDist], edges=None):
		# self.dists = { _key(rjd) : rjd for rjd in rjds }
		# self.dists = 
		self.dists = rjds # a list of RawJointDist, so that C[i] is the ith cluster.
		self.lookup = { frozenset([v.name for v in rjd.varlist]) : i 
			for i,rjd in enumerate(rjds) }

			# self.keys_big2small = sorted(
		# 	(frozenset([V.name for V in rjd.varlist]) for rjd in rjds),
		# 	key= lambda s: 
		# )

		# take maximum commonality spanning tree; don't see why guaranteed
		# to find one satisfying the running intersection property, but
		# it's the way pgmpy implements the junction tree algorithm, so...
		if edges is None:
			# possibly unnecessary O(n^2) computation here...
			complete_graph = nx.Graph()
			for i in range(len(rjds)):
				for j in range(i+1,len(rjds)):
					common = set(rjds[i].varlist) & set(rjds[j].varlist)
					complete_graph.add_edge(i,j, weight=-len(common))
			
			self.Gr = nx.minimum_spanning_tree(complete_graph)
		else:
			# graph is between integer indices.
			self.Gr = nx.Graph(edges)
			Gr_nodes = set(self.Gr.nodes())
			indices = set(range(len(rjds)))
			assert Gr_nodes.issubset(indices)
			self.Gr.add_nodes_from( indices - Gr_nodes )

		

		# calculate the union of all of the variable sets;
		self.varlist = []
		for rjd in rjds:
			for v in rjd.varlist:
				if v not in self.varlist:
					self.varlist.append(v)

		## assert that induced subtrees are connected
		for v in self.varlist:
			Cv =  [ i for i in range(len(rjds)) if v in rjds[i].varlist ]
			assert nx.is_connected(nx.induced_subgraph(self.Gr, Cv))
		
		# pre-calculate and save the separating sets, for convenience
		self.Ss = {}
		for (i,j) in self.edges:
			self.Ss[i,j] = list( set(rjds[i].varlist) & set(rjds[j].varlist) )


	@property
	def calibrated(self):
		## assert that all marginals are the same along tree
		return all(
			np.allclose( 
				self.dists[i][self.Ss[i,j]],
				self.dists[j][self.Ss[i,j]])
			for (i,j) in self.edges )

	def marginal_constraint_violation(self):
		return sum(
			np.abs(self.dists[i].conditional_marginal(self.Ss[i,j],query_mode='ndarray')
			 - self.dists[j].conditional_marginal(self.Ss[i,j],query_mode='ndarray')).sum()
			for (i,j) in self.edges
		)

	def renormalized(self):
		for d in self.dists:
			d.renormalized()
		return self

	@property
	def edges(self):
		return self.Gr.edges()

	@property
	def n_params(self):
		return sum(
			int(np.prod(dist.data.shape))
			for dist in self.dists
		)

	@property
	def _torch(self):
		return all(rjd._torch for rjd in self.dists)

	def _idxs(self, *varis, multi=False):
		return _idxs(self.varlist, *varis, multi=multi)
	
	# def __get_item__(): raise NotImplemented

	def conditional_marginal(self, vars, query_mode=None):
		""" 
		For now, only needs to handle the case where all variables fall within
		one distribution or the other. (Update: now uses pgmpy to handle 
		other cases also!)
		"""

		# print("querying ", [v.name for v in vars])

		for rjd in self.dists:
			try:
				## lol I don't even need to check. I can just try to do the thing.
				## also: the below doesn't work for some reason; I think it might
				## not handle atoms properly.
				# tarvars, cndvars = rjd._process_vars(vars)
				# if not all(v in rjd.varlist for v in itertools.chain(tarvars,cndvars)):
				# 	continue

				return rjd.conditional_marginal(vars, query_mode=query_mode)

			except ValueError:
				continue # this isn't the one.
			except RuntimeError as e:
				print("another error", e, "with rjd :", rjd)
				raise


		warnings.warn("Falling back on untested junction tree pgmpy query")
		tarvars, cndvars =_process_vars(self.varlist, vars)
		rjd = self._fallback_joint_query_bp(tarvars + cndvars)
		return rjd.conditional_marginal(vars, query_mode=query_mode)

		# raise NotImplementedError("not all variables are in the same cluster; this doesn't work yet.")

	def npify(self, inplace=True):
		if inplace:
			for d in self.dists:
				d.npify(inplace=True)
		else:
			return CliqueForest([d for d in self.dists], self.edges)

	# returns the marginal on a variable
	def __getitem__(self, vars):
		return self.conditional_marginal(vars)


	def to_pgmpy_jtree(self):
		""" may not be connected """
		from pgmpy.models import JunctionTree

		G = JunctionTree()
		namednodes = [ tuple(v.name for v in C.varlist) for C in self.dists ]
		G.add_nodes_from(namednodes)
		G.add_edges_from([ 
			(namednodes[i], namednodes[j]) for (i,j) in self.edges
				if len(set(namednodes[i]) & set(namednodes[j])) > 0
			])
		G.add_factors(*[rjd.to_pgmpy_discrete_factor() for rjd in self.dists])
		return G

	def to_pgmpy_jtrees(self):
		""" one for each subcomponent """
		from pgmpy.models import JunctionTree

		jj = []
		for idxset in nx.connected_components(self.Gr):
			G = JunctionTree()
			namednodes = [ tuple(v.name for v in self.dists[i].varlist) for i in idxset ]
			G.add_nodes_from(namednodes)
			G.add_edges_from([ 
				(namednodes[i], namednodes[j]) for (i,j) in self.edges
					if len(set(namednodes[i]) & set(namednodes[j])) > 0
					and i in idxset and j in idxset
				])
			G.add_factors(*[self.dists[i].to_pgmpy_discrete_factor() for i in idxset])
			jj.append(G)

		return jj

	def _fallback_joint_query_bp(self, varilist):
		J = self.to_pgmpy_jtree()
		bp = BeliefPropagation(J)

		ans = bp.query([v.name for v in varilist])
		return RawJointDist(ans.values, varilist)
    
	def _fallback_recalibrate_bp(self,avg_init=True):
		# J = self.to_pgmpy_jtree()
		# jj = [nx.induced_subgraph(J,ns) for ns in nx.connected_components(J)]
		jj = self.to_pgmpy_jtrees()

		for j in jj:
			if avg_init:
				bp = avg_init_pgmpy_BP_calibrate(j)
			else:
				bp = BeliefPropagation(j)
				bp.calibrate()

			for varname_tuple, df in bp.clique_beliefs.items():
				i = self.lookup[frozenset(varname_tuple)]
				idx1 = [*range(len(self.dists[i].varlist))]
				idx2 = [df.variables.index(v.name) for v in self.dists[i].varlist]
				self.dists[i].data[:] = \
					np.moveaxis(df.values, idx2,idx1)
		
		self.renormalized()
	
	def broadcast(self, cpt : CPT, vfrom=None, vto=None) -> np.array:
		return broadcast(cpt, self.varlist, vfrom, vto)

	def prob_matrix(self, *vars, given=None):
		""" A global, less user-friendly version of
		conditional_marginal(), which keeps indices for broadcasting.
		Does not handle duplicate dimensions. """
		for rjd in self.dists:
			try: 
				localpm = rjd.prob_matrix(*vars, given=given)
				# print(localpm.shape)

				# extend with ones for all missing dimensions;
				localpm = localpm.reshape(localpm.shape + 
					(1,)*(len(self.varlist) - len(rjd.varlist)))
				
				# build permutation
				permutation = []
				m = 0
				for V in self.varlist:
					if V in rjd.varlist:
						permutation.append(rjd.varlist.index(V))
					else:
						permutation.append(len(rjd.varlist) + m)
						m += 1
				# print(permutation)

				if rjd._torch:
					return localpm.permute( permutation )
				else:
					return np.moveaxis(localpm, permutation, [*range(len(self.varlist))])
			except ValueError: continue # this isn't the one.


		# if given is None:
		warnings.warn("Falling back on untested junction tree pgmpy query")
		tarvars, cndvars =_process_vars(self.varlist, vars, given=given)
		ans = self._fallback_joint_query_bp(tarvars + cndvars)
		return ans.prob_matrix(*vars,given=given)


		# raise NotImplementedError("not all variables are in the same cluster;"+
		# 	" this doesn't work yet.")
		# tarvars, cndvars = self._process_vars(vars, given=given)
		# # print([t.name for t in tarvars], "|", [c.name for c in cndvars])
		# idxt = self._idxs(*tarvars)
		# idxc = self._idxs(*cndvars)
		# # print("idxt: ", idxt, " \tidxc", idxc)
		# IDX = idxt + idxc

		# N = len(self.varlist)
		# dim_nocond = tuple(i for i in range(N) if i not in idxc )
		# dim_neither = tuple(i for i in range(N) if i not in IDX ) 
		# # want tosum across anything not in the index
		
		# if self._torch: # wow, torch's nonparamatricity of sum for dim=[] is crazy
		# 	collapsed = self.data.sum(dim=dim_neither,keepdim=True) if  len(dim_neither) \
		# 		else self.data

		# else: collapsed = self.data.sum(axis=dim_neither, keepdims=True)

		# if len(cndvars) > 0:
		# 	if self._torch: # nans are correct, but destroy the gradient. So we set them equal to zero.
		# 		denom = collapsed.sum(dim=dim_nocond, keepdim=True)
		# 		collapsed = torch.divide(collapsed, torch.where(denom==0, 1., denom))
		# 		# if denominator is zero, so is numerator, so at least this is a valid answer
		# 	else:
		# 		collapsed = np.ma.divide(collapsed, collapsed.sum(axis=dim_nocond, keepdims=True))

		# return collapsed


		
	def H(self, *vars, base=2, given=None):
		if vars == (Ellipsis,) and given is None:
			ent = 0
			for C in self.dists:
				ent += C.H(...)

			for (i,j) in self.Ss:
				ent -= self.dists[i].H(*self.Ss[i,j])
			# ## comptues the Kikuchi approximation.
			# ## edit: this is exponential time for no reason; with edges,
			# # we can actually make sense of this; otherwise, it's not clearly meaningful.

			# ent = 0.
			# # c = { frozenset(range(len(self.dists))) : 1 } # overcounting numbers
			# c = {}

			# # for S in powerset(self.dists, reverse=True):
			# for S in powerset(range(len(self.dists)), reverse=False):
			# 	if len(S) == 0: continue

			# 	common_names = set.intersection(*[
			# 		set([v.name for v in self.dists[i].varlist])  for i in S])

			# 	if len(common_names) == 0 \
			# 			or frozenset(common_names) in c: 
			# 		continue
				
			# 	# overcount_S = 1 - sum(cr for r,cr in c.items() if r.issubset(S))
			# 	# c[frozenset(S)] = overcount_S
			# 	overcount_S = 1 - sum(cr for r,cr in c.items() if r.issuperset(common_names))
			# 	c[frozenset(common_names)] = overcount_S
				
			# 	ent += overcount_S * self.dists[S[0]].H(*common_names)
			
			# # print(c)
			return ent


		for rjd in self.dists:
			try: return rjd.H(*vars, given=given)
			except ValueError: continue # this isn't the one.

		raise NotImplementedError("not all variables are in the same cluster;"+
			" this doesn't work yet.")