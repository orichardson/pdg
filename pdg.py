""""
This module contains the most important parts of 

"""

# %load_ext autoreload
# %autoreload 2

# import pandas as pd
# import sys # for printing.
import warnings
import random
import numpy as np
import networkx as nx

import collections
from numbers import Number

# import utils
from .util import dictwo
from .rv import Variable, ConditionRequest, Unit
from .fg import FactorGraph
from .dist import RawJointDist as RJD, CPT, CliqueForest#, Dist, CDist,
from .dist import z_mult, zz1_div


########### Conditional imports, depending on what is installed
try: # import torch? Often useful but not critical to use the library.
	import torch
	
	LOGZERO=1E12
	twhere,tlog = torch.where, torch.log
	def tzmul(prob, maybe_nan):
		return twhere(prob == 0, 0., twhere(torch.isnan(maybe_nan), 0., maybe_nan) * prob)   
except ImportError:
	print("No torch; only numpy backend")

try: ### import pgmpy?  If installed, we will provide hooks. Else, also fine.
	from pgmpy.models import BayesianNetwork
except ImportError:
	warnings.warn("pgmpy module not found; PGMPY conversions unavailable");


class CountLabeler:
	"""
	A simple class for generating names for things---mostly  for conditional probabilities. This default implementation simply names things `p{i}` for i=1,2,...
	"""
	def __init__(self, basenames=['p']):
		self._counter = 0
		self._basenames = basenames

	def fresh(self, vfrom, vto, **ctxt):
		self._counter += 1		
		return self._basenames[0] + str(self._counter)
		
	def copy(self):
		l = CountLabeler(self._basenames)
		l._counter = self._counter
		return l
		
	def __repr__(self):
		return f"CountLabeler(count={self._counter})"


####################################################
# Main PDG class starts here.
#####################################################
class PDG:

	def __init__(self, labeler = CountLabeler()):
		self.vars = {"1" : Unit } # varname => Variable
		self.edgedata = collections.defaultdict(dict)
			# { (nodefrom str, node to str, label) => { attribute dict } }
			# attributes: cpt, alpha, beta. context. (possible lambdas.). All optional.

		self.labeler = labeler
		self.graph = nx.MultiDiGraph()
		self.gamma_default = 1

		self._dist_repr = "atomic"


	@property
	def hypergraph_object(self):
		def _names(vs):
			return [v.name for v in vs]

		N = _names(self.atomic_vars)
		
		l_atomnames = ((l, _names(fr.atoms), _names(to.atoms)) for (l,fr,to) in self.edges('lXY'))

		# get rid of extra structural edges e.g., A×B ->> A
		# HE = list(filter(lambda FT: not set(FT[1]).issubset(FT[0]), HE))
		HE = { l : [F,T] for  l,F,T in  l_atomnames if not set(T).issubset(F) }

		return N, HE

	@property
	def stats(self):
		return dict(
			n_edges = len(self.edges()),
			n_worlds = int(np.prod(self.dshape)),
			n_params = int(sum(p.size for p in self.edges('P')))
		)

	@property
	def Ed(self):
		return [*self.edges('l')]
	
	def __call__(self, *INPUT, **kwargs):
		"""
		Custom string interpolation for interpreting PDG queries & command, making it
		easier to construct things and do work in the context of a PDG.
		
		Examples:
		---------
		M('AB')  replaces  Variable.product(M.vars['A'], M.vars['B'])
		M(p) 
		
		
		Functionality Wishlist:
		----
		M('A B -> C')  to return  a β-combination of cpts.
		M('A B -> B C := ', P)  to add a matrix with the appropriate types,
		and any missing variables with the right # of elements, if they are msising.
		"""
		connectives = ["->"]

		def interpret(token):
			if token in connectives:
				pass
			elif token in self.vars:
				return self.vars[token]

		if len(INPUT) == 1 and type(INPUT[0]) is str:
			objects = [interpret(t.strip()) for t in INPUT[0].split()]
			# print(objects)

			if all(isinstance(o,Variable) for o in objects):
				return Variable.product(*objects) if len(objects) != 1 else objects[0]


	def subpdg(self, *descriptors):
		minime = PDG(self.labeler)
		for vn, v in self.vars.items():
			if vn in descriptors:
				minime._include_var(v,vn)

		for ftl, attr in self.edgedata.items():
			if ftl[0] in descriptors and ftl[1] in descriptors:
				minime._set_edge(*ftl, **attr)

		return minime

	def copy(self) -> 'PDG':
		newme = PDG(self.labeler.copy())

		for vn, v in self.vars.items():
			newme._include_var(v,vn)

		for ftl, attr in self.edgedata.items():
			newme._set_edge(*ftl, **attr)

		newme._apply_structure();
		return newme


	@property
	def varlist(self):
		return self.getvarlist(self._dist_repr)

	def getvarlist(self, repr):
		if repr == "raw": return self.rawvarlist
		elif repr == "atomic": return self.atomic_vars
		return []

	@property
	def rawvarlist(self):
		return list(X for X in self.vars.values())

	@property
	def atomic_vars(self):
		# TODO do this more nicely but equally efficiently.
		atoms =  [v for v in self.rawvarlist if '×' not in v.name and len(v) > 1]
		ghostvars = [v for v in self.rawvarlist if '×' in v.name]

		# Sanity check...
		missing = [n for v in ghostvars for n in v.split(atomic=True) if n.name not in self.vars]
		assert len(missing)==0, "Missing Components: "+repr(missing)

		return atoms


	def getdshape(self, repr):
		return tuple(len(X) for X in self.getvarlist(repr))

	@property
	def dshape(self):
		return self.getdshape(self._dist_repr)

	@property
	def cpds(self):
		for (X,Y,cpd, α,β) in self:
			yield cpd

	def _get_edgekey(self, spec):
		gn = tn = label = None
		
		# print(spec)	
		if isinstance(spec, ConditionRequest):
			# print(spec.given.name, spec.target.name)
			gn,tn = spec.given.name, spec.target.name
			# raise KeyError("Multiple options possible.")
		elif type(spec) == tuple and type(spec[0]) is str:
			# normal strings can be looked up as a tuple
			gn,tn = spec[:2]
			if len(spec) == 3:
				label = spec[2]
		elif type(spec) is str:
			specX = specY = None
			if spec.find("|") > 0:
				specY, specX = map(str.strip, spec.split('|'))
			elif spec.find("->") > 0:
				specX, specY = map(str.strip, spec.split('->'))

			for X,Y,l in self.edges("X,Y,l"):
				if spec == ','.join((X.name,Y.name,l)): # could be (src, tgt, l)
					return X.name,Y.name,l
				
				if spec == l: # could just be label name
					if label is None: 
						gn, tn, label = X.name, Y.name, l
						break
					else: raise ValueError("Spec is not unique! Matching edges: ", 
							["%s : %s -> %s " % lxy for lxy in self.edges("l,Xn,Yn") if lxy[0] == spec])
				
				elif specX is not None and specY is not None and \
					all(a.name in specX for a in X.atoms) and all(a.name in specY for a in Y.atoms):
						if gn is None and tn is None:
							gn,tn = X.name, Y.name
							break
						else: raise ValueError(f"Couldn't uniquely decide on edge:  ({gn}->{tn})  vs ({X.name}->{Y.name})")
			else:
				# print("all edges: ", [','.join(xyl) for xyl in self.edges("Xn,Yn,l")])
				raise ValueError("no edge matches string specification \"%s\""%spec)
		else:
			raise ValueError("did not understand edge spec \"%s\"---not a conditionrequest, tuple, or string."%repr(spec))
		
		if label == None:
			if len(self.graph[gn][tn]) == 1:
				label = list(self.graph[gn][tn])[0]
			else:
				raise ValueError("Spec is not unique! Matches edges: ", repr(list(self.graph[gn][tn])))
		
		if (gn,tn,label) in self.edgedata:
			return gn,tn,label
		else:
			raise ValueError("no edge matching: '"+ repr(spec) +"'")
			

	def copy_with_params(self, **kwargs):
		rslt = self.copy()
		for param, val in kwargs.items():
			if type(val) is dict:
				for spec, defn in val.items():
					rslt.edgedata[rslt._get_edgekey(spec)][param] = defn
		return rslt
	
	def set_alpha(self, edge_spec, α):
		self.edgedata[self._get_edgekey(edge_spec)]['alpha'] = α
	
	def set_beta(self, edge_spec, β):
		self.edgedata[self._get_edgekey(edge_spec)]['beta'] = β
		

	def update_all_weights(self, a=None, b=None):
		for xyl in self.edges('Xn,Yn,l'):
			if a is not None:
				self.edgedata[xyl]['alpha'] = a 
			if b is not None:
				self.edgedata[xyl]['beta'] = b 

	def _apply_structure(self, focus=None):
		if focus is None:
			focus = self.vars

		for f in focus:
			var = self.vars[f] if type(f) is str else f

			for vstructure in var.structure:
				for relation in vstructure.gen_cpts_for(self):
					self += relation

	# should only need to use this during a copy.
	# def _rebuild_graph(self):
	#     for vn, v in self.vars.items():
	#         self.graph.add_node(vn, {'var' : v})
	#
	#     for (f,t,l), attr in self.edgedata:
	#         self.graph.add_edge(f,t, key=l, attr=dict(label=l,**attr))

	def _include_var(self, var, varname=None):
		if varname:
			if not hasattr(var,'name'):
				var.name = varname
			assert varname == var.name, "Variable has thinks its name is different from the PDG does..."

		if not hasattr(var,'name') or var.name == None:
			raise ValueError("Must name variable before incorporating into the PDG.")


		self.graph.add_node(var.name, var=var)
		if var.name in self.vars:
			self.vars[var.name] |= var.ordered
		else:
			self.vars[var.name] = var
			self._apply_structure([var])


	def _set_edge(self, nf: str, nt: str, l, **attr):
		edkey = self.graph.add_edge(nf,nt,l, **attr)
		self.edgedata[nf,nt,edkey] = self.graph.edges[nf,nt,edkey]


	def add_data(self, data, label=None):
		""" Include (collection of) (labeled) cpt(s) or variable(s) in this PDG.

		Adding a PDG itself computes a union; one can also add an individual
		cpt, or a variable, or a list or tuple of cpts.

		Can be given labels by
		```
		>  M: PDG  += "ℓ", p : CPT
		```
		"""

		if isinstance(data, PDG):
			for varname, var in data.vars.items():
				self._include_var(var,varname)

			for ftl, attr in data.edgedata.items():
				self._set_edge(*ftl, **attr)

		elif isinstance(data, CPT):
			self._include_var(data.src_var)
			self._include_var(data.tgt_var)
			# label = other.name if hasattr(other, 'name') else \
			#     self.labeler.fresh(other.src_var,other.tgt_var)
			if label is None:
				label = self.labeler.fresh(data.src_var.name, data.tgt_var.name)

			self._set_edge(data.src_var.name, data.tgt_var.name, label, cpd=data)
		
		elif isinstance(data, ConditionRequest):
			self._include_var(data.given)
			self._include_var(data.target)    
			if label is None:
				label = self.labeler.fresh(data.given.name, data.target.name)
			self._set_edge(data.given.name, data.target.name, label)

		elif isinstance(data, Variable):
			vari = data.copy()
			if label and not hasattr(vari, 'name'):
				vari.name = label
			self._include_var(vari)

		elif type(data) in (tuple,list):
			for o in data:
				self.add_data(o, label)
		elif type(data) is dict:
			# note: still handles \beta and \alpha properly.
			if 'from' in data and 'to' in data:
				X = data['from'], Y = data['to']
				if isinstance(X, Variable): XN = X.name
				if isinstance(Y,Variable):  YN = Y.name
			if 'cpd' in data:
				XN = data['cpd'].src_var.name
				YN = data['cpd'].tgt_var.name
			if 'label' in data:
				label = data['label']
			else:
				label = self.labeler.fresh(XN, YN)

			self._set_edge(XN, YN, label, **dictwo(data, ('from', 'to', 'label')))

		elif type(data) is str:
			idx1 = data.find('->')
			idx2 = data.find('(', idx1)
			idx3 = data.find(')', idx2)

			XN = data[0:idx1].strip()
			YN = data[idx1+2:idx2].strip()
			parens = data[idx2+1:idx3].strip()

			def kv(bit):
				head, tail = bit.split('=')
				try:
					return (head, float(tail))
				except:
					return (head, tail)

			print(kv(parens.split(';')[0]))
			self._set_edge(XN, YN, label, **(dict( kv(bit) for bit in parens.split(";")) if idx3 > idx2 else {}) )

		else:
			print("Warning: could not add data", data, "to PDG")
			return -1


	def __iadd__(self, other):
		if type(other) is tuple and len(other) > 1 and type(other[0]) in (str,int):
			self.add_data(other[1], label=other[0])
			self += other[2:]
		else:
			self.add_data(other)

		return self


	def __add__(self, other):
		rslt = self.copy()
		rslt += other;
		# return (self.copy() += other)
		return rslt

	def __delitem__(self, key):
		if isinstance(key, tuple):
			k = list(key)
			if len(key) in [2,3]:
				if isinstance(key[0], Variable):
					k[0] = key[0].name
				if isinstance(key[1], Variable):
					k[1] = key[1].name

				if(len(key) == 2):
					k += [next(iter(self.graph[k[0]][k[1]].keys()))]

				self.graph.remove_edge(*k)
				del self.edgedata[tuple(k)]

		if isinstance(key, str):
			if key in self.vars:
				del self.vars[key]
				self.graph.remove_node(key)
			else:
				if key in self.edges("l"):
					self.__delitem__(self._get_edgekey(key))

	def __getitem__(self, key):
		""" 
		takes an edge specification, and returns the relevant data as a namedtuple
		Alternatively, takes a string name."""
		# label = None
		# if isinstance(key, ConditionRequest):
		#     gn,tn = key.given.name, key.target.name
		#     # raise KeyError("Multiple options possible.")
		# elif type(key) == tuple and type(key[0]) is str:
		#     # normal strings can be looked up as a tuple
		#     gn,tn = key[:2]
		#     if len(key) == 3:
		#         label = key[2]
		# else:
		#     try:
		#         gn,tn,label = self._get_edgekey(key)
		#     except:
		#         print(key, 'is not a valid key')
		#         return
		# 
		# if label == None:
		#     if len(self.graph[gn][tn]) == 1:
		#         return self.graph[gn][tn]['cpd']
		# 
		#     return next(iter(self.graph[gn][tn].values()))['cpd']
		#
		# return self.edgedata[gn,tn,label]['cpd']
		return self.edgedata[self._get_edgekey(key)]['cpd']
		
	def __iter__(self):
		return self.edges("XYPαβ")

	def _fmted(self, XnameYnamel, fmt):
		"""
		Return data requested by `fmt` in the edge specified by `XnameYnamel`. 
		Concretely, the specified edge is the unique one from `Xname` to `Yname` with label `l`, where (Xname, Yname, l) = XnameYnamel.
		For details on `fmt`, see the documentation for the method `edges()`. 

		Gives defaults of α = 1 and β = 1  when unspecified.
		""" # TODO: revist the defaults. alpha = 0 is probably more appropriate, but it might break some things. Also it may be less interesting. 
		
		Xname, Yname, l = XnameYnamel
		data = self.edgedata[XnameYnamel]
		X,Y = self.vars[Xname], self.vars[Yname]
		
		# The keys  of the following dict are not present in data, but we add them now.
		# They are the cannonical names. 
		lookup = dict(X=X,Y=Y,Xname=Xname,Yname=Yname,alpha=1,beta=1,label=l)
		# TODO: maybe alpha=0?
		lookup.update(**data)
		
		lookup['S'] = lookup['X']
		lookup['T'] = lookup['Y']
		lookup['Sn'] = lookup['Xn'] = lookup['Xname']
		lookup['Tn'] = lookup['Yn'] = lookup['Yname']
		lookup['l'] = lookup['L'] = lookup['label']
		lookup['α'] = lookup['alpha']
		lookup['β'] = lookup['beta']
		lookup['P'] = lookup['p'] =  data.get('cpd',None)

		return tuple(lookup.get(s,None) for s in fmt) if len(fmt) > 1 \
			else lookup.get(fmt[0],None)
			
			
	### Next two methods _idx, _idxs, are stolen from dist. Still useful here.
	### # TODO: make them point to the same code 
	def _idxs(self, *varis, multi=False):
		idxs = []
		for V in varis:
			for a in V.atoms:
				i = self.varlist.index(a)
				if multi or (i not in idxs):
					idxs.append(i)
		return idxs

	def edges(self, fmt='XY'): # TODO change default to names, which is better for printing.
		"""
		A generator that iterates over edges, yielding data for each hyper-arc, of the requested data. 
		
		 - "X", "S", -> produce source variable : `rv.Variable`
		 - "Xname", "Xn", "Sn", "source" -> produce name of source variable : `str`
 		 - "Y", "T",  -> produce target variable : `rv.Variable`
 		 - "Yname", "Yn", "Tn", "target" -> produce name of source variable : `str`
		 - "L", "l", "label" -> produce name of the arc itself : `str`
		 - "cpd", "P" -> produce the probabilities : `dist.CPT`
		 - "alpha", "α" -> produce the qualitative weight : `float`
		 - "beta", "β" -> produce the quantitative weight : `float`
	 

		Examples usage:
		```
		M.edges("X,Y,cpd,α,β")
		M.edges("XYLp")
		M.edges(['X', 'Y'])
		```
		
		Functionality is based on the `_fmted` helper method.
		"""
		if type(fmt) is str: # could also be list, in which case we assume splitting has already been done. 
			delims = ',; ' # possible delimeters, in order of priority
			# if a delimeter is present, chunk by that delimeter
			for d in delims:
				if d in fmt:
					fmt = fmt.split(d)
					break
			# if no delimeter was present, then just use the chars of the string.
			# TODO: decide how to address problem where 'Xn' by itself produces  (X, none). 
			# Probably we want to depricate the super condensed style. We 

		for xn_yn_l in self.edgedata:
			yield self._fmted(xn_yn_l, fmt)

	def genΔ(self, kind=RJD.random, repr="atomic"):        
		d = kind(self.getvarlist(repr))
		return d

	##### OTHERS ##########
	def make_edge_mask(self, distrib):
		"""returns a PDG with the same shape as this one,
		but with the given distribution's marginals"""
		M = PDG()

		for name,V in self.vars.items():
			M += name, V
			# print(name, self.vars[name].structure, M.vars[name].structure)

		for X,Y, l in self.edges('XYl'):
			# print('edge  ', X.name,'->', Y.name,beta, self.edgedata)
			M += l, distrib.conditional_marginal(Y | X)
		return M

	def factor_product(self, repr="atomic", return_Z=False) -> RJD:
		""" pretend the PDG is a factor graph, with weights θ := β """ 
		# start with uniform
		# d = RJD.unif(self.atomic_vars)

		d = self.genΔ(RJD.unif, repr)
		# The above already has divided by |d|, messing up the computation
		# of the normalization constant.  
		
		for X,Y,cpt,β in self.edges("XYPβ"):
			if cpt is not None:
				#hopefully the broadcast works...
				d.data *= np.nan_to_num( d.broadcast(cpt) ** β, nan=1)
			# print(d.data)

		Z = d.data.sum()
		d.data /= Z
		Z *= d.data.size
		return (d,Z) if return_Z else d
	
	

	# semantics 1:
	def matches(self, mu):
		for X,Y, cpd in self.edges("XYP"):
			# print(mu[Y], '\n', mu[X], '\n', cpd)
			if( not np.allclose(mu[Y], mu[X] @ cpd) ):
				return False

		return True

########## INFERENCE ALGORITHMS ########



		
		
	def _build_fast_scorer(self, weightMods=None, gamma=None, repr="atomic", return_grads=True):
		N_WEIGHTS = 5
		if weightMods is None:
			weightMods = [lambda w : w] * N_WEIGHTS
		else:
			weightMods = [
					(lambda n: lambda b: n)(W) if isinstance(W, Number) else
					W if callable(W) else
					(lambda b: b)
				for W in weightMods]

		if gamma == None:
			gamma = self.gamma_default

		weights = np.zeros((N_WEIGHTS, len(self.edgedata)))
		for i,(X,Y,cpd, alpha,beta) in enumerate(self):
			w_suggest = [beta, -beta, alpha*gamma, 0, 0]
			for j, (wsug,wm) in enumerate(zip(w_suggest, weightMods)):
				weights[j,i] = wm(wsug)

		# SHAPE = self.dshape
		mu = self.genΔ(RJD.unif, repr)
		SHAPE = mu.data.shape
		Pr = mu.prob_matrix

		# print(weights)

		# PENALTY = 0
		# Maybe the penalty is the problem...
		PENALTY = 101.70300201
			# A large number unlikely to arise naturally with
			# nice numbers.
			# Very hacky but faster to do in this order.

		def score_vector_fast(distvec, debug=False):
			# enforce constraints here...
			# if(distvec.sum() < 0.001):
				# print(np.unique(distvec))
			if debug:
				print("penalty of ", PENALTY)
				print("weights initialized to ", weights)
				print('initial input sums to ', distvec.sum(), ',  has bounds  [', distvec.min(),', ',distvec.max(),'], and shape ', distvec.shape, ' but converted to ', SHAPE,'.\n')

			distvec = np.abs(distvec).reshape(*SHAPE)
			# distvec /= distvec.sum()
			mu.data = distvec # only create one object...

			gradient = np.zeros(distvec.shape)
			thescore = 0

			for i, (X,Y,cpd_df) in enumerate(self.edges("XYP")):
				# muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
				muxy = Pr([X, Y])
				muy_x = Pr(Y | X)
				# eq = (np.closecpt - muy_x
				if debug:
					print('\n\n')
					print("for edge ", X.name, " -> ", Y.name)
					print('weights', weights[:,i])
					# print('and the μ(%s | %s) = ' %(X.name,Y.name), muy_x)

				logcond_info = (-np.ma.log(muy_x)).filled(PENALTY)

				if cpd_df is None:
					logliklihood = 0
					logcond_claimed = 0
				else:
					cpt = mu.broadcast(cpd_df)
					claims = np.isfinite(cpt)

					if debug: print("the cpt is ", cpt)

					logliklihood = np.where(claims, (-np.ma.log(cpt)).filled(PENALTY), 0)
					logcond_claimed = np.where(claims, logcond_info, 0)

				# logextra = z_mult(mux * cpt, logcpt.filled(PENALTY))

				if debug: print('\# masked elements in log p:', np.ma.count_masked(np.ma.log(cpt)),
					' and after filling:  ', np.ma.count_masked((-np.ma.log(muy_x)).filled(PENALTY)),
					"\ngraident before:", gradient.reshape(-1))
				gradient += weights[0,i] * ( logliklihood )
				if debug: print('\# masked elements in log μ:', np.ma.count_masked(np.ma.log(muy_x)),
					' and after filling:  ', np.ma.count_masked((-np.ma.log(muy_x)).filled(PENALTY)),
					"\ngraident between:", gradient.reshape(-1))
				gradient += weights[1,i] * (  logcond_claimed )
				gradient += weights[2,i] * ( logcond_info )
				if debug: print("graident after:", gradient.reshape(-1))


				# print("masked", np.ma.count_masked(logliklihood), np.ma.count_masked(logcond_info), end='\t')

				thescore += weights[0,i] * z_mult(muxy, logliklihood).sum()
				thescore += weights[1,i] * z_mult(muxy, logcond_claimed).sum()
				thescore += weights[2,i] * z_mult(muxy, logcond_info).sum()
				# thescore += weights[2,i] * logextra.filled(PENALTY).sum()

			gradient += gamma * ( np.ma.log( distvec ))#.filled(PENALTY)
				# we now subtract + mu.H(...), but H is negative log, so it's still positive...
			gradient /= np.log(2)

			thescore /= np.log(2)
			thescore -= gamma * mu.H(...) # Returns log base 2 by default, so no need to divide
										# by log 2 again...
			gradient -= thescore


			# print(gradient.min(), gradient.max(), np.unravel_index(gradient.argmax(), gradient.shape))

			# print('score', thescore, 'entropy', mu.H(...), "distvec sum", distvec.sum())
			return (thescore, gradient.reshape(-1)) if return_grads else thescore

		# def score_gradient(distvec):
		#     distvec = np.abs(distvec).reshape(*SHAPE)
		#     distvec /= distvec.sum()
		#     mu.data = distvec # only create one object...
		#
		#     gradient = np.zeros(distvec.shape)
		#     for i, (X,Y,cpd_df,alpha,beta) in enumerate(self):
		#         # muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
		#         muy_x = Pr(Y | X)
		#         cpt = mu.broadcast(cpd_df)
		#
		#         logliklihood = - np.ma.log(cpt).filled(PENALTY)
		#         logcond_info = - np.ma.log(muy_x).filled(PENALTY)
		#         # logextra = z_mult(mux * cpt, logcpt.filled(PENALTY)
		#
		#         gradient += weights[0,i] * (logliklihood - distvec.dot(logliklihood))
		#         gradient += weights[1,i] * (logcond_info - distvec.dot(logcond_info))
		#         # thescore += weights[2,i] * logextra.filled(PENALTY).sum()
		#
		#     gradient += gamma * (np.ma.log( distvec ) + mu.H(...))
		#     gradient /= np.log(2)
		#     return gradient

		return score_vector_fast

	#  semantics 2
	def score(self, mu : RJD, weightMods=None, gamma=None):
		if gamma is None:
			gamma = self.gamma_default
		if weightMods is None:
			weightMods = [lambda w : w] * 5
		else:
			weightMods = [
				(lambda n: lambda b: n)(W) if isinstance(W, Number) else
				W if callable(W) else
				(lambda b: b)
					for W in weightMods]

		thescore = 0
		infoscores = np.zeros(mu.data.shape)

		for X,Y,cpd_df,alpha,beta in self:
			# This could easily be done 3x more efficiently
			# look here if optimization reqired.
			Pr = mu.prob_matrix
			muy_x, muxy, mux = Pr(Y | X), Pr([X, Y]), Pr(X)
			cpt = mu.broadcast(cpd_df)
			claims = np.isfinite(cpt)
			logcpt = - np.ma.log(cpt)


			### Liklihood.              E_mu log[ 1 / cpt(y | x) ]
			logliklihood = z_mult(muxy*claims, logcpt)
			### Local Normalization.    E_mu  log[ 1 / mu(y | x) ]
			logcond_info = z_mult(muxy, -np.ma.log(muy_x) )
			logcond_claimed = z_mult(muxy*claims, -np.ma.log(muy_x) )
			### Extra info.         E_x~mu cpt(y|x) log[1/cpt(y|x)]
			logextra = z_mult(mux * cpt, logcpt)
			### Dependency Network Thing.
			nonXorYs = [Z for Z in self.varlist if Z is not Y and Z is not X]
			dnterm = z_mult(muxy, -np.ma.log(Pr(Y|X,*nonXorYs)))

			weights = [beta, -beta, alpha*gamma, 0, 0]
			terms = [logliklihood, logcond_claimed, logcond_info, logextra, dnterm]

			# print(f"Weights for {X.name} -> {Y.name}", weights)
			for term, λ, mod in zip(terms, weights, weightMods):
				# adding the +1j here lets us count # of infinite
				# terms. Any real score is better than a complex ones
				thescore += mod(λ) * (term).astype(complex).filled(1j).sum()
				infoscores += mod(λ) * term # not an issue here


		################# λ4 ###################
		thescore /= np.log(2)
		thescore -= gamma * mu.H(...)

		return thescore.real if thescore.imag == 0 else thescore

	# TODO: Not sure if these are working properly in the presence of incomplete cpts..
	def Inc_ptwise(self, p, ed_vector=False):
		"""
		Used to be the implementation of Inc. 
		Computes a joint distribution of shape (m, *dshape),
		which might somehow be interesting, but then then sums over remaining axes,
		so it still doesn't expose this to the user.

		SOme downsides:
		 - doesn't work for cluster distributions and 
		 - wasn't even implemented for torch. 
		 - returned a complex number, confusingly.

		So it's been replaced by `Inc` for the most part.
		""" 
		if p._torch:
			raise NotImplementedError()
		
		Prp = p.prob_matrix
		# n_cpds = len(self.edgedata) # of edges
		Incv = np.zeros((len(self.edgedata),*p.shape),dtype=np.complex_)
		for i,(X,Y,cpd_df,alpha,β) in enumerate(self):
			cpd = p.broadcast(cpd_df)
			claims = np.isfinite(cpd)
			Incv[i,...] = β * p.data * (np.ma.where(claims, np.ma.log(  zz1_div(Prp(Y | X ), cpd) ), 0)) \
				.astype(np.complex_).filled(1j)

		Incv /= np.log(2)
		if ed_vector:
			return Incv.sum(axis=tuple(range(1, Incv.ndim)))
		return Incv.sum()

	def IDef_old(self, p, ed_vector=False):
		Prp = p.prob_matrix

		if p._torch and (ed_vector is False):  # if you insist on grads...
			idef = torch.tensor(0.)
			for i,(X,Y,α) in enumerate(self.edges("XYα")):
				# pxy = Prp(X,Y)
				pxy = p.data
				plogp = (pxy * torch.log( torch.where(pxy==0, 1., Prp(Y | X))))
				idef +=  - α * plogp.sum()
				# print(X.name, Y.name, temp.shape, temp.sum())
			idef = idef/np.log(2) -  p.H(...)
			return idef 
		
		else: # we can do the interesting thing (which is slower and maybe stupider)
			if p._torch:
				p = RJD(p.data.detach().numpy(), p.varlist, False)
				Prp = p.prob_matrix
			# n_cpds = len(self.edgedata) # of edges
			IDefv = np.zeros((len(self.edgedata)+1,*p.shape))
			for i,(X,Y,α) in enumerate(self.edges("XYα")):
				IDefv[i,...] = α * p.data * ( - np.ma.log(  Prp(Y | X ) ))

			IDefv[i+1,...]  += p.data * np.ma.log(p.data)
			IDefv /= np.log(2)

			if ed_vector:
				return IDefv.sum(axis=tuple(range(1, IDefv.ndim)))
			return IDefv.sum()


	def Inc(self, mu, ed_vector=False):
		""" New computation of Inc that works for torch and ClusterDists. """ 
		
		Pr = lambda *query : mu.conditional_marginal(query, query_mode="ndarray")
		# n_cpds = len(self.edgedata) # of edges
		# Incv = np.zeros((len(self.edgedata),*p.shape),dtype=np.complex_)
		m = len(self.edgedata)
		if mu._torch:
			inc = torch.zeros(m) if ed_vector else torch.tensor(0.)
		else: inc = np.zeros(m) if ed_vector else 0.

		for i,(X,Y,cpd_df,beta) in enumerate(self.edges("XYPβ")):
			# cpd = cpd_df.broadcast_to([Y,X])
			cpd = cpd_df.broadcast_to(list((Y&X).atoms))

			if mu._torch:
				claims = torch.isfinite(torch.tensor(cpd))
				# edgeinc = beta * torch.sum(Pr(Y,X) * torch.where(claims,
				# 	torch.log(Pr(Y|X)) - torch.log(torch.tensor(cpd)) , 0.))
				edgeinc = beta * torch.sum( tzmul( Pr(Y,X), torch.where(claims,
					torch.log(Pr(Y|X)) - torch.log(torch.tensor(cpd)) , 0.)))
			else:
				claims = np.isfinite(cpd)
				# edgeinc = beta * np.sum(Pr(Y,X) * (np.ma.where(claims,
				# 	np.ma.log(  zz1_div(Pr(Y|X), cpd) ), 0)).filled(np.inf))
				edgeinc = beta * z_mult(Pr(Y,X),
						np.ma.where(claims, np.ma.log(zz1_div(Pr(Y|X), cpd)), 0)
					).filled(np.inf).sum()  
			
			if ed_vector:
				inc[i] = edgeinc
			else:
				inc += edgeinc
		
		return inc
		
	def IDef(self, mu, ed_vector=False):
		Pr = lambda *query : mu.conditional_marginal(query, query_mode="ndarray")
	
		m = len(self.edgedata)
		if mu._torch:
			idef = torch.zeros(m) if ed_vector else torch.tensor(0.)
		else: idef = np.zeros(m) if ed_vector else 0.

		lib = torch if mu._torch else np

		for i,(X,Y,α) in enumerate(self.edges("XYα")):
			mu_yx = Pr(Y,X)
			
			mulogmu = (mu_yx * lib.log( lib.where(mu_yx==0, 1., Pr(Y|X))))
			wH = - α * mulogmu.sum() # weighted entropy 

			if ed_vector: idef[i] = wH
			else: idef +=  wH

		idef = idef/np.log(2) -  mu.H(...)
		return idef 

	def scoreII(self, mu, gamma):
		return self.Inc(mu) + self.IDef(mu) * gamma		

	####### SEMANTICS 3 ##########
	def optimize_score(self, gamma, repr="atomic", store_iters=False, **solver_kwargs ):
		scorer = self._build_fast_scorer(gamma=gamma, repr=repr,
			return_grads = not('jac' in solver_kwargs and solver_kwargs['jac'] == False))
		factordist = self.factor_product(repr=repr).data.reshape(-1)

		init = self.genΔ(RJD.unif).data.reshape(-1)
		# alternate start:
		# init = factordist
		if store_iters: iters = [ np.copy(init.data) ]

		from scipy.optimize import minimize, Bounds, LinearConstraint

		req0 = (factordist == 0) + 0
		# rslt = minimize(scorer,
		#     init,
		#     constraints = [LinearConstraint(np.ones(init.shape), 1,1), LinearConstraint(req0, 0,0.01)],
		#     bounds = Bounds(0,1),
		#         # callback=(lambda xk,w: print('..', round(f(xk),2), end=' \n')),
		#     method='trust-constr',
		#     options={'disp':False}) ;

		# solver_args = dict(
		#     method='trust-constr',
		#     bounds = Bounds(0,1),
		#     constraints = [LinearConstraint(np.ones(init.shape), 1,1)],
		#     jac=True, tol=1E-25)
		solver_args = dict(
			method = 'SLSQP',
			options = dict(ftol=1E-20, maxiter=500),
			bounds = Bounds(0,1),
			constraints = [LinearConstraint(np.ones(init.shape), 1,1)],
			jac=True)

		if store_iters:
			solver_args['callback'] = lambda xk: iters.append(xk)

		solver_args.update(**solver_kwargs)

		rslt = minimize(scorer,
			init +  np.random.rand(*init.shape)*1E-3 * (1-req0),
			**solver_args)
		self._opt_rslt = rslt

		rsltdata = abs(rslt.x).reshape(self.getdshape(repr))
		rsltdata /= rsltdata.sum()
		rsltdist =  RJD(rsltdata, self.getvarlist(repr))

		return (rsltdist, iters) if store_iters else rsltdist
		# TODO: figure out how to compute this. Gradient descent, probably. Because convex.
		# TODO: another possibly nice thing: a way of testing if this is truly the minimum distribution. Will require some careful thought and new theory.
		
	# def optimize_score_torch(self, gamma, store_iters=False, **solver_kwarg):
	#     pass


	def iter_GS_beta(self, 
			max_iters=600, tol=1E-30, 
			counterfactual_recalibration=False,
			init = None,
			store_iters=False, 
			repr='atomic') -> RJD:
		
		if init == None: init = RJD.unif
		if isinstance(init, RJD): dist = init
		else: dist = self.genΔ(init, repr)
		
		iters = [ np.copy(dist.data) ]
		totalβ = sum(β for β in self.edges("β"))
		
		τ = self.careful_cpd_transform if counterfactual_recalibration else self.GS_step 

		for it in range(max_iters):
			nextdist = np.zeros(dist.data.shape)
			for X,Y,cpd,β in self.edges("XYPβ"):
				nextdist += (β / totalβ) * τ(dist, (X,Y,cpd))

			change = np.sum((dist.data - nextdist) ** 2 )
			dist.data = nextdist

			if store_iters: iters.append(nextdist)
			else: iters[-1] = nextdist


			if change < tol:
				break
		else:
			warnings.warn('hit max iters but dist still changing; last change of size {:.2e} (tol = {:.2e})'.format(change,tol))

				# if change == 0: break
		return (dist, iters) if store_iters else dist

	def iter_GS_ordered(self, edge_order=None, 
			max_iters: Number = 200,  tol=1E-20, 
			counterfactual_recalibration=False,
			init = None,
			store_iters=False, 
			repr="atomic") -> RJD:
		"""
		Does the steps of iterative (pseudo-)(ordered-)Gibbs sampling, 
		but with a full distribution in
		
		:param init: the joint distribution we initialize to. Uniform if None.
		"""
		
		# First, sort out ordered_XYP list from the (supplied?) order
		if type(edge_order) is list:
			ordered_XYP = []
			for spec in edge_order:
				Xn,Yn,l = self._get_edgekey(spec)
				dat = self.edgedata[Xn,Yn,l]
				ordered_XYP.append((self.vars[Xn],self.vars[Yn],dat['cpd']))
			# The below should be the same.
			# ordered_XYP = [self._fmted(self._get_edgekey(spec), ['X','Y','P']) for spec in ordered_edges]
		elif edge_order is None or edge_order == "shuffle":
			ordered_XYP = list(self.edges("XYP"))
	
		if init == None: init = RJD.unif
		if isinstance(init, RJD): dist = init
		else: dist = self.genΔ(init, repr)
			
		iters = [ np.copy(dist.data) ]

		τ = self.careful_cpd_transform if counterfactual_recalibration else self.GS_step 
		
		for it in range(max_iters):
			for XYp in ordered_XYP:
				dist.data = τ(dist, XYp)

			change = ((dist.data - iters[-1]) ** 2 ).sum()

			if store_iters:
				iters.append(np.copy(dist.data))
			else:
				iters[-1] = dist.data


			if change < tol: break
			if edge_order == 'shuffle':
				random.shuffle(ordered_XYP)
		else:
			warnings.warn('hit max iters but dist still changing; last change of size {:.2e} (tol = {:.2e})'.format(change,tol))
		
		# return self.iterGS(init=self.genΔ(RJD.unif, repr), cpdgen=cpdgen)
		return (dist, iters) if store_iters else dist

	def GS_step(self, dist : RJD, XYP) -> RJD:
		"""
		Computes the transformation
			(Q(X,Y,Z), P(Y|X)) ↦ Q(XZ) P(Y|X)

		In other words, it performs a Gibbs Sampling procedure with a variable on the distribution `dist`
		according to the edge XYP = (X: Var, Y: Var, cpd:CPD[Y|X] ).
		"""
		X,Y,cpd = XYP
		not_target = list(v for v in self.rawvarlist if
			len(set(v.name.split('×')) & set(Y.name.split("×"))) == 0)
				# Get the cpd from all variables that do not share a name with target.
		return dist.prob_matrix([*not_target,X]) * dist.broadcast(cpd)

	def careful_cpd_transform(self, dist : RJD, XYP) -> RJD:
		""" 
		Computes the transformation
		( Q(XYZ), P(Y|X)) ↦ Q(X) P(Y|X) Q(Z|XY)
		Sometimes referred to elsewhere in code as 'counterfactual recalibration'. 
		"""
		X,Y,cpd = XYP
		# P(Y|X) * 
		return dist.broadcast(cpd) * dist.data * dist.prob_matrix(X) / dist.prob_matrix([X,Y])
	
	# def 
		
	def mk_edge_transformer(self, spec, reweight=True):
		X,Y,cpd = self._fmted(self._get_edgekey(spec), ['X','Y','P'])
		
		if reweight:
			def apply(dist):
				return RJD(dist.data * dist.broadcast(cpd) / dist.prob_matrix(Y|X), dist.varlist)
				# return RJD(dist.data * dist.prob_matrix(X) * dist.broadcast(cpd) / dist.prob_matrix(X,Y), dist.varlist)

		else:
			not_target = list(v for v in self.rawvarlist if
				len(set(v.name.split('×')) & set(Y.name.split("×"))) == 0)
			
			def apply(dist):
				return RJD(dist.prob_matrix(*not_target,X) * dist.broadcast(cpd), dist.varlist)

		return apply
		
		
		
	def MCMC(self, iters=200):
		# import random
		from pandas import DataFrame
		
		# history = []
		history_df = DataFrame(columns = [X.name for X in self.atomic_vars])
		try:
			sample = {X.name : X.default_value for X in self.atomic_vars }
		except:
			sample = {X.name : next(iter(X)) for X in self.atomic_vars }
		
		totalβ = 0 # sum(β for β in self.edges("β"))
		bkpts = []
		for i,(β,l)  in enumerate(self.edges("βl")):
			totalβ += β
			bkpts.append( (totalβ, l) )
			
		def draw_edge_by_beta():
			u = np.random.rand() * totalβ
			for (b, l) in bkpts:
				if u < b:
					return l
		
		for it in range (iters):
			l = draw_edge_by_beta()
			X,Y,cpd = self._fmted(self._get_edgekey(l), ["X", "Y", "P"])
			Xn, Yn = X.name, Y.name

			newy = cpd.sample(sample[Xn])
			
			# FIXME this is very inefficient  
			# for h in history: 
			#     if h[X.name] == sample[X.name] and h[Y.name] == newy:
			#         newsample = { x : v  for x,v in h.items() }
			#         random.shuffle(history) 
			#         break
			
			# history_df[X.name] == sample[X.name] & history_df
			rows = history_df[(history_df[Xn]==sample[Xn]) & (history_df[Yn]==newy)]
			if not rows.empty:
				newsample = rows.sample().iloc[0].to_dict()
			else:
				newsample = { x : v  for x,v in sample.items() }
				newsample[Y.name] = newy
			
		
			yield newsample
			# history.append(sample)
			history_df = history_df.append(sample,ignore_index=True)
			sample = newsample
	


	############# Utilities ##############
	def random_consistent_dists(self, how_many,
				transform_iters=1000,
				ret_initializations=False
			):
		"""
		Gives random distributions consistent with the PDG,
		by repeatedly running the consistency-improving transformer on it.
		"""
		dists = []
		for i in range(how_many):
			μ0 = self.genΔ(RJD.random)
			μ = self.iter_GS_ordered(edge_order="shuffle", 
				max_iters=transform_iters,
				init=μ0,
				counterfactual_recalibration=True,
				# store_iters=False
				)
			dists.append(μ)
		
		return dists
		

	def standard_library(self, repr='atomic'):
		from .store import TensorLibrary
				
		lib = TensorLibrary(decoder = lambda vec : RJD(vec, self.varlist))

		def store(*a, **b):
			def _store_inner(distrib, iterdatalist= []):
				lib(*a,**b).set(distrib)
				for i,io in enumerate(iterdatalist):
					lib(*a, **b, i=i).set(io)

			return _store_inner

		store('φ', 'product')(self.factor_product(repr))
		store('gibbs','GS','β')(*self.iter_GS_beta(repr=repr, store_iters=True))
		store('gibbs','GS','≺', 'ordered')(*self.iter_GS_ordered(repr=repr, store_iters=True))
		store('opt',γ=0)(*self.optimize_score(0, repr=repr, store_iters=True, tol=1E-20))

		STD_GAMMAS = [1E-20, 1E-10,1E-7, 1E-5, 1E-4, 1E-3, 1E-2, .1,
			.2, .5, .8, .9, .999, 1, 1.001, 1.1, 1.5, 2, 3]

		for γ in STD_GAMMAS:
			store('opt', γ=γ)( self.optimize_score(γ, repr=repr, store_iters=False, tol=1E-20) )
		return lib

	def draw(self):
		nx.draw_networkx(self.graph)

	# def random_consistent_dists():
	#     """ Algorithm:
	#     """
	#     pass
	#########  CONVERSIONS ########
	# @classmethod

	@staticmethod
	def from_BN(bn : BayesianNetwork):
		pdg = PDG();
		# varis = { vname : Variable(vals, name=vname) 
		# 	for (vname,vals) in  bn.states.items() }
		# for n,V in varis.items():
		# 	pdg += n, V

		for (vname,vals) in  bn.states.items():
			pdg += vname, Variable(vals, name=vname)

		for cpd in bn.cpds:
			pdg += CPT.from_pgmpy(cpd)

		return pdg
	

	@staticmethod
	def from_FG( fg : FactorGraph ):
		pdg = PDG()

		# TODO
		raise NotImplemented
		return pdg;



	def to_FG(self, via='β', drop_joints=False) -> FactorGraph:
		"""
		"""
		factors = []
		for L,X,Y,cpt,power in self.edges("LXYP"+via):
			if drop_joints and L[0] == 'π':
				continue
			
			if cpt is not None:
				br = cpt.broadcast_to(self.varlist)
				factors.append(np.nan_to_num(
						cpt.broadcast_to(self.varlist) ** power, nan=1))
		
		return FactorGraph(factors, self.varlist)
	
	def to_markov_net(self, via='β'):
		## This would be easiest but unfortunately doesn't always work
		## for a stupid reason: numpy arrays can only have 32 dimensions.
		# return self.to_FG(via).to_pgmpy_markov_net();
		## ... so instead,
		
		from pgmpy.models import MarkovNetwork
		from pgmpy.factors.discrete import DiscreteFactor
		from itertools import combinations
		
		mn = MarkovNetwork()
		mn.add_nodes_from([V.name for V in self.varlist])

		for L,X,Y,cpt,power in self.edges("LXYP"+via):
			# scope, card = zip(*[(v.name, len(v)) 
			# 	for (i,v) in enumerate(self._varlist) if f.shape[i] > 1])
			
			# don't need these extra coherence edges ...
			if L[0] == 'π': #  (it's a projection)
				continue 

			scope, card = zip(*[(v.name, len(v)) for v in (X&Y).atoms])
			
			# print(X.name, Y.name)

			mn.add_edges_from(combinations(scope,2));
			mn.add_factors( DiscreteFactor(scope, card, cpt.to_numpy() ** power ))

		# this is very stupid, but apparently all variables need factors
		# (because variables are just strings, so you don't know how many
		# values you have to make a distribution until all vars have factors).
		# Anyways, we go multiplying by one ...
		for vn in set(mn.nodes()) - set(mn.get_cardinality().keys()):
			vl = len(self.vars[vn])
			mn.add_factors( DiscreteFactor([vn], [vl], np.ones(vl)))

		return mn

	def to_markov_nets(self, via='β'):
		mm = self.to_markov_net(via=via)

		mms = [mm.subgraph(nodeset) for nodeset in nx.connected_components(mm)]

		for f in mm.factors:
			for mmi in mms:
				if set(mmi.nodes()).issuperset(f.variables):
					mmi.add_factors(f)
					break
			else:
				assert False
		
		return mms

	def to_uncalibrated_cforest(self, ctree, via="β"):
		dists = []

		for vvv in ctree.nodes():
			f = np.ones(tuple(
					len(self.vars[v]) for v in vvv
				))

			rjd = RJD(f, [self.vars[v] for v in vvv])
			dists.append(rjd)

		for L,X,Y,cpt,power in self.edges("LXYP"+via):
			if L[0] == 'π': #  (it's a projection)
				continue 

			for d in dists:
				if set((X&Y).atoms).issubset(d.varlist):
					d.data *= cpt.broadcast_to(d.varlist)
					break
			else:
				raise ValueError("given clusters don't cover cpd '%s'(%s|%s) "%(L,X.name,Y.name))

		return CliqueForest(dists, 
			nx.relabel_nodes(ctree, {vl:i for i,vl in enumerate(ctree.nodes())}))
