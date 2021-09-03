# %load_ext autoreload
# %autoreload 2

# import pandas as pd
import sys # for printing.
from random import random
import numpy as np
import networkx as nx

import collections
from numbers import Number

# import utils
from .utils import dictwo
from .rv import Variable, ConditionRequest, Unit
from .dist import RawJointDist as RJD, CPT #, Dist, CDist,
from .dist import z_mult, zz1_div


try:
    import torch
    
    LOGZERO=1E12
    twhere,tlog = torch.where, torch.log
    def zmul(prob, maybe_nan):
        return twhere(prob == 0, 0., twhere(torch.isnan(maybe_nan), 0., maybe_nan) * prob)
        # return twhere(prob == 0, 0., twhere(torch.isnan(maybe_nan), 0., maybe_nan) * prob)
    # zlog = lambda t : twhere(t==0, 0., tlog(twhere(t==0, LOGZERO, t)))    
except ImportError:
    print("No torch; only numpy backend")

class Labeler:
    # NAMES = ['p','q','r']

    def __init__(self, basenames=['p']):
        self._counter = 0
        self._basenames = basenames
        self._edge_specific_counts = collections.defaultdict(lambda: 0)
        # self._edge_specific_counts = {}

    def fresh(self, vfrom, vto, **ctxt):
        self._counter += 1
        self._edge_specific_counts[(vfrom,vto)] += 1
        
        return self._basenames[0] + str(self._counter)
        
    def copy(self):
        l = Labeler(self._basenames)
        l._counter = self._counter
        l._edge_specific_counts.update(self._edge_specific_counts)
        return l

class PDG:
    # By default, use the base labeleler, which
    # just gives a fresh label by incrementing a counter.
    def __init__(self, labeler: Labeler = Labeler()):
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
    
    def __call__(self, *INPUT, **kwargs):
        """
        Custom string interpolation for interpreting PDG queries & command, making it
        easier to construct things and do work in the context of a PDG.
        Examples:
        M('AB')  replaces  Variable.product(M.vars['A'], M.vars['B'])
        
        Future:
        M('A B -> C')  returns  a β-combination of cpts.
        M('A B -> B C := ', P)  adds a matrix with the appropriate types,
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

    # generates <node_name_from, node_name_to, edge_label>
    # @property
    # def E(self, include_cpts = False) -> Iterator[str, str, str, Number]:
    #     for i,j in self.cpds.keys():
    #         if include_cpts:
    #             for l, L in cpts[i,j].items():
    #                 yield (i,j,l), L
    #         else:
    #             for l in cpds[i,j].keys():
    #                 yield i,j,l

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
        rslt = PDG(self.labeler.copy())
        # rslt.vars = dict(**self.vars) # variables don't need a deep copy.

        for vn, v in self.vars.items():
            rslt._include_var(v,vn)

        for ftl, attr in self.edgedata.items():
            rslt._set_edge(*ftl, **attr)

        rslt._apply_structure();
        return rslt


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
        label = None
        if isinstance(spec, ConditionRequest):
            gn,tn = spec.given.name, spec.target.name
            # raise KeyError("Multiple options possible.")
        elif type(spec) == tuple and type(spec[0]) is str:
            # normal strings can be looked up as a tuple
            gn,tn = spec[:2]
            if len(spec) == 3:
                label = spec[2]
        elif type(spec) is str:
            for xyl in self.edgedata.keys():
                if ','.join(xyl) in spec or spec == xyl[-1]:
                    gn, tn, label = xyl
        else:
            raise ValueError("no edge matching: '"+ repr(spec) +"'")

        return gn,tn,label

    def with_params(self, **kwargs):
        rslt = self.copy()
        for param, val in kwargs.items():
            if type(val) is dict:
                for spec, defn in val.items():
                    self.edgedata[self._get_edgekey(spec)][param] = defn
        return rslt

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
        >>> M: PDG  += "ℓ", p : CPT
        """

        if isinstance(data, PDG):
            for varname, var in data.vars.items():
                self._include_var(var,varname)

            for ftl, attr in data.edgedata.items():
                self._set_edge(*ftl, **attr)

        elif isinstance(data, CPT):
            self._include_var(data.nfrom)
            self._include_var(data.nto)
            # label = other.name if hasattr(other, 'name') else \
            #     self.labeler.fresh(other.nfrom,other.nto)
            if label is None:
                label = self.labeler.fresh(data.nfrom.name, data.nto.name)

            self._set_edge(data.nfrom.name, data.nto.name, label, cpd=data)
        
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
            if 'from' in data and 'to' in data:
                X = data['from'], Y = data['to']
                if isinstance(X, Variable): XN = X.name
                if isinstance(Y,Variable):  YN = Y.name
            if 'cpd' in data:
                XN = data['cpd'].nfrom.name
                YN = data['cpd'].nto.name
            if 'label' in data:
                label = data['label']

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
        """ For convenience only.
        takes a pair (src, target) of variables, and returns the relevant cpt.
        Alternatively, takes a string name."""
        label = None
        if isinstance(key, ConditionRequest):
            gn,tn = key.given.name, key.target.name
            # raise KeyError("Multiple options possible.")
        elif type(key) == tuple and type(key[0]) is str:
            # normal strings can be looked up as a tuple
            gn,tn = key[:2]
            if len(key) == 3:
                label = key[2]
        else:
            try:
                gn,tn,label = self._get_edgekey(key)
            except:
                print(key, 'is not a valid key')
                return

        if label == None:
            if len(self.graph[gn][tn]) == 1:
                return next(iter(self.graph[gn][tn].values()))['cpd']

            return self.graph[gn][tn]['cpd']

        return self.edgedata[gn,tn,label]['cpd']

    def __iter__(self):
        return self.edges("XYPαβ")

    def edges(self, spec='XY'):
        """
        Examples:
            M.edges("X,Y,cpd,α,β")
            M.edges("XYLp")
            M.edges(['X', 'Y'])
        """
        if type(spec) is str:
            delims = ',; '
            for d in delims:
                if d in spec:
                    spec = spec.split(d)
                    break

        for (Xname, Yname, l), data in self.edgedata.items():
            X,Y = self.vars[Xname], self.vars[Yname]
            lookup = dict(src=X,tgt=Y,X=X,Y=Y,Xname=Xname,Yname=Yname, Xn=Xname, Yn =Yname,
                alpha=1,beta=1,l=l,label=l, L=l)
            lookup.update(**data)
            if 'α' not in lookup: lookup['α'] = lookup['alpha']
            if 'β' not in lookup: lookup['β'] = lookup['beta']
            lookup['P'] = lookup['p'] =  data.get('cpd',None)

            # alpha = data.get('alpha', 1)
            # beta = data.get('beta', 1)
            # yield X,Y, data.get('cpd', None), alpha, beta

            yield tuple(lookup.get(s,None) for s in spec) if len(spec) > 1 \
                else lookup.get(spec[0],None)

    # semantics 1:
    def matches(self, mu):
        for X,Y, cpd in self.edges("XYP"):
            # print(mu[Y], '\n', mu[X], '\n', cpd)
            if( not np.allclose(mu[Y], mu[X] @ cpd) ):
                return False

        return True

    def torch_score_alt(self, μ : RJD, γ):
        """ The simpler, linear version of semantics for small gamma """
        loss = torch.tensor(0.)
        for X,Y,cpd_df,α,β in self.edges("XYPαβ"):
            # print("For edge %s -> %s (α=%.2f, β=%.2f)"%(X.name,Y.name,α,β))
            # muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
            muxy = μ.prob_matrix(X, Y)
            muy_x = μ.prob_matrix(Y | X)

            logcond_info = - torch.log(twhere(muxy==0, 1., muy_x))
            # print(muxy*logcond_info)
            
            if cpd_df is None:
                logliklihood = 0.
                logcond_claimed = 0.
            else:
                cpt = torch.tensor(μ.broadcast(cpd_df), requires_grad=False)
                claims = np.isfinite(cpt)

                logliklihood = twhere(claims, -torch.log(twhere(cpt==0, LOGZERO, cpt)), 0.)
                logcond_claimed = twhere(claims, logcond_info, 0.)
                # logliklihood =  -torch.log(cpt)
                # logcond_claimed = -torch.log(muy_x)
        
            # print('cpt: ', cpt, '\nμ_{Y|X}', muy_x)
            # print('log ratio: ', muxy * (logliklihood - logcond_claimed))
            # print('... that sums to ', (muxy * (logliklihood - logcond_claimed)).sum())

            loss += β * zmul( muxy, (logliklihood-logcond_claimed)).sum()
            # loss += α*γ* zmul(muxy , logcond_info).sum()
            loss += α*γ* zmul(muxy , logliklihood).sum()
            # print('... loss now ', loss/np.log(2))

        loss /= np.log(2)
        loss -= γ * μ.H(...)
        # print('after entropy')
        # loss -= γ * zlog( μ )
        
        return loss
    
    def torch_score(self, μ : RJD, γ):
        loss = torch.tensor(0.)
        for X,Y,cpd_df,α,β in self.edges("XYPαβ"):
            # print("For edge %s -> %s (α=%.2f, β=%.2f)"%(X.name,Y.name,α,β))
            # muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
            muxy = μ.prob_matrix(X, Y)
            muy_x = μ.prob_matrix(Y | X)

            logcond_info = - torch.log(twhere(muxy==0, 1., muy_x))
            # print(muxy*logcond_info)
            
            if cpd_df is None:
                logliklihood = 0.
                logcond_claimed = 0.
            else:
                cpt = torch.tensor(μ.broadcast(cpd_df), requires_grad=False)
                claims = torch.isfinite(cpt)

                logliklihood = twhere(claims, -torch.log(twhere(cpt==0, LOGZERO, cpt)), 0.)
                logcond_claimed = twhere(claims, logcond_info, 0.)
                # logliklihood =  -torch.log(cpt)
                # logcond_claimed = -torch.log(muy_x)
        
            # print('cpt: ', cpt, '\nμ_{Y|X}', muy_x)
            # print('log ratio: ', muxy * (logliklihood - logcond_claimed))
            # print('... that sums to ', (muxy * (logliklihood - logcond_claimed)).sum())

            loss += β * zmul( muxy, (logliklihood-logcond_claimed)).sum()
            loss += α*γ* zmul(muxy , logcond_info).sum() # /plus/ E_mu ( - log mu)
            # print('... loss now ', loss/np.log(2))

        loss /= np.log(2)
        loss -= γ * μ.H(...)
        # print('after entropy')
        # loss -= γ * zlog( μ )
        
        return loss
        # Returns log base 2 by default, so base is correct already
    
    def _torch_opt_inc(self, gamma=None,    
            extraTemp = 1E-3, iters=350, 
            ret_losses:bool = True,
            ret_iterates:bool = False,
            representation:str = 'simplex', # or dist or softmax
            constraint_penalty = 1,
            init = torch.ones,
            optimizer = 'Adam',
            **optim_kwargs
            #, tol = 1E-8, max_iters=300 #unsupported
            ):
        """ = min_\mu inc(\mu, gamma) """
        optimizer = optimizer.lower()
        
        if gamma is None: # This is the target gamma
            gamma = self.gamma_default
        γ = gamma + extraTemp       
        
        Optims = {'adam' : torch.optim.Adam, 'sgd' : torch.optim.SGD, 
            'asgd' : torch.optim.ASGD, 'lbfgs' : torch.optim.LBFGS}
        
        # uniform starting position
        # μdata = torch.tensor(self.genΔ(RJD.unif).data, requires_grad=True)
        # μdata = torch.tensor(self.genΔ(RJD.unif).data, dtype=torch.double, requires_grad=True)
        normalize_init = True
        if representation in ['gibbs', 'exp+normalize']:
            representation = 'gibbs'
            normalize_init=False
            def todistrib(raw_data):
                nnμdata = torch.exp(raw_data)
                total = nnμdata.sum() 
                return ( nnμdata / total), constraint_penalty*(total-1)**2 
        elif representation in ['softclip+normalize', 'softmax+normalize', 'soft simplex']:
            representation = 'soft simplex'
            temp = [ 1E-3 ]
            def todistrib(raw_data):
                # temp = 1E-3
                # temp = lrsched1.get_last_lr()[0] * 1E-2
                # temp[0] *= 0.99
                ## slowly drift temp
                # temp += (lrsched1.get_last_lr()[0]*2 - temp) / 4
                nnμdata = temp[0]*torch.logsumexp(torch.stack([raw_data/temp[0], torch.zeros(raw_data.shape)], dim=raw_data.ndim), dim=-1) #soft max for +
                return ( nnμdata / nnμdata.sum()), 0
        elif representation in ['simplex', 'clip+normalize'] or True: # you gotta do something
            representation = 'simplex'
            def todistrib(raw_data):
                nnμdata = torch.clip(raw_data, min=0) # hard max for positivity
                total = nnμdata.sum() 
                if constraint_penalty == 0:
                    totneg = (torch.exp(torch.where(raw_data < 0., -raw_data , 0.)) - 1.).sum()
                else: totneg = 0
                return ( nnμdata / total), constraint_penalty*((total-1)**2 + totneg)
        
        # μdata = torch.zeros(self.dshape, dtype=torch.double, requires_grad=True)
        μdata = init(self.dshape, dtype=torch.double)
        if normalize_init: μdata /= μdata.sum()
        μdata.requires_grad = True
        
        if optimizer in ['sgd'] and 'lr' not in optim_kwargs:
            optim_kwargs['lr'] = 1E-3
        ozr = Optims[optimizer]([μdata], **optim_kwargs)
        
        μ = RJD(todistrib(μdata)[0], self.varlist, use_torch=True)

        # ozr = torch.optim.Adam([μdata], lr=5E-4)
        # ozr = torch.optim.SGD([μdata], lr=1E-3, momentum=0.8, dampening=0.0, nesterov=True)
        # ozr = torch.optim.ASGD([μdata], lr=1E-3)

        best_μdata = μdata.detach().clone()

        # def custom_lr(epoch):
        #     return 0.9999 ** (epoch ** 1.5)
        # lrsched1 = torch.optim.lr_scheduler.LambdaLR(ozr, custom_lr)
        # lrsched1 = torch.optim.lr_scheduler.ExponentialLR(ozr, 0.99999)
        lrsched1 = torch.optim.lr_scheduler.ExponentialLR(ozr, 1)
        # lrsched1 = torch.optim.lr_scheduler.ExponentialLR(ozr, 1.0001)
        # lrsched1 = torch.optim.lr_scheduler.CosineAnnealingLR(ozr, 10)
        # lrsched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(ozr, factor=0.1)
        
        bestl = float('inf')
        losses = [ float('inf') ] 
        if ret_iterates: iterates = [ μdata.detach() ]
        
        
        def closure():
            ozr.zero_grad(set_to_none=True)
            μ.data, penalty = todistrib(μdata)
            loss = self.torch_score(μ, γ)
            if constraint_penalty != 0: loss += loss + penalty
            loss.backward()
            return loss

        extrastr = ""
        numdig = str(len(str(iters)))
        for it in range(iters):
            l = closure().detach().item()
            # went_up = l > losses[-1]            
    
            if it%(1+iters//100) == 0:# or went_up: # Nice printing.
                sys.stdout.write(
                    ('\r[{ep:>'+numdig+'}/{its}]  loss:  {ls:.3e};  lr: {lr:.3e}; \t graient magnitude: {gm:.3e} \t ' + extrastr)\
                        .format(ep=it, ls=l, lr=lrsched1.get_last_lr()[0], its=iters, gm=torch.norm(μdata.grad)) )
            # if it%7 == 0:
                sys.stdout.flush()    
                # if True: sys.stdout.write('\n')
                # print(
                #     ('[{ep:>'+numdig+'}/{its}]  loss:  {ls:.3e};  lr: {lr:.3e}')\
                #         .format(ep=it, ls=loss.detach().item(), lr=lrsched1.get_last_lr()[0], its=iters) )
            
            ## The breadcrumbs we leave in case we get lost + a map of where we've been
            if ret_losses: losses.append(l)
            else: losses = [l]
            
            if ret_iterates: iterates.append(μ.data.detach().clone())
            else: iterates = [μ.data.detach().clone()]


            if l <= bestl:
                best_μdata = iterates[-1]
                bestl = l

            # if : # update the optimizer unless we just reset it
            # if optimizer == 'lbfgs':
            ozr.step(closure)
            # else:
            #     ozr.step()
            # if it % 10 == 0:
            lrsched1.step()
            
            # lrsched2.step(loss)
            
            γ += (gamma-γ) / 3. # anneal to original value gamma
            if representation == 'soft simplex':
                # temp[0] *= 0.999
                extrastr = "temp: {:.3e}".format(temp[0])
        
        μ.data = best_μdata
        
        to_ret = ()
        if ret_iterates: to_ret += (iterates,)
        if ret_losses: to_ret += (losses,)
        return (μ,)+to_ret if len(to_ret) else μ
            
        
        
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
                muxy = Pr(X, Y)
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
            muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
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
    def Inc(self, p, ed_vector=False):
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

    def IDef(self, p, ed_vector=False):
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
        
        else: # we can do the interesting thing
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
        
    def optimize_score_torch(self, gamma, store_iters=False, **solver_kwarg):
        pass


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

    def factor_product(self, repr="atomic") -> RJD:
        """ pretend the PDG is a factor graph, with weights θ := β """ 
        # start with uniform
        # d = RJD.unif(self.atomic_vars)

        d = self.genΔ(RJD.unif, repr)
        for X,Y,cpt,β in self.edges("XYPβ"):
            if cpt is not None:
                #hopefully the broadcast works...
                d.data *= np.nan_to_num( d.broadcast(cpt) ** β, nan=1)
            # print(d.data)

        d.data /= d.data.sum()
        return d

    def iter_GS_beta(self, max_iters=600, tol=1E-30, store_iters=False, repr='atomic') -> RJD:
        dist = self.genΔ(RJD.unif, repr)
        iters = [ np.copy(dist.data) ]
        totalβ = sum(β for β in self.edges("β"))

        for it in range(max_iters):
            nextdist = np.zeros(dist.data.shape)
            for X,Y,cpd,β in self.edges("XYPβ"):
                nextdist += (β / totalβ) * self.GS_step(dist, (X,Y,cpd))

            change = np.sum((dist.data - nextdist) ** 2 )
            dist.data = nextdist

            if store_iters: iters.append(nextdist)
            else: iters[-1] = nextdist


            if change < tol:
                break
        else:
            print('hit max iters, still changing at rate ', change, ' (tol = %f)'%tol)

                # if change == 0: break
        return (dist, iters) if store_iters else dist

    def iter_GS_ordered(self, ordered_edges=None, 
            max_iters: Number = 200,  tol=1E-30, 
            store_iters=False, repr="atomic") -> RJD:
        
        if ordered_edges is None:
            ordered_edges = list(self.edges("XYP"))


        dist = self.genΔ(RJD.unif, repr)
        iters = [ np.copy(dist.data) ]

        for it in range(max_iters):
            for XYp in ordered_edges:
                dist.data = self.GS_step(dist, XYp)

            change = ((dist.data - iters[-1]) ** 2 ).sum()

            if store_iters:
                iters.append(np.copy(dist.data))
            else:
                iters[-1] = dist.data


            if change < tol: break
        else:
            print('hit max iters, still changing at rate ', change, ' (tol = %f)'%tol)

        # return self.iterGS(init=self.genΔ(RJD.unif, repr), cpdgen=cpdgen)
        return (dist, iters) if store_iters else dist

    def GS_step(self, dist : RJD, XYP) -> RJD:
        """
        perform a Gibbs Sampling procedure with a variable on the distribution `dist`
        according to the edge XYP = (X: Var, Y: Var, cpd:CPD[Y|X] )
        """
        X,Y,cpd = XYP
        not_target = list(v for v in self.rawvarlist if
            len(set(v.name.split('×')) & set(Y.name.split("×"))) == 0)
                # Get the cpd from all variables that do not share a name with target.
        return dist.prob_matrix(*not_target,X) * dist.broadcast(cpd)



    ############# Utilities ##############

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
