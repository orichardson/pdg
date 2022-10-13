import sys # for printing.
# from random import random
import numpy as np
from functools import reduce

from ..fg import FactorGraph
from ..dist import RawJointDist as RJD

# try:
import torch

LOGZERO=1E12
twhere,tlog = torch.where, torch.log
# zlog = lambda t : twhere(t==0, 0., tlog(twhere(t==0, LOGZERO, t))) 
def zmul(prob, maybe_nan):
	# return twhere(prob == 0, 0., twhere(torch.isnan(maybe_nan), 0., maybe_nan) * prob)
	return twhere(prob == 0, 0., twhere(torch.isnan(maybe_nan), 0., maybe_nan) * prob)   
# except ImportError:
# 	print("No torch; only numpy backend")

Optims = {'adam' : torch.optim.Adam, 'sgd' : torch.optim.SGD, 
		'asgd' : torch.optim.ASGD, 'lbfgs' : torch.optim.LBFGS}


def torch_score_alt(pdg, μ : RJD, γ):
	""" The simpler, linear version of semantics for small gamma """
	loss = torch.tensor(0.)
	for X,Y,cpd_df,α,β in pdg.edges("XYPαβ"):
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

def torch_score(pdg, μ : RJD, γ):
	loss = torch.tensor(0.)
	for X,Y,cpd_df,α,β in pdg.edges("XYPαβ"):
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
	
	# Returns log base 2 by default, so base is correct already
	return loss

def approx_score(pdg, F : FactorGraph, γ):
	# TODO The below is in bad shape; needs help.
	broadcast = pdg.genΔ(kind=RJD.unif).broadcast
	
	loss = torch.tensor(0.)
	for X,Y,cpd_df,α,β in pdg.edges("XYPαβ"):
		# muxy = μ.prob_matrix(X, Y)
		muxy = F.gibbs_marginal_estimate([X,Y]) # TODO this is not implememnted
		# muy_x = μ.prob_matrix(Y | X)
		muy_x = muxy / muxy.sum(axis= pdg.varlist.index(X)) # TODO be more efficient.

		logcond_info = - torch.log(twhere(muxy==0, 1., muy_x))
		# print(muxy*logcond_info)
		
		if cpd_df is None:
			logliklihood = 0.
			logcond_claimed = 0.
		else:
			
			cpt = torch.tensor(broadcast(cpd_df), requires_grad=False)
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
	
	# TOOD: implement approx_entropy.
	loss -= γ * F.approx_entropy()
	# print('after entropy')
	# loss -= γ * zlog( μ )
	
	return loss


def opt_dist(pdg, gamma=None,
		extraTemp = 0, iters=350,
		ret_losses:bool = False,
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
		gamma = pdg.gamma_default
	γ = gamma + extraTemp       
	
	Optims = {'adam' : torch.optim.Adam, 'sgd' : torch.optim.SGD, 
		'asgd' : torch.optim.ASGD, 'lbfgs' : torch.optim.LBFGS}
	
	# uniform starting position
	# μdata = torch.tensor(pdg.genΔ(RJD.unif).data, requires_grad=True)
	# μdata = torch.tensor(pdg.genΔ(RJD.unif).data, dtype=torch.double, requires_grad=True)
	normalize_init = True
	if representation in ['gibbs', 'exp+normalize']:
		representation = 'gibbs'
		normalize_init=False
		def todistrib(raw_data):
			nnμdata = torch.exp(raw_data)
			total = nnμdata.sum() 
			return ( nnμdata / total), constraint_penalty*(total-1)**2 
	elif representation in ['softclip+normalize', 'softmax+normalize', 'soft-simplex', 'soft simplex']:
		representation = 'soft-simplex'
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
	
	# μdata = torch.zeros(pdg.dshape, dtype=torch.double, requires_grad=True)
	μdata = init(pdg.dshape, dtype=torch.double)
	if normalize_init: μdata /= μdata.sum()
	μdata.requires_grad = True
	
	if optimizer in ['sgd'] and 'lr' not in optim_kwargs:
		optim_kwargs['lr'] = 1E-3
	ozr = Optims[optimizer]([μdata], **optim_kwargs)
	
	μ = RJD(todistrib(μdata)[0], pdg.varlist, use_torch=True)

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
		loss = torch_score(pdg, μ, γ)
		if constraint_penalty != 0: loss += penalty
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


def optimize_via_FGs(pdg, gamma=0, init=None, iters=1000) -> RJD:
	"""
	Cover PDG with factors;

	#TODO : remove the joint distribution.
	"""
	HE = pdg.hypergraph_object[1]
	μ = pdg.genΔ().torchify(True) #init
	energies = [
		torch.tensor(-np.log(μ.broadcast(pdg[k])),
				dtype=torch.double,requires_grad=True) 
			for k in HE
		]
	# factors = [
	#     torch.tensor(μ.broadcast(pdg[k]),
	#             dtype=torch.double,requires_grad=True) 
	#         for k in HE
	#     ]
		
	# ozr = torch.optim.Adam(factors,lr=5E-4)
	ozr = torch.optim.Adam(energies,lr=2E-3)
	numdig = str(len(str(iters)))
	for it in range(iters):
		ozr.zero_grad(set_to_none=True)
		# pf = torch.clip(reduce(mul, factors), min=0.)
		pf = torch.exp(-reduce(torch.add, energies))
		μ.data = pf / pf.sum()
		# μ.data = factors[0]
		# print(μ.data.shape)
		loss = pdg.torch_score(μ, gamma)
		loss.backward()
		ozr.step()
		
		
		l = loss.detach().item()
		extrastr=""
		# went_up = l > losses[-1]            

		if it%(1+iters//100) == 0:# or went_up: # Nice printing.
			sys.stdout.write(
				('\r[{ep:>'+numdig+'}/{its}]  loss:  {ls:.3e};  lr: {lr:.3e}; \t graient magnitude: {gm:.3e} \t ' + extrastr)\
					.format(ep=it, ls=l, lr=2E-3, its=iters, gm=torch.norm(μ.data)) )
			sys.stdout.flush()    
		
	print(loss.detach().item())
	
	# μ.data = μ.data.detach()
	return μ
		