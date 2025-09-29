from pdg.pdg import PDG
from pdg.dist import RawJointDist as RJD
import numpy as np

class Automaton:
	def __init__(self,  M : PDG):
		self.pdg = M
		self.state = M.genΔ(kind=RJD.unif)


	def update(self, l, addconf):
		M = self.pdg
		if type(l) is list:
			for ll in l:
				if type(ll) is tuple and len(ll) == 2:
					self.update(ll[0], ll[1] * addconf)
				else:
					self.update(ll, addconf)

		elif l in M.edgedata or l in self.pdg.Ed:
			# XYl = M._get_edgekey(l)
			cpd = M[l]
			X,Y = cpd.nfrom, cpd.nto
			pY_X = self.state.broadcast(cpd)
			curr = self.state.prob_matrix(Y|X)

			ratio = pY_X / curr

			newjoint = self.state.data * ratio** (1 - np.exp(-addconf))
			self.state.data = newjoint / newjoint.sum()

	def vectorfield(self, l):
		M = self.pdg
		if l in M.edgedata or l in self.pdg.Ed:
			cpd = M[l]
			X,Y = cpd.nfrom, cpd.nto
			pY_X = self.state.broadcast(cpd)
			curr = self.state.prob_matrix(Y|X)

		surprise = self.state.data * np.log(pY_X/curr) 
		return surprise - self.state.data * surprise.sum()

	def plot_vectorfields(self, *labels):
		from matplotlib import pyplot as plt

		fig, axs = plt.subplots(1,len(labels)+1);
		vsum = 0;
		Mmax = 1E-4;
		

		vs = {}

		for l in labels:
			v = self.vectorfield(l)
			vsum += v
			vs[l] = v
			Mmax = max(Mmax, np.abs(self.vectorfield(l)).max())
		
		for l,ax in zip(labels, axs):
			v = vs[l]
			# M = max(M, np.abs(v).max())
			# M = np.abs(v).max()
			ax.set_title(l+"  ("+")")
			ax.matshow(v, cmap="PiYG", vmin=-Mmax, vmax=Mmax)
		
		axs[-1].matshow(vsum, cmap="PiYG", vmin=-Mmax, vmax=Mmax)
		axs[-1].set_title("sum (scale:{:.3f}".format(Mmax));

		plt.show()



		

