"""
VERY ROUGH DRAFT

An idea for an inference procedure based on Hinton and Carreira-Perpinan's work on Contrastive Divergence Learning.

https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
https://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf

"""

from ..rv import ConditionRequest, Variable as Var
from ..pdg import PDG

import numpy as np


class CD:
    """
    Idea: cover the PDG with a factor graph;
    run an MCMC sampler to do inference.
    """
    def __init__(self, pdg):
        self.pdg = pdg
        self.fg = pdg.FG_mask()  # initialize factor graph

    def query(cr : ConditionRequest):
        def step():
            pass

        # initialize to uniform cpd
        guess = np.ones(cr.shape) / len(cr.target)

        for i in range(100):
            # estimate gradient from MCMC sample of model.
            gradient = None
            guess += gradient
