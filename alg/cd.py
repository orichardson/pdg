# Contrastive divergence

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
            gradient =
            guess += gradient
