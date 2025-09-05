# param_cpd.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from typing import List
import numpy as np

# from .param_cpd import ParamCPD

class ParamCPD:

    def __init__(self,
                 X_card: int,
                 Y_card: int,
                 name: str = "",
                 cpd = None,
                 init: str | torch.Tensor = "from_cpd",
                 mask: Optional[torch.Tensor] = None,
                 dtype=torch.double,
                 device=None):
        self.name = name
        self.X_card = int(X_card)
        self.Y_card = int(Y_card)
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.cpd = cpd

        if init == "from_cpd":
            if cpd is None:
                raise ValueError("cpd must be provided when init='from_cpd'")
            if hasattr(cpd, "to_numpy"):
                arr = cpd.to_numpy()
            else:
                raise ValueError("cpd must have a to_numpy method")

            logits = torch.tensor(arr, dtype=dtype, device=self.device)


        elif init == "uniform":
            logits = torch.ones(self.X_card, self.Y_card, dtype=dtype, device=self.device)
        elif init == "random":
            logits = torch.randn(self.X_card, self.Y_card, dtype=dtype, device=self.device)
        elif isinstance(init, torch.Tensor):
            assert init.shape == (self.X_card, self.Y_card)
            logits = init.to(self.device, dtype)
        else:
            raise ValueError("init must be 'uniform', 'random', or a tensor of shape (|X|,|Y|)")

        self.logits = torch.nn.Parameter(logits, requires_grad=True)

        ## U: the mask may be useful later
        # if mask is None:
        #     self._mask = None
        # else:
        #     if mask.dtype != torch.bool:
        #         mask = mask.bool()
        #     assert mask.shape == (self.X_card, self.Y_card)
        #     self._mask = mask.to(self.device)

    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    ## U: the mask may be useful later
    # def mask(self) -> torch.Tensor:
    #     if self._mask is None:
    #         return torch.ones((self.X_card, self.Y_card), dtype=torch.bool, device=self.device)
    #     return self._mask

    def to_numpy(self):
        with torch.no_grad():
            return self.probs().detach().cpu().numpy()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.X_card, self.Y_card)




# ________________________________________________________________________________________
# ________________________________________________________________________________________

from .torch_opt_lir import opt_joint, torch_score
def _collect_learnables(pdg) -> List[Tuple[str, ParamCPD]]:
    out = []
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if isinstance(P, ParamCPD):
            out.append((L, P))
    return out

@torch.no_grad()
def _detach_mu(mu):
    mu.data = mu.data.detach().clone()
    return mu

def lir_train_simple(
    M,                          # PDG containing ParamCPDs (learnable θ)
    gamma: float = 0.0,
    T: int = 200,               # outer steps
    inner_iters: int = 300,     # μ*-solve iterations (opt_joint)
    lr: float = 1e-2,
    optimizer_ctor = torch.optim.Adam,
    verbose: bool = False,
    **inner_kwargs              
):
    """
    Simplified LIR:
      repeat T times:
        μ*  = argmin_μ  OInc_γ(M(θ); μ)        # inner solve (uses your opt_joint)
        θ  ← θ - η ∇_θ OInc_γ(M(θ); μ*)        # envelope theorem: μ* is treated constant
    """
    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    opt = optimizer_ctor([P.logits for (_, P) in learnables], lr=lr)

    last = None
    for t in range(T):
        # inner solve for μ*, given current θ
        μ_star = opt_joint(M, gamma=gamma, iters=inner_iters, verbose=False, **inner_kwargs)
        μ_star = _detach_mu(μ_star)

        # outer gradient on θ only
        opt.zero_grad(set_to_none=True)
        loss = torch_score(M, μ_star, gamma)
        loss.backward()
        opt.step()

        if verbose and (t % max(1, T // 10) == 0):
            val = float(loss.detach().cpu())
            delta = "" if last is None else f"  Δ={val - last:+.3e}"
            print(f"[LIR {t:4d}/{T}]  γ={gamma:.3g}  loss={val:.6e}{delta}")
            last = val

    return M  # parameters are updated in-place


