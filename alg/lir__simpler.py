from __future__ import annotations
from typing import Tuple
import torch
from typing import List
from pdg.dist import ParamCPD


from .torch_opt_lir import opt_joint, torch_score
def _collect_learnables(pdg) -> List[Tuple[str, ParamCPD]]:
    """
    Collects all learnable ParamCPD objects from the edges of a PDG in order to add them to an optimizer.
    """
    out = []
    # for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
    #     if isinstance(P, ParamCPD):
    #         out.append((L, P))
    for L, P in pdg.edges("l,P"):
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
    mu_init = None,
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
    
    mu_init = mu_init
    for t in range(T):

        def warm_start_init(shape, dtype=torch.double):
            if mu_init is not None:
                return mu_init.data.clone().to(dtype)
            else:
                return torch.ones(shape, dtype=dtype)
        # inner solve for μ*, given current θ
        μ_star = opt_joint(M, gamma=gamma, iters=inner_iters, verbose=False, init=warm_start_init, **inner_kwargs)
        μ_star = _detach_mu(μ_star)
        mu_init = μ_star.data.detach().clone()

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


