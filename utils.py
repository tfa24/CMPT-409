from contextlib import contextmanager

import torch
from torch.linalg import vector_norm


@contextmanager
def disabled_opacus_hooks(module):
    disabled = False
    if hasattr(module, "disable_hooks") and hasattr(module, "enable_hooks"):
        if getattr(module, "hooks_enabled", True):
            module.disable_hooks()
            disabled = True
    try:
        yield
    finally:
        if disabled:
            module.enable_hooks()


def get_lambda_max_H(g: list[torch.Tensor], params: list[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        v = [torch.randn_like(gi) for gi in g]
        v_flat = torch.cat([vi.reshape(-1) for vi in v])
        v_global_norm = vector_norm(v_flat)

        if not torch.isfinite(v_global_norm).item() or v_global_norm < 1e-8:
            v = [torch.zeros_like(vi) for vi in v]
        else:
            v = [vi / v_global_norm for vi in v]

    lambda_max = None

    for _ in range(64):
        hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True)
        hv = [hvi.detach() for hvi in hv]

        with torch.no_grad():
            v_flat = torch.cat([vi.reshape(-1) for vi in v])
            hv_flat = torch.cat([hvi.reshape(-1) for hvi in hv])

            lambda_max = torch.dot(v_flat, hv_flat)
            residual = vector_norm(hv_flat - lambda_max * v_flat)

            hv_global_norm = vector_norm(hv_flat)

            if not torch.isfinite(hv_global_norm).item() or hv_global_norm < 1e-8:
                lambda_max = hv_flat.new_zeros(())
                break

            if residual < 1e-3 * hv_global_norm:
                break

            v = [(hv_i / hv_global_norm) for hv_i in hv]

    return lambda_max.abs()


def get_lambda_max_P_inv_H(
    g: list[torch.Tensor],
    params: list[torch.Tensor],
    P_diag: list[torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        v = [torch.randn_like(gi) for gi in g]
        v_flat = torch.cat([vi.reshape(-1) for vi in v])
        v_global_norm = vector_norm(v_flat)

        if not torch.isfinite(v_global_norm).item() or v_global_norm < 1e-8:
            v = [torch.zeros_like(vi) for vi in v]
        else:
            v = [vi / v_global_norm for vi in v]

    lambda_max = None

    for _ in range(64):
        hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True)
        hv = [hvi.detach() for hvi in hv]

        with torch.no_grad():
            Av = [hvi / Pi for hvi, Pi in zip(hv, P_diag)]

            v_flat = torch.cat([vi.reshape(-1) for vi in v])
            hv_flat = torch.cat([hvi.reshape(-1) for hvi in hv])
            Pv_flat = torch.cat([(Pi * vi).reshape(-1) for vi, Pi in zip(v, P_diag)])
            Av_flat = torch.cat([Avi.reshape(-1) for Avi in Av])

            denom = torch.dot(v_flat, Pv_flat)
            if not torch.isfinite(denom).item() or denom.abs() < 1e-8:
                lambda_max = hv_flat.new_zeros(())
                break
            lambda_max = torch.dot(v_flat, hv_flat) / denom

            residual = vector_norm(hv_flat - lambda_max * Pv_flat)
            Av_global_norm = vector_norm(Av_flat)
            hv_global_norm = vector_norm(hv_flat)

            if not torch.isfinite(Av_global_norm).item() or Av_global_norm < 1e-8:
                lambda_max = Av_flat.new_zeros(())
                break

            if residual < 1e-3 * hv_global_norm:
                break

            v = [Avi / Av_global_norm for Avi in Av]

    return lambda_max.abs()


def get_P_diag(
    opt,
    params: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    P_diag = []
    for p in params:
        st = opt.state.get(p)
        P_diag.append(st["preconditioner"])

    return P_diag
