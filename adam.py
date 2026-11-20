import math
from typing import Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self.dp_bc = False
        self.dp_noise_var = 0.0

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["preconditioner"] = torch.empty_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                if self.dp_bc:
                    v_hat = exp_avg_sq / bias_correction2
                    v_hat_corr = v_hat - self.dp_noise_var
                    v_hat_corr.clamp_(min=1e-8)

                    denom = v_hat_corr.sqrt().add_(eps)
                    state["preconditioner"].copy_(denom)
                else:
                    denom = (
                        exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                    )

                    precond_diag = denom.mul(bias_correction1)
                    state["preconditioner"].copy_(precond_diag)

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
