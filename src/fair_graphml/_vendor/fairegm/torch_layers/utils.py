from typing import Optional

import torch


def apply_initializer(param: torch.nn.Parameter, initializer: Optional[str]) -> None:
    if initializer is None:
        return
    name = initializer.lower()
    with torch.no_grad():
        if name in {'glorot_uniform', 'xavier_uniform'}:
            torch.nn.init.xavier_uniform_(param)
        elif name in {'glorot_normal', 'xavier_normal'}:
            torch.nn.init.xavier_normal_(param)
        elif name == 'zeros':
            torch.nn.init.zeros_(param)
        elif name == 'ones':
            torch.nn.init.ones_(param)
        else:
            raise ValueError(f"Unsupported initializer '{initializer}'.")


def apply_constraint(param: torch.nn.Parameter, constraint: Optional[str]) -> None:
    if constraint is None:
        return
    name = constraint.lower()
    if name in {'nonneg', 'non_neg'}:
        param.data.clamp_(min=0.0)
    elif name == 'unit_norm':
        dim = param.data.dim() - 1
        param.data = torch.nn.functional.normalize(param.data, dim=dim)
    else:
        raise ValueError(f"Unsupported constraint '{constraint}'.")
