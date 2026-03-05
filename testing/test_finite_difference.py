import math
from collections.abc import Callable

import pytest
import torch

from causal_flare import flare_chunk_triton
from causal_flare.recurrent import RecurrentFLARE
from causal_flare.torch import flare_causal_chunked, flare_causal_reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
device = torch.device("cuda")


_SEPARATE_DECODE_FLAGS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


_IMPLEMENTATIONS = [
    ("reference", "bnhd", 1e-3, 5e-2),
    ("pytorch_chunked", "bnhd", 2e-3, 6e-2),
    ("recurrent", "bhnd", 2e-3, 6e-2),
    ("triton_chunked", "bnhd", 3e-3, 7e-2),
]


def _make_base_tensors(*, batch: int, seq: int, heads: int, q_tokens: int, head_dim: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "q": torch.randn((heads, q_tokens, head_dim), device=device, dtype=dtype),
        "k": torch.randn((batch, seq, heads, head_dim), device=device, dtype=dtype),
        "v": torch.randn((batch, seq, heads, head_dim), device=device, dtype=dtype),
        "q_dec_rand": torch.randn((batch, seq, heads, head_dim), device=device, dtype=dtype),
        "k_dec_rand": torch.randn((heads, q_tokens, head_dim), device=device, dtype=dtype),
    }


def _resolve_decode_tensors(
    values: dict[str, torch.Tensor], *, q_dec_mode: str, k_dec_mode: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if q_dec_mode == "none":
        q_dec = None
    elif q_dec_mode == "k_enc":
        q_dec = values["k"]
    elif q_dec_mode == "rand":
        q_dec = values["q_dec_rand"]
    else:
        raise ValueError(f"Unknown q_dec_mode: {q_dec_mode}")

    if k_dec_mode == "none":
        k_dec = None
    elif k_dec_mode == "q_enc":
        k_dec = values["q"]
    elif k_dec_mode == "rand":
        k_dec = values["k_dec_rand"]
    else:
        raise ValueError(f"Unknown k_dec_mode: {k_dec_mode}")

    return q_dec, k_dec


def _decode_modes_from_separate_flags(*, separate_q_dec: bool, separate_k_dec: bool) -> tuple[str, str]:
    q_dec_mode = "rand" if separate_q_dec else "none"
    k_dec_mode = "rand" if separate_k_dec else "none"
    return q_dec_mode, k_dec_mode


def _target_names(*, separate_q_dec: bool, separate_k_dec: bool) -> tuple[str, ...]:
    targets = ["q", "k", "v"]
    if separate_q_dec:
        targets.append("q_dec_rand")
    if separate_k_dec:
        targets.append("k_dec_rand")
    return tuple(targets)


def _run_directional_finite_difference(
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    base_values: dict[str, torch.Tensor],
    targets: tuple[str, ...],
    *,
    eps: float,
    atol: float,
    rtol: float,
) -> None:
    grad_values = {
        name: tensor.detach().clone().requires_grad_(name in targets)
        for name, tensor in base_values.items()
    }
    loss = loss_fn(grad_values)
    grads = torch.autograd.grad(loss, [grad_values[name] for name in targets], allow_unused=False)

    for name, grad in zip(targets, grads):
        direction = torch.randn_like(base_values[name])
        direction = direction / direction.norm().clamp_min(1e-12)

        pos_values = dict(base_values)
        neg_values = dict(base_values)
        pos_values[name] = base_values[name] + eps * direction
        neg_values[name] = base_values[name] - eps * direction

        with torch.no_grad():
            loss_pos = loss_fn(pos_values)
            loss_neg = loss_fn(neg_values)

        finite_diff = (loss_pos - loss_neg) / (2.0 * eps)
        autodiff = (grad.detach() * direction).sum()
        err = float((finite_diff - autodiff).abs().item())
        scale = max(float(finite_diff.abs().item()), float(autodiff.abs().item()))
        limit = atol + rtol * scale

        assert err <= limit, (
            f"Directional finite-difference check failed for {name}: "
            f"err={err:.3e}, limit={limit:.3e}, finite_diff={float(finite_diff.item()):.3e}, "
            f"autodiff={float(autodiff.item()):.3e}"
        )


@pytest.mark.parametrize(("impl_name", "out_layout", "atol", "rtol"), _IMPLEMENTATIONS)
@pytest.mark.parametrize(("separate_q_dec", "separate_k_dec"), _SEPARATE_DECODE_FLAGS)
def test_causal_flare_backward_matches_directional_finite_difference(
    impl_name: str,
    out_layout: str,
    atol: float,
    rtol: float,
    separate_q_dec: bool,
    separate_k_dec: bool,
):
    torch.manual_seed(3500)

    batch = 1
    seq = 16
    heads = 2
    q_tokens = 16
    head_dim = 16
    chunk_size = 16
    dtype = torch.float32
    scale = 1.0 / math.sqrt(head_dim)

    base_values = _make_base_tensors(
        batch=batch,
        seq=seq,
        heads=heads,
        q_tokens=q_tokens,
        head_dim=head_dim,
        dtype=dtype,
    )
    if out_layout == "bhnd":
        grad_out = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    elif out_layout == "bnhd":
        grad_out = torch.randn((batch, seq, heads, head_dim), device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown output layout: {out_layout}")

    q_dec_mode, k_dec_mode = _decode_modes_from_separate_flags(
        separate_q_dec=separate_q_dec, separate_k_dec=separate_k_dec
    )

    def loss_fn(values: dict[str, torch.Tensor]) -> torch.Tensor:
        q_dec, k_dec = _resolve_decode_tensors(values, q_dec_mode=q_dec_mode, k_dec_mode=k_dec_mode)
        if impl_name == "reference":
            y = flare_causal_reference(values["q"], values["k"], values["v"], Q_dec=q_dec, K_dec=k_dec, scale=scale)
        elif impl_name == "pytorch_chunked":
            y = flare_causal_chunked(
                values["q"],
                values["k"],
                values["v"],
                Q_dec=q_dec,
                K_dec=k_dec,
                scale=scale,
                chunk_size=chunk_size,
            )
        elif impl_name == "recurrent":
            y = RecurrentFLARE.apply(values["q"], values["k"], values["v"], scale, None, None, q_dec, k_dec, 16)
        elif impl_name == "triton_chunked":
            y = flare_chunk_triton(
                Q=values["q"],
                K=values["k"],
                V=values["v"],
                scale=scale,
                chunk_size=chunk_size,
                Q_dec=q_dec,
                K_dec=k_dec,
            )
        else:
            raise ValueError(f"Unknown implementation: {impl_name}")
        return (y * grad_out).sum()

    _run_directional_finite_difference(
        loss_fn,
        base_values,
        _target_names(separate_q_dec=separate_q_dec, separate_k_dec=separate_k_dec),
        eps=1e-3,
        atol=atol,
        rtol=rtol,
    )
