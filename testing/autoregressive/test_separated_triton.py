import pytest
import torch

from causal_flare.autoregressive.separated_triton import flare_autoregressive_separated_trition


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_triton_is_explicitly_disabled() -> None:
    pytest.importorskip("triton")
    U = torch.randn((1, 8, 2, 16), device="cuda", dtype=torch.float32)
    retain = torch.rand((1, 8, 2), device="cuda", dtype=torch.float32)
    write = torch.randn((1, 8, 2, 4), device="cuda", dtype=torch.float32)
    decode = torch.randn((1, 8, 2, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="intentionally disabled"):
        flare_autoregressive_separated_trition(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode,
            chunk_size=16,
        )
