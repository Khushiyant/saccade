"""Verify the RoPE port: BlockCausalAttention is accuracy-faithful to V-JEPA2."""
import torch
from transformers import AutoModel
from saccade.streaming.causal_attention import BlockCausalAttention, block_causal_mask
from saccade.streaming.state_cache import StateCache

dev, dt = "cuda", torch.float16
m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=dt).to(dev).eval()
attn = m.encoder.layer[0].attention
tpf = attn.grid_size ** 2
N = 2 * tpf
torch.manual_seed(0)
x = torch.randn(1, N, 1024, device=dev, dtype=dt)

bca = BlockCausalAttention.from_attention(attn, tokens_per_frame=tpf).to(dev)
print(f"rope_enabled={bca.rope_enabled} grid={bca.grid_size} d/h/w={bca.rope_d_dim}/{bca.rope_h_dim}/{bca.rope_w_dim} tpf={tpf}")

with torch.no_grad():
    hf_out = attn(x)[0]
    full_mask = torch.ones(N, N, dtype=torch.bool, device=dev)
    bca_full = bca._forward_parallel(x, full_mask)
    denom = hf_out.abs().max().item()
    rope_rel = (hf_out - bca_full).abs().max().item() / denom
    print(f"[1] RoPE port  max|HF - BCA(bidirectional)| rel = {rope_rel:.5f}  -> {'PASS' if rope_rel < 0.02 else 'FAIL'}")

    cmask = block_causal_mask(2, tpf, device=dev)
    par = bca._forward_parallel(x, cmask)
    cache = StateCache(num_layers=1, max_frames=64, tokens_per_frame=tpf)
    outs = [bca._forward_incremental(x[:, f * tpf:(f + 1) * tpf], cache, 0) for f in range(2)]
    inc = torch.cat(outs, dim=1)
    inc_max = (par - inc).abs().max().item()
    inc_rel = inc_max / par.abs().max().item()
    print(f"[2] incremental == parallel-causal  max={inc_max:.5f} rel={inc_rel:.5f}  -> {'PASS' if inc_rel < 0.02 else 'FAIL'}")

    causal_vs_bidir = (par[:, :tpf] - bca_full[:, :tpf]).abs().max().item()
    print(f"[3] causal frame0 differs from bidirectional frame0: {causal_vs_bidir:.4f} (should be > 0)")

    last_rel = (par[:, tpf:] - bca_full[:, tpf:]).abs().max().item() / bca_full[:, tpf:].abs().max().item()
    print(f"[4] last-frame causal == bidirectional rel={last_rel:.5f} -> {'PASS' if last_rel < 0.02 else 'FAIL'}")
