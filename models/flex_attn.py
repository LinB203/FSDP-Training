# import torch
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# # 你之前的设置
# torch._dynamo.config.cache_size_limit = 512
# torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention)


# # mask_mod: causal (query index >= kv index => allowed)
# def causal_mask(b, h, q_idx, kv_idx):
#     # 注意：这里返回 True 表示 q_idx >= kv_idx（即允许访问当前或之前的位置）
#     return q_idx >= kv_idx


# # 一些超参
# B = 2  # batch
# H = 16  # heads
# L = 4096  # 序列长度 (query length = kv length for causal)
# D_head = 128  # 每个 head 的维度
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(0)

# # reshape 到 (B, H, L, D_head)
# q = torch.randn(
#     B,
#     H,
#     L,
#     D_head,
#     dtype=torch.float32,
#     device=device,
# )  # (B, H, L, D)
# k = torch.randn(
#     B,
#     H,
#     L,
#     D_head,
#     dtype=torch.float32,
#     device=device,
# )  # (B, H, L, D)
# v = torch.randn(
#     B,
#     H,
#     L,
#     D_head,
#     dtype=torch.float32,
#     device=device,
# )  # (B, H, L, D)

# block_mask = create_block_mask(
#     causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device, _compile=True
# )

# # 调用 flex_attention
# out = flex_attention(q, k, v, block_mask=block_mask)

# # out 的形状通常是 (B, H, L, D_head)
# print("flex_attention output shape:", out.shape)

# # --- 参考实现：标准 SDPA + 因果 mask，用来做数值比对 ---
# ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
# print("sdpa output shape:", ref_out.shape)

# # 比较
# max_abs = (out - ref_out).abs().max().item()
# mean_abs = (out - ref_out).abs().mean().item()
# print(f"差异: max_abs={max_abs}, mean_abs={mean_abs}")


import time
import statistics
import torch
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

B = 2
H = 32
L = 8192
D_head = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP = 100
ITERS = 500

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

try:
    flex_attention = torch.compile(flex_attention)
except Exception as e:
    print("torch.compile(flex_attention) failed or not necessary:", e)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


torch.manual_seed(0)
q = torch.randn(B, H, L, D_head, dtype=torch.bfloat16, device=device)
k = torch.randn(B, H, L, D_head, dtype=torch.bfloat16, device=device)
v = torch.randn(B, H, L, D_head, dtype=torch.bfloat16, device=device)

block_mask = create_block_mask(
    causal_mask, B=B, H=H, Q_LEN=L, KV_LEN=L, device=device, _compile=True
)


def call_flex(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)


def call_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def benchmark(fn, name, warmup=WARMUP, iters=ITERS, use_block_mask=False):
    device_is_cuda = device.startswith("cuda")
    if device_is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for i in range(warmup):
        out = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
        if device_is_cuda:
            torch.cuda.synchronize()
    times = []
    for i in range(iters):
        if device_is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
        if device_is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = statistics.mean(times)
    med = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    mn = min(times)
    mx = max(times)
    # throughput: tokens/sec = B * L / avg_time
    toks_per_sec = (B * L) / avg if avg > 0 else float("inf")

    peak_mem_gb = None
    if device_is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_bytes / (1024**3)

    results = {
        "name": name,
        "avg_s": avg,
        "med_s": med,
        "stdev_s": stdev,
        "min_s": mn,
        "max_s": mx,
        "tokens_per_sec": toks_per_sec,
        "peak_mem_gb": peak_mem_gb,
    }
    return results


def main():
    print(f"Device: {device}  B={B} H={H} L={L} D_head={D_head}")
    results = []
    try:
        print("Benchmarking flex_attention ...")
        r = benchmark(call_flex, "flex_attention", use_block_mask=True)
        results.append(r)
    except Exception as e:
        print("flex_attention benchmark failed:", e)

    try:
        print("Benchmarking scaled_dot_product_attention (SDPA) ...")
        r = benchmark(call_sdpa, "scaled_dot_product_attention")
        results.append(r)
    except Exception as e:
        print("SDPA benchmark failed:", e)

    print("\n--- Results ---")
    for r in results:
        print(f"{r['name']}:")
        print(f"  avg time     : {r['avg_s'] * 1000:.3f} ms")
        print(f"  median time  : {r['med_s'] * 1000:.3f} ms")
        print(f"  stdev        : {r['stdev_s'] * 1000:.3f} ms")
        print(
            f"  min / max    : {r['min_s'] * 1000:.3f} ms / {r['max_s'] * 1000:.3f} ms"
        )
        print(
            f"  tokens/sec   : {r['tokens_per_sec'] / 1e6:.3f} Mtokens/s ({r['tokens_per_sec']:.0f} toks/s)"
        )
        if r["peak_mem_gb"] is not None:
            print(f"  peak memory  : {r['peak_mem_gb']:.3f} GB")
        print("")


if __name__ == "__main__":
    main()
