"""Time a 32-token read-only forward, then mark one CUDA-graph replay for nsys.

The eager samples and the replay are the same work: 32 tokens attending to a
prefix that is already in the cache. The captured range is a single replay.
"""

import torch

from run_forward import load_model

BLOCK = 32


def event_ms(fn, repeats=5):
    """Median GPU time of `fn`. The caller warms the allocator up first."""
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    samples.sort()
    return samples[len(samples) // 2]


def main():
    model = load_model()
    token_ids = torch.randint(0, 1000, (1, BLOCK), device="cuda")
    with torch.no_grad():
        norm = model.model.layers[0].input_layernorm
        sample = torch.randn(2, 8, norm.weight.numel(), device="cuda", dtype=torch.bfloat16)
        got = norm(sample)
        sample32 = sample.float()
        ref = (
            sample32
            * torch.rsqrt(sample32.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
            * norm.weight
        ).to(sample.dtype)
        print(f"rmsnorm_max_abs_err {(got - ref).abs().max().item():.4e}", flush=True)

        prefix = model(input_ids=token_ids, use_cache=True, store_kv=True, logits_to_keep=0)
        cache = prefix.past_key_values
        length = cache.get_seq_length()

        def denoise():
            return model(
                input_ids=token_ids,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
                logits_to_keep=0,
            )

        denoise()
        eager_ms = event_ms(denoise)
        if cache.get_seq_length() != length:
            raise RuntimeError("read-only forward wrote the cache")
        print(f"eager_ms {eager_ms:.2f}", flush=True)

        warmed = denoise()
        static_logits = torch.empty_like(warmed.logits)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            denoise()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = denoise()
            static_logits.copy_(output.logits)

        def replay():
            graph.replay()

        replay_ms = event_ms(replay)
        print(f"replay_ms {replay_ms:.2f}", flush=True)
        token_ids[0, 0] = (token_ids[0, 0] + 17) % 1000
        replay()
        replayed = static_logits.float().cpu()
        del graph, static_logits, warmed
        torch.cuda.empty_cache()
        eager = denoise().logits.float().cpu()
        mismatch = (replayed - eager).abs().max().item()
        print(f"replay_logit_max_abs_err {mismatch:.4e}", flush=True)
    print("checked graphed 32-token forward", flush=True)


if __name__ == "__main__":
    main()
