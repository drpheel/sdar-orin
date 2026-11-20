"""Block-diffusion inference for SDAR-1.7B-Chat-b32.

The same sampler builds a teacher trajectory and runs a short test on this
board. A block starts as mask tokens. Each pass reads the prefix cache, scores
the block, and commits the most confident masked positions. The cache is
written once, after the block is finished, so a noisy pass cannot pollute it.

A trajectory records the commit step of every new token. Masking everything
committed later than step t rebuilds the input the model saw at that step,
which is the record a few-step distillation run needs.
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field

# Must be set before the first CUDA allocation. run_forward sets the same
# variable, but this module imports torch first.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer

from run_forward import load_model, stop_token_ids

MASK_TOKEN = "<|MASK|>"
BLOCK_SIZE = 32


@dataclass
class BlockTrajectory:
    """One generated block and the denoising step that committed each token."""

    token_ids: list
    commit_step: list
    passes: int

    def masks_at_step(self, step, mask_id):
        """Input ids at the start of `step`: tokens committed earlier stay visible."""
        return [
            token if committed < step else mask_id
            for token, committed in zip(self.token_ids, self.commit_step)
        ]


@dataclass
class Generation:
    prompt_ids: list
    blocks: list = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    stopped: str = ""

    @property
    def new_token_ids(self):
        return [token for block in self.blocks for token in block.token_ids]

    @property
    def passes(self):
        return sum(block.passes for block in self.blocks)


def static_commit_count(block_size, steps, step, remaining):
    """How many masked positions a fixed schedule commits on this pass.

    Every pass takes ceil(block / steps) positions. The last pass takes whatever
    is left, so the block always finishes in `steps` passes.
    """
    if remaining <= 0:
        return 0
    if step + 1 >= steps:
        return remaining
    return min(math.ceil(block_size / steps), remaining)


def select_positions(confidence, masked, count):
    """Indices of the `count` most confident positions that are still masked."""
    count = min(count, int(masked.sum()))
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=confidence.device)
    scores = confidence.masked_fill(~masked, float("-inf"))
    return torch.topk(scores, count).indices


def dynamic_positions(confidence, masked, threshold, minimum):
    """Positions at or above `threshold`, or the best `minimum` if none qualify."""
    chosen = (confidence >= threshold) & masked
    if int(chosen.sum()) >= minimum:
        return chosen.nonzero(as_tuple=False).flatten()
    return select_positions(confidence, masked, minimum)


class SDAREngine:
    """Prefill a prompt into the cache, then denoise new blocks against it."""

    def __init__(self, model, tokenizer, block_size=BLOCK_SIZE):
        self.model = model
        self.tokenizer = tokenizer
        mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        if not isinstance(mask_id, int) or mask_id < 0:
            raise RuntimeError(f"tokenizer has no {MASK_TOKEN}")
        self.mask_id = mask_id
        self.block_size = block_size
        self.stop_ids = stop_token_ids(tokenizer)
        self._graph = None
        self._graph_len = None
        self._static_ids = None
        self._static_logits = None
        self._warmup_hold = None
        self._host_logits = None

    def _cache_len(self, cache):
        if cache is None:
            return -1
        return cache.get_seq_length()

    def _run(self, ids, cache, store_kv):
        """One forward. `ids` is a [1, length] CUDA tensor kept at a fixed address for a graph."""
        return self.model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=store_kv,
            logits_to_keep=0,
        )

    def _capture(self, tokens, cache):
        """Graph the read-only 32-token forward for this cache length.

        Two eager passes on this stream initialize the 32-token GEMMs. The graph
        recorded on the first launch does not match them. The capture is the
        first scored pass; later passes of this cache length replay it.
        """
        if self._static_ids is None:
            self._static_ids = torch.empty(
                1, self.block_size, dtype=torch.long, device="cuda"
            )
        self._static_ids[0].copy_(tokens)
        if self._host_logits is None:
            # Pinned destination for the replayed logits. Scoring this on the
            # CPU avoids another GPU allocation while the graph pool is alive.
            self._host_logits = torch.empty(
                self.block_size,
                self.model.config.vocab_size,
                dtype=self.model.dtype,
                pin_memory=True,
            )
        # The first 32-token launch and the graph of that launch disagree with
        # a later eager pass. Two eager passes, then the capture, match.
        first = self._run(self._static_ids, cache, store_kv=False)
        warmed = self._run(self._static_ids, cache, store_kv=False)
        del first
        self._static_logits = torch.empty_like(warmed.logits)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._run(self._static_ids, cache, store_kv=False)
            self._static_logits.copy_(output.logits)
        self._graph = graph
        self._graph_len = self._cache_len(cache)
        # Keep the eager output alive so the caching allocator cannot hand its
        # storage to the logit buffer.
        self._warmup_hold = warmed

    def _forward_read(self, tokens, cache):
        """Score a full block without writing the cache. Replays the graph after the first pass.

        `tokens` is a pinned CPU tensor. The only GPU writes are into the static
        buffers captured with the graph, so the denoise loop does not allocate.
        """
        if self._graph is None or self._graph_len != self._cache_len(cache):
            self._capture(tokens, cache)
        else:
            self._static_ids[0].copy_(tokens)
            self._graph.replay()
        self._host_logits.copy_(self._static_logits[0])
        return self._host_logits.float(), cache

    def _forward(self, token_ids, cache, store_kv):
        """One block forward. `store_kv` writes the block; otherwise the cache is read only."""
        ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        output = self._run(ids, cache, store_kv)
        return output.logits[0].float(), output.past_key_values

    def _prefill(self, prompt_ids):
        """Store the prompt in block-sized pieces so each piece sees only earlier pieces."""
        cache = None
        for start in range(0, len(prompt_ids), self.block_size):
            piece = prompt_ids[start : start + self.block_size]
            _, cache = self._forward(piece, cache, store_kv=True)
        return cache

    def _scores(self, logits, masked):
        """Greedy token and its probability, for masked positions only."""
        probs = torch.softmax(logits, dim=-1)
        chosen = probs.argmax(dim=-1)
        # A mask prediction would leave the position empty. Take the next-best token.
        mask_hit = chosen == self.mask_id
        if bool(mask_hit.any()):
            top2 = probs.topk(2, dim=-1).indices
            chosen = torch.where(mask_hit, top2[:, 1], chosen)
        confidence = probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
        confidence = confidence.masked_fill(~masked, 0)
        return chosen, confidence

    def _commit_details(self, logits, positions, chosen, confidence):
        """Confidence and runner-up probability for the positions this pass commits.

        Both come from the same row of the softmax. The runner-up is the best
        token other than the one committed, so a mask that was skipped still
        counts as an alternative.
        """
        pos = positions.detach().to(dtype=torch.long, device=logits.device)
        rows = logits.index_select(0, pos)
        probs = torch.softmax(rows, dim=-1)
        chosen_ids = chosen.index_select(0, pos).unsqueeze(1)
        probs.scatter_(1, chosen_ids, 0)
        runner_p, runner_i = probs.max(dim=-1)
        conf = confidence.index_select(0, pos)
        return [
            {
                "index": int(pos[i]),
                "confidence": float(conf[i]),
                "runner_up_id": int(runner_i[i]),
                "runner_up_p": float(runner_p[i]),
            }
            for i in range(int(pos.numel()))
        ]

    def _denoise_block(self, cache, steps, strategy, threshold, record_trace=False):
        """Fill one block. Returns the trajectory and the cache after the commit forward."""
        tokens = torch.full(
            (self.block_size,), self.mask_id, dtype=torch.long, pin_memory=True
        )
        commit_step = [-1] * self.block_size
        confidence_at = [0.0] * self.block_size
        runner_id = [0] * self.block_size
        runner_p = [0.0] * self.block_size
        passes = 0
        for step in range(steps):
            masked = tokens == self.mask_id
            remaining = int(masked.sum())
            if remaining == 0:
                break
            logits, _ = self._forward_read(tokens, cache)
            chosen, confidence = self._scores(logits, masked)
            if strategy == "static":
                count = static_commit_count(self.block_size, steps, step, remaining)
                positions = select_positions(confidence, masked, count)
            else:
                positions = dynamic_positions(confidence, masked, threshold, minimum=1)
            if positions.numel() == 0:
                break
            if record_trace:
                for detail in self._commit_details(logits, positions, chosen, confidence):
                    index = detail["index"]
                    confidence_at[index] = detail["confidence"]
                    runner_id[index] = detail["runner_up_id"]
                    runner_p[index] = detail["runner_up_p"]
            tokens[positions] = chosen[positions]
            for index in positions.tolist():
                commit_step[index] = step
            passes += 1
            if int((tokens == self.mask_id).sum()) == 0:
                break
        if any(step < 0 for step in commit_step):
            raise RuntimeError("block finished with positions still masked")
        # The graph pool holds a second copy of the forward workspace. Drop it
        # before the commit pass, which has to allocate the cached keys.
        self._graph = None
        self._static_logits = None
        self._warmup_hold = None
        torch.cuda.empty_cache()
        _, cache = self._forward(tokens.tolist(), cache, store_kv=True)
        passes += 1
        block = BlockTrajectory(tokens.tolist(), commit_step, passes)
        if record_trace:
            # Not a dataclass field, so the distillation record stays unchanged.
            block.trace = {
                "confidence": confidence_at,
                "runner_up_id": runner_id,
                "runner_up_p": runner_p,
            }
        return block, cache

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        steps,
        strategy="static",
        threshold=0.9,
        max_new_tokens=128,
        record_trace=False,
        stop_texts=None,
        on_block=None,
    ):
        """Denoise blocks until an end token is committed or `max_new_tokens` is reached.

        `record_trace` stores, on each block, the probability of the committed
        token and of its runner-up. `stop_texts` ends generation once one of
        those strings appears in the decoded completion. `on_block` is called
        with the generation and the block just finished.
        """
        if strategy not in ("static", "dynamic"):
            raise ValueError("strategy must be 'static' or 'dynamic'")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        prefilled = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        cache = self._prefill(prompt_ids)
        prefilled.record()
        result = Generation(prompt_ids=list(prompt_ids))
        produced = 0
        while produced < max_new_tokens:
            block, cache = self._denoise_block(
                cache, steps, strategy, threshold, record_trace=record_trace
            )
            result.blocks.append(block)
            produced += self.block_size
            if on_block is not None:
                on_block(result, block)
            if any(token in self.stop_ids for token in block.token_ids):
                result.stopped = "end_token"
                break
            if stop_texts:
                decoded = self.tokenizer.decode(result.new_token_ids)
                if any(stop and stop in decoded for stop in stop_texts):
                    result.stopped = "stop_text"
                    break
        if not result.stopped:
            result.stopped = "max_new_tokens"
        finished.record()
        torch.cuda.synchronize()
        result.prefill_s = started.elapsed_time(prefilled) / 1000
        result.decode_s = prefilled.elapsed_time(finished) / 1000
        return result

    def training_record(self, prompt_ids, generation):
        """JSON-ready trajectory. `commit_step` rebuilds each denoising input."""
        return {
            "prompt_ids": prompt_ids,
            "mask_id": self.mask_id,
            "block_size": self.block_size,
            "blocks": [asdict(block) for block in generation.blocks],
        }


def _check_schedule():
    """The fixed schedule covers a block of 32 in the requested number of passes."""
    for steps, expected in ((1, [32]), (4, [8, 8, 8, 8]), (3, [11, 11, 10]), (32, [1] * 32)):
        remaining = 32
        counts = []
        for step in range(steps):
            count = static_commit_count(32, steps, step, remaining)
            counts.append(count)
            remaining -= count
        if counts != expected or remaining != 0:
            raise RuntimeError(f"schedule {steps}: {counts}, left {remaining}")


def main():
    parser = argparse.ArgumentParser(description="Denoise one SDAR prompt and write its trajectory.")
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--strategy", choices=("static", "dynamic"), default="static")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--trajectory", default="")
    args = parser.parse_args()
    _check_schedule()

    from run_forward import MODEL

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    meta = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if meta:
        raise RuntimeError(f"parameters still on meta: {meta[:8]}")
    engine = SDAREngine(model, tokenizer)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    print(f"prompt_tokens {len(prompt_ids)}  steps {args.steps}  strategy {args.strategy}", flush=True)
    generation = engine.generate(
        prompt_ids,
        steps=args.steps,
        strategy=args.strategy,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
    )
    new_ids = generation.new_token_ids
    stop_at = len(new_ids)
    for index, token in enumerate(new_ids):
        if token in engine.stop_ids:
            stop_at = index + 1
            break
    visible = new_ids[:stop_at]
    print(
        f"prefill_s {generation.prefill_s:.3f}  decode_s {generation.decode_s:.3f}  "
        f"passes {generation.passes}  stop {generation.stopped}",
        flush=True,
    )
    if generation.decode_s > 0:
        print(f"decode_tok_s {len(visible) / generation.decode_s:.2f}", flush=True)
    print(f"decoded {tokenizer.decode(visible)!r}", flush=True)
    if args.trajectory:
        record = engine.training_record(prompt_ids, generation)
        with open(args.trajectory, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        print(f"trajectory {args.trajectory}", flush=True)


if __name__ == "__main__":
    main()
