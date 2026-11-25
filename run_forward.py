"""One BF16 prefill and a greedy decode of SDAR-1.7B-Chat-b32 on the Orin GPU.

The board has one memory pool. The safetensors file is 3.8 GB and the
GPU weights are another 3.8 GB, so both cannot stay resident. This loader
reserves one CUDA buffer, copies each tensor into it, and drops the CPU
bytes before reading the next tensor.
"""

import gc
import json
import os
import struct
import time

# cudaMallocAsync on this nvgpu driver stops near 2.5 GiB. The native caching
# allocator can take the 3.8 GiB weight buffer. Expandable segments let the
# forward's next allocation grow that pool instead of asking for a fresh 1 GiB
# block that the 7.4 GiB board no longer has free.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Checkpoint on the NVMe. The modeling file there is a symlink back into this project.
MODEL = "/mnt/nvme/models/SDAR-1.7B-Chat-b32"
WEIGHTS = f"{MODEL}/model.safetensors"
# Pad each tensor in the packed buffer so BF16 views stay aligned.
ALIGN = 256
# Stop earlier than this only when an end token is chosen. This cap is a backstop.
MAX_NEW_TOKENS = 512
# Decode tok/s is reported after this many new-token forwards, when a run reaches them.
PASS_MARKS = (8, 16, 32, 64, 128)
PROMPTS = (
    "Who are you?",
    "Say hello in one sentence.",
    "Name three prime numbers.",
    "Explain what an SSD is in two sentences.",
)

# Safetensors dtype tags mapped to the torch dtypes stored in the file.
_SAFETENSORS_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_header(path):
    """Read the safetensors header and return the data offset plus one record per tensor."""
    with open(path, "rb") as handle:
        # First 8 bytes are the little-endian length of the JSON header.
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        # Tensor bytes begin immediately after that header.
        data_base = 8 + header_len
    tensors = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        tensors.append(
            {
                "name": name,
                "start": start,
                "nbytes": end - start,
                "dtype": _SAFETENSORS_DTYPE[meta["dtype"]],
                "shape": tuple(meta["shape"]),
            }
        )
    return data_base, tensors


# Q, K, V rows, in the order the fused GEMM splits them.
_FUSED_ROWS = (
    ("self_attn.q_proj.weight", "qkv", 0),
    ("self_attn.k_proj.weight", "qkv", 1),
    ("self_attn.v_proj.weight", "qkv", 2),
)


def fusion_slot(name):
    """Return (group, order) when this weight is part of a packed GEMM, else (None, None)."""
    for suffix, kind, index in _FUSED_ROWS:
        if name.endswith(suffix):
            return name[: -len(suffix)] + kind, index
    return None, None


def _checked_nbytes(item):
    numel = 1
    for dim in item["shape"]:
        numel *= dim
    expect = numel * item["dtype"].itemsize
    if expect != item["nbytes"]:
        raise RuntimeError(
            f"{item['name']} is {item['nbytes']} bytes, shape {item['shape']} wants {expect}"
        )
    return item["nbytes"]


def plan_offsets(tensors):
    """Assign offsets in the single CUDA buffer.

    Standalone tensors stay 256-byte aligned. Q, K, and V are packed with no gap
    so one GEMM can view them as a single matrix without copying weights.
    """
    groups = {}
    singles = []
    for item in tensors:
        key, index = fusion_slot(item["name"])
        if key is None:
            singles.append(item)
        else:
            groups.setdefault(key, []).append((index, item))

    offset = 0

    def place(item, at):
        item["offset"] = at
        return at + _checked_nbytes(item)

    for item in singles:
        offset = (offset + ALIGN - 1) & ~(ALIGN - 1)
        offset = place(item, offset)
    for members in groups.values():
        members.sort(key=lambda pair: pair[0])
        offset = (offset + ALIGN - 1) & ~(ALIGN - 1)
        for _, item in members:
            offset = place(item, offset)
    return offset


def assign_weight(model, name, tensor):
    """Replace one meta parameter with a view into the CUDA buffer. No copy."""
    module = model
    parts = name.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    leaf = parts[-1]
    # requires_grad=False keeps the view a leaf. The buffer owns the storage.
    module.register_parameter(leaf, nn.Parameter(tensor, requires_grad=False))


def load_model():
    """Build the model on the meta device, then fill one CUDA buffer from disk."""
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # Meta tensors allocate no storage, so the module skeleton does not cost 3.8 GB.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    data_base, tensors = read_header(WEIGHTS)
    total = plan_offsets(tensors)
    print(f"reserving {total / 1024 ** 3:.3f} GiB for {len(tensors)} tensors", flush=True)
    # One allocation for every parameter. Views into this buffer are the weights.
    storage = torch.empty(total, dtype=torch.uint8, device="cuda")
    # Reused CPU staging area, sized to the largest tensor (the 593 MiB embedding).
    max_bytes = max(item["nbytes"] for item in tensors)
    cpu = torch.empty(max_bytes, dtype=torch.uint8)

    expected = {name for name, _ in model.named_parameters()}
    seen = set()
    with open(WEIGHTS, "rb") as handle:
        for index, item in enumerate(tensors, start=1):
            name = item["name"]
            nbytes = item["nbytes"]
            handle.seek(data_base + item["start"])
            stage = cpu.narrow(0, 0, nbytes)
            # Read straight into the staging tensor. It is the only CPU copy.
            got = handle.readinto(stage.numpy())
            if got != nbytes:
                raise RuntimeError(f"short read for {name}: {got} != {nbytes}")
            gpu = storage.narrow(0, item["offset"], nbytes)
            gpu.copy_(stage)
            # Reinterpret those bytes as the parameter. This shares the CUDA storage.
            weight = gpu.view(item["dtype"]).reshape(item["shape"])
            if name not in expected:
                raise KeyError(f"checkpoint tensor is not a model parameter: {name}")
            assign_weight(model, name, weight)
            seen.add(name)
            if index % 40 == 0 or nbytes >= 64 * 1024 * 1024:
                print(f"loaded {index} tensors  last {name}", flush=True)

    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"missing weights: {missing[:8]}")

    # Drop the staging buffer before the forward so only the GPU weights stay live.
    del cpu
    gc.collect()

    # RoPE's inv_freq buffer is not in the checkpoint. Rebuild it on the GPU.
    rotary = model.model.rotary_emb.__class__(config, device="cuda")
    model.model.rotary_emb = rotary
    qkv_layers = model.bind_fused_projections()
    print(f"fused qkv {qkv_layers}", flush=True)
    # Keep the packed storage alive for as long as the parameter views exist.
    model._weight_storage = storage
    model.eval()
    return model


def stop_token_ids(tokenizer):
    """End-of-turn ids. Decoding stops once one of these is chosen."""
    stop_ids = set()
    eos = tokenizer.eos_token_id
    if eos is not None:
        stop_ids.add(eos)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)
    return stop_ids


def generate_until_end(model, tokenizer, prompt, stop_ids):
    """Prefill one prompt, then decode one token per pass until an end token.

    The first new token comes from the prefill. Each later pass feeds the
    previous token and appends its keys and values. Times are taken after a
    CUDA sync, so they include the GPU work for that pass.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    # store_kv writes this prompt into the cache. logits_to_keep=1 scores the last token.
    prefill = model(**inputs, use_cache=True, store_kv=True, logits_to_keep=1)
    torch.cuda.synchronize()
    decode_start = time.perf_counter()

    cache = prefill.past_key_values
    generated = [int(prefill.logits[0, -1].argmax())]
    # One timestamp per decode pass, measured from the end of the prefill.
    pass_ends = []
    reason = "end_token" if generated[-1] in stop_ids else None
    while reason is None and len(generated) < MAX_NEW_TOKENS:
        token = torch.tensor([[generated[-1]]], device="cuda")
        step = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
            logits_to_keep=1,
        )
        torch.cuda.synchronize()
        pass_ends.append(time.perf_counter())
        cache = step.past_key_values
        generated.append(int(step.logits[0, -1].argmax()))
        passes = len(pass_ends)
        if passes in PASS_MARKS:
            rate = passes / (pass_ends[-1] - decode_start)
            print(f"  passes {passes}  decode_tok_s {rate:.2f}", flush=True)
        if generated[-1] in stop_ids:
            reason = "end_token"
    if reason is None:
        reason = "max_new_tokens"

    return {
        "prompt_tokens": inputs["input_ids"].shape[-1],
        "generated": generated,
        "prefill_s": decode_start - prefill_start,
        "decode_start": decode_start,
        "pass_ends": pass_ends,
        "reason": reason,
        "cache_tokens": cache.get_seq_length(),
        "logits_shape": tuple(prefill.logits.shape),
        "logits_dtype": prefill.logits.dtype,
    }


def decode_tok_s(result, passes):
    """New tokens per second over the first `passes` decode forwards."""
    elapsed = result["pass_ends"][passes - 1] - result["decode_start"]
    return passes / elapsed


def report_prompt(tokenizer, prompt, result):
    """Print the completion and tok/s at each recorded pass count."""
    passes = len(result["pass_ends"])
    print(f"prompt {prompt!r}", flush=True)
    print(f"prompt_tokens {result['prompt_tokens']}", flush=True)
    print(
        f"prefill_s {result['prefill_s']:.3f}  "
        f"logits {result['logits_shape']} {result['logits_dtype']}",
        flush=True,
    )
    for mark in PASS_MARKS:
        if passes < mark:
            continue
        print(f"passes {mark}  decode_tok_s {decode_tok_s(result, mark):.2f}", flush=True)
    final_rate = decode_tok_s(result, passes) if passes else 0.0
    print(f"passes {passes}  decode_tok_s {final_rate:.2f}  stop {result['reason']}", flush=True)
    print(f"new_tokens {len(result['generated'])}", flush=True)
    print(f"cache_tokens {result['cache_tokens']}", flush=True)
    print(f"decoded {tokenizer.decode(result['generated'])!r}", flush=True)
    print(flush=True)


def main():
    """Load the model, then decode each prompt until its end token and time the passes."""
    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    # A leftover meta parameter would crash the forward on the first use.
    meta = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if meta:
        raise RuntimeError(f"parameters still on meta: {meta[:8]}")

    stop_ids = stop_token_ids(tokenizer)
    with torch.inference_mode():
        for prompt in PROMPTS:
            print(f"prompt {prompt!r}", flush=True)
            result = generate_until_end(model, tokenizer, prompt, stop_ids)
            report_prompt(tokenizer, prompt, result)

    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"max_allocated_mib {allocated:.0f}", flush=True)


if __name__ == "__main__":
    main()
