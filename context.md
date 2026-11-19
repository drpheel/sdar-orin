# Handoff

Continue SDAR-1.7B-Chat-b32 inference on this Jetson Orin Nano Super. The goal is BF16 weights, a long prefix stored on the NVMe, and a short resident KV window in RAM. Step 1 (one short BF16 prefill) is not finished. The forward has never succeeded.

## Board

- Jetson Orin Nano Super, JetPack 6, L4T R36.4.7, CUDA 12.6, sm_87, power mode 25 W.
- 7.4 GiB unified memory. CPU and GPU share it. About 6 GiB is free when the machine is idle.
- NVMe: SanDisk SSD Plus 250GB on PCIe 3.0 x4, mounted at `/mnt/nvme`, about 215 GB free before the checkpoint.
- GPU streaming reads measured at 56 GB/s (1 GB buffer) and 66 GB/s (24 MB reads) at 918 MHz. Spec peak is 102 GB/s. Do not use 93 GB/s.
- NVMe direct reads: 2 MiB random 1.55 GB/s, 14–28 MiB random about 2.1–2.2 GB/s. An earlier 1 MiB `dd` read was 1.5 GB/s.

## Model

Checkpoint: `/mnt/nvme/models/SDAR-1.7B-Chat-b32` from https://huggingface.co/JetLM/SDAR-1.7B-Chat-b32

- 2.03B parameters, BF16, `model.safetensors` is 3.8 GB.
- 28 layers, hidden 2048, intermediate 6144, 16 query heads, 8 KV heads, head dim 128.
- `max_position_embeddings` is 32768. RoPE theta is 1e6. Keys are written to the cache after RoPE.
- KV is 112 KiB per token in BF16 (56 KiB in INT8). 8192 tokens is 896 MB BF16 or 450 MB INT8. 32768 tokens is 3.5 GiB and does not fit next to the weights.
- Block size for this checkpoint is 32. The published recipe for the block-4 models is `denoising_steps=4`. Step count for b32 was not measured here.
- Non-embedding weights streamed every pass are 3.44 GB. At 66 GB/s that floor is 52 ms. A 32-token pass is about 60 ms once an 8192-token INT8 window is included.

## Speed we already settled

Exact attention over a paged prefix is disk-bound. One pass over a 32768-token BF16 cache is 3.5 GiB, about 2.5 s at 1.5 GB/s, about 13 tok/s if the whole block commits in one pass and about 3 tok/s at 4 passes.

The 30–130 tok/s band is only for a sliding window that already sits in RAM, with NVMe used to append the new 32-token page (about 1 ms) or to prefetch pages known before the block starts. A full 450 MB INT8 reload is about 200 ms and does not hide behind a short 4-pass block. Replacing the whole hot set after the block is decided drops a 4-pass run from about 130 tok/s to about 70.

Those rates are ceilings from the bandwidth measurements, not timed generation.

## What is done

- PyTorch in `~/.local` is `2.8.0` from the JetPack 6 cu126 index. `torch.cuda.is_available()` is true and a BF16 tensor allocates on the Orin. The previous wheel was `2.12.0+cu130` and could not see this GPU. That swap is system-wide for this user.
- `accelerate` 1.15.0 is installed. `transformers` is 5.10.2, which is newer than the 4.52.3 the checkpoint was exported with.
- `/mnt/nvme/models/SDAR-1.7B-Chat-b32/modeling_sdar.py` is a symlink to `/home/ameyades/sdar-orin/modeling_sdar.py`. The original file is `modeling_sdar.py.upstream` in the checkpoint dir.
- `configuration_sdar.py` in this project is a symlink back to the checkpoint.
- Modeling patch, in `modeling_sdar.py`:
  - Removed the hard `flash_attn` import. Triton RMSNorm is not on this board.
  - RMSNorm is the plain PyTorch implementation that was commented out in the upstream file.
  - Both attention paths use `F.scaled_dot_product_attention` with `enable_gqa=True`. An all-true mask is treated as no mask.
  - `LossKwargs` is a local TypedDict. Transformers 5.10 removed it.
  - `padding_idx` uses `getattr(config, "pad_token_id", None)`. The config has no pad id.
  - RoPE type `"default"` is computed locally. `ROPE_INIT_FUNCTIONS` in transformers 5.10 no longer has that key.
- `run_forward.py` builds the model under `init_empty_weights`, then copies safetensors onto CUDA one tensor at a time. It sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` before importing torch.

## Where it stopped

`python3 /home/ameyades/sdar-orin/run_forward.py` has never printed a next token.

1. `from_pretrained(..., device_map="cuda")` hit `NVML_SUCCESS == r` inside `CUDACachingAllocator.cpp`. nvidia-smi on this board reports memory as unsupported, and the caching allocator asserts on that NVML query. Skipping `caching_allocator_warmup` was not enough. The same assert happens on `tensor.to("cuda")`.
2. `backend:cudaMallocAsync` gets past the assert. Loading still dies with `torch.OutOfMemoryError` / `NvMapMemAllocInternalTagged ... error 12`. The one-tensor loop printed `loaded 80 tensors` and then failed on the next copy. A single 2 GB `torch.empty` succeeded in a separate process, including while a stuck loader was resident, so the failure is the growing set of parameter allocations plus the CPU copy, not a hard 1 GB ceiling.
3. One loader was left resident at about 3.6 GB RSS and had to be killed. Check `ps` for `run_forward.py` before retrying.

Do not keep retrying `from_pretrained` unchanged. The next load should reserve one CUDA buffer for the parameters, copy each tensor into it, and drop the CPU copy before reading the next one. Unified memory means the safetensors bytes and the GPU weights cannot both stay live.

Agreed order:

1. One short BF16 prefill. This is the open step.
****ONLY AFTER HUMAN INTERVENTION - ASK AMEYA (USER) FIRST****
2. Online-softmax attention over a few thousand tokens in RAM, checked against a short reference.
3. Append 32-token KV pages to the NVMe and measure tok/s on a sliding window.
4. Only then grow toward an 8192-token resident window and a 32768-token stored prefix. Positions past 32768 are outside this checkpoint.
