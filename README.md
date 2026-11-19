# SDAR on the Orin Nano

BF16 inference for [JetLM/SDAR-1.7B-Chat-b32](https://huggingface.co/JetLM/SDAR-1.7B-Chat-b32) on the Jetson Orin Nano Super.

Weights stay on the NVMe at `/mnt/nvme/models/SDAR-1.7B-Chat-b32`. This project holds the loader and the modeling patch. The checkpoint's `modeling_sdar.py` is a symlink back here, so the model loads the PyTorch RMSNorm and SDPA path instead of FlashAttention.

```bash
python3 /home/ameyades/sdar-orin/run_forward.py
```

That runs one prefill of "Who are you?" in BF16 and prints the next-token logits.
