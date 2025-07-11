#!/usr/bin/env python3
# Two-Jetson TinyLlama pipeline – CPU hop avoids Gloo CUDA bug.

import argparse, datetime, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
def init_pg(rank, world, addr, port):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"tcp://{addr}:{port}",
        timeout=datetime.timedelta(minutes=10),
    )
    print(f"[Rank {rank}] ✓ process group", flush=True)

# ---------------  CPU-hop send / recv  ---------------------------------
def send_gpu_tensor(tensor_gpu, dst):
    t_cpu  = tensor_gpu.to("cpu", non_blocking=False).contiguous()
    shape  = torch.tensor(t_cpu.shape, dtype=torch.int64)   # CPU
    dist.send(shape, dst);   dist.send(t_cpu, dst)

def recv_gpu_tensor(src, device):
    shape = torch.empty(3, dtype=torch.int64)               # CPU
    dist.recv(shape, src)
    t_cpu = torch.empty(tuple(shape.tolist()), dtype=torch.float16)
    dist.recv(t_cpu, src)
    return t_cpu.to(device, non_blocking=False)
# ----------------------------------------------------------------------

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world_size", type=int, default=2)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--master_addr", default="192.168.3.15")
    ap.add_argument("--master_port", type=int, default=29500)
    ap.add_argument("--prompt", default="TinyLlama on Jetson is")
    ap.add_argument("--model_name",
                    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    cfg = ap.parse_args()

    init_pg(cfg.rank, cfg.world_size, cfg.master_addr, cfg.master_port)
    dev = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    mdl.config.use_cache = False

    mdl.model.embed_tokens = mdl.model.embed_tokens.to(dev)

    layers = mdl.model.layers
    layers0, layers1 = layers[:cfg.k], layers[cfg.k:]
    for blk in layers0 + layers1:
        blk.to(dev)
    norm = mdl.model.norm.to(dev)
    head = mdl.lm_head.to(dev)
    print(f"[Rank {cfg.rank}] ✓ shards ready", flush=True)

    if cfg.rank == 0:                          # ---------- master ----------
        ids = tok(cfg.prompt, return_tensors="pt").input_ids.to(dev)
        pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        h   = mdl.model.embed_tokens(ids)
        for blk in layers0:
            h = blk(h, position_ids=pos, use_cache=False)[0]
        send_gpu_tensor(h, 1)

        logits = recv_gpu_tensor(1, dev)
        nxt_id = torch.argmax(logits[0, -1]).unsqueeze(0)
        print(">> Next token:", tok.decode(nxt_id), flush=True)

    else:                                     # ---------- worker ----------
        h = recv_gpu_tensor(0, dev)
        pos = torch.arange(h.shape[1], device=dev).unsqueeze(0)
        for blk in layers1:
            h = blk(h, position_ids=pos, use_cache=False)[0]
        h = norm(h)
        logits = head(h)
        send_gpu_tensor(logits, 0)

if __name__ == "__main__":
    main()

