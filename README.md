# TinyLlama Distributed Demo on Two Jetsons

```
llm_demo/
├── jetson_tinyllama_pipeline.py    # Python script that splits TinyLlama across 2 GPUs
├── run_tinyllama_pipeline.sh       # One‑click launcher (master or worker)
└── README.md                       # (this file)
```

## 0. Prerequisites

| Item | How to check / install |
|------|------------------------|
| **JetPack 6.x** | `cat /etc/nv_tegra_release` → shows L4T 36.x |
| **Docker** | `sudo apt install -y docker.io` |
| User in `docker` group | `sudo usermod -aG docker $USER && newgrp docker` |
| **Two Jetsons on the same LAN** | Verify they can ping each other. |
| **Disk space** | ≈3 GB free under `~/hf_cache` (model weights). |

No Hugging Face token is needed (TinyLlama is public).

---

## 1. Get the files

```bash
mkdir -p ~/llm_demo && cd ~/llm_demo
# copy these three files into this folder:
#   jetson_tinyllama_pipeline.py
#   run_tinyllama_pipeline.sh
#   README.md  (this file)
chmod +x run_tinyllama_pipeline.sh
```

---

## 2. Choose roles / IPs

* Pick one board as **master** (rank 0). Note its IP, e.g. `192.168.3.15`.
* The other is **worker** (rank 1).  
* Choose one free TCP port (default `29500`).

---

## 3. Launch the worker **first**

```bash
cd ~/llm_demo
./run_tinyllama_pipeline.sh worker 192.168.3.15 29500
```

Wait until you see:

```
[Rank 1] waiting for hidden …
```

---

## 4. Launch the master

```bash
cd ~/llm_demo
./run_tinyllama_pipeline.sh master 192.168.3.15 29500
```

You should eventually see:

```
>> Next token: blazing
```

---

## 5. How it works (quick overview)

```
jetson_tinyllama_pipeline.py   # PyTorch script
├─ moves TinyLlama to fp16 CUDA
├─ splits layers: first k on master, rest on worker
├─ sends tensors over TCP (Gloo) via CPU hop
└─ prints next‑token result

run_tinyllama_pipeline.sh      # wrapper
├─ pulls NVIDIA Jetson PyTorch container
├─ mounts ~/llm_demo + ~/hf_cache
├─ installs transformers + deps (first run only)
└─ runs the script with correct rank/env
```

---

## 6. FAQ

| Question | Answer |
|----------|--------|
| **Port already used?** | Pick another, e.g. `29600`, and pass to both commands. |
| **OOM on master Jetson?** | Lower `--k` inside the script (fewer layers on rank 0). |
| **Try Llama‑2‑7B?** | Change `--model_name` in the script and export a valid HF token. |

Happy distributed inferencing!

