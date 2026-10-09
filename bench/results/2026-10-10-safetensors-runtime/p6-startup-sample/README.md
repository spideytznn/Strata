# Bounded CPU startup sample, 2026-10-10

Windows, Ryzen 9950X3D, 96 GB RAM; original checkpoint on D: BIWIN X570 PRO.
The existing RTX 5090 user service was running; this tool uses no CUDA context,
performs no generation and does not restart the server.

Command after the CPU-only build:

```powershell
.\build-safetensors\safetensors_batch_bench.exe D:\迅雷下载\Qwen3.8-flash-next-nvfp4 0 64 4
```

The JSON records one warmup per mode, then three rounds alternating execution
order. Timed legacy mode loads one expert, copies into the arena and frees its
temporary. Batch mode reuses bounded host scratch and packs directly into the
arena with four workers. Both arenas are byte-compared after every round, with
separate checks of all input scales and rejection of source reads after sealing.
Each timed invocation reads 176,948,736 expert bytes, no ngram bytes.

| Mode | Median ms | Source read calls per invocation |
| --- | ---: | ---: |
| Legacy serial | 84.8902 | 192 |
| Batch 64, four workers, reused scratch | 34.7159 | 141 |

This measures warm loading of 64 layer-0 experts, including allocations in each
loader's timed call. It does not measure full-model cold startup, full-arena
locking, GPU cache population, prefill, quality or inference speed. Do not
extrapolate the sample ratio to total startup time. The checkpoint remains
read-only; no disk weight cache is generated.
