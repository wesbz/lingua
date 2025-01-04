# Errors zoo

I believe it makes sense to list the issues I encountered while trying to use lingua on H2.

## 0. No issue

Most nodes have no issue as long as you use less than 5 GPUs.
Here are nodes that can run lingua at 8 GPUs:
- `learnfair7518`

## 1. `pynvml.nvml.NVMLError_NotSupported: Not Supported`

```bash
[rank2]: Traceback (most recent call last):
[rank2]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank2]:   File "<frozen runpy>", line 88, in _run_code
[rank2]:   File "/private/home/wesbz/Projects/lingua/apps/main/train.py", line 653, in <module>
[rank2]:     main()
[rank2]:   File "/private/home/wesbz/Projects/lingua/apps/main/train.py", line 649, in main
[rank2]:     train(cfg)
[rank2]:   File "/private/home/wesbz/Projects/lingua/apps/main/train.py", line 295, in train
[rank2]:     logger.info(f"GPU memory usage: {gpu_memory_monitor}")
[rank2]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/private/home/wesbz/Projects/lingua/lingua/metrics.py", line 177, in __str__
[rank2]:     mem_stats = self.get_peak_stats()
[rank2]:                 ^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/private/home/wesbz/Projects/lingua/lingua/metrics.py", line 155, in get_peak_stats
[rank2]:     power_draw = torch.cuda.power_draw()
[rank2]:                  ^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/cuda/__init__.py", line 1209, in power_draw
[rank2]:     return pynvml.nvmlDeviceGetPowerUsage(handle)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/pynvml/nvml.py", line 2404, in nvmlDeviceGetPowerUsage
[rank2]:     _nvmlCheckReturn(ret)
[rank2]:   File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/pynvml/nvml.py", line 833, in _nvmlCheckReturn
[rank2]:     raise NVMLError(ret)
[rank2]: pynvml.nvml.NVMLError_NotSupported: Not Supported
```

Nodes where this behavior is observed:
- `learnfair7691`

## 2. `RuntimeErrorCUDA driver error: invalid device ordinal`

```bash
Traceback (most recent call last):
    File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
      return f(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^
    File "/private/home/wesbz/Projects/lingua/apps/main/train.py", line 651, in main
      train(cfg)
    File "/private/home/wesbz/Projects/lingua/apps/main/train.py", line 233, in train
      setup_torch_distributed(args.distributed)
    File "/private/home/wesbz/Projects/lingua/lingua/distributed.py", line 277, in setup_torch_distributed
      torch.distributed.init_process_group(init_method="env://", backend="nccl")
    File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
      return func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
    File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 97, in wrapper
      func_return = func(*args, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^
    File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1527, in init_process_group
      default_pg, _ = _new_process_group_helper(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/private/home/wesbz/.conda/envs/lingua_241021/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1771, in _new_process_group_helper
      backend_class = ProcessGroupNCCL(
                      ^^^^^^^^^^^^^^^^^
  RuntimeError: CUDA driver error: invalid device ordinal
```

Nodes where this behavior is observed:
- `learnfair2315`
- `learnfair2368` (for 4 GPUs)
- `learnfair1834`

### Solution

This is due to the environment variable `ENABLE_INTRA_NODE_COMM`. Set it to `0` or `1`, idk.