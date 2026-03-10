import torch


def get_torch_runtime_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "device_type": "cpu",
        "device_name": "CPU",
        "device_label": "CPU",
    }

    if info["cuda_available"]:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        info.update(
            {
                "device_type": "cuda",
                "device_name": device_name,
                "device_label": f"GPU: {device_name}",
                "device_index": device_index,
                "device_capability": torch.cuda.get_device_capability(device_index),
            }
        )

    return info


def optimize_predictor_runtime(predictor):
    runtime_info = get_torch_runtime_info()
    predictor.runtime_info = runtime_info
    predictor.runtime_device_label = runtime_info["device_label"]
    predictor._use_channels_last = False
    predictor._non_blocking_transfers = False

    if predictor.device.type != "cuda":
        return runtime_info

    torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    predictor.model = predictor.model.to(
        device=predictor.device,
        memory_format=torch.channels_last,
    )
    predictor._use_channels_last = True
    predictor._non_blocking_transfers = True
    return runtime_info


def warmup_predictor_runtime(predictor):
    if predictor.device.type != "cuda":
        return

    input_channels = 1 if predictor.use_grayscale else 3
    dummy_batch = torch.zeros(
        (64, input_channels, 32, 32),
        dtype=torch.float32,
        device=predictor.device,
    )
    if getattr(predictor, "_use_channels_last", False):
        dummy_batch = dummy_batch.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        predictor.model(dummy_batch)
    torch.cuda.synchronize(predictor.device)
