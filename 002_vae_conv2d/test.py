import argparse
import importlib.util
import random
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
TORCH_REF_PATH = ROOT_DIR / "00_torch_ref" / "kernel.py"
TRITON_PATH = ROOT_DIR / "01_triton" / "kernel.py"

DEFAULT_NUM_GROUPS = 32
DEFAULT_KERNEL_SIZE = 3

EDGE_CASES = [
    {"batch_size": 1, "channels": 64, "height": 1, "width": 1, "kernel_size": 3},
    {"batch_size": 1, "channels": 64, "height": 1, "width": 7, "kernel_size": 3},
    {"batch_size": 1, "channels": 64, "height": 7, "width": 1, "kernel_size": 3},
    {"batch_size": 1, "channels": 32, "height": 2, "width": 2, "kernel_size": 3},
    {"batch_size": 1, "channels": 32, "height": 3, "width": 3, "kernel_size": 3},
    {"batch_size": 2, "channels": 64, "height": 31, "width": 17, "kernel_size": 3},
    {"batch_size": 2, "channels": 64, "height": 17, "width": 31, "kernel_size": 3},
    {"batch_size": 3, "channels": 96, "height": 5, "width": 13, "kernel_size": 3},
    {"batch_size": 4, "channels": 160, "height": 4, "width": 4, "kernel_size": 3},
    {"batch_size": 1, "channels": 192, "height": 5, "width": 22, "kernel_size": 3},
]

LARGE_CASES = [
    {"batch_size": 8, "channels": 64, "height": 32, "width": 32, "kernel_size": 3},
    {"batch_size": 8, "channels": 128, "height": 32, "width": 32, "kernel_size": 3},
    {"batch_size": 8, "channels": 256, "height": 16, "width": 16, "kernel_size": 3},
    {"batch_size": 4, "channels": 256, "height": 32, "width": 32, "kernel_size": 3},
    {"batch_size": 4, "channels": 256, "height": 48, "width": 48, "kernel_size": 3},
    {"batch_size": 4, "channels": 256, "height": 64, "width": 64, "kernel_size": 3},
    {"batch_size": 2, "channels": 256, "height": 64, "width": 96, "kernel_size": 3},
    {"batch_size": 1, "channels": 256, "height": 128, "width": 128, "kernel_size": 3},
    {"batch_size": 1, "channels": 256, "height": 96, "width": 160, "kernel_size": 3},
]

SANA_CASES = [
    {"batch_size": 1, "channels": 256, "height": 8, "width": 8, "kernel_size": 3},
    {"batch_size": 1, "channels": 256, "height": 16, "width": 16, "kernel_size": 3},
    {"batch_size": 1, "channels": 256, "height": 32, "width": 32, "kernel_size": 3},
    {"batch_size": 2, "channels": 256, "height": 8, "width": 8, "kernel_size": 3},
    {"batch_size": 2, "channels": 256, "height": 16, "width": 16, "kernel_size": 3},
    {"batch_size": 2, "channels": 256, "height": 32, "width": 32, "kernel_size": 3},
    {"batch_size": 4, "channels": 256, "height": 8, "width": 8, "kernel_size": 3},
    {"batch_size": 4, "channels": 256, "height": 16, "width": 16, "kernel_size": 3},
    {"batch_size": 1, "channels": 256, "height": 7, "width": 11, "kernel_size": 3},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the 002 VAE Conv2D residual block against the torch reference",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--impl",
        default="auto",
        choices=("auto", "torch_ref", "triton"),
        help="Implementation to validate. 'auto' picks triton on CUDA and torch_ref otherwise.",
    )
    parser.add_argument(
        "--suite",
        choices=("edge", "sana", "large", "all"),
        default="all",
    )
    parser.add_argument("--num-groups", type=int, default=DEFAULT_NUM_GROUPS)
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list. Defaults to --seed when omitted.",
    )
    parser.add_argument("--random-tests", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def _resolve_impl_name(impl_arg: str, device: torch.device) -> str:
    if impl_arg == "auto":
        return "triton" if device.type == "cuda" else "torch_ref"
    if impl_arg == "triton" and device.type != "cuda":
        raise RuntimeError("The Triton implementation requires a CUDA device")
    return impl_arg


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_seed_list(seed_arg: str | None, default_seed: int) -> list[int]:
    if seed_arg is None:
        return [default_seed]

    seeds = []
    for item in seed_arg.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("at least one seed must be provided")
    return seeds


def _build_random_cases(args: argparse.Namespace, seed: int) -> list[dict[str, int]]:
    rng = random.Random(seed)
    channel_choices = [32, 64, 96, 128, 160, 192, 224, 256]
    cases = []
    for _ in range(args.random_tests):
        while True:
            case = {
                "batch_size": rng.randint(1, 4),
                "channels": rng.choice(channel_choices),
                "height": rng.randint(1, 48),
                "width": rng.randint(1, 48),
                "kernel_size": args.kernel_size,
            }
            if case["channels"] % args.num_groups == 0:
                cases.append(case)
                break
    return cases


def _build_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    if args.suite == "edge":
        return list(EDGE_CASES)
    if args.suite == "sana":
        return list(SANA_CASES)
    if args.suite == "large":
        return list(LARGE_CASES)
    return list(EDGE_CASES + SANA_CASES + LARGE_CASES)


def _make_inputs(
    case: dict[str, int],
    device: torch.device,
    eps: float,
) -> dict[str, torch.Tensor | float]:
    batch_size = case["batch_size"]
    channels = case["channels"]
    height = case["height"]
    width = case["width"]
    kernel_size = case["kernel_size"]

    x = (torch.randn(batch_size, channels, height, width, device=device) * 0.1).to(torch.float32)

    conv_scale = (channels * kernel_size * kernel_size) ** -0.5
    conv1_weight = (
        torch.randn(channels, channels, kernel_size, kernel_size, device=device) * conv_scale
    ).to(torch.float32)
    conv2_weight = (
        torch.randn(channels, channels, kernel_size, kernel_size, device=device) * conv_scale
    ).to(torch.float32)

    norm1_weight = (1.0 + 0.1 * torch.randn(channels, device=device)).to(torch.float32)
    norm1_bias = (0.1 * torch.randn(channels, device=device)).to(torch.float32)
    norm2_weight = (1.0 + 0.1 * torch.randn(channels, device=device)).to(torch.float32)
    norm2_bias = (0.1 * torch.randn(channels, device=device)).to(torch.float32)

    return {
        "x": x.contiguous(),
        "conv1_weight": conv1_weight.contiguous(),
        "norm1_weight": norm1_weight.contiguous(),
        "norm1_bias": norm1_bias.contiguous(),
        "conv2_weight": conv2_weight.contiguous(),
        "norm2_weight": norm2_weight.contiguous(),
        "norm2_bias": norm2_bias.contiguous(),
        "eps": float(eps),
    }

def _run_case(
    seed: int,
    case_idx: int,
    case: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    ref_run,
    impl_run,
    impl_name: str,
) -> tuple[bool, float]:
    case_seed = seed + case_idx
    torch.manual_seed(case_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(case_seed)

    inputs = _make_inputs(case, device=device, eps=args.eps)
    ref_out = ref_run(**inputs)
    out = impl_run(**inputs)

    allclose = torch.allclose(out, ref_out, atol=args.atol, rtol=args.rtol)
    max_abs_diff = (out - ref_out).abs().max().item()
    print(
        f"[{args.suite}] seed={seed} case={case_idx:02d} impl={impl_name} "
        f"shape=({case['batch_size']},{case['channels']},{case['height']},{case['width']}) "
        f"allclose={allclose} max_abs_diff={max_abs_diff:.8e}"
    )
    return allclose, max_abs_diff


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    impl_name = _resolve_impl_name(args.impl, device)

    if args.num_groups != DEFAULT_NUM_GROUPS:
        raise ValueError(
            f"only --num-groups {DEFAULT_NUM_GROUPS} is supported by the current implementations"
        )
    if args.kernel_size != DEFAULT_KERNEL_SIZE:
        raise ValueError(
            f"only --kernel-size {DEFAULT_KERNEL_SIZE} is supported by this benchmark"
        )

    torch_ref_module = _load_module(TORCH_REF_PATH, "vae_conv2d_torch_ref")
    if impl_name == "triton":
        impl_module = _load_module(TRITON_PATH, "vae_conv2d_triton")
    else:
        impl_module = torch_ref_module

    seeds = _parse_seed_list(args.seeds, args.seed)

    print(f"device={device}")
    print(f"impl={impl_name}")
    print(
        f"suite={args.suite} num_groups={args.num_groups} kernel_size={args.kernel_size} "
        f"eps={args.eps} seeds={seeds} random_tests={args.random_tests}"
    )

    failures = []
    overall_max = 0.0
    total_cases = 0

    for seed in seeds:
        cases = _build_cases(args)
        if args.random_tests > 0:
            cases.extend(_build_random_cases(args, seed))

        for case_idx, case in enumerate(cases):
            allclose, max_abs_diff = _run_case(
                seed=seed,
                case_idx=case_idx,
                case=case,
                args=args,
                device=device,
                ref_run=torch_ref_module.run,
                impl_run=impl_module.run,
                impl_name=impl_name,
            )
            total_cases += 1
            overall_max = max(overall_max, max_abs_diff)
            if not allclose:
                failures.append((seed, case_idx, case, max_abs_diff))

    print(f"total_cases={total_cases}")
    print(f"overall_max_abs_diff={overall_max:.8e}")
    print(f"passed={len(failures) == 0}")

    if failures:
        first_seed, first_idx, first_case, first_diff = failures[0]
        raise AssertionError(
            f"first failing seed={first_seed} case={first_idx} case={first_case} "
            f"max_abs_diff={first_diff}"
        )


if __name__ == "__main__":
    main()
