import argparse
import importlib.util
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
TORCH_REF_PATH = ROOT_DIR / "00_torch_ref" / "kernel.py"
TRITON_PATH = ROOT_DIR / "01_triton" / "kernel.py"
CUTILE_PATH   = ROOT_DIR / "02_cutile"   / "kernel.py"
CUTEDSL_PATH  = ROOT_DIR / "03_cutedsl"  / "kernel.py"
CUBLASLT_PATH = ROOT_DIR / "04_cudnn"    / "reference.py"   # pure-Python stand-in; C++ ext loaded separately

HIDDEN_SIZE = 2560

DEFAULT_CASES = (
    {"batch_size": 1,  "seq_len": 64},
    {"batch_size": 4,  "seq_len": 128},
    {"batch_size": 16, "seq_len": 512},
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the attention output projection implementation against the torch reference",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--impl",
        default="auto",
        choices=("auto", "torch_ref", "triton", "cutile", "cutedsl", "cublaslt"),
        help="Implementation to validate. 'auto' picks cutedsl on CUDA and torch_ref otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-tests", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_impl_name(impl_arg: str, device: torch.device) -> str:
    if impl_arg == "auto":
        return "cutedsl" if device.type == "cuda" else "torch_ref"
    if impl_arg in ("triton", "cutile", "cutedsl", "cublaslt") and device.type != "cuda":
        raise RuntimeError(f"The {impl_arg} implementation requires a CUDA device")
    return impl_arg


def _collect_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    if args.batch_size is not None or args.seq_len is not None:
        return [{"batch_size": args.batch_size or 4, "seq_len": args.seq_len or 128}]
    return list(DEFAULT_CASES)


def _random_case() -> dict[str, int]:
    return {
        "batch_size": int(torch.randint(1, 9, ()).item()),
        "seq_len": int(torch.randint(1, 9, ()).item()) * 64,
    }


def _get_inputs(axes: dict[str, int], device: torch.device) -> dict:
    B, S, H = axes["batch_size"], axes["seq_len"], HIDDEN_SIZE
    return {
        "attn_output":  torch.randn(B, S, H, dtype=torch.bfloat16, device=device),
        "residual":     torch.randn(B, S, H, dtype=torch.bfloat16, device=device),
        "o_proj_weight": torch.randn(H, H, dtype=torch.bfloat16, device=device),
    }


def _run_case(
    case_id: int,
    axes: dict[str, int],
    device: torch.device,
    ref_run,
    impl_run,
    impl_name: str,
    atol: float,
    rtol: float,
) -> bool:
    inputs = _get_inputs(axes, device)

    ref_out = ref_run(**inputs)
    out = impl_run(**inputs)

    allclose = torch.allclose(out, ref_out, atol=atol, rtol=rtol)
    max_diff = (out.float() - ref_out.float()).abs().max().item()

    print(
        f"case={case_id} impl={impl_name} axes={axes} "
        f"allclose={allclose} max_abs_diff={max_diff:.6f}"
    )
    return allclose


if __name__ == "__main__":
    args = _parse_args()
    device = _resolve_device(args.device)
    impl_name = _resolve_impl_name(args.impl, device)

    torch_ref_module = _load_module(TORCH_REF_PATH, "atten_res_torch_ref")
    if impl_name == "triton":
        impl_module = _load_module(TRITON_PATH, "atten_res_triton")
    elif impl_name == "cutile":
        impl_module = _load_module(CUTILE_PATH, "atten_res_cutile")
    elif impl_name == "cutedsl":
        impl_module = _load_module(CUTEDSL_PATH, "atten_res_cutedsl")
    elif impl_name == "cublaslt":
        impl_module = _load_module(CUBLASLT_PATH, "atten_res_cublaslt")
    else:
        impl_module = torch_ref_module

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cases = _collect_cases(args)
    for _ in range(args.random_tests):
        cases.append(_random_case())

    print(f"device={device}")
    print(f"impl={impl_name}")
    print(f"seed={args.seed}")
    print(f"num_cases={len(cases)}")

    num_passed = 0
    for case_id, axes in enumerate(cases):
        if _run_case(
            case_id, axes, device,
            ref_run=torch_ref_module.run,
            impl_run=impl_module.run,
            impl_name=impl_name,
            atol=args.atol,
            rtol=args.rtol,
        ):
            num_passed += 1

    print(f"passed={num_passed}/{len(cases)}")
    if num_passed != len(cases):
        raise SystemExit(1)
