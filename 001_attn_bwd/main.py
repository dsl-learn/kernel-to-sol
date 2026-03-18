import argparse
import importlib.util
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
TORCH_REF_PATH = ROOT_DIR / "00_torch_ref" / "submission.py"
TRITON_PATH = ROOT_DIR / "01_triton" / "submission.py"


DEFAULT_CASES = (
    {"batch_size": 1, "seq_len_q": 8, "seq_len_kv": 16},
    {"batch_size": 2, "seq_len_q": 16, "seq_len_kv": 32},
    {"batch_size": 4, "seq_len_q": 32, "seq_len_kv": 64},
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the attention backward implementation against the torch reference",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--impl",
        default="auto",
        choices=("auto", "torch_ref", "triton"),
        help="Implementation to validate. 'auto' picks triton on CUDA and torch_ref otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len-q", type=int, default=None)
    parser.add_argument("--seq-len-kv", type=int, default=None)
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
        return "triton" if device.type == "cuda" else "torch_ref"
    if impl_arg == "triton" and device.type != "cuda":
        raise RuntimeError("The Triton implementation requires a CUDA device")
    return impl_arg


def _collect_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    if (
        args.batch_size is not None
        or args.seq_len_q is not None
        or args.seq_len_kv is not None
    ):
        return [
            {
                "batch_size": args.batch_size or 2,
                "seq_len_q": args.seq_len_q or 16,
                "seq_len_kv": args.seq_len_kv or 32,
            }
        ]
    return list(DEFAULT_CASES)


def _random_case() -> dict[str, int]:
    return {
        "batch_size": int(torch.randint(1, 5, ()).item()),
        "seq_len_q": int(torch.randint(1, 5, ()).item()) * 8,
        "seq_len_kv": int(torch.randint(1, 9, ()).item()) * 8,
    }


def _run_case(
    case_id: int,
    axes_and_scalars: dict[str, int],
    device: torch.device,
    ref_run,
    impl_run,
    impl_name: str,
    get_inputs,
    atol: float,
    rtol: float,
) -> bool:
    inputs = get_inputs(axes_and_scalars, device=device)

    ref_grad_attn_scores, ref_grad_value_states = ref_run(**inputs)
    out_grad_attn_scores, out_grad_value_states = impl_run(**inputs)

    scores_allclose = torch.allclose(
        out_grad_attn_scores,
        ref_grad_attn_scores,
        atol=atol,
        rtol=rtol,
    )
    values_allclose = torch.allclose(
        out_grad_value_states,
        ref_grad_value_states,
        atol=atol,
        rtol=rtol,
    )

    scores_max_abs_diff = (
        out_grad_attn_scores.float() - ref_grad_attn_scores.float()
    ).abs().max().item()
    values_max_abs_diff = (
        out_grad_value_states.float() - ref_grad_value_states.float()
    ).abs().max().item()

    print(
        f"case={case_id} impl={impl_name} axes={axes_and_scalars} "
        f"scores_allclose={scores_allclose} values_allclose={values_allclose} "
        f"scores_max_abs_diff={scores_max_abs_diff:.8f} "
        f"values_max_abs_diff={values_max_abs_diff:.8f}"
    )
    return scores_allclose and values_allclose


if __name__ == "__main__":
    args = _parse_args()
    device = _resolve_device(args.device)
    impl_name = _resolve_impl_name(args.impl, device)

    torch_ref_module = _load_module(TORCH_REF_PATH, "attn_bwd_torch_ref")
    if impl_name == "triton":
        impl_module = _load_module(TRITON_PATH, "attn_bwd_triton")
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
    for case_id, axes_and_scalars in enumerate(cases):
        if _run_case(
            case_id,
            axes_and_scalars,
            device,
            ref_run=torch_ref_module.run,
            impl_run=impl_module.run,
            impl_name=impl_name,
            get_inputs=torch_ref_module.get_inputs,
            atol=args.atol,
            rtol=args.rtol,
        ):
            num_passed += 1

    print(f"passed={num_passed}/{len(cases)}")
    if num_passed != len(cases):
        raise SystemExit(1)
