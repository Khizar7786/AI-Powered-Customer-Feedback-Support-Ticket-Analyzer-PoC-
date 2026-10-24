"""Inspect the installed environment and run tiny device operations, not models."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch


def check_environment(device_choice: str = "auto") -> dict:
    cuda_available = torch.cuda.is_available()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in (
                "feedback-triage-ai", "pip", "setuptools", "torch", "transformers",
                "datasets", "accelerate", "scikit-learn", "streamlit", "pydantic",
                "python-dotenv", "matplotlib", "pytest",
            )
        },
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "requested_device": device_choice,
        "used_device": "cuda" if cuda_available and device_choice == "auto" else "cpu",
        "scope": "Tiny tensor operations only; no model loading, inference, or training.",
    }
    device = torch.device(report["used_device"])
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report["gpu"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_mib": round(total / 1024**2, 2),
            "free_memory_mib_at_check": round(free / 1024**2, 2),
            "compiled_architectures": torch.cuda.get_arch_list(),
        }
    with torch.inference_mode():
        matrix = torch.ones((2, 2), dtype=torch.float32, device=device)
        result = matrix @ matrix
        report["float32_matmul_passed"] = bool(torch.all(result == 2).item())
        report["float16_cuda_matmul_passed"] = None
        if device.type == "cuda":
            half = matrix.to(dtype=torch.float16)
            report["float16_cuda_matmul_passed"] = bool(torch.all(half @ half == 2).item())
            torch.cuda.synchronize()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    report = check_environment(args.device)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["float32_matmul_passed"] or report["float16_cuda_matmul_passed"] is False:
        raise SystemExit("Device operation failed; inspect the report before training.")


if __name__ == "__main__":
    main()
