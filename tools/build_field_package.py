from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.preflight import run_preflight


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "runs",
    "X-AnyLabeling-main",
}
EXCLUDED_SUFFIXES = {".pyc", ".tmp"}


def package_files(include_wheelhouse: bool) -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if relative.parts and relative.parts[0] == "wheelhouse" and not include_wheelhouse:
            continue
        if relative.parts and relative.parts[0] == "field_packages":
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成国赛现场离线备份 ZIP")
    parser.add_argument("--config", default=str(ROOT / "configs" / "field.yaml"))
    parser.add_argument("--include-wheelhouse", action="store_true")
    parser.add_argument("--allow-preflight-fail", action="store_true")
    args = parser.parse_args()

    report = run_preflight(args.config, hardware=False, root=ROOT)
    if report.failures and not args.allow_preflight_fail:
        print(f"自检存在 {report.failures} 个阻止项，未生成比赛包。")
        print("修复后重试；只为备份当前状态时可加 --allow-preflight-fail。")
        return 1

    output_dir = ROOT / "field_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"robotgame_field_{stamp}.zip"
    files = package_files(args.include_wheelhouse)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("robotgame") / path.relative_to(ROOT))
    print(f"FIELD_PACKAGE_OK: {destination}")
    print(f"files={len(files)} size={destination.stat().st_size / 1024 / 1024:.1f} MiB")
    print("wheelhouse=" + ("included" if args.include_wheelhouse else "not included"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
