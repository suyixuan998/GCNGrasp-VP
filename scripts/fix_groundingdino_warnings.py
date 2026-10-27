"""Apply reversible compatibility fixes to the installed GroundingDINO 0.1.0.

Run with the target environment's Python. No warning filters are installed.
Original files, a unified diff, and hashes are saved under runs/ by default.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile


REPLACEMENTS = {
    "models/GroundingDINO/fuse_modules.py": [
        ("from timm.models.layers import DropPath", "from timm.layers import DropPath"),
    ],
    "models/GroundingDINO/backbone/swin_transformer.py": [
        ("from timm.models.layers import DropPath, to_2tuple, trunc_normal_",
         "from timm.layers import DropPath, to_2tuple, trunc_normal_"),
        ("torch.meshgrid([coords_h, coords_w])",
         'torch.meshgrid([coords_h, coords_w], indexing="ij")'),
        ("if self.use_checkpoint:",
         "if self.use_checkpoint and self.training and torch.is_grad_enabled():"),
        ("checkpoint.checkpoint(blk, x, attn_mask)",
         "checkpoint.checkpoint(blk, x, attn_mask, use_reentrant=False)"),
    ],
    "models/GroundingDINO/transformer.py": [
        ("torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),\n            )",
         'torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),\n                indexing="ij",\n            )'),
        ("if self.use_checkpoint:",
         "if self.use_checkpoint and self.training and torch.is_grad_enabled():"),
        ("if self.use_transformer_ckpt:",
         "if self.use_transformer_ckpt and self.training and torch.is_grad_enabled():"),
        ("                        text_attention_mask,\n                    )",
         "                        text_attention_mask,\n                        use_reentrant=False,\n                    )"),
        ("                    key_padding_mask,\n                )",
         "                    key_padding_mask,\n                    use_reentrant=False,\n                )"),
        ("torch.cuda.amp.autocast(enabled=False)",
         'torch.amp.autocast("cuda", enabled=False)'),
    ],
    "models/GroundingDINO/utils.py": [
        ("torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),\n        )",
         'torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),\n            indexing="ij",\n        )'),
    ],
    "models/GroundingDINO/bertwarper.py": [
        ("attention_mask, input_shape, device\n        )",
         "attention_mask, input_shape\n        )"),
    ],
    "util/box_ops.py": [
        ("torch.meshgrid(y, x)", 'torch.meshgrid(y, x, indexing="ij")'),
    ],
    "util/inference.py": [
        ('torch.load(model_checkpoint_path, map_location="cpu")',
         'torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)'),
    ],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    # A same-sized edit within one second must not reuse stale bytecode.
    for bytecode in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        bytecode.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs/groundingdino_warning_patch")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate without changing files")
    mode.add_argument("--restore", action="store_true", help="restore this script's saved originals")
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("groundingdino")
    root = Path(distribution.locate_file("groundingdino")).resolve()
    manifest_path = args.backup_dir / "manifest.json"

    if args.restore:
        manifest = json.loads(manifest_path.read_text())
        if Path(manifest["package_root"]) != root:
            raise SystemExit("Backup belongs to a different environment; use that environment's Python.")
        plans = []
        for entry in manifest["files"]:
            target = root / entry["relative_path"]
            backup = args.backup_dir / entry["backup"]
            if digest(backup.read_bytes()) != entry["original_sha256"]:
                raise SystemExit(f"Invalid backup: {backup}")
            current = digest(target.read_bytes())
            if current not in (entry["original_sha256"], entry["patched_sha256"]):
                raise SystemExit(f"File has other edits; refusing to overwrite: {target}")
            plans.append((target, backup.read_bytes()))
        for target, original in plans:
            atomic_write(target, original)
        print(f"Restored {len(plans)} files in {root}")
        return

    if distribution.version != "0.1.0":
        raise SystemExit(f"Expected GroundingDINO 0.1.0; found {distribution.version}.")
    plans = []
    for relative, replacements in REPLACEMENTS.items():
        path = root / relative
        original = path.read_bytes()
        source = original.decode("utf-8")
        for old, new in replacements:
            count = source.count(old)
            if count == 1:
                source = source.replace(old, new, 1)
            elif count == 0 and new in source:
                continue
            else:
                raise SystemExit(f"Unexpected source in {path}: expected one occurrence of {old!r}.")
        ast.parse(source, filename=str(path))
        updated = source.encode("utf-8")
        if original != updated:
            plans.append((relative, path, original, updated))
    if args.check:
        print(f"Compatible: {len(plans)} files need patching in {root}")
        return
    if not plans:
        print(f"Already patched: {root}")
        return

    # Validate every replacement before writing any installed package file.
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    entries, diffs = [], []
    for relative, path, original, updated in plans:
        backup = Path("originals") / relative
        backup_path = args.backup_dir / backup
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists() and backup_path.read_bytes() != original:
            raise SystemExit(f"Different backup already exists: {backup_path}")
        backup_path.write_bytes(original)
        entries.append({"relative_path": relative, "backup": str(backup),
                        "original_sha256": digest(original), "patched_sha256": digest(updated)})
        diffs.extend(difflib.unified_diff(original.decode().splitlines(True),
                                        updated.decode().splitlines(True),
                                        fromfile=f"original/{relative}", tofile=f"patched/{relative}"))
    manifest_path.write_text(json.dumps({"package_root": str(root), "version": distribution.version,
                                       "source": json.loads(distribution.read_text("direct_url.json") or "null"),
                                       "files": entries}, indent=2) + "\n")
    (args.backup_dir / "changes.patch").write_text("".join(diffs))
    for _, path, _, updated in plans:
        atomic_write(path, updated)
    print(f"Patched {len(plans)} files in {root}")
    print(f"Backups and diff: {args.backup_dir}")


if __name__ == "__main__":
    main()
