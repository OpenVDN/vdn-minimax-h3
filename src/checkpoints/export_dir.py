"""The exploded checkpoint: one artifact as a DIRECTORY, config and weights apart.

    python -m src.checkpoints.export_dir IN.pt OUT_DIR [--dtype bfloat16|keep]

    OUT_DIR/
      model_spec.json                    the ModelSpec: base reference, transforms, adapters
      metadata.json                      kind, format version, the trainer's metadata
      linear_branch/
        config.json                      the hybrid_attention transform config (a copy of
                                         model_spec.transforms[0].config, for reading)
        model.safetensors                every non-LoRA tensor the transform introduced
      adapters/<name>/
        adapter_spec.json                that adapter's spec entry (rank, alpha, targets)
        adapter_model.safetensors        its lora_A / lora_B tensors, peft names kept

The adapter's JSON is `adapter_spec.json` rather than peft's `adapter_config.json`
spelling: the schema is this project's ModelSpec entry, and a peft-named file peft
cannot read fails confusingly. Nothing reads it -- it is the spec, per adapter, to read.

Same content as the single .pt (`kind: weights`), readable by the same `load_checkpoint`;
inference does not know which one it got. Config is JSON you can open, weights are
safetensors you can inspect per component, and a Stage-DMD artifact shows its two adapters
as two directories instead of one 10 GB pickle. Only `weights` artifacts are exported:
a train_state's optimizer moments belong to the trainer, not to a release.
"""
import argparse
import json
import os
import re
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

from src.checkpoints.schema import CheckpointArtifact, UnsupportedCheckpointFormat
from src.models.model_spec import ModelSpec

SPEC_FILE = "model_spec.json"
META_FILE = "metadata.json"
BRANCH_DIR = "linear_branch"
ADAPTERS_DIR = "adapters"
_LORA_KEY = re.compile(r"\.lora_[AB]\.([^.]+)\.")


def adapter_name_of(key: str):
    """`...to_q.lora_A.<name>.weight` -> <name>; None for a non-LoRA tensor."""
    m = _LORA_KEY.search(key)
    return m.group(1) if m else None


def split_weights(weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
                                                             Dict[str, Dict[str, torch.Tensor]]]:
    branch, adapters = {}, {}
    for key, tensor in weights.items():
        name = adapter_name_of(key)
        if name is None:
            branch[key] = tensor
        else:
            adapters.setdefault(name, {})[key] = tensor
    return branch, adapters


def _adapter_spec(spec: Dict[str, Any], name: str, index: int) -> Dict[str, Any]:
    """Match an adapter's tensors to its spec entry: by `config.name` when the spec names
    it, else by position (the first, unnamed entry is peft's `default`)."""
    entries = spec.get("adapters", [])
    for entry in entries:
        if entry["config"].get("name") == name:
            return entry
    if name == "default" and entries and "name" not in entries[0]["config"]:
        return entries[0]
    if index < len(entries):
        return entries[index]
    raise UnsupportedCheckpointFormat(f"adapter {name!r} has tensors but no spec entry")


def export_checkpoint_dir(artifact: CheckpointArtifact, out_dir: str,
                          dtype: str = "bfloat16") -> str:
    if artifact.kind != "weights":
        raise UnsupportedCheckpointFormat(
            f"only `weights` artifacts are exported as directories, got {artifact.kind!r}")
    ModelSpec.from_dict(artifact.model_spec)
    cast = None if dtype == "keep" else getattr(torch, dtype)

    def prepared(tensors):
        return {k: (t.to(cast) if cast is not None and t.is_floating_point() else t)
                .contiguous() for k, t in tensors.items()}

    branch, adapters = split_weights(artifact.weights)
    tmp = out_dir.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        raise FileExistsError(f"{tmp} exists; a previous export did not finish")
    os.makedirs(os.path.join(tmp, BRANCH_DIR))

    with open(os.path.join(tmp, SPEC_FILE), "w") as f:
        json.dump(artifact.model_spec, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(tmp, META_FILE), "w") as f:
        json.dump({"kind": artifact.kind, "checkpoint_format_version": 2,
                   "weights_dtype": dtype, "metadata": artifact.metadata},
                  f, indent=2, sort_keys=True)
        f.write("\n")

    transforms = artifact.model_spec.get("transforms", [])
    with open(os.path.join(tmp, BRANCH_DIR, "config.json"), "w") as f:
        json.dump(transforms[0] if transforms else {}, f, indent=2, sort_keys=True)
        f.write("\n")
    save_file(prepared(branch), os.path.join(tmp, BRANCH_DIR, "model.safetensors"))

    for index, (name, tensors) in enumerate(adapters.items()):
        adir = os.path.join(tmp, ADAPTERS_DIR, name)
        os.makedirs(adir)
        with open(os.path.join(adir, "adapter_spec.json"), "w") as f:
            json.dump(_adapter_spec(artifact.model_spec, name, index), f, indent=2,
                      sort_keys=True)
            f.write("\n")
        save_file(prepared(tensors), os.path.join(adir, "adapter_model.safetensors"))

    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists; remove it first")
    os.replace(tmp, out_dir)
    return out_dir


def is_checkpoint_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, SPEC_FILE))


def load_checkpoint_dir(path: str) -> CheckpointArtifact:
    with open(os.path.join(path, SPEC_FILE)) as f:
        model_spec = json.load(f)
    with open(os.path.join(path, META_FILE)) as f:
        meta = json.load(f)
    if meta.get("checkpoint_format_version") != 2 or meta.get("kind") != "weights":
        raise UnsupportedCheckpointFormat(f"{path}: not an exploded `weights` artifact")
    ModelSpec.from_dict(model_spec)

    weights = load_file(os.path.join(path, BRANCH_DIR, "model.safetensors"))
    adapters_root = os.path.join(path, ADAPTERS_DIR)
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            weights.update(load_file(os.path.join(adapters_root, name,
                                                  "adapter_model.safetensors")))
    return CheckpointArtifact(kind="weights", model_spec=model_spec, weights=weights,
                              metadata=meta.get("metadata", {}))


def main():
    from src.checkpoints.loader import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", help="a v2 .pt `weights` artifact")
    p.add_argument("out_dir")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "keep"),
                   help="floating tensors are cast on the way out (inference runs the "
                        "branch in bf16 anyway); `keep` stores them as saved")
    args = p.parse_args()

    art = load_checkpoint(args.checkpoint)
    out = export_checkpoint_dir(art, args.out_dir, dtype=args.dtype)
    branch, adapters = split_weights(art.weights)
    print(f"wrote {out}: {len(branch)} branch tensors, "
          + ", ".join(f"adapter {n}: {len(t)} tensors" for n, t in adapters.items()))


if __name__ == "__main__":
    main()
