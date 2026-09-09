"""Publish a checkpoint directory as a diffusers-loadable component.

    python -m src.diffusers_export.export ckpts/stage-dmd-step-250 OUT_DIR \
        [--index OUT_DIR/../modular_model_index.json]

OUT_DIR is what goes to the Hub as `<checkpoint>/diffusers/`: a `config.json`, the
`modeling_vdn_h3.py` entrypoint and the `src.models` closure it imports, flattened to
one directory. It holds NO weights -- `config.json` points back at the base transformer
under `h3-base/` and at the checkpoint's own `linear_branch/` and `adapters/`, so the
published layout has one copy of every tensor and this repository's own loaders keep
reading exactly the files they read today.

`--index` additionally writes the pipeline index. It belongs at the repository ROOT:
`ModularPipeline` has no `subfolder` argument for its own config, and a root
`modular_model_index.json` is also the only path Hugging Face's download counter
matches for a diffusers repo. One index, naming the default checkpoint; a second
checkpoint is selected per component at load time:

    pipe.load_components(trust_remote_code=True,
                         subfolder={"transformer": "stage-b-step-2000/diffusers"})

The flattening is mechanical and total: any `src.` import the rewriter cannot map is a
hard error, so this fails loudly when `src/models/` grows a dependency rather than
publishing a tree that half-imports.
"""
import argparse
import ast
import json
import os
import re
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTRY = "src/diffusers_export/modeling.py"
ENTRY_MODULE = "modeling_vdn_h3"
CLASS_NAME = "VDNMiniMaxH3Transformer3DModel"

_IMPORT = re.compile(r"^\s*import\s+src\.", re.M)
# What diffusers' own loader sees. It is a regex over the raw text, not a parse, so the
# published files must not contain a relative-import SPELLING anywhere it can reach --
# a usage example inside a docstring counts, and a module that appears to import itself
# sends `get_cached_module_file` into unbounded recursion.
_LOADER_SEES = re.compile(r"^\s*from\s+\.(\S+)\s+import", re.M)


def flat_name(dotted: str) -> str:
    """`src.models.linear_attention.branch` -> `vdn_models_linear_attention_branch`.
    Flat because the Hub's dynamic module loader resolves a relative import against
    the entrypoint's own directory: a real subpackage would fetch its siblings from
    the wrong level."""
    return "vdn_" + dotted[len("src."):].replace(".", "_")


def source_of(dotted: str):
    """The file behind a dotted module, module before package. None if neither."""
    base = os.path.join(REPO_ROOT, *dotted.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def closure(entry: str):
    """Every `src.` module reachable from `entry`, as {dotted: file}."""
    found, queue = {}, [entry]
    while queue:
        path = queue.pop()
        text = open(path).read()
        if _IMPORT.search(text):
            raise SystemExit(f"{path}: `import src.x` cannot be rewritten to a relative "
                             "import; use `from src.x import y`")
        for dotted, _ in src_imports(text):
            if dotted in found:
                continue
            source = source_of(dotted)
            if source is None:
                raise SystemExit(f"{path}: no file behind {dotted!r}")
            found[dotted] = source
            queue.append(source)
    return found


def src_imports(text: str):
    """(module, line number) for every real `from src.x import ...` statement. AST, not
    a regex: the same spelling inside a docstring is prose, and rewriting it would put a
    relative import where the Hub's loader can see one that no import graph explains."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.ImportFrom) and not node.level
                and node.module and node.module.startswith("src.")):
            out.append((node.module, node.lineno))
    return out


def rewrite(text: str, names) -> str:
    lines = text.splitlines(keepends=True)
    for dotted, lineno in src_imports(text):
        if dotted not in names:
            raise SystemExit(f"unmapped import {dotted!r}")
        lines[lineno - 1] = re.sub(rf"from\s+{re.escape(dotted)}\s+import",
                                   f"from .{names[dotted]} import", lines[lineno - 1])
    return "".join(lines)


def check_graph(out_dir: str, written):
    """Read the output back the way diffusers reads it, and refuse a graph it cannot
    walk: a missing file, or a cycle (which its recursion does not terminate on --
    the guard it uses to stop tests a path without the `.py` it later appends)."""
    graph = {}
    for name in written:
        text = open(os.path.join(out_dir, name)).read()
        graph[name[:-3]] = sorted(set(_LOADER_SEES.findall(text)))
        for target in graph[name[:-3]]:
            if not os.path.isfile(os.path.join(out_dir, target + ".py")):
                raise SystemExit(f"{name}: relative import of {target!r}, which is not "
                                 "a file here")

    def walk(node, path):
        if node in path:
            raise SystemExit("import cycle the Hub loader would recurse on forever: "
                             + " -> ".join(path[path.index(node):] + [node]))
        for child in graph.get(node, ()):
            walk(child, path + [node])

    for node in graph:
        walk(node, [])


def manifest(modules) -> str:
    """Name every vendored module from the entrypoint.

    diffusers resolves remote code differently for the two sources: from a Hub repo it
    recurses through relative imports and fetches the whole graph, but from a LOCAL
    directory it copies only the imports the entrypoint itself declares. Naming them all
    here is what makes `from_pretrained(<a downloaded directory>)` work too. `__doc__`
    because the import must bind something and that binding must not shadow anything.
    """
    lines = "\n".join(f"from .{module} import __doc__ as _  # noqa: F401"
                       for module in modules)
    return ("\n\n# Every module this component needs, named here so a LOCAL directory load\n"
            "# copies them all; see src/diffusers_export/export.py:manifest.\n"
            f"{lines}\n")


def write_code(out_dir: str):
    entry = os.path.join(REPO_ROOT, ENTRY)
    modules = closure(entry)
    names = {dotted: flat_name(dotted) for dotted in modules}
    collisions = {n for n in names.values() if list(names.values()).count(n) > 1}
    if collisions:
        raise SystemExit(f"flattened name collision: {sorted(collisions)}")

    written = [f"{ENTRY_MODULE}.py"]
    with open(os.path.join(out_dir, written[0]), "w") as f:
        f.write(rewrite(open(entry).read(), names))
        f.write(manifest(sorted(names.values())))
    for dotted, source in sorted(modules.items()):
        written.append(names[dotted] + ".py")
        with open(os.path.join(out_dir, written[-1]), "w") as f:
            f.write(rewrite(open(source).read(), names))

    check_graph(out_dir, written)
    return written


def write_config(checkpoint_dir: str, out_dir: str, base_source: str, base_subfolder: str):
    with open(os.path.join(checkpoint_dir, "model_spec.json")) as f:
        spec = json.load(f)

    adapters_root = os.path.join(checkpoint_dir, "adapters")
    adapters = [f"adapters/{name}/adapter_model.safetensors"
                for name in sorted(os.listdir(adapters_root))] \
        if os.path.isdir(adapters_root) else []

    config = {
        "_class_name": CLASS_NAME,
        "auto_map": {"AutoModel": f"{ENTRY_MODULE}.{CLASS_NAME}"},
        # Read by modeling_vdn_h3: where the pieces are, and how to assemble them.
        # Weight paths are relative to the checkpoint directory holding this one.
        "vdn": {
            # No revision: the spec's one pins MiniMaxAI/MiniMax-H3, and `source` is this
            # repository's byte-identical copy of it, whose own commit pins the weights.
            "base": {"source": base_source, "subfolder": base_subfolder, "revision": None},
            "transform": spec["transforms"][0]["config"],
            "branch": "linear_branch/model.safetensors",
            "adapters": adapters,
            "inference_kernels": True,
            "softmax_backend": "auto",
        },
    }
    path = os.path.join(out_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return len(adapters)


def write_index(base_index: str, out_path: str, repo: str, subfolder: str):
    """The base pipeline index with `transformer` re-pointed at the exported component.
    `type_hint: [null, null]` on purpose: that is what sends the component through
    `AutoModel.from_pretrained`, the only loader that honours `auto_map`."""
    with open(base_index) as f:
        index = json.load(f)
    if "transformer" not in index:
        raise SystemExit(f"{base_index}: no `transformer` component to replace")

    index["transformer"] = [None, None, {
        "type_hint": [None, None],
        "pretrained_model_name_or_path": repo,
        "subfolder": subfolder,
        "variant": None,
        "revision": None,
    }]
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", help="an exported checkpoint directory (model_spec.json)")
    p.add_argument("out_dir", help="written fresh; the Hub's <checkpoint>/diffusers/")
    p.add_argument("--repo", default="OpenVDN/vdn-minimax-h3")
    p.add_argument("--base-subfolder", default="h3-base/transformer",
                   help="where the base transformer lives inside the base source")
    p.add_argument("--base-source", default=None,
                   help="repo id or local directory holding the base transformer "
                        "(default: --repo). A local path is for testing an export "
                        "before the weights are on the Hub.")
    p.add_argument("--index", help="also write the root pipeline index here")
    p.add_argument("--base-index", default="h3-base/modular_model_index.json",
                   help="the index to re-point, relative to the checkpoint's parent")
    args = p.parse_args()

    checkpoint = os.path.abspath(args.checkpoint.rstrip("/"))
    name = os.path.basename(checkpoint)
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    written = write_code(args.out_dir)
    adapters = write_config(checkpoint, args.out_dir, args.base_source or args.repo,
                            args.base_subfolder)
    print(f"{args.out_dir}: {len(written)} modules, {adapters} adapters referenced")

    if args.index:
        write_index(os.path.join(os.path.dirname(checkpoint), args.base_index),
                    args.index, args.repo, f"{name}/diffusers")
        print(f"{args.index}: transformer -> {args.repo} :: {name}/diffusers")


if __name__ == "__main__":
    main()
