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
import sys

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


# The merged file's sections, in reading order: the shared vocabulary first, then the
# primitives, then each branch bottom-up, then the layer that joins them, and the model
# class last. `validate_order` checks this is a legal definition order, so a new
# dependency that breaks the layout fails the export instead of the import.
SECTIONS = (
    "src.checkpoints.key_mapping",
    "src.models.sequence_layout",
    "src.models.attention_gates",
    "src.models.ops.rms_norm",
    "src.models.ops.temporal_conv",
    "src.models.ops.fp8_linear",
    "src.models.ops.fused_block",
    "src.models.linear_attention.kernels",
    "src.models.linear_attention.delta_rule",
    "src.models.linear_attention.features",
    "src.models.linear_attention.layers",
    "src.models.linear_attention.scan",
    "src.models.linear_attention.branch",
    "src.models.softmax_attention.kernels",
    "src.models.softmax_attention.window",
    "src.models.softmax_attention.flex_attention",
    "src.models.softmax_attention.decomposed",
    "src.models.softmax_attention.dense_processor",
    "src.models.hybrid_attention",
    "src.models.hybrid_transform",
    "src.inference.utils.lora",
)


def source_of(dotted: str):
    """The file behind a dotted module, module before package. None if neither."""
    base = os.path.join(REPO_ROOT, *dotted.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def src_imports(text: str):
    """(module, node) for every real `from src.x import ...` statement. AST, not a
    regex: the same spelling inside a docstring is prose, and one module's docstring
    does name its own module in a usage example."""
    return [(node.module, node) for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom) and not node.level
            and node.module and node.module.startswith("src.")]


def closure(entry: str):
    """Every `src.` module reachable from `entry`, as {dotted: file}."""
    found, queue = {}, [entry]
    while queue:
        path = queue.pop()
        text = open(path).read()
        if _IMPORT.search(text):
            raise SystemExit(f"{path}: `import src.x` cannot be merged; use "
                             "`from src.x import y`")
        for dotted, _ in src_imports(text):
            if dotted in found:
                continue
            source = source_of(dotted)
            if source is None:
                raise SystemExit(f"{path}: no file behind {dotted!r}")
            found[dotted] = source
            queue.append(source)
    return found


def is_reexport(path: str) -> bool:
    """A package `__init__.py` that only re-exports. It carries nothing into a merged
    file, where every name is already module-global."""
    body = ast.parse(open(path).read()).body
    return all(isinstance(n, (ast.Import, ast.ImportFrom)) or
               (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
               for n in body)


def validate_order(modules, order):
    """Every module must come after the ones it imports. Class bases, decorators and
    default arguments are evaluated when the definition runs, so this is correctness,
    not taste -- the reading order above just has to also be a legal one."""
    missing = set(modules) - set(order)
    if missing:
        raise SystemExit(f"SECTIONS does not place {sorted(missing)}")
    rank = {name: i for i, name in enumerate(order)}
    for dotted, path in modules.items():
        for needed, _ in src_imports(open(path).read()):
            if needed in rank and rank[needed] > rank[dotted]:
                raise SystemExit(f"SECTIONS puts {needed} after {dotted}, which needs "
                                 "it defined first")


def top_level(text: str):
    """Module docstring, `src.` imports at ANY depth, other top-level imports, and
    everything else. Depth matters: one module imports a sibling lazily inside a
    function, and a merged file that keeps that line raises ModuleNotFoundError the
    first time the function runs, on a machine that does not have this repository."""
    tree = ast.parse(text)
    doc, rest = None, list(tree.body)
    if rest and isinstance(rest[0], ast.Expr) and isinstance(rest[0].value, ast.Constant) \
            and isinstance(rest[0].value.value, str):
        doc, rest = rest[0], rest[1:]
    drop = [node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and not node.level and node.module
            and node.module.startswith("src.")]
    imports = [node for node in rest
               if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in drop]
    return doc, drop, imports, rest


def bindings(nodes):
    """Module-level name -> its defining node, for collision detection."""
    out = {}
    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def rename(text: str, old: str, new: str) -> str:
    """Rewrite every reference to a module-level name, by AST position. Positional so
    that the same word in a comment or a docstring is left alone."""
    spots = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name) and node.id == old:
            spots.append((node.lineno, node.col_offset))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == old:
            line = text.splitlines()[node.lineno - 1]
            spots.append((node.lineno, line.index(old, node.col_offset)))
    lines = text.splitlines(keepends=True)
    for lineno, col in sorted(spots, reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col] + new + line[col + len(old):]
    return "".join(lines)


def strip(text: str, nodes) -> str:
    """`text` without those top-level nodes, comments and spacing otherwise intact."""
    cut = set()
    for node in nodes:
        cut.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "".join(line for n, line in enumerate(text.splitlines(keepends=True), 1)
                   if n not in cut)


def banner(dotted: str, doc) -> str:
    """A module's docstring becomes the section header it already was."""
    rule = "# " + "=" * 76
    body = "\n".join(f"# {line}".rstrip()
                      for line in (doc.value.value.strip().splitlines() if doc else []))
    return f"\n\n{rule}\n# {dotted}\n{rule}\n{body}\n\n" if body else \
           f"\n\n{rule}\n# {dotted}\n{rule}\n\n"


def collect(imports, nodes):
    """Gather import statements across modules: `import x` by its binding, `from x
    import ...` merged per module, so the same name imported twice appears once."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports["plain"].add((alias.name, alias.asname))
        else:
            names = imports["from"].setdefault(node.module, set())
            names.update((alias.name, alias.asname) for alias in node.names)


def render_imports(imports) -> str:
    """The standard library first, then everything else, each block sorted."""
    def spell(name, asname):
        return f"{name} as {asname}" if asname else name

    # `import torch.nn as nn` and `from torch import nn` bind the same name to the same
    # module; two modules spelling it differently should not show up as two imports.
    same = {(module, alias or name)
            for module, names in imports["from"].items() for name, alias in names}
    plain = {True: [], False: []}
    forms = {True: [], False: []}
    for name, asname in imports["plain"]:
        if "." in name and (name.rsplit(".", 1)[0], asname or name) in same:
            continue
        plain[name.split(".")[0] in sys.stdlib_module_names].append(
            f"import {spell(name, asname)}")
    for module, names in imports["from"].items():
        spelled = ", ".join(spell(*n) for n in sorted(names))
        forms[module.split(".")[0] in sys.stdlib_module_names].append(
            f"from {module} import {spelled}")

    blocks = ["\n".join(sorted(plain[std]) + sorted(forms[std])) for std in (True, False)]
    return "\n\n".join(blocks) + "\n"


def merge(entry: str, modules) -> str:
    """One file, because the Hub's loader is only reliable with one. Multi-file remote
    code costs an explicit manifest (its local-directory path copies the entrypoint's
    own imports and no deeper), a cycle check (its de-duplication tests a path without
    the `.py` it appends, so it never fires), and the docstring hazard above. None of
    that exists when there are no relative imports left to resolve."""
    validate_order(modules, SECTIONS)
    order = [name for name in SECTIONS if name in modules]

    seen, imports, chunks = {}, {"plain": set(), "from": {}}, []
    for dotted in order:
        text = open(modules[dotted]).read()

        # Two modules may define the same module-level name. Identical definitions are
        # kept once; different ones are a collision the merge has to break, and it is
        # the later module that gives way.
        drop_duplicate = []
        for name, node in bindings(top_level(text)[3]).items():
            if name not in seen:
                seen[name] = ast.dump(node)
            elif ast.dump(node) == seen[name]:
                drop_duplicate.append(name)
            else:
                text = rename(text, name,
                              f"_{dotted.split('.')[-1]}_{name.lstrip('_')}")

        doc, drop_src, module_imports, rest = top_level(text)
        collect(imports, module_imports)

        # Hoisted, so the originals go: an import halfway down a file is not a
        # style anyone writes by hand.
        drop = list(drop_src) + list(module_imports) + ([doc] if doc else [])
        duplicates = bindings(rest)
        drop += [duplicates[name] for name in drop_duplicate]
        chunks.append(banner(dotted, doc) + strip(text, drop).strip("\n") + "\n")

    entry_text = open(entry).read()
    doc, drop_src, module_imports, _ = top_level(entry_text)
    collect(imports, module_imports)

    contents = "\n".join(f"    {i:2d}. {name}" for i, name in enumerate(order, 1))
    header = doc.value.value.strip() if doc else ""
    body = strip(entry_text, list(drop_src) + list(module_imports)
                 + ([doc] if doc else []))
    text = ('"""' + header + "\n\nGENERATED by src/diffusers_export/export.py -- edit "
            "that, or the modules it merges, never this file.\n\nSections, in order:\n"
            + contents + '\n"""\n' + render_imports(imports) + "\n"
            + "".join(chunks) + banner(ENTRY_MODULE, None)
            + body.strip("\n") + "\n")
    # Hoisting the imports out of twenty-one module bodies leaves gaps behind them.
    # Two blank lines is what separates top-level definitions everywhere else here.
    return re.sub(r"\n{4,}", "\n\n\n", text)


def write_code(out_dir: str):
    entry = os.path.join(REPO_ROOT, ENTRY)
    modules = {d: p for d, p in closure(entry).items() if not is_reexport(p)}
    text = merge(entry, modules)

    tree = ast.parse(text)
    left = _LOADER_SEES.findall(text)
    if left:
        raise SystemExit(f"merged file still has relative imports: {sorted(set(left))}")
    unresolved = sorted({node.module for node in ast.walk(tree)
                         if isinstance(node, ast.ImportFrom) and node.module
                         and node.module.startswith("src.")})
    if unresolved:
        raise SystemExit(f"merged file still imports {unresolved} -- nothing on the "
                         "Hub can satisfy that")

    name = f"{ENTRY_MODULE}.py"
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(text)
    return [name], len(modules), text.count("\n")


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

    written, sections, lines = write_code(args.out_dir)
    adapters = write_config(checkpoint, args.out_dir, args.base_source or args.repo,
                            args.base_subfolder)
    print(f"{args.out_dir}: {written[0]}, {sections} modules merged into "
          f"{lines} lines, {adapters} adapters referenced")

    if args.index:
        write_index(os.path.join(os.path.dirname(checkpoint), args.base_index),
                    args.index, args.repo, f"{name}/diffusers")
        print(f"{args.index}: transformer -> {args.repo} :: {name}/diffusers")


if __name__ == "__main__":
    main()
