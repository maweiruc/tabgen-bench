"""Conservative AST import policy: all imports, including TYPE_CHECKING, count.

Dynamic import calls are rejected, even when literal, to keep this DAG auditable.
This checks maintainability boundaries, not adversarial Python sandboxing.
"""

import ast
from importlib.util import resolve_name
from pathlib import Path

MODULES = frozenset(
    {
        "core",
        "datasets",
        "generators",
        "metrics",
        "evaluation",
        "experiments",
        "reporting",
        "interfaces",
    }
)
ALLOWED = {
    "core": frozenset(),
    "datasets": frozenset({"core"}),
    "generators": frozenset({"core"}),
    "metrics": frozenset({"core"}),
    "evaluation": frozenset({"core"}),
    "experiments": frozenset({"core", "datasets", "generators", "evaluation"}),
    "reporting": frozenset({"core"}),
    "interfaces": MODULES - {"interfaces"},
}


def check_source(source: str, module: str, *, is_package: bool = False):
    parts = module.split(".")
    owner = parts[1] if len(parts) > 1 else None
    package = module if is_package else module.rpartition(".")[0]
    errors, edges = [], set()
    tree = ast.parse(source)
    importlib_names = {"importlib"}
    dynamic_names = {"__import__"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module or ""
            )
            imports.append((base, node.lineno))
            imports.extend((f"{base}.{a.name}", node.lineno) for a in node.names if a.name != "*")
            if base in {"importlib", "builtins"}:
                dynamic_names.update(
                    a.asname or a.name
                    for a in node.names
                    if a.name in {"import_module", "__import__"}
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            dynamic = isinstance(func, ast.Name) and func.id in dynamic_names
            dynamic |= (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in importlib_names
                and func.attr == "import_module"
            )
            if dynamic:
                errors.append(f"{module}:{node.lineno}: dynamic imports forbidden")
    for imported, line in imports:
        root = imported.split(".")[0]
        if root in {"sdv", "torch"} and module != "tabgen_bench.generators.sdv_adapter":
            errors.append(f"{module}:{line}: optional ML import outside sdv_adapter: {imported}")
        if root == "tests":
            errors.append(f"{module}:{line}: runtime cannot import tests")
        if root != "tabgen_bench":
            continue
        target_parts = imported.split(".")
        if len(target_parts) < 2:
            continue
        target = target_parts[1]
        if target not in MODULES:
            if target != "__version__":
                errors.append(f"{module}:{line}: unknown top-level module: {imported}")
            continue
        if target == owner:
            continue
        edges.add((owner, target))
        if owner is None or target not in ALLOWED[owner]:
            errors.append(f"{module}:{line}: forbidden dependency {owner} -> {target}")
    return errors, edges


def check_tree(root: Path):
    errors, edges = [], set()
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(root).with_suffix("")
        is_package = relative.name == "__init__"
        path = relative.parts[:-1] if is_package else relative.parts
        module = ".".join(("tabgen_bench", *path))
        found, dependencies = check_source(file.read_text(), module, is_package=is_package)
        errors.extend(found)
        edges.update(dependencies)
    return errors, edges
