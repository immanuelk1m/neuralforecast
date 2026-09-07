"""Load explicit, trusted research checkouts without global `layers` collisions.

This is import isolation, not a security sandbox. Installing a checkout is an
explicit decision to execute its Python code. The preparation script pins the
reviewed upstream revisions; no downloads happen at package import or prediction.
"""

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import threading


SOURCE_REVISIONS = {
    "dag": ("decisionintelligence/DAG", "0758990e2c73bb54138ea3e7b11a35cbc5476bcc"),
    "kite": ("decisionintelligence/KITE", "3140ee824cbd80c5ec7fdf2b54666210519d0b39"),
    "apt": ("blisky-li/APT", "98a4c9c017666207b02029f842eff92818f0eab8"),
    "glaff": ("ForestsKing/GLAFF", "4dedf10e0028b519780645ef5824810f4b1bdb55"),
    "tgtsf": ("VEWOXIC/TGTSF", "fdf10ceea422c0bf13b0013c9d6ca48179bf9de7"),
    "spectf": ("hiepnh137/SpecTF", "85185c7b883fed7de40098d76ce6782ba1eba016"),
    "chronosx": ("amazon-science/chronos-forecasting", "2b52bfc500e3ebab1ce846f0a2a60ee5ef2a14a7"),
    "uni2ts": ("SalesforceAIResearch/uni2ts", "cfd46d4510ed8896f263116f32928eede05b0a75"),
}
_LOCK = threading.RLock()


class _Imports(ast.NodeTransformer):
    def __init__(self, prefix, roots):
        self.prefix, self.roots = prefix, roots

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module.split(".")[0] in self.roots:
            node.module = self.prefix + "." + node.module
        return node

    def visit_Import(self, node):
        result = []
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.roots:
                result.append(ast.Import(names=[alias]))
            elif alias.asname or "." not in alias.name:
                result.append(ast.Import(names=[ast.alias(
                    name=self.prefix + "." + alias.name,
                    asname=alias.asname or root,
                )]))
            else:
                result.append(ast.Import(names=[ast.alias(
                    name=self.prefix + "." + alias.name,
                )]))
                result.append(ast.ImportFrom(
                    module=self.prefix, names=[ast.alias(name=root)], level=0,
                ))
        return [ast.copy_location(item, node) for item in result]


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path, prefix, roots):
        self.path, self.prefix, self.roots = path, prefix, roots

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_text(encoding="utf-8"), filename=str(self.path))
        tree = ast.fix_missing_locations(_Imports(self.prefix, self.roots).visit(tree))
        module.__file__ = str(self.path)
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, prefix):
        self.root, self.prefix = root, prefix
        self.roots = {p.stem for p in root.iterdir() if p.is_dir() or p.suffix == ".py"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.prefix and not fullname.startswith(self.prefix + "."):
            return None
        suffix = fullname[len(self.prefix):].lstrip(".").split(".")
        source = self.root.joinpath(*suffix).resolve()
        if not source.is_relative_to(self.root):
            raise ImportError("Official source import escaped its checkout.")
        if source.is_dir():
            # Import the requested architecture, not benchmark-wide __init__ files
            # which eagerly import unrelated models, dataset engines and services.
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(source)]
            return spec
        source = source.with_suffix(".py").resolve()
        if not source.is_relative_to(self.root):
            raise ImportError("Official source file escaped its checkout.")
        if source.is_file():
            return importlib.util.spec_from_loader(
                fullname, _SourceLoader(source, self.prefix, self.roots),
                origin=str(source),
            )
        return None


def source_module(source_dir, module_name):
    """Import a reviewed module from a caller-supplied checkout or its src folder."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Official source directory does not exist: {root}")
    if not module_name or not all(part.isidentifier() for part in module_name.split(".")):
        raise ValueError("module_name must be a dotted Python module name.")
    prefix = "_nf_official_" + hashlib.sha256(str(root).encode()).hexdigest()[:20]
    with _LOCK:
        if not any(isinstance(f, _SourceFinder) and f.prefix == prefix for f in sys.meta_path):
            sys.meta_path.insert(0, _SourceFinder(root, prefix))
        return importlib.import_module(prefix + "." + module_name)
