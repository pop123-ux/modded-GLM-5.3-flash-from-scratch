"""Restricted verifier for the included tiny tasks, with per-candidate timeouts.

This is a local experiment harness, not an OS sandbox for hostile code.
"""
from __future__ import annotations
import ast
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .tasks import CodingTask

DENIED = (ast.AsyncFunctionDef, ast.ClassDef, ast.ImportFrom, ast.With, ast.AsyncWith,
          ast.Lambda, ast.Global, ast.Nonlocal, ast.While, ast.For, ast.AsyncFor,
          ast.Try, ast.Raise, ast.Delete, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
SAFE_BUILTINS = {name: getattr(__import__('builtins'), name) for name in ('sum','len','abs','min','max','bool','int','str')}

@dataclass(frozen=True)
class Evaluation:
    status: str
    passed: bool
    tests_passed: int
    tests_total: int
    pass_fraction: float
    message: str = ''
    def to_dict(self):
        return asdict(self)

def validate_source(source: str, entry_point: str) -> ast.Module:
    if len(source.encode('utf-8')) > 2048:
        raise ValueError('source too large')
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != entry_point:
        raise ValueError('required function missing or extra function present')
    for node in tree.body:
        if node is not functions[0] and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise ValueError('only the requested function is allowed at top level')
    if functions[0].decorator_list or functions[0].returns or functions[0].args.defaults or functions[0].args.kw_defaults:
        raise ValueError('decorators, annotations and defaults are disallowed')
    for node in ast.walk(tree):
        if isinstance(node, DENIED):
            raise ValueError(f'disallowed syntax: {type(node).__name__}')
        if isinstance(node, ast.arg) and node.annotation:
            raise ValueError('annotations disallowed')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('dunder access disallowed')
        if isinstance(node, ast.Import):
            if len(node.names) != 1 or node.names[0].name != 'math' or node.names[0].asname:
                raise ValueError('only import math is permitted')
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == 'math' and node.attr in ('exp','factorial')):
                raise ValueError('attribute access disallowed')
        if isinstance(node, ast.Call):
            allowed = isinstance(node.func, ast.Name) and node.func.id in SAFE_BUILTINS
            allowed_math = isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' and node.func.attr in ('exp','factorial')
            if not (allowed or allowed_math):
                raise ValueError('unsafe call')
    return tree

def _safe_import(name, *args, **kwargs):
    if name != 'math':
        raise ImportError(name)
    return math

def _run(payload):
    tree = validate_source(payload['source'], payload['entry_point'])
    ns = {'__builtins__': dict(SAFE_BUILTINS, __import__=_safe_import), 'math': math}
    exec(compile(tree, 'candidate.py', 'exec'), ns)
    passed = 0
    for arguments, expected in payload['cases']:
        try:
            actual = ns[payload['entry_point']](*arguments)
            equal = (math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
                     if type(expected) is float and type(actual) is float else type(actual) is type(expected) and actual == expected)
            passed += int(equal)
        except Exception:
            pass
    total = len(payload['cases'])
    return Evaluation('passed' if passed == total else 'failed', passed == total, passed, total, passed/total).to_dict()

def evaluate_source(task: CodingTask, source: str, *, timeout: float = 2.0) -> Evaluation:
    total = len(task.cases)
    try:
        validate_source(source, task.entry_point)
    except (SyntaxError, ValueError, TypeError) as error:
        return Evaluation('invalid', False, 0, total, 0.0, str(error))
    try:
        completed = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker'],
            input=json.dumps({'source': source, 'entry_point': task.entry_point, 'cases': task.cases}),
            text=True, capture_output=True, timeout=timeout, check=False)
        if completed.returncode:
            return Evaluation('failed', False, 0, total, 0.0, 'worker exited without a result')
        return Evaluation(**json.loads(completed.stdout))
    except subprocess.TimeoutExpired:
        return Evaluation('timeout', False, 0, total, 0.0, 'candidate time limit exceeded')
    except (ValueError, TypeError):
        return Evaluation('failed', False, 0, total, 0.0, 'invalid worker result')

def completion_is_parseable(task: CodingTask, completion: str) -> bool:
    try:
        validate_source(task.prompt + completion, task.entry_point)
        return completion.endswith('\n')
    except (SyntaxError, ValueError):
        return False

if __name__ == '__main__':
    # POSIX resource limits complement the wall-clock timeout; Windows uses timeout only.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    except (ImportError, ValueError, OSError):
        pass
    try:
        print(json.dumps(_run(json.load(sys.stdin))))
    except Exception:
        sys.exit(1)
