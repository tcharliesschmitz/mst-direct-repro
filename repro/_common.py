"""Shared helpers for the reproduction scripts (paths are relative to the package root)."""
import os, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / '.mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
REF = ROOT / 'results_reference'
sys.path.insert(0, str(ROOT / 'mst_direct-2.0.0' / 'src'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compare(new, ref, cols, label, key=None, tol=1e-9):
    """Print max |difference| between a regenerated table and the published reference."""
    import pandas as pd
    a, b = pd.read_csv(new), pd.read_csv(ref)
    if key:
        a, b = a.set_index(key), b.set_index(key)
        b = b.loc[a.index]
    diff = max(float((a[c] - b[c]).abs().max()) for c in cols)
    status = 'MATCH' if diff <= tol else 'DIFFERS'
    print(f'[{label}] max |regenerated - reference| = {diff:.3e} -> {status}')
    return diff
