"""Orchestrator for the MST reproducibility package.

Usage:
  python run_all.py tests           # unit tests of the mst_direct package (~1 min)
  python run_all.py quick           # tests + MST calibration (Table 5, MST rows) + one Jura seed (~3 min)
  python run_all.py table2          # synthetic benchmark, all four methods (~10-20 min)
  python run_all.py table3_4        # Jura nested validation + historical split (~25 min)
  python run_all.py ablation        # Jura reference field vs. MST, 10 seeds x 500 fields
  python run_all.py table5          # interval calibration + comparators (needs table2 first)
  python run_all.py figures         # Figures 1-8 (needs table2, table3_4, table5)
  python run_all.py all             # everything, in order

All outputs go to ./outputs and to the module 'resultados/' folders inside this package.
The published values in ./results_reference are never overwritten; each step prints the
maximum difference between the regenerated table and the published one.
"""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
R = ROOT / 'repro'
STEPS = {
    'tests': [[sys.executable, '-m', 'pytest', '-q', str(ROOT / 'mst_direct-2.0.0' / 'tests')]],
    'table2': [[sys.executable, str(R / 'table2_synthetic_benchmark.py')]],
    'table3_4': [[sys.executable, str(R / 'table3_4_jura_nested.py')]],
    'ablation': [[sys.executable, str(R / 'jura_ablation.py')]],
    'table5': [[sys.executable, str(R / 'table5_calibration.py')]],
    'figures': [[sys.executable, str(R / 'figures.py')]],
}
STEPS['quick'] = STEPS['tests'] + [[sys.executable, str(R / 'table5_calibration.py'), '--mst-only'],
                                   [sys.executable, str(R / 'jura_one_seed_check.py')]]
STEPS['all'] = sum((STEPS[k] for k in ['tests', 'table2', 'table3_4', 'ablation', 'table5', 'figures']), [])

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(__doc__); sys.exit(1)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    for cmd in STEPS[sys.argv[1]]:
        print('>>', ' '.join(Path(c).name for c in cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT / 'repro')
