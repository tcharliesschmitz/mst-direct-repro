"""Table 5 / Fig. 8 (complementary interval-calibration experiment).
MST: 8 calibration truths + 8 evaluation truths (seeds 2026110501-08 / 2026110601-08), 60 fields each
(about 1 min). Comparators (FFT-MA, SGS, PPMT) on the same evaluation truths: requires the PPMT fit
cache ppmt_pymgt_v1/ajuste_pymgt.pkl, created by table2_synthetic_benchmark.py (seed 9101).
Note: the calibration factors are estimated with the known synthetic generator (see paper, Sect. 3)."""
import argparse, sys, json, shutil
import numpy as np, pandas as pd
from _common import ROOT, OUTPUTS, REF, compare
C = ROOT / 'calibracao_cobertura_v1'
sys.path.insert(0, str(C))

def main(mst_only=False):
    OUTPUTS.mkdir(exist_ok=True)
    import experimento as e
    cal = e.generate('calibration'); factor = e.fit(cal)
    ev = e.generate('evaluation')
    per, _ = e.evaluate(ev, factor, 'Evaluation')
    summary = per.groupby('intervalo', sort=False)[['cobertura', 'largura_norm', 'interval_score_norm', 'MAE_norm']].mean().reset_index()
    out = OUTPUTS / 'table5_mst_calibration.csv'; summary.to_csv(out, index=False)
    compare(out, REF / 'table5_mst_calibration_03_resumo_avaliacao.csv', ['cobertura', 'largura_norm', 'interval_score_norm', 'MAE_norm'], 'Table 5 (MST)', key='intervalo')
    if mst_only:
        return
    if not (ROOT / 'ppmt_pymgt_v1' / 'ajuste_pymgt.pkl').exists():
        raise SystemExit('Run table2_synthetic_benchmark.py first: it creates the PPMT fit cache used by the comparators.')
    import comparar_metodos as cm
    cm.display = lambda *x, **k: None
    cm.compare()
    shutil.copy2(e.OUT / '08_comparacao_metodos.csv', OUTPUTS / 'table5_comparison.csv')
    compare(e.OUT / '08_comparacao_metodos.csv', REF / 'table5_08_comparacao_metodos.csv', ['cobertura', 'largura_norm', 'interval_score_norm', 'MAE_norm'], 'Table 5 (all methods)', key='metodo')

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--mst-only', action='store_true'); main(a.parse_args().mst_only)
