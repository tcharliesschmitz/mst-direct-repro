"""Table 2 (synthetic benchmark): MST, Gaussian copula + FFT-MA, local SGS and PPMT (pymgt + tail adapter).

Writes fields to ppmt_pymgt_v1/benchmark/resultados/<METHOD>/ and the summary to
ppmt_pymgt_v1/resumo_comparativo_pymgt.csv and outputs/table2_summary.csv.
Seeds: cloud 6301, truths 401/402/403, anchors 7301, realizations 81000-81059 (+1000 conditional),
large grids 98001-98003/98101/98102, PPMT fit 9101 (stopping threshold from seed 9102).
"""
import shutil
from _common import ROOT, OUTPUTS, REF, load, compare
import sys
from threadpoolctl import threadpool_limits
import pandas as pd

B = ROOT / 'ppmt_pymgt_v1' / 'benchmark'
sys.path.insert(0, str(B)); sys.path.insert(0, str(ROOT / 'ppmt_pymgt_v1'))
m = load('motor', B / 'motor.py')
cp = load('continuar_ppmt', ROOT / 'ppmt_pymgt_v1' / 'continuar_ppmt.py')


def main():
    OUTPUTS.mkdir(exist_ok=True)
    package = cp.find_package()
    with threadpool_limits(limits=2):
        for name in ['MST', 'FFTMA', 'SGS']:
            m.run_method(name)
        m.run_large(['MST', 'FFTMA', 'SGS'])
        cp.install_adapter(m, package)   # pymgt with the audited tail adapter (seed 9101)
        cp.gates(m)                      # inversion / repeatability gates
        cp.core(m)                       # PPMT fields (same seeds and anchors)
    a = load('avaliacao', B / 'avaliacao.py')
    a.display = lambda *x, **k: None
    a.prepare_tables(); a._TABLES = None
    rows = []
    for name in m.METHODS:
        for k in [3, 4, 8, 11, 12]:
            a.render(name, k)
        folder = m.OUT / name
        t = {k: pd.read_csv(folder / f'{k:02d}_tabela.csv') for k in [3, 4, 8, 11, 12]}
        cr = t[12][t[12].nominal == .9]
        rows.append(dict(metodo=name, W1_normalizado=t[3].W1_normalizado_media.mean(), JS_bits=t[4].JS_bits_media.mean(),
                         RMSE_normalizado=t[11].RMSE_normalizado.mean(), CRPS_normalizado=cr.CRPS_normalizado.mean(),
                         cobertura90=cr.cobertura.mean(), erro_ancoras=t[8].erro_final_max.max()))
    out = ROOT / 'ppmt_pymgt_v1' / 'resumo_comparativo_pymgt.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    shutil.copy2(out, OUTPUTS / 'table2_summary.csv')
    compare(out, REF / 'table2_resumo_comparativo_pymgt.csv',
            ['W1_normalizado', 'JS_bits', 'RMSE_normalizado', 'CRPS_normalizado', 'cobertura90'], 'Table 2', key='metodo')


if __name__ == '__main__':
    main()
