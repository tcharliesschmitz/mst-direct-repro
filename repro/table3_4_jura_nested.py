"""Tables 3/4 and Figs. 6-7 (Jura): nested spatial validation (3 outer x 2 inner X-strips) and the
retrospective historical split (seeds 20261111-20261120, 200 fields each). About 25 min on 2 threads.
Requires jura_real/dados/jura_train.csv and jura_validation.csv (see data/fetch_jura.py)."""
import runpy, shutil
from _common import ROOT, OUTPUTS, REF, compare

def main():
    OUTPUTS.mkdir(exist_ok=True)
    runpy.run_path(str(ROOT / 'validacao_publicacao_v1' / 'validar_jura.py'), run_name='__main__')
    res = ROOT / 'validacao_publicacao_v1' / 'resultados'
    for f in ['resumo_final.csv', 'metricas_outer.csv', 'metricas_jura_externo.csv']:
        shutil.copy2(res / f, OUTPUTS / ('table3_4_' + f))
    compare(res / 'resumo_final.csv', REF / 'table3_4_resumo_final.csv', ['MAE_Cd', 'RMSE_Cd'], 'Tables 3/4', key=['cenario', 'metodo'])

if __name__ == '__main__':
    main()
