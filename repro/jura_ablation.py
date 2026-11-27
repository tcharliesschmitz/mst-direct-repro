"""Jura ablation: cokriging + FFT-MA reference field alone ("Backbone") vs. MST transport over it.
10 seeds (20261029-20261038) x 500 fields; Cd MAE of the ensemble median on the 100 historical targets.
Published values in results_reference/jura_ablation_backbone_vs_mst_10_tabela.csv (0.4452 -> 0.4064 mean).
Use --seeds N to run only the first N seeds and --reps R to reduce the fields (smoke test)."""
import argparse
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from _common import ROOT, OUTPUTS, REF, load

def main(nseeds=10, reps=500):
    OUTPUTS.mkdir(exist_ok=True)
    ex = load('estabilidade_fftma_v11', ROOT / 'jura_cokrigagem_fftma_v1_1' / 'estabilidade_fftma_v11.py')
    ex.REPS = reps; ex.p.REPS = reps
    st = ex.p.state(); y = st['va'].Cd.to_numpy(); rows = []
    with threadpool_limits(limits=2):
        for seed in ex.SEEDS[:nseeds]:
            backbone, mst, _ = ex.run_one(seed)
            for label, s in [('Backbone', backbone), ('MST', mst)]:
                rows.append(dict(semente=seed, etapa=label, **ex.p.metrics(s[:, :, 0], y)))
            print(seed, 'done', flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUTPUTS / 'jura_ablation.csv', index=False)
    ref = pd.read_csv(REF / 'jura_ablation_backbone_vs_mst_10_tabela.csv')
    m = df.merge(ref, on=['semente', 'etapa'], suffixes=('', '_pub'))
    print(m[['semente', 'etapa', 'MAE_mediana', 'MAE_mediana_pub']].to_string(index=False))
    print(df.groupby('etapa').MAE_mediana.mean())

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--seeds', type=int, default=10); a.add_argument('--reps', type=int, default=500)
    x = a.parse_args(); main(x.seeds, x.reps)
