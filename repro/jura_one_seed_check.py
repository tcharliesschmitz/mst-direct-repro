"""Quick check (~1 min): regenerate one historical Jura seed (default 20261111, 200 fields) for the
default MST (lambda=0, 48 candidates) and the selected configuration (lambda=1, 48 candidates) and
compare the Cd MAE of the median with the published value."""
import sys
import pandas as pd
from threadpoolctl import threadpool_limits
from _common import ROOT, OUTPUTS, REF, load

def main(seed=20261111, reps=200):
    v = load('validar_jura', ROOT / 'validacao_publicacao_v1' / 'validar_jura.py')
    tr = pd.read_csv(ROOT / 'jura_real/dados/jura_train.csv')[v.p.XY + v.p.V]
    va = pd.read_csv(ROOT / 'jura_real/dados/jura_validation.csv')[v.p.XY + v.p.V]
    st = v.prepare(tr, va)
    cfg = [c for c in v.CONFIGS if c['id'] in ('lambda0_nn48', 'lambda1_nn48')]
    with threadpool_limits(limits=2):
        fs, _ = v.simulate(st, cfg, reps, [seed], f'check_{seed}')
    ref = pd.read_csv(REF / 'table3_4_metricas_jura_externo.csv')
    for c in cfg:
        mae = v.p.metrics(fs[c['id']][:, :, 0], va.Cd.to_numpy())['MAE_mediana']
        pub = ref[(ref.seed == seed) & (ref.config == c['id'])].MAE_mediana
        print(f"seed {seed} {c['id']}: MAE regenerated {mae:.6f} | published {float(pub.iloc[0]) if len(pub) else float('nan'):.6f}")

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20261111)
