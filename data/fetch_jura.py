"""Download the Jura data (Atteia et al., 1994; distributed with the R package gstat) and write
jura_real/dados/jura_train.csv (259 points) and jura_validation.csv (100 points).

The data are NOT redistributed in this package; they are obtained from the gstat distribution.
The files used in the paper have these SHA-256 digests (checked after download):
  jura_train.csv       afd7e84b5762424a145c4ce15648245ae696612e81ae8da891289d5cef9796e8
  jura_validation.csv  8dfe16fd3fd8139b5d35e8e95a751097c1a05fcdf1ed9ea7a66b4eac5abc102d

Usage:
  python data/fetch_jura.py                 # download from the gstat R-universe CSV export
  python data/fetch_jura.py --from-dir DIR  # use prediction.dat.csv / validation.dat.csv exported
                                            # from R: write.csv(gstat::prediction.dat, ..., row.names=FALSE)
"""
import argparse, hashlib, io, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'jura_real' / 'dados'
SOURCES = {'jura_train.csv': 'https://r-spatial.r-universe.dev/gstat/data/prediction.dat/csv',
           'jura_validation.csv': 'https://r-spatial.r-universe.dev/gstat/data/validation.dat/csv'}
LOCAL = {'jura_train.csv': 'prediction.dat.csv', 'jura_validation.csv': 'validation.dat.csv'}
SHA = {'jura_train.csv': 'afd7e84b5762424a145c4ce15648245ae696612e81ae8da891289d5cef9796e8',
       'jura_validation.csv': '8dfe16fd3fd8139b5d35e8e95a751097c1a05fcdf1ed9ea7a66b4eac5abc102d'}
COLS = ['Xloc', 'Yloc', 'Landuse', 'Rock', 'Cd', 'Co', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
N = {'jura_train.csv': 259, 'jura_validation.csv': 100}


def main(from_dir=None):
    DEST.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in SOURCES.items():
        raw = (Path(from_dir) / LOCAL[name]).read_bytes() if from_dir else urllib.request.urlopen(url, timeout=30).read()
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine='python')
        missing = set(COLS) - set(df.columns)
        if missing:
            sys.exit(f'{name}: missing columns {missing}')
        df = df[COLS]
        if len(df) != N[name] or not np.isfinite(df[COLS[4:]].to_numpy(float)).all():
            sys.exit(f'{name}: unexpected content')
        out = DEST / name
        df.to_csv(out, index=False)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        match = digest == SHA[name]
        ok &= match
        print(f'{name}: {len(df)} rows, sha256 {"matches the paper" if match else "DIFFERS: " + digest}')
    if not ok:
        print('Content differs byte-wise from the files used in the paper (e.g. number formatting); '
              'check values before comparing results.')


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--from-dir'); main(a.parse_args().from_dir)
