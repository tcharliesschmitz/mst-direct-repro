"""Parada calibrada somente em amostras Gaussianas; não recebe verdades ou âncoras."""
import sys,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'vendor/pymgt-main/pymgt-main/src'))
from pymgt.ppmt_utils import find_next_best_direction_gd
if __name__=='__main__':
 rng=np.random.default_rng(9102);indices=[];start=time.perf_counter()
 with threadpool_limits(limits=2):
  for k in range(50):
   x=rng.normal(size=(4096,6));_,index=find_next_best_direction_gd(x);indices.append(float(index))
   if (k+1)%10==0:print('Nulo Gaussiano',k+1,'/50',flush=True)
 result=dict(target=float(np.quantile(indices,.95)),seed=9102,replicates=50,n=4096,d=6,quantile=.95,indices=indices,criterio='Quantil 95% do máximo índice Friedman buscado pelo mesmo otimizador em amostras iid Gaussianas do mesmo tamanho',motivo='Revisão técnica anterior a qualquer métrica: target .001 abaixo do nível de flutuação amostral; composição de 100 passos falhou inversão fora do treino.',seconds=time.perf_counter()-start)
 (HERE/'parada_gaussiana.json').write_text(json.dumps(result,indent=2));print(result['target'])
