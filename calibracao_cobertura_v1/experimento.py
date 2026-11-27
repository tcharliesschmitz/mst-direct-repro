"""Calibração empírica de intervalos MST com verdades simuladas independentes."""
from pathlib import Path
import sys,os,json,hashlib,time,io
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_cobertura')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Image
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'benchmark_comum_v1'))
import motor
OUT=HERE/'resultados';OUT.mkdir(exist_ok=True)
DATA=ROOT/'ppmt_pymgt_v1/benchmark/dados_congelados.npz'
with np.load(DATA) as z:dat={k:z[k] for k in z.files}
FREE=dat['free'];ANCHORS=dat['anchors'];SD=dat['cloud'].std(0);V=[f'Z{i+1}' for i in range(6)]
PROTOCOL={'calibration_truth_seeds':list(range(2026110501,2026110509)),'evaluation_truth_seeds':list(range(2026110601,2026110609)),'calibration_field_seed':202612000,'evaluation_field_seed':202613000,'realizations_per_truth':60,'anchors':48,'free_nodes':352,'nominal':.9,'factor':'Por propriedade: máximo entre 1 e quantil empírico 0.90 superior de |y-centro|/semilargura, com calibração em oito verdades independentes.','test_use':'Oito verdades novas; histórico apenas como diagnóstico secundário. Nenhuma seleção posterior ao teste.','scope':'Calibração marginal por simulação no mesmo gerador, grade e geometria; não é garantia conformal espacial nem teste no Jura.'}
def freeze_protocol():
 (OUT/'protocolo_congelado.json').write_text(json.dumps(PROTOCOL,ensure_ascii=False,indent=2))
def show(fig,name):
 buf=io.BytesIO();fig.savefig(OUT/(name+'.png'),dpi=140,bbox_inches='tight');fig.savefig(buf,format='png',dpi=140,bbox_inches='tight');plt.close(fig);display(Image(data=buf.getvalue()))
def table(df,name):df.to_csv(OUT/(name+'.csv'),index=False);display(df)
def generate(group):
 freeze_protocol();path=OUT/(group+'_fields.npz');seeds=PROTOCOL[group+'_truth_seeds'];base=PROTOCOL[group+'_field_seed']
 if path.exists():
  with np.load(path) as z:return {k:z[k] for k in z.files}
 start=time.perf_counter();truth=[];fields=[];geo=motor.spatial(20);L=np.linalg.cholesky(geo['C']+1e-12*np.eye(400))
 with threadpool_limits(limits=2):
  method=motor.Method('MST',dat['cloud'])
  for k,seed in enumerate(seeds):
   y=motor.truth_transform(L@np.random.default_rng(seed).normal(size=(400,6)));ensemble=[]
   for j in range(60):
    f,_=method.simulate(20,ANCHORS,y[ANCHORS],base+1000*k+j);assert np.array_equal(f[ANCHORS],y[ANCHORS]);assert np.isfinite(f).all();ensemble.append(f)
   truth.append(y);fields.append(ensemble);print(group,k+1,'/8 concluído',flush=True)
 data={'truth':np.array(truth),'fields':np.array(fields),'seeds':np.array(seeds)};np.savez_compressed(path,**data);(OUT/(group+'_runtime.json')).write_text(json.dumps({'seconds':time.perf_counter()-start}));return data

def bounds(data):
 low,high=np.quantile(data['fields'][:,: ,FREE,:],[.05,.95],axis=1);return low,high

def fit(data):
 low,high=bounds(data);center=(low+high)/2;half=(high-low)/2;score=abs(data['truth'][:,FREE,:]-center)/np.maximum(half,1e-12)
 factor=np.maximum(1,np.quantile(score.reshape(-1,6),.9,axis=0,method='higher'))
 result={'factors':factor.tolist(),'calibration_seeds':PROTOCOL['calibration_truth_seeds'],'calibration_sha256':hashlib.sha256((OUT/'calibration_fields.npz').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'protocolo_congelado.json').read_bytes()).hexdigest()}
 (OUT/'fatores_congelados.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return factor

def evaluate(data,factor,label):
 low,high=bounds(data);center=(low+high)/2;half=(high-low)/2;y=data['truth'][:,FREE,:];median=np.median(data['fields'][:,:,FREE,:],axis=1);rows=[];pertruth=[]
 for method,mult in [('Original',np.ones(6)),('Calibrado',factor)]:
  lo=center-half*mult;hi=center+half*mult;covered=(y>=lo)&(y<=hi);width=(hi-lo)/SD;score=(hi-lo+20*(lo-y)*(y<lo)+20*(y-hi)*(y>hi))/SD
  for j,var in enumerate(V):rows.append({'conjunto':label,'intervalo':method,'variavel':var,'cobertura':covered[:,:,j].mean(),'largura_norm':width[:,:,j].mean(),'interval_score_norm':score[:,:,j].mean(),'MAE_norm':(abs(y-median)/SD)[:,:,j].mean()})
  for k in range(len(y)):pertruth.append({'conjunto':label,'verdade':k+1,'intervalo':method,'cobertura':covered[k].mean(),'largura_norm':width[k].mean(),'interval_score_norm':score[k].mean()})
 return pd.DataFrame(rows),pd.DataFrame(pertruth)
