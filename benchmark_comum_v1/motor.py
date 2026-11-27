"""Benchmark sintético comum: verdade independente; adaptadores auditáveis.
Não contém resultados bibliográficos nem DMS não reproduzido.
"""
import os,sys,time,json,hashlib,pickle
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_benchmark_comum')
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.stats import norm,rankdata
from scipy.ndimage import label
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'mst_direct-2.0.0/src'))
from mst_direct import ScalableMST,spherical
from mst_direct.ppmt import PPMT
HERE=Path(__file__).resolve().parent;OUT=HERE/'resultados';OUT.mkdir(exist_ok=True)
METHODS=['MST','FFTMA','SGS','PPMT']
LABELS={'MST':'Matching for Structural Transport oficial','FFTMA':'Cópula gaussiana + FFT-MA','SGS':'SGS multivariada — implementação local','PPMT':'PPMT compacto local + FFT-MA'}
N=20;D=6;RANGE=5.;REPS=60;REF_SEEDS=[401,402,403];REAL_SEEDS=list(range(81000,81000+REPS));ANCHOR_SEED=7301
V=[f'Z{i+1}' for i in range(D)]

def truth_transform(g):
 x=g[:,0]
 return np.column_stack([x,1.8*(x>0)-.9+.25*g[:,1],.8*x+np.where(g[:,2]>0,1.,-1.)*(1.5+.3*x)+.12*g[:,3],np.sin(1.7*x)+.15*g[:,3],.6*x+(.15+.5*abs(x))*g[:,4],x*x+.2*g[:,5]])
def spatial(n):
 yy,xx=np.indices((n,n));coords=np.column_stack([xx.ravel(),yy.ravel()]).astype(float);C=1-spherical(cdist(coords,coords),rng=RANGE) if n<=40 else None
 m=2**int(np.ceil(np.log2(max(2*n,4*RANGE))))
 while True:
  a=np.minimum(np.arange(m),m-np.arange(m));dist=np.hypot(a[:,None],a[None,:]);cov=1-spherical(dist,rng=RANGE);spec=np.fft.fft2(cov).real
  if spec.min()>=-1e-10:break
  m*=2
  if m>1024:raise ValueError('embedding não positivo')
 return dict(n=n,coords=coords,C=C,spectrum=np.maximum(spec,0),m=m)
def fft_fields(geo,seed):
 rng=np.random.default_rng(seed);white=rng.normal(size=(geo['m'],geo['m'],D));z=np.fft.ifft2(np.fft.fft2(white,axes=(0,1))*np.sqrt(geo['spectrum'])[:,:,None],axes=(0,1)).real
 return z[:geo['n'],:geo['n']].reshape(-1,D)
def make_data():
 geo=spatial(N);rng=np.random.default_rng(6301);cloud=truth_transform(rng.normal(size=(4096,D)));L=np.linalg.cholesky(geo['C']+1e-12*np.eye(N*N));truth=[]
 for seed in REF_SEEDS:truth.append(truth_transform(L@np.random.default_rng(seed).normal(size=(N*N,D))))
 truth=np.array(truth);anchors=np.sort(np.random.default_rng(ANCHOR_SEED).choice(N*N,48,replace=False));free=np.ones(N*N,dtype=bool);free[anchors]=False
 i,j=np.triu_indices(N*N,1);dist=np.linalg.norm(geo['coords'][i]-geo['coords'][j],axis=1);eligible=np.where((dist>0)&(dist<=10))[0];chosen=np.sort(np.random.default_rng(6401).choice(eligible,min(6000,len(eligible)),replace=False));i=i[chosen];j=j[chosen];dist=dist[chosen];edges=np.linspace(0,10,9);binid=np.minimum(np.searchsorted(edges,dist,side='right')-1,7)
 np.savez_compressed(HERE/'dados_congelados.npz',cloud=cloud,truth=truth,anchors=anchors,free=free,coords=geo['coords'],pair_i=i,pair_j=j,pair_bin=binid,lags=(edges[:-1]+edges[1:])/2)
 return dict(cloud=cloud,truth=truth,anchors=anchors,free=free,coords=geo['coords'],pair_i=i,pair_j=j,pair_bin=binid,lags=(edges[:-1]+edges[1:])/2)
def data():
 with np.load(HERE/'dados_congelados.npz') as a:return {k:a[k] for k in a.files}

class Marginals:
 def fit(self,x):
  self.x=[];self.g=[]
  for j in range(x.shape[1]):
   vals,counts=np.unique(x[:,j],return_counts=True);probs=(np.cumsum(counts)-counts/2)/len(x);self.x.append(vals);self.g.append(norm.ppf(probs))
  return self
 def forward(self,x):return np.column_stack([np.interp(x[:,j],self.x[j],self.g[j]) for j in range(x.shape[1])])
 def inverse(self,g):return np.column_stack([np.interp(g[:,j],self.g[j],self.x[j]) for j in range(g.shape[1])])


def robust_interp(query,xp,fp):
 out=np.interp(query,xp,fp);k=min(len(xp)-1,max(8,int(.01*len(xp))))
 slo=(fp[k]-fp[0])/(xp[k]-xp[0]);shi=(fp[-1]-fp[-1-k])/(xp[-1]-xp[-1-k]);lo=query<xp[0];hi=query>xp[-1]
 out[lo]=fp[0]+(query[lo]-xp[0])*slo;out[hi]=fp[-1]+(query[hi]-xp[-1])*shi;return out

def ppmt_forward(x,pp):
 w=(x-pp.mean_)@pp.whiten_
 for direction,xp,gp in pp.steps_:
  z=w@direction;w=w+np.outer(robust_interp(z,xp,gp)-z,direction)
 return w

def ppmt_inverse(g,pp):
 w=g.copy()
 for direction,xp,gp in reversed(pp.steps_):
  z=w@direction;w=w+np.outer(robust_interp(z,gp,xp)-z,direction)
 return pp.mean_+w@pp.dewhiten_

class Method:
 def __init__(self,name,cloud):
  self.name=name;self.cloud=cloud;self.mu=cloud.mean(0);self.sd=cloud.std(0);self.pp=None;start=time.perf_counter();self.geometry_seconds=0.;self.geom={};self.paths={}
  if name in ['FFTMA','SGS']:
   self.marg=Marginals().fit(cloud);g=self.marg.forward(cloud);cross=np.corrcoef(g,rowvar=False);self.crossL=np.linalg.cholesky(cross+1e-10*np.eye(D));self.training_roundtrip=float(abs(self.marg.inverse(g)-cloud).max());self.factor_cross=cross
  elif name=='PPMT':
   self.pp=PPMT(n_iter=30,n_dirs=60,random_state=9101).fit(cloud);g=ppmt_forward(cloud,self.pp);self.fmu=g.mean(0);self.fsd=g.std(0);self.factor_cross=np.corrcoef(g,rowvar=False);self.training_roundtrip=float(abs(ppmt_inverse(g,self.pp)-cloud).max())
   if self.training_roundtrip>1e-6:raise ValueError('PPMT: inversão do treino inconsistente')
  else:self.factor_cross=np.corrcoef((cloud-self.mu)/self.sd,rowvar=False);self.training_roundtrip=0.
  self.fit_seconds=time.perf_counter()-start
 def geometry(self,n,anchors):
  key=(n,tuple(np.asarray(anchors,dtype=int)))
  if key not in self.geom:
   start=time.perf_counter();g=spatial(n);a=np.asarray(anchors,dtype=int);g['anchors']=a
   if len(a) and self.name!='SGS':
    Caa=1-spherical(cdist(g['coords'][a],g['coords'][a]),rng=RANGE);Cga=1-spherical(cdist(g['coords'],g['coords'][a]),rng=RANGE);g['W']=cho_solve(cho_factor(Caa+1e-10*np.eye(len(a)),lower=True),Cga.T).T
   else:g['W']=None
   if self.name=='SGS':
    if n<=40:g['near']=np.argsort(cdist(g['coords'],g['coords']),axis=1,kind='stable')
    else:g['tree']=cKDTree(g['coords']);g['near']=g['tree'].query(g['coords'],k=min(128,n*n))[1]
   self.geom[key]=g;self.geometry_seconds+=time.perf_counter()-start
  return self.geom[key]
 def sgs_plan(self,geo,seed):
  key=(geo['n'],tuple(geo['anchors']),int(seed))
  if key in self.paths:return self.paths[key]
  start=time.perf_counter();nn=geo['n']**2;known=np.zeros(nn,dtype=bool);known[geo['anchors']]=True;path=np.random.default_rng(seed+99000).permutation(np.where(~known)[0]);plan=[]
  for node in path:
   nei=geo['near'][node][known[geo['near'][node]]][:32]
   if geo['C'] is None and len(nei)<min(32,int(known.sum())):
    if known.sum()<2048:
     avail=np.where(known)[0];nei=avail[np.argsort(np.linalg.norm(geo['coords'][avail]-geo['coords'][node],axis=1),kind='stable')[:32]]
    else:
     k=256
     while len(nei)<32:
      candidates=np.atleast_1d(geo['tree'].query(geo['coords'][node],k=min(k,nn))[1]);nei=candidates[known[candidates]][:32];k*=2
   if len(nei):
    Cnn=geo['C'][np.ix_(nei,nei)] if geo['C'] is not None else 1-spherical(cdist(geo['coords'][nei],geo['coords'][nei]),rng=RANGE);c=geo['C'][node,nei] if geo['C'] is not None else 1-spherical(np.linalg.norm(geo['coords'][nei]-geo['coords'][node],axis=1),rng=RANGE);w=cho_solve(cho_factor(Cnn+1e-10*np.eye(len(nei)),lower=True),c);sd=np.sqrt(max(1-w@c,0.))
   else:w=np.empty(0);sd=1.
   plan.append((node,nei,w,sd));known[node]=True
  self.paths[key]=plan;self.geometry_seconds+=time.perf_counter()-start;return plan
 def forward(self,x):
  if self.name=='MST':return (x-self.mu)/self.sd
  if self.name=='PPMT':return (ppmt_forward(x,self.pp)-self.fmu)/self.fsd
  return self.marg.forward(x)
 def inverse(self,g):
  if self.name=='PPMT':return ppmt_inverse(self.fmu+self.fsd*g,self.pp)
  return self.marg.inverse(g)
 def simulate(self,n,anchors,values,seed):
  anchors=np.asarray(anchors,dtype=int);geo=self.geometry(n,anchors);nn=n*n;start=time.perf_counter();gauss_data=self.forward(values) if len(anchors) else np.empty((0,D));raw_anchor_error=0.
  if self.name=='SGS':
   plan=self.sgs_plan(geo,seed);z=np.zeros((nn,D));z[anchors]=gauss_data;innov=np.random.default_rng(seed).normal(size=(nn,D))@self.crossL.T
   for node,nei,w,sd in plan:z[node]=(w@z[nei] if len(nei) else 0.)+sd*innov[node]
  else:
   z=fft_fields(geo,seed)
   if self.name=='FFTMA':z=z@self.crossL.T
   if len(anchors):z=z+geo['W']@(gauss_data-z[anchors]);z[anchors]=gauss_data
  if self.name=='MST':
   rng=np.random.default_rng(seed+50000);donors=rng.integers(len(self.cloud),size=nn-len(anchors));pool=np.vstack([values,self.cloud[donors]]) if len(anchors) else self.cloud[donors];m=ScalableMST(random_state=seed);out=m.simulate(z,pool,pinned=(anchors,np.arange(len(anchors))) if len(anchors) else None,coords=geo['coords']);assert np.array_equal(out,pool[m.assignment_]) and len(np.unique(m.assignment_))==nn
  else:
   out=self.inverse(z)
   if len(anchors):raw_anchor_error=float(abs(out[anchors]-values).max());out[anchors]=values
  assert np.isfinite(out).all()
  if len(anchors):assert np.array_equal(out[anchors],values)
  return out,dict(seconds=time.perf_counter()-start,anchor_error_before_hard_imposition=raw_anchor_error,anchor_error_final=float(abs(out[anchors]-values).max()) if len(anchors) else 0.)

def gamma_fields(fields,dat):
 z=fields/dat['cloud'].std(0);diff=z[:,dat['pair_i']]-z[:,dat['pair_j']];curves=[]
 for lag in range(8):
  dd=diff[:,dat['pair_bin']==lag];curves.append(np.einsum('rpi,rpj->rij',dd,dd)/(2*dd.shape[1]))
 return np.stack(curves,axis=1)
def crps(samples,y):
 m=len(samples);w=(2*np.arange(1,m+1)-m-1)[:,None,None];return np.mean(abs(samples-y),axis=0)-np.sum(w*np.sort(samples,axis=0),axis=0)/m**2

def run_method(name):
 dat=data();out=OUT/name;out.mkdir(exist_ok=True);start=time.perf_counter()
 with threadpool_limits(limits=2):
  method=Method(name,dat['cloud']);un=[];conds=[];diag=[]
  for r,seed in enumerate(REAL_SEEDS):
   field,info=method.simulate(N,[],np.empty((0,D)),seed);un.append(field);diag.append(dict(scenario='incondicional',referencia=-1,realizacao=r,**info))
  for ref in range(3):
   fields=[]
   for r,seed in enumerate(REAL_SEEDS):
    field,info=method.simulate(N,dat['anchors'],dat['truth'][ref,dat['anchors']],seed+1000);fields.append(field);diag.append(dict(scenario='condicional',referencia=ref,realizacao=r,**info))
   conds.append(fields);print(name,'referência',ref+1,'/3 concluída',flush=True)
  un=np.array(un);conds=np.array(conds);np.savez_compressed(out/'campos.npz',unconditional=un,conditional=conds);pd.DataFrame(diag).to_csv(out/'execucao.csv',index=False)
  np.savez_compressed(out/'variogramas.npz',unconditional=gamma_fields(un,dat),conditional=np.array([gamma_fields(s,dat) for s in conds]),truth=gamma_fields(dat['truth'],dat))
  scale=[]
  for n in [16,20,24]:
   t=time.perf_counter()
   for seed in [94001,94002,94003]:method.simulate(n,[],np.empty((0,D)),seed)
   dt=time.perf_counter()-t;scale.append(dict(lado=n,nos=n*n,realizacoes=3,seconds_total_generation_plus_new_geometry=dt,seconds_per_realization=dt/3))
  pd.DataFrame(scale).to_csv(out/'escala.csv',index=False)
  stats=dict(method=name,label=LABELS[name],fit_seconds=method.fit_seconds,geometry_and_sgs_plan_seconds=method.geometry_seconds,total_seconds=time.perf_counter()-start,training_roundtrip_max=method.training_roundtrip,n_iter_ppmt=30 if name=='PPMT' else None,n_dirs_ppmt=60 if name=='PPMT' else None,SGS_neighbors=32 if name=='SGS' else None,PPMT_forward_tails='linear_robust_1pct_mutually_inverse' if name=='PPMT' else None,cross_training=method.factor_cross.tolist(),all_conditional_anchors_exact=True)
  (out/'manifesto_metodo.json').write_text(json.dumps(stats,indent=2))
 return stats


def run_large(names=None):
 dat=data();frozen=HERE/'dados_grandes_congelados.npz'
 if frozen.exists():
  with np.load(frozen) as a:truth100=a['truth100'];anchors=a['anchors100'];truth200=a['truth200']
 else:
  g100=spatial(100);truth100=truth_transform(fft_fields(g100,98001));anchors=np.sort(np.random.default_rng(98002).choice(10000,200,replace=False));truth200=truth_transform(fft_fields(spatial(200),98003));np.savez_compressed(frozen,truth100=truth100,anchors100=anchors,truth200=truth200)
 for name in (METHODS if names is None else names):
  t=time.perf_counter()
  with threadpool_limits(limits=2):
   method=Method(name,dat['cloud']);tu=time.perf_counter();u,du=method.simulate(200,[],np.empty((0,D)),98101);tu=time.perf_counter()-tu;tc=time.perf_counter();c,dc=method.simulate(100,anchors,truth100[anchors],98102);tc=time.perf_counter()-tc
  np.savez_compressed(OUT/name/'campos_grandes.npz',unconditional200=u,conditional100=c);pd.DataFrame([dict(cenario='incondicional200',nos=40000,ancoras=0,seconds_including_new_geometry=tu,**du),dict(cenario='condicional100',nos=10000,ancoras=200,seconds_including_new_geometry=tc,**dc)]).to_csv(OUT/name/'execucao_grandes.csv',index=False)
  print('GRANDES',name,'concluído',round(time.perf_counter()-t,1),'s',flush=True)
