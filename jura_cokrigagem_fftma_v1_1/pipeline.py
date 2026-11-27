"""Cokrigagem simples LMC + FFT-MA multivariado + condicionamento + MST oficial.
Caminhos relativos ao modulo; nenhum caminho absoluto do Desktop.
"""
import os,sys,json,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_cokrigagem_fftma')
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from IPython.display import display,Markdown
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'mst_direct-2.0.0/src'))
from mst_direct import ScalableMST,spherical
OUT=Path(__file__).parent/'resultados';OUT.mkdir(exist_ok=True)
V=['Cd','Ni','Zn'];XY=['Xloc','Yloc'];REPS=500;SEED=20261029;DX=.1
PAIRS=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
PARAMS=dict(beta=6.,n_candidates=48,n_iter=150,lam_relational=0.,n_relax=0)
def load():
 p=ROOT/'jura_real/dados';return pd.read_csv(p/'jura_train.csv')[XY+V],pd.read_csv(p/'jura_validation.csv')[XY+V]
def project_psd(A,floor=1e-7):
 w,U=np.linalg.eigh((A+A.T)/2);return (U*np.maximum(w,floor))@U.T

def fit_lmc(tr):
 z=np.log(tr[V].to_numpy());mu=z.mean(0);sd=z.std(0);zs=(z-mu)/sd;coords=tr[XY].to_numpy();i,j=np.triu_indices(len(tr),1);d=np.linalg.norm(coords[i]-coords[j],axis=1);edges=np.linspace(0,3,16);lags=(edges[:-1]+edges[1:])/2;diff=zs[i]-zs[j];gam=[];counts=[]
 for q in range(15):
  mask=(d>=edges[q])&(d<edges[q+1]);gam.append(np.einsum('ni,nj->ij',diff[mask],diff[mask])/(2*mask.sum()));counts.append(mask.sum())
 gam=np.array(gam);weights=np.array(counts)/np.sum(counts);best=None;candidates=[]
 for r1 in [.15,.3,.6]:
  for r2 in [1.,1.8,3.]:
   X=np.column_stack([np.ones_like(lags),spherical(lags,rng=r1),spherical(lags,rng=r2)]);H=X.T@(weights[:,None]*X);rhs=np.einsum('hq,h,hij->qij',X,weights,gam);coef=np.linalg.solve(H+1e-10*np.eye(3),rhs.reshape(3,-1)).reshape(3,3,3);coef=np.array([project_psd(a,.03 if q==0 else 1e-7) for q,a in enumerate(coef)]);lr=.45/max(np.linalg.eigvalsh(H))
   for _ in range(150):
    grad=np.einsum('qr,rij->qij',H,coef)-rhs;coef=np.array([project_psd(a,.03 if q==0 else 1e-7) for q,a in enumerate(coef-lr*grad)])
   # Normalize the total diagonal variance to one in standardized log space.
   invsd=1/np.sqrt(np.diag(coef.sum(0)));coef=coef*invsd[None,:,None]*invsd[None,None,:];pred=np.einsum('hq,qij->hij',X,coef);err=float(np.sum(weights[:,None,None]*(pred-gam)**2));row=dict(alcance_curto_km=r1,alcance_longo_km=r2,erro_ajuste=err);candidates.append(row)
   if best is None or err<best[0]:best=(err,r1,r2,coef)
 err,r1,r2,A=best;return dict(mu=mu,sd=sd,zs=zs,lags=lags,gamma=gam,counts=np.array(counts),ranges=np.array([r1,r2]),A=A[1:],B0=A[0],candidates=pd.DataFrame(candidates).sort_values('erro_ajuste'))

def embedding(coords,ranges):
 origin=np.floor(coords.min(0)/DX)*DX-2*DX;end=np.ceil(coords.max(0)/DX)*DX+2*DX;nx=int(round((end[0]-origin[0])/DX))+1;ny=int(round((end[1]-origin[1])/DX))+1
 size=2**int(np.ceil(np.log2(max(2*nx,2*ny,4*max(ranges)/DX))))
 while True:
  a=np.minimum(np.arange(size),size-np.arange(size))*DX;dist=np.hypot(a[:,None],a[None,:]);covs=np.array([1-spherical(dist,rng=r) for r in ranges]);spec=np.fft.fft2(covs,axes=(-2,-1)).real
  if spec.min()>=-1e-10:break
  size*=2
  if size>2048:raise ValueError('Embedding FFT sem positividade espectral; revisar modelo/grade.')
 # Roundoff clipping only; all covariances below use the actual resulting spectrum.
 spec=np.maximum(spec,0);covs=np.fft.ifft2(spec,axes=(-2,-1)).real
 pos=(coords-origin)/DX;ij=np.floor(pos).astype(int);t=pos-ij
 gx=np.stack([ij[:,0],ij[:,0]+1,ij[:,0],ij[:,0]+1],axis=1);gy=np.stack([ij[:,1],ij[:,1],ij[:,1]+1,ij[:,1]+1],axis=1)
 w=np.column_stack([(1-t[:,0])*(1-t[:,1]),t[:,0]*(1-t[:,1]),(1-t[:,0])*t[:,1],t[:,0]*t[:,1]])
 return dict(origin=origin,nx=nx,ny=ny,size=size,spec=spec,covs=covs,gx=gx,gy=gy,w=w)
def spatial_at_sites(e):
 n=len(e['w']);m=e['size'];result=np.zeros((2,n,n))
 for a in range(4):
  for b in range(4):
   dx=(e['gx'][:,a,None]-e['gx'][None,:,b])%m;dy=(e['gy'][:,a,None]-e['gy'][None,:,b])%m;result+=e['covs'][:,dy,dx]*e['w'][None,:,a,None]*e['w'][None,None,:,b]
 return result

def setup():
 tr,va=load();f=fit_lmc(tr);coords=np.vstack([tr[XY].to_numpy(),va[XY].to_numpy()]);e=embedding(coords,f['ranges']);C=spatial_at_sites(e);K=sum(np.kron(c,a) for c,a in zip(C,f['A']))+np.kron(np.eye(len(coords)),f['B0']);nt=len(tr);n=len(coords)
 observed=np.r_[np.arange(3*nt),(3*np.arange(nt,n)[:,None]+np.array([1,2])[None,:]).ravel()];free=3*np.arange(nt,n)
 # Only target Ni/Zn enter this conditioning system; target Cd is never read here.
 ztrain=f['zs'].ravel();zsecondary=((np.log(va[['Ni','Zn']].to_numpy())-f['mu'][1:])/f['sd'][1:]).ravel();data=np.r_[ztrain,zsecondary]
 Kdd=K[np.ix_(observed,observed)];Kfd=K[np.ix_(free,observed)];jitter=1e-9
 with threadpool_limits(limits=2):
  chol=cho_factor(Kdd+jitter*np.eye(len(observed)),lower=True);W=cho_solve(chol,Kfd.T).T
 mean=W@data;cc=K[np.ix_(free,free)]-W@Kfd.T;cc=(cc+cc.T)/2;ev=np.linalg.eigvalsh(cc)
 if ev.min()<-1e-6:raise ValueError('Covariância condicional indefinida')
 cdtrain=3*np.arange(nt);Kcd=K[np.ix_(cdtrain,cdtrain)];wcd=cho_solve(cho_factor(Kcd+jitter*np.eye(nt),lower=True),K[np.ix_(cdtrain,free)]).T;mean_kriging=wcd@f['zs'][:,0]
 return dict(tr=tr,va=va,f=f,e=e,K=K,coords=coords,observed=observed,free=free,data=data,W=W,mean=mean,cc=cc,mean_kriging=mean_kriging,jitter=jitter)
_STATE=None
def state():
 global _STATE
 if _STATE is None:_STATE=setup()
 return _STATE

def simulate_fft(st):
 e=st['e'];A=st['f']['A'];rng=np.random.default_rng(SEED);factors=[np.linalg.cholesky(a) for a in A];raw=[];first=None;t=time.perf_counter()
 for rep in range(REPS):
  grid=np.zeros((e['size'],e['size'],3))
  for q in range(2):
   white=rng.normal(size=(e['size'],e['size'],3));filtered=np.fft.ifft2(np.fft.fft2(white,axes=(0,1))*np.sqrt(e['spec'][q])[:,:,None],axes=(0,1)).real;grid+=filtered@factors[q].T
  if first is None:first=grid[:e['ny'],:e['nx']].copy()
  sites=np.sum(grid[e['gy'],e['gx']]*e['w'][:,:,None],axis=1);sites+=rng.normal(size=sites.shape)@np.linalg.cholesky(st['f']['B0']).T;raw.append(sites)
 return np.array(raw),first,time.perf_counter()-t

def conditional(raw,st):
 flat=raw.reshape(len(raw),-1);cond=np.empty_like(flat);cond[:,st['observed']]=st['data'];res=flat[:,st['free']]-flat[:,st['observed']]@st['W'].T;cond[:,st['free']]=st['mean']+res
 return cond.reshape(raw.shape),res

def finish(k,df,fig,note):
 df=pd.DataFrame(df);df.to_csv(OUT/f'{k:02d}_tabela.csv',index=False);fig.tight_layout();fig.savefig(OUT/f'{k:02d}_figura.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/f'{k:02d}_figura.pdf',bbox_inches='tight');(OUT/f'{k:02d}_nota.md').write_text(note);display(df);display(fig);display(Markdown(note));plt.close(fig)
def variograms(sims,truth,coords,sd):
 s=np.log(sims)/sd;y=np.log(truth)/sd;i,j=np.triu_indices(len(coords),1);d=np.linalg.norm(coords[i]-coords[j],axis=1);edges=np.linspace(0,3,11);lags=(edges[:-1]+edges[1:])/2;rows=[]
 for a,b in PAIRS:
  for q in range(10):
   mask=(d>=edges[q])&(d<edges[q+1]);g=.5*np.mean((s[:,i[mask],a]-s[:,j[mask],a])*(s[:,i[mask],b]-s[:,j[mask],b]),axis=1);obs=.5*np.mean((y[i[mask],a]-y[j[mask],a])*(y[i[mask],b]-y[j[mask],b]));rows.append(dict(par=V[a]+'/'+V[b],lag_km=lags[q],pares=int(mask.sum()),observado=obs,mediana=np.median(g),q05=np.quantile(g,.05),q95=np.quantile(g,.95)))
 return pd.DataFrame(rows)
def metrics(samples,y):
 m=len(samples);pred=np.median(samples,axis=0);avg=samples.mean(0);lo,hi=np.quantile(samples,[.05,.95],axis=0);w=(2*np.arange(1,m+1)-m-1)[:,None];crps=np.mean(abs(samples-y),axis=0)-np.sum(w*np.sort(samples,axis=0),axis=0)/m**2
 return dict(MAE_mediana=abs(pred-y).mean(),RMSE_mediana=np.sqrt(np.mean((pred-y)**2)),MAE_media=abs(avg-y).mean(),RMSE_media=np.sqrt(np.mean((avg-y)**2)),CRPS=crps.mean(),cobertura90=((y>=lo)&(y<=hi)).mean(),largura90=(hi-lo).mean())
def step(k):
 st=state();tr=st['tr'];va=st['va'];f=st['f'];e=st['e'];nt=len(tr)
 if k==1:
  df=pd.DataFrame([dict(grupo='259 pontos completos',Cd='observado',Ni='observado',Zn='observado'),dict(grupo='100 alvos',Cd='oculto até avaliação',Ni='observado antes da cokrigagem',Zn='observado antes da cokrigagem')]);fig,axs=plt.subplots(1,2,figsize=(12,5));axs[0].scatter(tr.Xloc,tr.Yloc,s=12,label='259 completos');axs[0].scatter(va.Xloc,va.Yloc,s=22,facecolors='none',edgecolors='red',label='100 alvos Ni/Zn');axs[0].legend();axs[0].set(xlabel='X (km)',ylabel='Y (km)',aspect='equal');axs[1].axis('off')
  for yy,text in zip([.9,.68,.46,.24],['1. Modelo conjunto de covariância (treino)','2. FFT-MA multivariado (ruído)','3. Cokrigagem: média + residual condicionado','4. MST: transporte das tuplas']):axs[1].text(.01,yy,text,fontsize=11)
  finish(k,df,fig,'Este teste usa Ni/Zn dos alvos no condicionamento ANTES do MST; é diferente do experimento sequencial de lookup posterior. Cd externo não entra no ajuste, cokrigagem ou transporte. As médias/desvios de log e o modelo espacial são ajustados no treino. Simple cokriging usa médias plug-in estimadas no treino, não ordinary cokriging. Não há regressão Cd~Ni/Zn separada.')
 elif k==2:
  rows=[];fig,axs=plt.subplots(2,3,figsize=(14,8));pred=f['B0'][None,:,:]+sum(spherical(f['lags'],rng=r)[:,None,None]*a for r,a in zip(f['ranges'],f['A']))
  for ax,(a,b) in zip(axs.ravel(),PAIRS):
   ax.plot(f['lags'],f['gamma'][:,a,b],'o-',label='Experimental treino');ax.plot(f['lags'],pred[:,a,b],'--',label='LMC PSD');ax.set(title=V[a]+' / '+V[b],xlabel='Distância (km)',ylabel='Semivariância log padronizada');ax.legend(fontsize=7)
   for q,M in enumerate([f['B0'],*f['A']]):rows.append(dict(par=V[a]+'/'+V[b],estrutura='pepita' if q==0 else str(q),alcance_km=0. if q==0 else f['ranges'][q-1],patamar_cruzado=M[a,b],menor_autovalor_matriz=np.linalg.eigvalsh(M).min()))
  f['candidates'].to_csv(OUT/'grade_ajuste_LMC.csv',index=False);np.savez_compressed(OUT/'modelo_LMC.npz',ranges=f['ranges'],A=f['A'],B0=f['B0'],mu_log=f['mu'],sd_log=f['sd'])
  finish(k,pd.DataFrame(rows),fig,'LMC com pepita multivariada B0 e duas estruturas esféricas: C(h)=B0*I(h=0)+A1*c1(h)+A2*c2(h). As três matrizes de patamar são projetadas para PSD, garantindo validade do modelo conjunto. Alcances escolhidos numa grade pequena pelo ajuste aos seis variogramas experimentais do treino; não otimizados por Cd externo. Somas dos patamares diretos são normalizadas a 1 em log padronizado. Microestrutura pontual ajustada no treino, com piso PSD de 0,03 antes da normalização para evitar degenerescência; esse piso é regularização fixada antes da avaliação. Jitter de 1e-9 é somente numérico, não representa a pepita. A grade suaviza estruturas subgrade: a microestrutura nos pontos evita impor falsas relações lineares exatas a observações irregulares próximas. Referência: https://r-spatial.github.io/gstat/reference/fit.lmc.html .')
 elif k==3:
  sd=f['sd'][0];mu=f['mu'][0];var=np.diag(st['cc']);median=np.exp(mu+sd*st['mean']);mean=np.exp(mu+sd*st['mean']+.5*sd**2*var)
  df=pd.DataFrame(dict(alvo=np.arange(len(va)),media_cokrig_log_pad=st['mean'],variancia_cond_log_pad=var,mediana_Cd_mgkg=median,media_Cd_mgkg=mean,mediana_krigagem_so_Cd=np.exp(mu+sd*st['mean_kriging'])));df.to_csv(OUT/'cokrigagem_alvos.csv',index=False)
  fig,axs=plt.subplots(1,3,figsize=(15,5))
  for ax,vals,title in zip(axs,[median,mean,np.sqrt(var)*sd],['Mediana condicional Cd','Média condicional Cd','Desvio condicional de log Cd']):im=ax.scatter(va.Xloc,va.Yloc,c=vals,s=28);ax.set(title=title,xlabel='X (km)',ylabel='Y (km)',aspect='equal');fig.colorbar(im,ax=ax)
  finish(k,df,fig,'Cokrigagem simples multivariada: m=C_fd*(C_dd+jI)^(-1)*z_d; C_cond=C_ff−C_fd*(C_dd+jI)^(-1)*C_df. Há 977 valores conhecidos: 259×3 + 100×2; só os 100 Cd dos alvos estão livres. Na escala gaussiana média=mediana; após exponenciar elas diferem. As fórmulas mostradas em mg/kg assumem lognormalidade do backbone, antes do MST. Não se usa Cd externo para produzir estes mapas.')
 elif k==4:
  raw,grid,seconds=simulate_fft(st);np.savez_compressed(OUT/'fftma_nao_condicionado.npz',raw=raw,first_grid=grid)
  expected=np.diag(st['K']).reshape(len(st['coords']),3);emp=raw.var(0,ddof=1);df=pd.DataFrame([dict(metal=v,media_empirica=raw[:,:,j].mean(),variancia_empirica_media=emp[:,j].mean(),variancia_modelo_interpolada_media=expected[:,j].mean(),RMSE_variancias=np.sqrt(np.mean((emp[:,j]-expected[:,j])**2)),grade_dx_km=DX,embedding_lado=e['size'],tempo_FFT_total_s=seconds) for j,v in enumerate(V)])
  fig,axs=plt.subplots(2,3,figsize=(14,8))
  for j,v in enumerate(V):
   im=axs[0,j].imshow(grid[:,:,j],origin='lower',extent=[e['origin'][0],e['origin'][0]+(e['nx']-1)*DX,e['origin'][1],e['origin'][1]+(e['ny']-1)*DX],aspect='equal');axs[0,j].set(title=v+' — primeira FFT-MA',xlabel='X (km)',ylabel='Y (km)');fig.colorbar(im,ax=axs[0,j]);axs[1,j].scatter(expected[:,j],emp[:,j],s=9);axs[1,j].plot([0,1],[0,1],'k--');axs[1,j].set(title=v+' — variância entre 500 realizações',xlabel='Variância teórica interpolada',ylabel='Variância empírica')
  finish(k,df,fig,'FFT-MA real: filtra ruído branco por sqrt(FFT(covariância)) em embedding periódico ampliado, mistura três componentes pela Cholesky de cada matriz do LMC e soma as duas estruturas. Acrescenta nos pontos um ruído multivariado independente entre posições com covariância B0, representando a pepita ajustada; ele não é chamado de FFT-MA. A grade mostrada contém só a parte estruturada. 500 realizações, semente fixa. Não normaliza empiricamente cada realização: isso alteraria a covariância. Interpolação bilinear leva o campo aos pontos irregulares; a cokrigagem usa EXATAMENTE a covariância desse mesmo campo interpolado. A discretização de 0,1 km afeta principalmente o curto alcance e é uma limitação, não escondida como simulação exata no contínuo.')
 elif k==5:
  with np.load(OUT/'fftma_nao_condicionado.npz') as a:raw=a['raw']
  cond,res=conditional(raw,st);np.savez_compressed(OUT/'backbone_condicionado.npz',standardized=cond,residual_Cd=res,conditional_cov_Cd=st['cc'])
  err=np.max(abs(cond.reshape(REPS,-1)[:,st['observed']]-st['data']));assert err==0
  var=np.diag(st['cc']);naive=st['mean']+raw[:,nt:,0]*np.sqrt(var/np.diag(st['K'])[st['free']]);prior=st['K'][np.ix_(st['free'],st['free'])];D=np.sqrt(var/np.diag(prior));naive_cov=prior*D[:,None]*D[None,:]
  df=pd.DataFrame(dict(alvo=range(len(va)),media_teorica=st['mean'],media_empirica=cond[:,nt:,0].mean(0),variancia_teorica=var,variancia_empirica=cond[:,nt:,0].var(0,ddof=1),erro_maximo_observados=err));df.to_csv(OUT/'momentos_condicionais.csv',index=False)
  fig,axs=plt.subplots(2,2,figsize=(12,10));axs[0,0].scatter(var,df.variancia_empirica,s=18);mx=max(var.max(),df.variancia_empirica.max());axs[0,0].plot([0,mx],[0,mx],'k--');axs[0,0].set(xlabel='Variância condicional teórica',ylabel='Variância empírica FFT-MA condicionado');axs[0,1].scatter(st['mean'],df.media_empirica,s=18);lo=min(st['mean']);hi=max(st['mean']);axs[0,1].plot([lo,hi],[lo,hi],'k--');axs[0,1].set(xlabel='Média cokrigagem',ylabel='Média entre realizações')
  for ax,M,title in zip(axs[1],[st['cc'],naive_cov-st['cc']],['Covariância condicional correta','Apenas escalar variância: covariância errada − correta']):im=ax.imshow(M,cmap='coolwarm');ax.set(title=title,xlabel='Índice do alvo',ylabel='Índice do alvo');fig.colorbar(im,ax=ax)
  finish(k,df,fig,'Passo central: B_cond=m + [B_FFT,f − W*B_FFT,d]. A correção do ruído produz a covariância condicional inteira; não basta multiplicar ruído pelo desvio local. Observados são impostos exatamente após o jitter numérico (efeito de ordem 1e-9 na fórmula livre). O painel inferior direito mostra a discrepância teórica do atalho de escalar apenas variâncias. Médias/variâncias empíricas têm erro Monte Carlo de 500 realizações. Referência: https://geostatisticslessons.com/lessons/conditioningbykriging .')
 elif k==6:
  with np.load(OUT/'backbone_condicionado.npz') as a:bb=a['standardized']
  actual=np.exp(f['mu']+f['sd']*bb);fig,axs=plt.subplots(2,3,figsize=(14,8));rows=[]
  for j,v in enumerate(V):
   for r in [0,1,2]:axs[0,j].scatter(va.Xloc,actual[r,nt:,j],s=9,alpha=.5,label=f'Realização {r}')
   axs[0,j].set(title=v+' — três realizações nos alvos',xlabel='X local (km)',ylabel='mg/kg');axs[0,j].legend(fontsize=6);before=actual[:,nt:,j];known=va[v].to_numpy() if j else None
   if j:err=np.max(abs(before-known));axs[1,j].scatter(known,before[0],s=15);lo=min(known);hi=max(known);axs[1,j].plot([lo,hi],[lo,hi],'k--');axs[1,j].set(xlabel=v+' observado',ylabel=v+' backbone');rows.append(dict(metal=v,erro_maximo_alvos_observados=err,desvio_entre_realizacoes_medio=before.std(0).mean()))
   else:axs[1,j].hist(before.ravel(),bins=35,density=True);axs[1,j].set(xlabel='Cd simulado nos alvos',ylabel='Densidade');rows.append(dict(metal=v,erro_maximo_alvos_observados=np.nan,desvio_entre_realizacoes_medio=before.std(0).mean()))
  finish(k,pd.DataFrame(rows),fig,'ANTES do matcher, todos os 259 trios de treino e os Ni/Zn dos 100 alvos são respeitados. Só Cd varia nos alvos. Não se condiciona Cd externo. Estes são campos gaussianos em log e não preservam a nuvem multivariada empírica do treino; é essa a transformação que será solicitada ao MST na próxima etapa.')
 elif k==7:
  with np.load(OUT/'backbone_condicionado.npz') as a:bb=a['standardized']
  rng=np.random.default_rng(SEED+1);sims=[];donors=[];assign=[];rows=[]
  for rep in range(REPS):
   donor=rng.integers(len(tr),size=len(va));pool=np.vstack([np.log(tr[V].to_numpy()),np.log(tr[V].to_numpy())[donor]]);model=ScalableMST(**PARAMS,random_state=rep);t=time.perf_counter();s=model.simulate(bb[rep],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords']);dt=time.perf_counter()-t
   assert np.array_equal(s,pool[model.assignment_]) and len(np.unique(model.assignment_))==len(pool);assert np.array_equal(s[:nt],np.log(tr[V].to_numpy()))
   out=np.exp(s[nt:]);rows.append(dict(realizacao=rep,segundos_matching=dt,erro_treino_log=0.,bijeção=True,MAE_Ni=abs(out[:,1]-va.Ni.to_numpy()).mean(),MAE_Zn=abs(out[:,2]-va.Zn.to_numpy()).mean()));sims.append(out);donors.append(donor);assign.append(model.assignment_)
  sims=np.array(sims);np.savez_compressed(OUT/'mst_500.npz',sims=sims,pool_donors=donors,assignments=assign);df=pd.DataFrame(rows);fig,axs=plt.subplots(1,3,figsize=(15,4));axs[0].plot(df.realizacao,df.segundos_matching);axs[0].set(xlabel='Realização',ylabel='Tempo matching (s)');axs[1].plot(df.realizacao,df.MAE_Ni);axs[1].set(xlabel='Realização',ylabel='MAE Ni nos alvos (mg/kg)');axs[2].plot(df.realizacao,df.MAE_Zn);axs[2].set(xlabel='Realização',ylabel='MAE Zn nos alvos (mg/kg)')
  finish(k,df,fig,'Matcher ScalableMST oficial intacto: beta=6, 48 candidatos, 150 iterações, sem termo relacional. Pool por realização = 259 tuplas originais + 100 tuplas bootstrap do treino, com treino fixo por pinning. O matcher padroniza novamente o backbone e permuta tuplas completas; ele NÃO implementa condicionamento parcial. Portanto, é esperado que Ni/Zn dos alvos deixem de ser exatos. Não substituímos esses valores após o transporte para esconder esse efeito. Isso é medido nos painéis e na tabela.')
 elif k==8:
  with np.load(OUT/'backbone_condicionado.npz') as a:bb=a['standardized']
  with np.load(OUT/'mst_500.npz') as a:mst=a['sims']
  backbone=np.exp(f['mu']+f['sd']*bb[:,nt:]);rows=[]
  for name,s in [('FFT-MA + cokrigagem (antes MST)',backbone),('MST (depois do transporte)',mst)]:
   for j,v in enumerate(V):
    m=metrics(s[:,:,j],va[v].to_numpy())
    if name.startswith('FFT-MA') and j>0:m['cobertura90']=np.nan
    rows.append(dict(etapa=name,metal=v,**m))
  df=pd.DataFrame(rows);truth=va.Cd.to_numpy();fig,axs=plt.subplots(1,2,figsize=(12,5))
  for name,s in [('Backbone',backbone[:,:,0]),('MST',mst[:,:,0])]:
   axs[0].scatter(truth,np.median(s,axis=0),s=14,alpha=.6,label=name);cov=[]
   for n in [.5,.8,.9,.95]:lo,hi=np.quantile(s,[(1-n)/2,1-(1-n)/2],axis=0);cov.append(((truth>=lo)&(truth<=hi)).mean())
   axs[1].plot([.5,.8,.9,.95],cov,'o-',label=name)
  mx=max(truth);axs[0].plot([0,mx],[0,mx],'k--');axs[0].set(xlabel='Cd observado (mg/kg)',ylabel='Mediana preditiva Cd (mg/kg)');axs[0].legend();axs[1].plot([0,1],[0,1],'k--');axs[1].set(xlabel='Cobertura nominal',ylabel='Cobertura observada Cd',ylim=(0,1));axs[1].legend()
  finish(k,df,fig,'Cd externo é revelado somente para avaliação, após todos os ajustes, geração e transporte. Tabela separa média e mediana como preditores, sem escolher retrospectivamente o melhor; CRPS e cobertura usam o ensemble inteiro. Ni/Zn no backbone são valores condicionantes, portanto erro próximo de zero nesses metais não é previsão fora da amostra. Cobertura de Ni/Zn no backbone não é uma métrica preditiva: são observados fixos, e está marcada como não aplicável. Sua mudança após MST mede perda de condicionamento parcial. O conjunto externo já foi visto em testes anteriores, não é um novo teste cego.')
 elif k==9:
  with np.load(OUT/'backbone_condicionado.npz') as a:bb=a['standardized']
  with np.load(OUT/'mst_500.npz') as a:mst=a['sims']
  backbone=np.exp(f['mu']+f['sd']*bb[:,nt:]);dfs=[];fig,axs=plt.subplots(2,3,figsize=(15,9))
  for name,s in [('Backbone',backbone),('MST',mst)]:d=variograms(s,va[V].to_numpy(),va[XY].to_numpy(),f['sd']);d['etapa']=name;dfs.append(d)
  df=pd.concat(dfs,ignore_index=True)
  for ax,(a,b) in zip(axs.ravel(),PAIRS):
   par=V[a]+'/'+V[b]
   for name in ['Backbone','MST']:
    g=df[(df.par==par)&(df.etapa==name)];ax.plot(g.lag_km,g.mediana,label=name);ax.fill_between(g.lag_km,g.q05,g.q95,alpha=.15)
   ax.plot(g.lag_km,g.observado,'ko--',ms=3,label='Observado');ax.set(title=par,xlabel='Distância (km)',ylabel='Semivariância log padronizada');ax.legend(fontsize=6)
  finish(k,df,fig,'Variogramas diretos e cruzados calculados POR REALIZAÇÃO, antes/depois do MST, nunca do campo médio. As bandas são 5–95% entre realizações, não intervalos de confiança da referência. Antes do MST, Ni/Zn dos alvos são exatamente observados, então seus variogramas coincidem por construção. Depois do MST, qualquer diferença revela o efeito do transporte. O condicionamento muda a distribuição espacial em relação ao modelo estacionário não condicionado; não se exige que cada realização tenha exatamente o variograma ajustado.')
 else:raise ValueError(k)
