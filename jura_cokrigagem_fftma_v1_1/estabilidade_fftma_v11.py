import sys,json,time,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Markdown
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mst_fftma_v11_pipeline',HERE/'pipeline.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=HERE/'resultados';OUT.mkdir(exist_ok=True);SERIES=OUT/'series';SERIES.mkdir(exist_ok=True)
SEEDS=list(range(20261029,20261039));REPS=500

def finish(k,df,fig,note):
 p.display=display;p.finish(k,df,fig,note)
def run_one(seed):
 st=p.state();tr=st['tr'];va=st['va'];f=st['f'];nt=len(tr);p.SEED=seed
 raw,first,fft_s=p.simulate_fft(st);bb,res=p.conditional(raw,st);backbone=np.exp(f['mu']+f['sd']*bb[:,nt:]);rng=np.random.default_rng(seed+1);logtrain=np.log(tr[p.V].to_numpy());sims=[];donors=[];assign=[];times=[]
 assert np.array_equal(bb.reshape(REPS,-1)[:,st['observed']],np.broadcast_to(st['data'],(REPS,len(st['data']))))
 for rep in range(REPS):
  donor=rng.integers(nt,size=len(va));pool=np.vstack([logtrain,logtrain[donor]]);model=p.ScalableMST(**p.PARAMS,random_state=rep);t=time.perf_counter();s=model.simulate(bb[rep],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords']);times.append(time.perf_counter()-t)
  assert np.array_equal(s,pool[model.assignment_]);assert len(np.unique(model.assignment_))==len(pool);assert np.array_equal(s[:nt],logtrain)
  sims.append(np.exp(s[nt:]));donors.append(donor);assign.append(model.assignment_)
 mst=np.array(sims);np.savez_compressed(SERIES/f'semente_{seed}.npz',raw=raw,backbone=backbone,mst=mst,pool_donors=donors,assignments=assign)
 return backbone,mst,dict(fft_s=fft_s,matching_s=float(sum(times)),bijeções=REPS,erro_ancoras_log=0.,erro_observados_pad=0.)
def seed_arrays(seed):
 if seed==SEEDS[0]:
  st=p.state()
  with np.load(OUT/'backbone_condicionado.npz') as a:backbone=np.exp(st['f']['mu']+st['f']['sd']*a['standardized'][:,len(st['tr']):])
  with np.load(OUT/'mst_500.npz') as a:mst=a['sims']
  return backbone,mst
 with np.load(SERIES/f'semente_{seed}.npz') as a:return a['backbone'],a['mst']
def evaluate(seed,backbone,mst,diag):
 st=p.state();rows=[];var=[];constraints=[]
 for label,s in [('Backbone',backbone),('MST',mst)]:
  rows.append(dict(semente=seed,etapa=label,**p.metrics(s[:,:,0],st['va'].Cd.to_numpy())))
  d=p.variograms(s,st['va'][p.V].to_numpy(),st['va'][p.XY].to_numpy(),st['f']['sd']);d['semente']=seed;d['etapa']=label;var.append(d)
  for j,v in [(1,'Ni'),(2,'Zn')]:constraints.append(dict(semente=seed,etapa=label,metal=v,MAE_por_realizacao_media=float(abs(s[:,:,j]-st['va'][v].to_numpy()).mean()),erro_maximo=float(abs(s[:,:,j]-st['va'][v].to_numpy()).max()),MAE_mediana=float(abs(np.median(s[:,:,j],axis=0)-st['va'][v].to_numpy()).mean())))
 return rows,pd.concat(var,ignore_index=True),constraints,dict(semente=seed,**diag)
def step(k):
 if k<=9:
  p.display=display;p.step(k);return
 st=p.state();va=st['va']
 if k==10:
  rows=[];curves=[];constraints=[];audits=[];start=time.perf_counter()
  for i,seed in enumerate(SEEDS):
   if i==0:
    backbone,mst=seed_arrays(seed);d=pd.read_csv(OUT/'07_tabela.csv');diag=dict(fft_s=np.nan,matching_s=d.segundos_matching.sum(),bijeções=REPS,erro_ancoras_log=float(d.erro_treino_log.max()),erro_observados_pad=0.)
   else:backbone,mst,diag=run_one(seed)
   r,v,c,a=evaluate(seed,backbone,mst,diag);rows.extend(r);curves.append(v);constraints.extend(c);audits.append(a)
   pd.DataFrame(rows).to_csv(OUT/'metricas_sementes.csv',index=False);pd.concat(curves,ignore_index=True).to_csv(OUT/'variogramas_sementes.csv',index=False);pd.DataFrame(constraints).to_csv(OUT/'condicionamento_sementes.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'auditoria_sementes.csv',index=False)
   print(f'Semente {i+1}/10 concluída: {seed}; MAE Cd MST={r[1]["MAE_mediana"]:.5f}; total {time.perf_counter()-start:.1f}s',flush=True)
  p.SEED=SEEDS[0];df=pd.DataFrame(rows);fig,axs=plt.subplots(1,2,figsize=(13,5))
  for label,g in df.groupby('etapa'):axs[0].plot(np.arange(1,11),g.MAE_mediana,'o-',label=label);axs[1].plot(np.arange(1,11),g.cobertura90,'o-',label=label)
  axs[0].set(xlabel='Execução (sementes pré-fixadas)',ylabel='MAE mediana Cd (mg/kg)');axs[0].legend();axs[1].axhline(.9,color='k',ls='--');axs[1].set(xlabel='Execução',ylabel='Cobertura nominal 90%');axs[1].legend()
  finish(k,df,fig,'10 sementes pré-fixadas, 500 realizações cada: total 5.000 campos. A primeira é a referência repetida nas etapas 1–9. Outras nove variam o FFT-MA, a pepita e o bootstrap do pool; modelo LMC, dados, parâmetros MST e random_state=índice da realização no matcher ficam fixos. Cada comparação antes/depois é pareada pelo mesmo backbone. Não se escolhe a melhor semente e não há ajuste por Cd externo. Isso mede estabilidade Monte Carlo nesta divisão de dados, não incerteza de generalização.')
 elif k==11:
  d=pd.read_csv(OUT/'metricas_sementes.csv');metrics=['MAE_mediana','RMSE_mediana','CRPS','cobertura90'];rows=[];fig,axs=plt.subplots(2,2,figsize=(12,9))
  for ax,m in zip(axs.ravel(),metrics):
   a=d[d.etapa=='Backbone'].set_index('semente')[m];b=d[d.etapa=='MST'].set_index('semente')[m];delta=b-a
   rows.append(dict(metrica=m,backbone_media=a.mean(),backbone_desvio_sementes=a.std(ddof=1),MST_media=b.mean(),MST_desvio_sementes=b.std(ddof=1),diferenca_pareada_media=delta.mean(),diferenca_min=delta.min(),diferenca_max=delta.max(),sementes_MST_menor=int((delta<0).sum())))
   for seed in SEEDS:ax.plot([0,1],[a.loc[seed],b.loc[seed]],'o-',alpha=.65)
   ax.set(xticks=[0,1],xticklabels=['Backbone','MST'],ylabel=m)
   if m=='cobertura90':ax.axhline(.9,color='k',ls='--')
  df=pd.DataFrame(rows);finish(k,df,fig,'Médias e desvios calculados ENTRE dez execuções de 500 realizações, sem tratar as 5.000 realizações como novos pontos de validação. Diferença = MST − backbone; negativa indica melhora para erros/CRPS, mas não necessariamente para cobertura. Para cobertura, o alvo é 0,90. Os desvios medem efeito das sementes, não um intervalo de confiança populacional. As mesmas 100 medições são reutilizadas em todas as execuções.')
 elif k==12:
  d=pd.read_csv(OUT/'variogramas_sementes.csv');rows=[];fig,axs=plt.subplots(2,3,figsize=(15,9))
  for ax,(a,b) in zip(axs.ravel(),p.PAIRS):
   par=p.V[a]+'/'+p.V[b];sub=d[d.par==par]
   for label in ['Backbone','MST']:
    h=sub[sub.etapa==label]
    for seed,g in h.groupby('semente'):
     if label=='MST':ax.plot(g.lag_km,g.mediana,color='#d97706',alpha=.18,lw=1)
     rmse=np.sqrt(np.average((g.mediana-g.observado)**2,weights=g.pares));rows.append(dict(par=par,etapa=label,semente=seed,RMSE_curva_mediana=float(rmse)))
    g=h.groupby('lag_km',as_index=False)[['mediana','observado']].mean();ax.plot(g.lag_km,g.mediana,lw=2,label=label+' — média das medianas')
   ax.plot(g.lag_km,g.observado,'ko--',ms=3,label='Observado');ax.set(title=par,xlabel='Distância (km)',ylabel='Semivariância log padronizada');ax.legend(fontsize=6)
  detail=pd.DataFrame(rows);detail.to_csv(OUT/'erro_variogramas_detalhado.csv',index=False);df=detail.groupby(['par','etapa'],as_index=False).agg(RMSE_medio=('RMSE_curva_mediana','mean'),desvio_sementes=('RMSE_curva_mediana','std'))
  finish(k,df,fig,'Variogramas diretos/cruzados calculados por realização e resumidos pela mediana de cada execução. Linhas finas mostram as dez medianas MST, não intervalos de confiança. RMSE ponderado pelo número de pares compara a curva mediana ao variograma observado nos mesmos 100 alvos; é um diagnóstico, não critério usado no ajuste. Antes do MST, Ni/Zn são observados e coincidem por construção; depois, discrepâncias expõem a alteração pelo transporte. Estabilidade das curvas não implica ajuste perfeito.')
 elif k==13:
  c=pd.read_csv(OUT/'condicionamento_sementes.csv');a=pd.read_csv(OUT/'auditoria_sementes.csv');df=c.merge(a[['semente','bijeções','erro_ancoras_log','erro_observados_pad']],on='semente');fig,axs=plt.subplots(1,3,figsize=(15,4))
  for j,v in enumerate(['Ni','Zn']):
   for label,g in c[c.metal==v].groupby('etapa'):axs[j].plot(range(1,11),g.MAE_por_realizacao_media,'o-',label=label)
   axs[j].set(title=v,xlabel='Execução',ylabel='MAE médio por realização (mg/kg)');axs[j].legend(fontsize=7)
  axs[2].plot(range(1,11),a.erro_ancoras_log,'o-',label='Treino após MST');axs[2].plot(range(1,11),a.erro_observados_pad,'x--',label='Conhecidos antes MST');axs[2].set(xlabel='Execução',ylabel='Erro máximo em log padronizado/log');axs[2].legend(fontsize=7)
  assert a['bijeções'].sum()==5000 and a.erro_ancoras_log.max()==0 and a.erro_observados_pad.max()==0
  finish(k,df,fig,'Todas as 5.000 atribuições verificam bijeção, tuplas do pool e pinning do treino. Antes do MST, os 977 valores conhecidos são impostos exatamente na escala padronizada. Depois, Ni/Zn dos alvos deixam de ser exatos em todas as execuções; não são sobrescritos para esconder o efeito. MAE médio por realização e MAE da mediana são conceitos distintos, ambos registrados. Bootstrap e FFT variam; não há condicionamento parcial novo no matcher.')
 elif k==14:
  rows=[];fig,axs=plt.subplots(1,2,figsize=(12,5))
  for seed in SEEDS:
   a,b=seed_arrays(seed)
   for n in [100,250,500]:
    for label,s in [('Backbone',a),('MST',b)]:rows.append(dict(semente=seed,realizacoes=n,etapa=label,**p.metrics(s[:n,:,0],va.Cd.to_numpy())))
  df=pd.DataFrame(rows)
  for label,g in df.groupby('etapa'):
   a=g.groupby('realizacoes').MAE_mediana.agg(['mean','min','max']);axs[0].plot(a.index,a['mean'],'o-',label=label);axs[0].fill_between(a.index,a['min'],a['max'],alpha=.15)
   b=g.groupby('realizacoes').cobertura90.agg(['mean','min','max']);axs[1].plot(b.index,b['mean'],'o-',label=label);axs[1].fill_between(b.index,b['min'],b['max'],alpha=.15)
  axs[0].set(xlabel='Realizações por execução',ylabel='MAE médio entre sementes');axs[0].legend();axs[1].axhline(.9,color='k',ls='--');axs[1].set(xlabel='Realizações por execução',ylabel='Cobertura 90% média entre sementes');axs[1].legend()
  finish(k,df,fig,'Prefixos de 100/250/500 realizações do mesmo conjunto, sem novas simulações. Faixas mostram mínimo/máximo entre sementes, não intervalos de confiança. Não selecionamos o prefixo com menor erro: 500 realizações por execução continua sendo o resultado principal. Esta análise permite avaliar se a dispersão das métricas diminui com o orçamento Monte Carlo; não demonstra convergência assintótica nem estabilidade em outras divisões dos dados.')
 else:raise ValueError(k)
