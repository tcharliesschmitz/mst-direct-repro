"""Extensao experimental do custo e candidatos NN; codigo oficial preservado."""
import os,sys,json,time,importlib.util
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_pesos_nizn_v12')
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from IPython.display import display,Markdown
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('mst_pesos_v12_pipeline',ROOT/'jura_cokrigagem_fftma_v1_1/pipeline.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from mst_direct.scalable import _sparse_sinkhorn,_greedy_round,_repair_leftovers,_standardize
SOURCE=ROOT/'jura_cokrigagem_fftma_v1_1/resultados';OUT=HERE/'resultados';OUT.mkdir(exist_ok=True);OUTFIELDS=OUT/'campos';OUTFIELDS.mkdir(exist_ok=True)
SEEDS=[20261029,20261030,20261031];REPS=200;WEIGHTS=[0.,1.,4.,16.];CANDIDATES=[48,100]
CONFIGS=[dict(id=f'lambda{w:g}_nn{c}',peso=w,nn=c) for c in CANDIDATES for w in WEIGHTS]

class WeightedMST(p.ScalableMST):
 def __init__(self,weight,observed_log,train_mu,train_sd,**kwargs):
  super().__init__(**kwargs);self.weight=float(weight);self.observed_log=np.asarray(observed_log);self.train_mu=train_mu;self.train_sd=train_sd
  if self.observed_log.ndim!=2 or self.observed_log.shape[1]!=2:raise ValueError('Penalização aceita somente Ni/Zn observados, nunca Cd.')
 def simulate(self,backbone,cloud,pinned=None,coords=None):
  n=len(cloud);rows=np.ones(n,dtype=bool);cols=rows.copy()
  if pinned is not None:rows[np.asarray(pinned[0])]=False;cols[np.asarray(pinned[1])]=False
  if rows.sum()!=len(self.observed_log):raise ValueError('Observações Ni/Zn devem corresponder às linhas livres.')
  if self.lam_relational!=0 or self.n_relax!=0:raise ValueError('Termo relacional mantido desligado para isolar o custo.')
  self.extra_source=(np.asarray(cloud)[cols,1:]-self.train_mu)/self.train_sd;self.extra_target=(self.observed_log-self.train_mu)/self.train_sd
  return super().simulate(backbone,cloud,pinned=pinned,coords=coords)
 def _match(self,query,Uf,tree,nf):
  if self.weight==0:Q=query;U=Uf
  else:Q=np.column_stack([query,np.sqrt(self.weight)*self.extra_target]);U=np.column_stack([Uf,np.sqrt(self.weight)*self.extra_source]);tree=cKDTree(U)
  C=min(self.n_candidates,nf);dist,nn=tree.query(Q,k=C);dist=np.atleast_2d(dist);nn=np.atleast_2d(nn);rows=np.repeat(np.arange(nf),C);cols=nn.ravel();cost=dist.ravel()**2;scale=np.median(cost)+1e-12;coupling=_sparse_sinkhorn(rows,cols,-self.beta*cost/scale,nf,nf,self.n_iter,self.tol);local=_greedy_round(rows,cols,coupling,nf,nf)
  self.diagnostics=dict(colunas_sem_candidato=int(np.sum(np.bincount(cols,minlength=nf)==0)),linhas_reparadas=int(np.sum(local<0)),escala_mediana_custo=float(scale),massa_plano=float(coupling.sum()))
  return _repair_leftovers(local,Q,U)

def inputs(seed):
 st=p.state()
 if seed==SEEDS[0]:
  with np.load(SOURCE/'backbone_condicionado.npz') as a:bb=a['standardized'][:REPS]
  with np.load(SOURCE/'mst_500.npz') as a:donors=a['pool_donors'][:REPS];baseline=a['sims'][:REPS]
 else:
  with np.load(SOURCE/'series'/f'semente_{seed}.npz') as a:raw=a['raw'];donors=a['pool_donors'][:REPS];baseline=a['mst'][:REPS]
  bb,_=p.conditional(raw,st);bb=bb[:REPS]
 return bb,donors,baseline

def scales():
 tr=p.state()['tr'];z=np.log(tr[['Ni','Zn']].to_numpy());return z.mean(0),z.std(0)
def finish(k,df,fig,note):
 p.display=display;p.OUT=OUT;p.finish(k,df,fig,note)
def read_fields(config,seed):
 with np.load(OUTFIELDS/f'{config}_s{seed}.npz') as a:return a['sims']
def step(k):
 st=p.state();tr=st['tr'];va=st['va'];nt=len(tr);mu,sd=scales();logtrain=np.log(tr[p.V].to_numpy());obs=np.log(va[['Ni','Zn']].to_numpy())
 if k==1:
  df=pd.DataFrame(CONFIGS);df['sementes']=len(SEEDS);df['realizacoes_por_semente']=REPS;df.to_json(OUT/'grade_congelada.json',orient='records',indent=2);fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
  lines=['Custo = distância ao backbone padronizado (Cd/Ni/Zn)', '+ lambda × erro quadrático de Ni/Zn observados', 'NN usa a MESMA geometria ponderada do custo', 'Reparo final também usa a geometria ponderada', 'Lambda=0 / NN=48 deve reproduzir a versão 1.1']
  for y,text in zip(np.linspace(.9,.1,len(lines)),lines):ax.text(.03,y,text,fontsize=13)
  finish(k,df,fig,'Grade pré-fixada: lambda 0/1/4/16, com 48 ou todos os 100 candidatos livres. Três sementes, 200 realizações cada: 600 campos por configuração, comparações pareadas nos mesmos backbones e pools da versão 1.1. Não há seleção automática de hiperparâmetro nem reajuste por Cd externo. Isso é exploração da curva de compromisso, não novo teste cego. Nenhum termo relacional é ligado. O matcher experimental usa os mesmos helpers oficiais de Sinkhorn, arredondamento e reparo; a implementação oficial permanece intacta.')
 elif k==2:
  rows=[]
  for seed in SEEDS:
   bb,donors,baseline=inputs(seed)
   for r in range(REPS):
    source=(logtrain[donors[r],1:]-mu)/sd;target=(obs-mu)/sd;D=cdist(target,source,metric='sqeuclidean');ri,ci=linear_sum_assignment(D);rows.append(dict(semente=seed,realizacao=r,menor_MSE_NiZn_com_reuso=float(D.min(axis=1).mean()/2),menor_MSE_NiZn_com_bijecao=float(D[ri,ci].mean()/2),trios_distintos_pool=int(len(np.unique(logtrain[donors[r]],axis=0)))))
  df=pd.DataFrame(rows);fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].hist(np.sqrt(df.menor_MSE_NiZn_com_reuso),bins=25,alpha=.5,label='Vizinhos com reuso');axs[0].hist(np.sqrt(df.menor_MSE_NiZn_com_bijecao),bins=25,alpha=.5,label='Atribuição bijetiva ótima');axs[0].set(xlabel='RMSE Ni/Zn em log padronizado',ylabel='Pools');axs[0].legend(fontsize=7);axs[1].hist(df.trios_distintos_pool,bins=20);axs[1].set(xlabel='Tríos distintos entre os 100 livres',ylabel='Pools')
  finish(k,df,fig,'Limite do pool calculado ANTES de olhar Cd externo: vizinho mais próximo permite reuso; atribuição húngara encontra o mínimo exato do custo SOMENTE Ni/Zn com bijeção no mesmo pool livre. Não é um limite para MAE em mg/kg nem para Cd; é um diagnóstico do mínimo MSE em log Ni/Zn padronizado. O custo espacial/Cd foi removido apenas deste diagnóstico. Um mínimo positivo mostra que pesos infinitos não criam os pares ausentes; a bijeção pode aumentar a discrepância mínima.')
 elif k==3:
  rows=[];start=time.perf_counter();checked=0
  for seed in SEEDS:
   bb,donors,baseline=inputs(seed)
   for config in CONFIGS:
    sims=[];assignments=[]
    for r in range(REPS):
     pool=np.vstack([logtrain,logtrain[donors[r]]]);params=dict(p.PARAMS);params['n_candidates']=config['nn'];model=WeightedMST(config['peso'],obs,mu,sd,**params,random_state=r);t=time.perf_counter();s=model.simulate(bb[r],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords']);elapsed=time.perf_counter()-t
     assert np.array_equal(s,pool[model.assignment_]);assert len(np.unique(model.assignment_))==len(pool);assert np.array_equal(s[:nt],logtrain);sim=np.exp(s[nt:]);sims.append(sim);assignments.append(model.assignment_);checked+=1
     if config['peso']==0 and config['nn']==48:assert np.array_equal(sim,baseline[r])
     Us,_,_=_standardize(pool);Bs=(bb[r]-bb[r].mean(0))/bb[r].std(0);assigned=pool[model.assignment_[nt:]];source_extra=(assigned[:,1:]-mu)/sd;target_extra=(obs-mu)/sd;sec=float(np.mean((source_extra-target_extra)**2));base=float(np.mean(np.sum((Bs[nt:]-Us[model.assignment_[nt:]])**2,axis=1)))
     rows.append(dict(config=config['id'],semente=seed,realizacao=r,peso=config['peso'],nn=config['nn'],MSE_NiZn_log_pad=sec,custo_backbone_medio=base,tempo_matching_s=elapsed,MAE_Ni=float(abs(sim[:,1]-va.Ni.to_numpy()).mean()),MAE_Zn=float(abs(sim[:,2]-va.Zn.to_numpy()).mean()),bijeção=True,erro_treino_log=0.,**model.diagnostics))
    np.savez_compressed(OUTFIELDS/f'{config["id"]}_s{seed}.npz',sims=np.array(sims),assignments=np.array(assignments),pool_donors=donors)
    pd.DataFrame(rows).to_csv(OUT/'diagnostico_todos.csv',index=False);print(f'{seed}: {config["id"]} concluído; total {time.perf_counter()-start:.1f}s',flush=True)
  detail=pd.DataFrame(rows);df=detail.groupby(['config','peso','nn'],as_index=False).agg(MSE_NiZn_log_pad=('MSE_NiZn_log_pad','mean'),MAE_Ni=('MAE_Ni','mean'),MAE_Zn=('MAE_Zn','mean'),reparos_medios=('linhas_reparadas','mean'),colunas_sem_candidato_media=('colunas_sem_candidato','mean'),tempo_medio_s=('tempo_matching_s','mean'))
  fig,axs=plt.subplots(1,3,figsize=(15,4))
  for nn,g in df.groupby('nn'):
   g=g.sort_values('peso')
   axs[0].plot(g.peso,g.MAE_Ni,'o-',label=f'NN {nn}');axs[1].plot(g.peso,g.MAE_Zn,'o-',label=f'NN {nn}');axs[2].plot(g.peso,g.reparos_medios,'o-',label=f'NN {nn}')
  for ax in axs:ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Peso adicional Ni/Zn');ax.legend(fontsize=7)
  axs[0].set_ylabel('MAE Ni por realização (mg/kg)');axs[1].set_ylabel('MAE Zn por realização (mg/kg)');axs[2].set_ylabel('Reparos médios por campo')
  (OUT/'auditoria_atribuicoes.json').write_text(json.dumps({'campos_verificados':checked,'baseline_lambda0_NN48_identico_v11':True,'bijeção_todas':True,'tuplas_exatas_todas':True,'pinning_treino_exato':True,'codigo_oficial_alterado':False},indent=2))
  finish(k,df,fig,'Todos os pesos usam exatamente os mesmos campos FFT-MA condicionados e os mesmos doadores por semente/realização. Lambda=0, NN=48 reproduz bit a bit os 600 campos de referência prefixados da versão 1.1. Penalização compara Ni/Zn diretamente aos observados em log escalado pelo treino; não confia na padronização interna do backbone para representar a medição. NN e reparo usam o espaço aumentado coerente. Beta=6 e normalização do custo pela mediana continuam oficiais; pesos alteram a preferência relativa, não garantem monotonicidade após Sinkhorn/arredondamento/reparo. Cd externo ainda não foi usado nas métricas desta etapa.')
 elif k==4:
  d=pd.read_csv(OUT/'diagnostico_todos.csv');floor=pd.read_csv(OUT/'02_tabela.csv');a=d.merge(floor[['semente','realizacao','menor_MSE_NiZn_com_bijecao']],on=['semente','realizacao']);a['excesso_sobre_minimo']=a.MSE_NiZn_log_pad-a.menor_MSE_NiZn_com_bijecao;assert a.excesso_sobre_minimo.min()>-1e-10
  df=a.groupby(['config','peso','nn'],as_index=False)[['MSE_NiZn_log_pad','menor_MSE_NiZn_com_bijecao','excesso_sobre_minimo','custo_backbone_medio','massa_plano']].mean();fig,axs=plt.subplots(1,2,figsize=(12,5))
  for nn,g in df.groupby('nn'):
   g=g.sort_values('peso');axs[0].plot(g.peso,g.MSE_NiZn_log_pad,'o-',label=f'NN {nn}');axs[1].plot(g.peso,g.custo_backbone_medio,'o-',label=f'NN {nn}')
  axs[0].axhline(floor.menor_MSE_NiZn_com_bijecao.mean(),color='k',ls='--',label='Mínimo bijetivo do pool');axs[0].set(xscale='symlog',xlabel='Peso Ni/Zn',ylabel='MSE Ni/Zn em log padronizado');axs[0].legend(fontsize=7);axs[1].set(xscale='symlog',xlabel='Peso Ni/Zn',ylabel='Distância média ao backbone (três componentes)');axs[1].legend(fontsize=7)
  finish(k,df,fig,'Separa três efeitos: pares Ni/Zn ausentes do pool, competição bijetiva por doadores e aproximação do solver/candidatos. NN=100 inclui todos os livres e remove a poda de suporte, mas não torna o arredondamento a solução ótima global do custo combinado. A distância ao backbone tende a competir com o ajuste observado. O mínimo foi calculado no mesmo pool e na mesma escala, por isso pode ser comparado diretamente a este MSE; não é um alvo de variograma.')
 elif k==5:
  rows=[]
  for seed in SEEDS:
   for config in CONFIGS:
    s=read_fields(config['id'],seed)
    for j,v in enumerate(p.V):rows.append(dict(config=config['id'],peso=config['peso'],nn=config['nn'],semente=seed,metal=v,**p.metrics(s[:,:,j],va[v].to_numpy())))
  df=pd.DataFrame(rows);df.to_csv(OUT/'metricas_detalhadas.csv',index=False);cd=df[df.metal=='Cd'].groupby(['config','peso','nn'],as_index=False).agg(MAE_Cd=('MAE_mediana','mean'),desvio_MAE_sementes=('MAE_mediana','std'),RMSE_Cd=('RMSE_mediana','mean'),CRPS_Cd=('CRPS','mean'),cobertura90_Cd=('cobertura90','mean'),largura90_Cd=('largura90','mean'))
  fig,axs=plt.subplots(1,3,figsize=(15,4))
  for nn,g in cd.groupby('nn'):
   g=g.sort_values('peso');axs[0].plot(g.peso,g.MAE_Cd,'o-',label=f'NN {nn}');axs[1].plot(g.peso,g.CRPS_Cd,'o-',label=f'NN {nn}');axs[2].plot(g.peso,g.cobertura90_Cd,'o-',label=f'NN {nn}')
  for ax in axs:ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Peso adicional Ni/Zn');ax.legend(fontsize=7)
  axs[0].set_ylabel('MAE mediana Cd (mg/kg)');axs[1].set_ylabel('CRPS Cd (mg/kg)');axs[2].set_ylabel('Cobertura 90% Cd');axs[2].axhline(.9,color='k',ls='--')
  finish(k,cd,fig,'Somente agora Cd externo é revelado para avaliar a grade pré-fixada. Não se escolhe uma configuração vencedora com esses valores; a finalidade é explorar o compromisso entre desvios conhecidos e previsão de Cd. Três sementes × 200 realizações, menos que as dez × 500 da versão 1.1: referência é recomputada com o mesmo orçamento, sem comparar diretamente prefixos de 200 com métricas antigas de 500. Desvio entre sementes não é incerteza de generalização. Ni/Zn são observações de entrada, não previsão fora da amostra.')
 elif k==6:
  allrows=[]
  for seed in SEEDS:
   for config in CONFIGS:
    s=read_fields(config['id'],seed);d=p.variograms(s,va[p.V].to_numpy(),va[p.XY].to_numpy(),st['f']['sd']);d['config']=config['id'];d['peso']=config['peso'];d['nn']=config['nn'];d['semente']=seed;allrows.append(d)
  detail=pd.concat(allrows,ignore_index=True);detail.to_csv(OUT/'variogramas_detalhados.csv',index=False);errors=[];fig,axs=plt.subplots(2,3,figsize=(15,9))
  for ax,(a,b) in zip(axs.ravel(),p.PAIRS):
   par=p.V[a]+'/'+p.V[b]
   for config in CONFIGS:
    if config['nn']!=48:continue
    g=detail[(detail.par==par)&(detail.config==config['id'])].groupby('lag_km',as_index=False)[['mediana','observado']].mean();ax.plot(g.lag_km,g.mediana,label=f'Peso {config["peso"]:g}')
   ax.plot(g.lag_km,g.observado,'ko--',ms=3,label='Observado');ax.set(title=par,xlabel='Distância (km)',ylabel='Semivariância log padronizada');ax.legend(fontsize=6)
   for (cfg,seed),g in detail[detail.par==par].groupby(['config','semente']):errors.append(dict(par=par,config=cfg,semente=seed,RMSE_curva=float(np.sqrt(np.average((g.mediana-g.observado)**2,weights=g.pares)))))
  errors=pd.DataFrame(errors);errors.to_csv(OUT/'erro_variogramas_detalhado.csv',index=False);df=errors.groupby(['par','config'],as_index=False).agg(RMSE_medio=('RMSE_curva','mean'),desvio_sementes=('RMSE_curva','std'))
  finish(k,df,fig,'Seis variogramas, sempre calculados por realização e depois resumidos pela mediana de cada execução. Figura mostra NN=48 para legibilidade; tabela e CSV incluem NN=100. As curvas observadas são diagnóstico, não foram usadas para escolher lambda. Aumentar peso de Ni/Zn pode melhorar algumas curvas e piorar Cd ou relações cruzadas; preservar o pool não equivale a preservar todos os variogramas.')
 elif k==7:
  cd=pd.read_csv(OUT/'05_tabela.csv');diag=pd.read_csv(OUT/'03_tabela.csv');floor=pd.read_csv(OUT/'04_tabela.csv');df=cd.merge(diag,on=['config','peso','nn']).merge(floor[['config','excesso_sobre_minimo','custo_backbone_medio']],on='config');fig,axs=plt.subplots(1,2,figsize=(12,5))
  for nn,g in df.groupby('nn'):
   g=g.sort_values('peso')
   axs[0].plot(g.MAE_Ni,g.MAE_Cd,'o-',label=f'NN {nn}');axs[1].plot(g.MAE_Zn,g.MAE_Cd,'o-',label=f'NN {nn}')
   for _,r in g.iterrows():axs[0].annotate(f'{r.peso:g}',(r.MAE_Ni,r.MAE_Cd),fontsize=8);axs[1].annotate(f'{r.peso:g}',(r.MAE_Zn,r.MAE_Cd),fontsize=8)
  axs[0].set(xlabel='MAE Ni médio por realização',ylabel='MAE da mediana Cd');axs[1].set(xlabel='MAE Zn médio por realização',ylabel='MAE da mediana Cd');axs[0].legend();axs[1].legend()
  finish(k,df,fig,'Mapa de compromisso: reduzir erro de observações conhecidas e reduzir erro preditivo de Cd são objetivos distintos. Rótulos indicam lambda. Todos os campos preservam tuplas completas e âncoras, mas Ni/Zn continuam aproximados. Nenhuma configuração é promovida automaticamente a padrão; uma escolha informada exigiria critério pré-fixado e validação interna/repetição mais ampla. O limite de pool e o número de reparos ajudam a decidir o próximo teste.')
 elif k==8:
  with np.load(OUTFIELDS/f'lambda0_nn48_s{SEEDS[0]}.npz') as a:base=a['sims']
  bb,donors,source=inputs(SEEDS[0]);assert np.array_equal(base,source)
  config=CONFIGS[-1];pool=np.vstack([logtrain,logtrain[donors[0]]]);params=dict(p.PARAMS);params['n_candidates']=config['nn'];m=WeightedMST(config['peso'],obs,mu,sd,**params);first=m.simulate(bb[0],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords']);alternative=va.copy();alternative['Cd']=va.Cd.iloc[::-1].to_numpy()*100;m2=WeightedMST(config['peso'],np.log(alternative[['Ni','Zn']].to_numpy()),mu,sd,**params);second=m2.simulate(bb[0],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords']);assert np.array_equal(first,second)
  try:WeightedMST(1,np.log(va[p.V].to_numpy()),mu,sd)
  except ValueError:pass
  else:raise AssertionError('API aceitou Cd')
  df=pd.DataFrame([dict(verificacao='Lambda 0 / NN48 reproduz referência bit a bit',passou=True),dict(verificacao='API de penalização rejeita Cd',passou=True),dict(verificacao='Alterar Cd externo não muda a atribuição',passou=True),dict(verificacao='4.800 atribuições com bijeção, tuplas e pinning',passou=True),dict(verificacao='Código oficial permanece preservado',passou=True)]);diag=pd.read_csv(OUT/'diagnostico_todos.csv');assert len(diag)==4800 and diag['bijeção'].all() and diag.erro_treino_log.max()==0
  fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].plot(base[0,:,0],source[0,:,0],'.');lo=min(base[0,:,0]);hi=max(base[0,:,0]);axs[0].plot([lo,hi],[lo,hi],'k--');axs[0].set(xlabel='Cd da referência v1.1',ylabel='Cd lambda=0 / NN48');axs[1].bar(['Campos verificados','Falhas'],[len(diag),0]);axs[1].set_ylabel('Atribuições')
  finish(k,df,fig,'Valida a compatibilidade com a referência e a separação das entradas. A adaptação não recebe Cd observado no custo e não altera os dados oficiais. A exigência de bijeção continua valendo para o pool completo, incluindo tuplas repetidas como colunas distintas. Integridade é necessária, mas não garante previsão ou estrutura espacial correta. O índice do projeto aponta para este notebook e descreve os protocolos anteriores.')
 else:raise ValueError(k)
