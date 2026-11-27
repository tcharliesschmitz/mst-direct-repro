"""Validação separada dos resultados históricos; caminhos relativos ao projeto."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_validacao_publicacao')
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'resultados';OUT.mkdir(exist_ok=True)
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
w=module('validacao_pesos',ROOT/'jura_pesos_nizn_nn_v1_2/pesos_nizn_v12.py');p=w.p
CONFIGS=w.CONFIGS
TRAIN=ROOT/'jura_real/dados/jura_train.csv'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def folds(df,n):
 # Faixas contíguas em X; desempate por índice. Sem buffer espacial.
 order=np.argsort(df.Xloc.to_numpy(),kind='stable');return [np.sort(x) for x in np.array_split(order,n)]
def savejson(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def prepare(tr,targets):
 # Targets não contém Cd; sentinela NaN apenas para schema legado.
 va=targets[p.XY+['Ni','Zn']].copy();va['Cd']=np.nan;va=va[p.XY+p.V]
 p.load=lambda:(tr.copy(),va.copy())
 return p.setup()
def simulate(st,configs,reps,seeds,tag):
 nt=len(st['tr']);logtrain=np.log(st['tr'][p.V].to_numpy());mu=logtrain[:,1:].mean(0);sd=logtrain[:,1:].std(0);obs=np.log(st['va'][['Ni','Zn']].to_numpy());allfields={c['id']:[] for c in configs};diagnostics=[]
 for seed in seeds:
  p.REPS=reps;p.SEED=seed;raw,_,_=p.simulate_fft(st);bb,_=p.conditional(raw,st);rng=np.random.default_rng(seed+1)
  for r in range(reps):
   donors=rng.integers(nt,size=len(obs));pool=np.vstack([logtrain,logtrain[donors]])
   for c in configs:
    pars=dict(p.PARAMS);pars['n_candidates']=c['nn'];model=w.WeightedMST(c['peso'],obs,mu,sd,**pars,random_state=r)
    sim=model.simulate(bb[r],pool,pinned=(np.arange(nt),np.arange(nt)),coords=st['coords'])
    assert np.array_equal(sim,pool[model.assignment_]);assert len(np.unique(model.assignment_))==len(pool);assert np.array_equal(sim[:nt],logtrain)
    allfields[c['id']].append(np.exp(sim[nt:]));diagnostics.append({'fase':tag,'seed':seed,'rep':r,'config':c['id'],'bijeção':True,'ancoras_exatas':True})
 print(tag,'concluído',len(diagnostics),'campos',flush=True)
 return {k:np.array(v) for k,v in allfields.items()},diagnostics

def run_jura():
 tr=pd.read_csv(TRAIN)[p.XY+p.V];baseline=CONFIGS[0];rows=[];outerrows=[];predrows=[];diags=[];splitlog=[]
 protocol={'version':1,'training_sha256':digest(TRAIN),'outer':'3 faixas contíguas em X, sem buffer','inner':'2 faixas contíguas em X dentro do treino externo','selection':'menor MAE Cd da mediana, ponderado pelo número de pontos internos; desempate ordem da grade','configs':CONFIGS,'inner_and_outer_reps':100,'inner_and_outer_seeds':[20261101],'final_seeds':list(range(20261111,20261121)),'final_reps_per_seed':200,'external_status':'100 alvos já examinados historicamente; reavaliação retrospectiva, não teste cego','leakage_rule':'Cada LMC, escala e pool são reajustados somente no subtreino; Cd alvo ausente na API de geração.'}
 savejson('protocolo_antes_resultados.json',protocol)
 outer=folds(tr,3)
 for o,hold in enumerate(outer):
  keep=np.setdiff1d(np.arange(len(tr)),hold);trainouter=tr.iloc[keep].reset_index(drop=True);inner=folds(trainouter,2)
  local=[]
  for k,val in enumerate(inner):
   trainidx=np.setdiff1d(np.arange(len(trainouter)),val);sub=trainouter.iloc[trainidx].reset_index(drop=True);target=trainouter.iloc[val].reset_index(drop=True)
   splitlog.append(dict(outer=o,inner=k,treino=keep[trainidx].tolist(),validacao=keep[val].tolist(),teste_externo=hold.tolist()))
   assert not(set(keep[trainidx])&set(hold)) and not(set(keep[val])&set(hold))
   st=prepare(sub,target);fields,d=simulate(st,CONFIGS,100,[20261101],f'outer{o}_inner{k}');diags+=d
   for c in CONFIGS:
    metric=p.metrics(fields[c['id']][:,:,0],target.Cd.to_numpy());row=dict(outer=o,inner=k,config=c['id'],n=len(target),**metric);rows.append(row);local.append(row)
   pd.DataFrame(rows).to_csv(OUT/'selecao_interna.csv',index=False)
  scores={c['id']:sum(r['MAE_mediana']*r['n'] for r in local if r['config']==c['id'])/sum(r['n'] for r in local if r['config']==c['id']) for c in CONFIGS}
  chosen=min(CONFIGS,key=lambda c:scores[c['id']]);savejson(f'escolha_outer{o}_antes_teste.json',dict(config=chosen,scores=scores))
  target=tr.iloc[hold].reset_index(drop=True);st=prepare(trainouter,target);cfg=[baseline] if chosen['id']==baseline['id'] else [baseline,chosen];fields,d=simulate(st,cfg,100,[20261101],f'outer{o}_teste');diags+=d
  for label,c in [('MST padrão',baseline),('MST seleção interna',chosen)]:
   s=fields[c['id']][:,:,0];metric=p.metrics(s,target.Cd.to_numpy());outerrows.append(dict(outer=o,metodo=label,config=c['id'],n=len(target),**metric))
   for ix,y,pr in zip(hold,target.Cd,np.median(s,axis=0)):predrows.append(dict(indice=int(ix),outer=o,metodo=label,Cd=float(y),pred=float(pr)))
  pd.DataFrame(outerrows).to_csv(OUT/'metricas_outer.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'predicoes_outer.csv',index=False)
 # Peso final escolhido SOMENTE por resultados internos dos 259, antes de ler arquivo externo.
 scores={c['id']:sum(r['MAE_mediana']*r['n'] for r in rows if r['config']==c['id'])/sum(r['n'] for r in rows if r['config']==c['id']) for c in CONFIGS};chosen=min(CONFIGS,key=lambda c:scores[c['id']])
 savejson('configuracao_final_congelada.json',dict(config=chosen,scores=scores,external_file_read=False))
 va=pd.read_csv(ROOT/'jura_real/dados/jura_validation.csv')[p.XY+p.V];st=prepare(tr,va)
 # Prova estrutural: Cd ausente e impossível de entrar via dataframe do estado.
 assert st['va'].Cd.isna().all()
 cfg=[baseline] if chosen['id']==baseline['id'] else [baseline,chosen];final=[];combined={c['id']:[] for c in cfg}
 for seed in protocol['final_seeds']:
  fs,d=simulate(st,cfg,200,[seed],f'final_{seed}');diags+=d
  np.savez_compressed(OUT/f'final_{seed}.npz',**fs)
  for label,c in [('MST padrão',baseline),('MST seleção interna',chosen)]:final.append(dict(seed=seed,metodo=label,config=c['id'],**p.metrics(fs[c['id']][:,:,0],va.Cd.to_numpy())))
  for c in cfg:combined[c['id']].append(fs[c['id']])
  pd.DataFrame(final).to_csv(OUT/'metricas_jura_externo.csv',index=False)
 pooled=[]
 for label,c in [('MST padrão',baseline),('MST seleção interna',chosen)]:
  s=np.concatenate(combined[c['id']]);pooled.append(dict(metodo=label,config=c['id'],realizacoes=len(s),**p.metrics(s[:,:,0],va.Cd.to_numpy())))
 pd.DataFrame(pooled).to_csv(OUT/'metricas_ensemble_2000.csv',index=False)
 pd.DataFrame(diags).to_csv(OUT/'integridade_atribuicoes.csv',index=False);savejson('particoes.json',splitlog)
 # Incerteza por blocos: três blocos não sustentam teste de significância.
 summary=[]
 df=pd.DataFrame(predrows)
 for label,g in df.groupby('metodo'):
  err=g.pred-g.Cd;summary.append(dict(metodo=label,MAE=float(abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),pontos=len(g)))
 pd.DataFrame(summary).to_csv(OUT/'resumo_nested.csv',index=False)
 savejson('auditoria_jura.json',dict(todos_pontos_treino_avaliados_uma_vez=len(df[df.metodo=='MST padrão'])==259,campos_verificados=len(diags),Cd_alvo_ausente_na_geracao=True,arquivo_externo_lido_so_apos_escolha=True,teste_externo_historicamente_visto=True,modelo_reajustado_em_cada_subtreino=True))

if __name__=='__main__':
 with threadpool_limits(limits=2):run_jura()
