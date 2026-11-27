import os,json,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_benchmark_comum')
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance,ks_2samp,spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.spatial import cKDTree
from scipy.ndimage import label
from IPython.display import display,Markdown
import motor as m
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
HERE=m.HERE;OUT=m.OUT;V=m.V;PAIRS=[(a,b) for a in range(6) for b in range(a+1,6)]
TITLES=['Dados, verdade independente e protocolo','Crossplots e distribuição conjunta','Marginais: histogramas, KS e Wasserstein','Dependência multivariada e histogramas bivariados','Variogramas diretos — incondicional','Variogramas cruzados — incondicional','Campos condicionais: verdade, realização e mediana','Reprodução dos pontos condicionantes','Variogramas diretos — condicional','Variogramas cruzados — condicional','Previsão pontual nos nós livres','Calibração, largura e CRPS nos nós livres','Desempenho por distância aos condicionantes','Extremos, conectividade e caudas conjuntas','Continuidade direcional','Tempo e sensibilidade ao tamanho da grade','Escala original: incondicional 200×200','Escala original: condicional 100×100, 200 pontos']
NOTES=[
'Três verdades independentes geradas por Cholesky de campos gaussianos latentes e transformação não linear explícita, sem usar MST ou transformações ajustadas. Distribuição-alvo conhecida por 4.096 amostras iid independentes; 48 posições condicionantes fixas, 352 nós livres. Todos recebem essa mesma informação e alcance latente prescrito 5. É simulação com distribuição conhecida, não inferência da distribuição somente a partir de 48 pontos. Não usa valores livres da verdade no pool ou no ajuste.',
'Exibe a primeira realização incondicional pré-fixada, nunca a melhor. Mesmas posições dos painéis, limites, amostras de referência e cores em todos os notebooks. A referência cinza contém relações não lineares que não se resumem à correlação de Pearson. A diagonal mostra histogramas normalizados.',
'Métricas por realização e variável, resumidas em média/desvio entre 60 campos. Wasserstein dividido pelo desvio da distribuição de referência; KS é a distância entre CDFs, sem interpretar p-valores iid em dados espacialmente dependentes. Histogramas agregam todas as realizações; métricas não são calculadas apenas nesse agregado. Bins e limites compartilhados entre métodos.',
'JS em bits entre histogramas bivariados da referência e de cada realização, com bins de quantis pré-fixados pela referência e caudas abertas, depois média entre campos. Mesmo orçamento de 400 tuplas por realização. Descreve marginais e dependências em conjunto; não é um teste isolado de cópula. Correlações Pearson/Spearman são diagnósticos complementares, não resumo completo das formas não lineares.',
'Variogramas calculados por realização sobre os mesmos 6.000 pares sorteados e oito lags. Escala normalizada pelo desvio da referência, não pelo de cada campo. Linha preta: média das três verdades independentes; faixa azul: quantis 5–95% entre realizações incondicionais, sem tratá-la como intervalo de confiança da curva verdadeira. O alcance prescrito vale no espaço gaussiano; transformações não lineares mudam variogramas na escala das propriedades.',
'Todos os 15 pares de variáveis, calculados nos mesmos pares espaciais usados nos variogramas diretos. Linha de referência e bandas têm o mesmo significado da etapa anterior. Preservar crossplots não garante reprodução espacial cruzada. Os métodos recebem o mesmo alcance prescrito, mas representam as dependências multivariadas de formas diferentes.',
'Primeira verdade, primeira realização e mediana do ensemble na mesma grade; posições condicionantes marcadas. A realização não foi escolhida pelo erro. A mediana é previsão pontual e NÃO é apresentada como campo simulado. Métricas da tabela excluem condicionantes e agregam as três verdades.',
'Imposição de dados completos, não condicionamento parcial. Todos os adaptadores preservam os valores observados. Para transformações empíricas, impomos explicitamente os valores conhecidos depois da transformação inversa, inclusive se houver truncamento de cauda; o erro anterior a essa imposição é registrado. Isso não deve ser confundido com garantir continuidade perfeita ao redor das âncoras. No MST, tuplas são fixadas por pinning.',
'Variogramas das realizações condicionadas, não do mapa mediano. A referência agrega três verdades; as bandas juntam os 180 campos e incluem variação entre referências e sementes. Curvas são diagnósticos no domínio completo, inclusive pontos conhecidos; a previsão e cobertura são avaliadas separadamente apenas nos nós livres.',
'Variogramas cruzados condicionais dos 15 pares, com limites e posição de painéis idênticos entre métodos. Não se pressupõe que um modelo gaussiano preserve dependências não lineares, nem que uma permutação preserve o modelo espacial após transporte. Os dados condicionantes e os pares amostrados são os mesmos.',
'MAE/RMSE/viés da mediana preditiva nos 352 nós livres, por referência e variável; normalização pelos desvios da distribuição-alvo. Cada verdade recebe peso igual. Reutilizar os mesmos pontos em 60 realizações não aumenta o número de observações independentes. Nenhum parâmetro foi escolhido com esses erros.',
'Intervalos de 50/80/90/95%, larguras e CRPS empírico por propriedade e referência, somente nós livres. Não atribuímos intervalos binomiais iid à cobertura espacial. Apenas três verdades: resultado exploratório, não calibração universal. A distribuição de referência foi fornecida aos métodos por desenho.',
'Distância ao condicionante mais próximo, agrupada por ordem em quatro grupos com a mesma contagem. Empates de distância são desempatados pela ordem fixa das coordenadas; os limites podem coincidir entre grupos. Mesmo grupo de nós em todos os métodos. A curva de erro usa RMSE normalizado e a cobertura usa intervalos de 90%. Faixas com diferenças de estrutura podem ter comportamento não monotônico; não é um experimento causal sobre distância.',
'Cauda superior definida pelo quantil 95% da distribuição-alvo, sem mudar limiar por método. Conectividade por quatro vizinhos em 2D; comparamos componentes e maior componente entre as verdades e os campos condicionados. Caudas conjuntas usam as realizações incondicionais. Frequências globais e conectividade espacial são propriedades distintas.',
'Direções horizontal, vertical e diagonal, usando pares exatos da grade e offsets 1 a 5. Curvas normalizadas e resumidas entre seis propriedades. A tabela mantém as seis separadas. A verdade geradora é isotrópica em distribuição, mas campos finitos exibem flutuações direcionais. Não inferimos anisotropia sistemática de uma diferença visual.',
'Tempos locais, não comparação de complexidade assintótica. Curva em grades 16/20/24 com três realizações cada e transformação ajustada reutilizada; inclui geometria nova quando necessária. A grade 20 já tinha geometria em cache, e SGS prepara caminhos novos. Tabela separa ajuste de transformação, preparação e tempo total. Tempo inclui características desta implementação Python; não é desempenho de softwares industriais. Não se reporta memória de arrays como se fosse RSS.',
'Reproduz o tamanho 200×200 do teste inicial, agora com todos os métodos e distribuição/alcance comuns. Uma realização por método, seed pré-fixada: teste estrutural e de custo, não calibração de ensemble. Limites de mapas compartilhados. A verdade de comparação espacial é independente e gerada por FFT de latentes com a mesma lei, sem MST. Transformações diferentes induzem variogramas diferentes; não se exige igualdade por construção.',
'Grade 100×100 e os mesmos 200 pontos completos para todos. Uma realização por método; a tabela mede erro do CAMPO ÚNICO e variograma, não erro de um estimador médio nem cobertura. A verdade é independente e não entra no pool, salvo os condicionantes observados. Duas linhas: verdade e realização. Sem reamostragem ou seleção posterior de campos.'
]
_CACHE=None
def cache():
 global _CACHE
 if _CACHE is not None:return _CACHE
 dat=m.data();models={}
 for name in m.METHODS:
  with np.load(OUT/name/'campos.npz') as a:u=a['unconditional'];c=a['conditional']
  with np.load(OUT/name/'variogramas.npz') as a:g={k:a[k] for k in a.files}
  with np.load(OUT/name/'campos_grandes.npz') as a:large={k:a[k] for k in a.files}
  models[name]=dict(u=u,c=c,g=g,large=large,meta=json.loads((OUT/name/'manifesto_metodo.json').read_text()))
 with np.load(HERE/'dados_grandes_congelados.npz') as a:large_data={k:a[k] for k in a.files}
 limits=[]
 for j in range(6):
  arrays=[dat['cloud'][:,j],dat['truth'][:,:,j].ravel(),large_data['truth100'][:,j],large_data['truth200'][:,j]]
  for model in models.values():arrays.extend([model['u'][:,:,j].ravel(),model['c'][:,:,:,j].ravel(),model['large']['unconditional200'][:,j],model['large']['conditional100'][:,j]])
  lo=min(a.min() for a in arrays);hi=max(a.max() for a in arrays);pad=.025*max(hi-lo,1);limits.append([float(lo-pad),float(hi+pad)])
 _CACHE=dict(dat=dat,models=models,large_data=large_data,limits=limits);return _CACHE

def vtable(curves,target,dat,pairs):
 rows=[]
 for a,b in pairs:
  for lag in range(8):
   values=curves[:,lag,a,b];rows.append(dict(par=V[a]+'/'+V[b],lag=float(dat['lags'][lag]),pares=int(np.sum(dat['pair_bin']==lag)),referencia=float(target[:,lag,a,b].mean()),mediana=float(np.median(values)),q05=float(np.quantile(values,.05)),q95=float(np.quantile(values,.95))))
 return pd.DataFrame(rows)
def large_vario(field,truth,n,sd):
 rng=np.random.default_rng(99101+n);base=rng.integers(0,n,size=(16000,2));off=rng.integers(-10,11,size=(16000,2));other=base+off;valid=(other>=0).all(1)&(other<n).all(1)&(np.linalg.norm(off,axis=1)>0);base=base[valid];other=other[valid];off=off[valid];i=base[:,0]*n+base[:,1];j=other[:,0]*n+other[:,1];dist=np.linalg.norm(off,axis=1);binid=np.minimum((dist/2).astype(int),6);g=[];t=[]
 for q in range(7):
  mask=binid==q;g.append(.5*np.mean(((field[i[mask]]-field[j[mask]])/sd)**2,axis=0));t.append(.5*np.mean(((truth[i[mask]]-truth[j[mask]])/sd)**2,axis=0))
 return np.sqrt(np.mean((np.array(g)-np.array(t))**2,axis=0))

def prepare_tables():
 C=cache();dat=C['dat'];cloud=dat['cloud'];sd=cloud.std(0);free=dat['free'];tables={};edges=[]
 for j in range(6):e=np.quantile(cloud[:,j],np.linspace(0,1,21));e[0]=-np.inf;e[-1]=np.inf;edges.append(np.unique(e))
 P={pair:np.histogram2d(cloud[:,pair[0]],cloud[:,pair[1]],bins=[edges[pair[0]],edges[pair[1]]])[0].ravel() for pair in PAIRS}
 distance=cKDTree(dat['coords'][dat['anchors']]).query(dat['coords'][free])[0];groups=np.array_split(np.argsort(distance,kind='stable'),4);dbin=np.empty(len(distance),dtype=int)
 for q,group in enumerate(groups):dbin[group]=q
 for name,model in C['models'].items():
  u=model['u'];cond=model['c'];g=model['g'];table={};table[1]=pd.DataFrame([dict(item='Método',valor=m.LABELS[name]),dict(item='Grade principal',valor='20×20, seis variáveis'),dict(item='Referências independentes',valor=3),dict(item='Realizações incondicionais',valor=60),dict(item='Realizações condicionais por referência',valor=60),dict(item='Âncoras completas',valor=48),dict(item='Nós livres',valor=352),dict(item='Distribuição conhecida iid',valor=4096),dict(item='Alcance latente prescrito',valor=5)])
  rows=[];up=u.reshape(-1,6)
  for a,b in PAIRS:rows.append(dict(par=V[a]+'/'+V[b],Pearson_referencia=np.corrcoef(cloud[:,a],cloud[:,b])[0,1],Pearson_simulado=np.corrcoef(up[:,a],up[:,b])[0,1],Spearman_referencia=spearmanr(cloud[:,a],cloud[:,b]).statistic,Spearman_simulado=spearmanr(up[:,a],up[:,b]).statistic))
  table[2]=pd.DataFrame(rows);rows=[]
  for j in range(6):
   w=[wasserstein_distance(s[:,j],cloud[:,j])/sd[j] for s in u];ks=[ks_2samp(s[:,j],cloud[:,j],method='asymp').statistic for s in u];rows.append(dict(variavel=V[j],W1_normalizado_media=np.mean(w),W1_normalizado_desvio=np.std(w,ddof=1),KS_media=np.mean(ks),KS_desvio=np.std(ks,ddof=1)))
  table[3]=pd.DataFrame(rows);rows=[]
  for a,b in PAIRS:
   vals=[jensenshannon(P[(a,b)],np.histogram2d(s[:,a],s[:,b],bins=[edges[a],edges[b]])[0].ravel(),base=2)**2 for s in u];rows.append(dict(par=V[a]+'/'+V[b],JS_bits_media=np.mean(vals),JS_bits_desvio=np.std(vals,ddof=1)))
  table[4]=pd.DataFrame(rows);table[5]=vtable(g['unconditional'],g['truth'],dat,[(j,j) for j in range(6)]);table[6]=vtable(g['unconditional'],g['truth'],dat,PAIRS)
  pred=np.median(cond,axis=1);rows=[]
  for ref in range(3):
   for j in range(6):
    err=pred[ref,free,j]-dat['truth'][ref,free,j];rows.append(dict(referencia=ref,variavel=V[j],MAE=float(abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),MAE_normalizado=float(abs(err).mean()/sd[j]),RMSE_normalizado=float(np.sqrt(np.mean(err**2))/sd[j]),vies_normalizado=float(err.mean()/sd[j])))
  table[11]=pd.DataFrame(rows);table[7]=table[11].groupby('variavel',as_index=False).mean(numeric_only=True).drop(columns='referencia');diag=pd.read_csv(OUT/name/'execucao.csv');rows=[]
  for ref in range(3):
   d=diag[(diag.scenario=='condicional')&(diag.referencia==ref)];rows.append(dict(referencia=ref,ancoras=48,realizacoes=60,erro_antes_imposicao_max=d.anchor_error_before_hard_imposition.max(),erro_final_max=float(abs(cond[ref][:,dat['anchors']]-dat['truth'][ref,dat['anchors']]).max())))
  table[8]=pd.DataFrame(rows);cg=g['conditional'].reshape(-1,8,6,6);table[9]=vtable(cg,g['truth'],dat,[(j,j) for j in range(6)]);table[10]=vtable(cg,g['truth'],dat,PAIRS);rows=[]
  for ref in range(3):
   s=cond[ref][:,free];y=dat['truth'][ref,free];cr=m.crps(s,y).mean(0)/sd
   for nominal in [.5,.8,.9,.95]:
    lo,hi=np.quantile(s,[(1-nominal)/2,1-(1-nominal)/2],axis=0)
    for j in range(6):rows.append(dict(referencia=ref,variavel=V[j],nominal=nominal,cobertura=float(((y[:,j]>=lo[:,j])&(y[:,j]<=hi[:,j])).mean()),largura_normalizada=float((hi[:,j]-lo[:,j]).mean()/sd[j]),CRPS_normalizado=float(cr[j])))
  table[12]=pd.DataFrame(rows);rows=[]
  for q in range(4):
   mask=dbin==q
   for j in range(6):
    errs=[];covers=[]
    for ref in range(3):
     y=dat['truth'][ref,free,j][mask];s=cond[ref][:,free,j][:,mask];lo,hi=np.quantile(s,[.05,.95],axis=0);errs.extend(((np.median(s,axis=0)-y)/sd[j])**2);covers.extend((y>=lo)&(y<=hi))
    rows.append(dict(faixa=q+1,variavel=V[j],nos=int(mask.sum()),dist_min=distance[mask].min(),dist_max=distance[mask].max(),RMSE_normalizado=np.sqrt(np.mean(errs)),cobertura90=np.mean(covers)))
  table[13]=pd.DataFrame(rows);rows=[];threshold=np.quantile(cloud,.95,axis=0)
  for j in range(6):
   def components(field):
    mask=(field[:,j]>=threshold[j]).reshape(m.N,m.N);lab,n=label(mask);counts=np.bincount(lab.ravel())[1:];return n,counts.max() if len(counts) else 0,mask.mean()
   refs=np.array([components(y) for y in dat['truth']]);sims=np.array([components(s) for s in cond.reshape(-1,m.N*m.N,6)]);rows.append(dict(tipo='conectividade',item=V[j],componentes_referencia=refs[:,0].mean(),componentes_simulados=sims[:,0].mean(),maior_referencia=refs[:,1].mean(),maior_simulado=sims[:,1].mean(),fracao_referencia=refs[:,2].mean(),fracao_simulada=sims[:,2].mean()))
  for a,b in PAIRS:rows.append(dict(tipo='cauda_conjunta',item=V[a]+'/'+V[b],fracao_referencia=np.mean((cloud[:,a]>=threshold[a])&(cloud[:,b]>=threshold[b])),fracao_simulada=np.mean((u[:,:,a]>=threshold[a])&(u[:,:,b]>=threshold[b]))))
  table[14]=pd.DataFrame(rows);rows=[];grid=cond.reshape(-1,m.N,m.N,6)/sd;truth=dat['truth'].reshape(3,m.N,m.N,6)/sd
  for direction,dy,dx in [('Horizontal',0,1),('Vertical',1,0),('Diagonal',1,1)]:
   for h in range(1,6):
    ya=slice(None,-h*dy) if dy else slice(None);yb=slice(h*dy,None) if dy else slice(None);xa=slice(None,-h*dx) if dx else slice(None);xb=slice(h*dx,None) if dx else slice(None);gs=.5*np.mean((grid[:,ya,xa]-grid[:,yb,xb])**2,axis=(1,2));gt=.5*np.mean((truth[:,ya,xa]-truth[:,yb,xb])**2,axis=(1,2))
    for j in range(6):rows.append(dict(direcao=direction,lag=h*np.hypot(dx,dy),variavel=V[j],referencia=gt[:,j].mean(),simulado=np.median(gs[:,j])))
  table[15]=pd.DataFrame(rows);scale=pd.read_csv(OUT/name/'escala.csv');meta=model['meta'];scale['fit_transform_seconds']=meta['fit_seconds'];scale['geometry_seconds_total_core']=meta['geometry_and_sgs_plan_seconds'];scale['total_core_plus_scale_seconds']=meta['total_seconds'];table[16]=scale
  big=model['large'];bd=C['large_data'];bigdiag=pd.read_csv(OUT/name/'execucao_grandes.csv');rows=[];var=large_vario(big['unconditional200'],bd['truth200'],200,sd)
  for j in range(6):rows.append(dict(variavel=V[j],W1_normalizado=wasserstein_distance(big['unconditional200'][:,j],cloud[:,j])/sd[j],KS=ks_2samp(big['unconditional200'][:,j],cloud[:,j],method='asymp').statistic,RMSE_variograma_normalizado=var[j],seconds=float(bigdiag.iloc[0].seconds_including_new_geometry),nos=40000))
  table[17]=pd.DataFrame(rows);rows=[];var=large_vario(big['conditional100'],bd['truth100'],100,sd);freebig=np.ones(10000,dtype=bool);freebig[bd['anchors100']]=False
  for j in range(6):
   err=(big['conditional100'][freebig,j]-bd['truth100'][freebig,j])/sd[j];rows.append(dict(variavel=V[j],MAE_campo_unico_normalizado=float(abs(err).mean()),RMSE_campo_unico_normalizado=float(np.sqrt(np.mean(err**2))),erro_ancoras_max=float(abs(big['conditional100'][bd['anchors100'],j]-bd['truth100'][bd['anchors100'],j]).max()),RMSE_variograma_normalizado=var[j],seconds=float(bigdiag.iloc[1].seconds_including_new_geometry),nos=10000,ancoras=200))
  table[18]=pd.DataFrame(rows)
  for k,df in table.items():df.to_csv(OUT/name/f'{k:02d}_tabela.csv',index=False)
  tables[name]=table
 (HERE/'config_visual.json').write_text(json.dumps({'limites_variaveis':C['limits'],'dimensoes_png_fixas':True,'mesmos_bins_histogramas':True,'indices_exibidos':'primeira referencia e primeira realizacao','ordem_testes':TITLES},indent=2))
 return tables

_TABLES=None
def tables():
 global _TABLES
 if _TABLES is None:_TABLES={name:{k:pd.read_csv(OUT/name/f'{k:02d}_tabela.csv') for k in range(1,19)} for name in m.METHODS}
 return _TABLES

def shared_max(test,column,default=1.):
 vals=[t[test][column].max() for t in tables().values()];return max(max(vals)*1.12,default*1e-3)
def vario_limits(test,par):
 lows=[];highs=[]
 for name,t in tables().items():
  g=t[test][t[test].par==par];lows.extend(g[['referencia','q05']].to_numpy().ravel());highs.extend(g[['referencia','q95']].to_numpy().ravel())
 lo=min(0.,min(lows));hi=max(highs);pad=.08*max(hi-lo,.01);return lo-pad,hi+pad

def figure(name,k):
 C=cache();dat=C['dat'];model=C['models'][name];df=tables()[name][k];sd=dat['cloud'].std(0);limits=C['limits'];u=model['u'];cond=model['c']
 if k==1:
  fig,axs=plt.subplots(1,2,figsize=(12,6));axs[0].scatter(dat['coords'][:,0],dat['coords'][:,1],s=9,c='#b9c3cc');a=dat['coords'][dat['anchors']];axs[0].scatter(a[:,0],a[:,1],s=25,c='red',label='48 condicionantes completos');axs[0].legend(fontsize=8);axs[0].set(xlabel='X',ylabel='Y',aspect='equal');axs[1].axis('off');texts=['Verdade: processo latente + relações não lineares','Distribuição conhecida: 4.096 tuplas iid','3 verdades × 60 realizações condicionais','60 realizações incondicionais','Grade 20×20; cenários grandes 100×100 / 200×200']
  for y,text in zip(np.linspace(.9,.1,len(texts)),texts):axs[1].text(.01,y,text,fontsize=10)
 elif k==2:
  fig,axs=plt.subplots(6,6,figsize=(13,13));ref=dat['cloud'];sim=u[0]
  for a in range(6):
   for b in range(6):
    ax=axs[a,b]
    if a==b:
     bins=np.linspace(*limits[b],31);ax.hist(ref[:,b],bins=bins,density=True,histtype='step',color='black',lw=1);ax.hist(sim[:,b],bins=bins,density=True,alpha=.5,color='#2e81ac');ax.set_xlim(limits[b])
     ymax=0
     for mod in C['models'].values():ymax=max(ymax,np.histogram(mod['u'][0,:,b],bins=bins,density=True)[0].max())
     ymax=max(ymax,np.histogram(ref[:,b],bins=bins,density=True)[0].max());ax.set_ylim(0,ymax*1.1)
    else:ax.scatter(ref[::4,b],ref[::4,a],s=2,c='#bbbbbb',alpha=.18);ax.scatter(sim[:,b],sim[:,a],s=3,c='#2e81ac',alpha=.35);ax.set(xlim=limits[b],ylim=limits[a])
    if a==5:ax.set_xlabel(V[b])
    if b==0:ax.set_ylabel(V[a])
    ax.tick_params(labelsize=5)
 elif k==3:
  fig,axs=plt.subplots(2,3,figsize=(12,7))
  for j,ax in enumerate(axs.ravel()):
   bins=np.linspace(*limits[j],31);ax.hist(dat['cloud'][:,j],bins=bins,density=True,histtype='step',color='black',lw=1.5,label='Referência iid');ax.hist(u[:,:,j].ravel(),bins=bins,density=True,alpha=.45,color='#2e81ac',label='Ensemble');ymax=max(np.histogram(dat['cloud'][:,j],bins=bins,density=True)[0].max(),max(np.histogram(mod['u'][:,:,j].ravel(),bins=bins,density=True)[0].max() for mod in C['models'].values()));ax.set(title=V[j],xlabel='Valor',ylabel='Densidade',xlim=limits[j],ylim=(0,ymax*1.1));ax.legend(fontsize=7)
 elif k==4:
  fig,ax=plt.subplots(figsize=(9,7));matrix=np.full((6,6),np.nan)
  for _,r in df.iterrows():a,b=[V.index(x) for x in r.par.split('/')];matrix[a,b]=matrix[b,a]=r.JS_bits_media
  im=ax.imshow(matrix,vmin=0,vmax=shared_max(4,'JS_bits_media'),cmap='magma');ax.set(xticks=range(6),yticks=range(6),xticklabels=V,yticklabels=V)
  for a in range(6):
   for b in range(6):ax.text(b,a,'—' if a==b else f'{matrix[a,b]:.3f}',ha='center',va='center',color='black' if a==b else 'white')
  fig.colorbar(im,ax=ax,label='JS média (bits), menor é melhor')
 elif k in [5,6,9,10]:
  pairs=[(j,j) for j in range(6)] if k in [5,9] else PAIRS;fig,axs=plt.subplots(2,3,figsize=(12,7)) if len(pairs)==6 else plt.subplots(3,5,figsize=(16,10))
  for ax,(a,b) in zip(axs.ravel(),pairs):
   par=V[a]+'/'+V[b];g=df[df.par==par];ax.plot(g.lag,g.referencia,'k--',label='Referência (média 3)');ax.plot(g.lag,g.mediana,color='#2e81ac',label='Mediana das realizações');ax.fill_between(g.lag,g.q05,g.q95,alpha=.2,color='#2e81ac',label='5–95%');ax.set(title=par,xlabel='Distância',ylabel='Semivariância normalizada',ylim=vario_limits(k,par));ax.legend(fontsize=5)
 elif k==7:
  fig,axs=plt.subplots(3,6,figsize=(18,9));fields=[dat['truth'][0],cond[0,0],np.median(cond[0],axis=0)];labels=['Verdade','Realização 1','Mediana (não é realização)']
  for row in range(3):
   for j in range(6):
    im=axs[row,j].imshow(fields[row][:,j].reshape(m.N,m.N),origin='lower',vmin=limits[j][0],vmax=limits[j][1],cmap='viridis');a=dat['coords'][dat['anchors']];axs[row,j].scatter(a[:,0],a[:,1],s=4,c='red');axs[row,j].set_title(V[j]+' — '+labels[row],fontsize=8);fig.colorbar(im,ax=axs[row,j],shrink=.7)
 elif k==8:
  fig,axs=plt.subplots(1,2,figsize=(12,5));values=df[['erro_antes_imposicao_max','erro_final_max']].to_numpy();mx=max(shared_max(8,'erro_antes_imposicao_max'),1e-12);im=axs[0].imshow(values,vmin=0,vmax=mx,cmap='Blues',aspect='auto');axs[0].set(xticks=[0,1],xticklabels=['Antes da imposição explícita','Final'],yticks=[0,1,2],yticklabels=['Ref. 1','Ref. 2','Ref. 3'])
  for i in range(3):
   for j in range(2):axs[0].text(j,i,f'{values[i,j]:.2g}',ha='center',va='center',color='black')
  fig.colorbar(im,ax=axs[0],label='Erro máximo nas âncoras');axs[1].axis('off');axs[1].text(.08,.65,'48 pontos × 6 variáveis × 3 referências',fontsize=12);axs[1].text(.08,.4,'Valor final imposto = valor observado',fontsize=12);axs[1].text(.08,.2,'Não é avaliação de previsão nos nós livres.',fontsize=10)
 elif k==11:
  fig,axs=plt.subplots(1,2,figsize=(12,5));g=df.groupby('variavel').mean(numeric_only=True).reindex(V)
  for ax,col,title in zip(axs,['MAE_normalizado','RMSE_normalizado'],['MAE normalizado','RMSE normalizado']):ax.bar(V,g[col],color='#2e81ac');ax.set(ylabel=title,ylim=(0,shared_max(11,col)))
 elif k==12:
  fig,axs=plt.subplots(1,2,figsize=(12,5));g=df.groupby(['variavel','nominal'],as_index=False).mean(numeric_only=True)
  for v,sub in g.groupby('variavel'):axs[0].plot(sub.nominal,sub.cobertura,'o-',label=v)
  axs[0].plot([0,1],[0,1],'k--');axs[0].set(xlabel='Cobertura nominal',ylabel='Cobertura observada',xlim=(0,1),ylim=(0,1));axs[0].legend(fontsize=7);sub=g[g.nominal==.9].set_index('variavel').reindex(V);axs[1].bar(V,sub.largura_normalizada,color='#2e81ac');axs[1].set(ylabel='Largura média normalizada (90%)',ylim=(0,shared_max(12,'largura_normalizada')))
 elif k==13:
  fig,axs=plt.subplots(1,2,figsize=(12,5));g=df.groupby('faixa').mean(numeric_only=True);axs[0].plot(g.index,g.RMSE_normalizado,'o-');axs[0].set(xlabel='Quartil de distância',ylabel='RMSE normalizado médio',ylim=(0,shared_max(13,'RMSE_normalizado')));axs[1].plot(g.index,g.cobertura90,'o-');axs[1].axhline(.9,color='k',ls='--');axs[1].set(xlabel='Quartil de distância',ylabel='Cobertura 90% média',ylim=(0,1))
 elif k==14:
  fig,axs=plt.subplots(1,2,figsize=(12,5));g=df[df.tipo=='conectividade'];x=np.arange(6);axs[0].bar(x-.18,g.maior_referencia,width=.36,label='Verdade');axs[0].bar(x+.18,g.maior_simulado,width=.36,label='Simulações');axs[0].set(xticks=x,xticklabels=V,ylabel='Maior componente da cauda',ylim=(0,max(shared_max(14,'maior_simulado'),shared_max(14,'maior_referencia'))));axs[0].legend();g=df[df.tipo=='cauda_conjunta'];axs[1].scatter(g.fracao_referencia,g.fracao_simulada,s=25);mx=max(shared_max(14,'fracao_referencia'),shared_max(14,'fracao_simulada'));axs[1].plot([0,mx],[0,mx],'k--');axs[1].set(xlabel='Cauda conjunta na referência iid',ylabel='Cauda conjunta simulada',xlim=(0,mx),ylim=(0,mx))
 elif k==15:
  fig,axs=plt.subplots(1,3,figsize=(14,4))
  for ax,(direction,g) in zip(axs,df.groupby('direcao',sort=True)):
   t=g.groupby('lag').mean(numeric_only=True);ax.plot(t.index,t.referencia,'k--',label='Referência');ax.plot(t.index,t.simulado,'o-',label='Mediana simulações');ax.set(title=direction,xlabel='Distância',ylabel='Semivariância média normalizada',ylim=(0,max(shared_max(15,'referencia'),shared_max(15,'simulado'))));ax.legend(fontsize=7)
 elif k==16:
  fig,axs=plt.subplots(1,2,figsize=(12,5));axs[0].plot(df.nos,df.seconds_per_realization,'o-');maxy=shared_max(16,'seconds_per_realization');miny=min(t[16].seconds_per_realization.min() for t in tables().values());axs[0].set(xlabel='Nós',ylabel='Segundos por realização, log',yscale='log',ylim=(max(miny*.5,1e-6),maxy));stats=model['meta'];axs[1].bar(['Ajuste transformação','Total protocolo + escala'],[stats['fit_seconds'],stats['total_seconds']]);maxy=max(x['meta']['total_seconds'] for x in C['models'].values())*1.1;axs[1].set(ylabel='Segundos, log',yscale='log',ylim=(1e-5,maxy));axs[1].tick_params(axis='x',labelsize=7)
 elif k==17:
  fig,axs=plt.subplots(2,3,figsize=(12,8));field=model['large']['unconditional200']
  for j,ax in enumerate(axs.ravel()):im=ax.imshow(field[:,j].reshape(200,200),origin='lower',vmin=limits[j][0],vmax=limits[j][1],cmap='viridis');ax.set(title=V[j],xlabel='X',ylabel='Y');fig.colorbar(im,ax=ax)
 elif k==18:
  fig,axs=plt.subplots(2,6,figsize=(18,7));fields=[C['large_data']['truth100'],model['large']['conditional100']];anchors=C['large_data']['anchors100']
  for row in range(2):
   for j in range(6):im=axs[row,j].imshow(fields[row][:,j].reshape(100,100),origin='lower',vmin=limits[j][0],vmax=limits[j][1],cmap='viridis');axs[row,j].scatter(anchors%100,anchors//100,s=2,c='red');axs[row,j].set_title(V[j]+' — '+('Verdade' if row==0 else 'Realização única'),fontsize=8);fig.colorbar(im,ax=axs[row,j],shrink=.7)
 else:raise ValueError(k)
 fig.suptitle(f'{k:02d} · {TITLES[k-1]} · {m.LABELS[name]}',fontsize=12);fig.tight_layout(rect=(0,0,1,.97));return fig

def render(name,k):
 df=tables()[name][k];fig=figure(name,k);path=OUT/name
 fig.savefig(path/f'{k:02d}_figura.png',dpi=120);fig.savefig(path/f'{k:02d}_figura.pdf');(path/f'{k:02d}_nota.md').write_text(NOTES[k-1]);display(df);display(fig);display(Markdown(NOTES[k-1]));plt.close(fig)
