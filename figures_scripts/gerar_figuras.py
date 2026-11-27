import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_artigo_figuras')
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'figuras';OUT.mkdir(exist_ok=True)
B=ROOT/'ppmt_pymgt_v1/benchmark';R=B/'resultados';J=ROOT/'validacao_publicacao_v1/resultados';METHODS=['MST','FFTMA','SGS','PPMT'];LABELS=['MST','Cópula + FFT-MA','SGS local','PPMT (pymgt)'];COLORS=['#1764a0','#d07822','#32825b','#a44050']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
def finish(n,fig):
 fig.savefig(OUT/f'figura_{n:02d}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'figura_{n:02d}.pdf',bbox_inches='tight');plt.close(fig)
with np.load(B/'dados_congelados.npz') as a:dat={k:a[k] for k in a.files}
sumdf=pd.read_csv(ROOT/'ppmt_pymgt_v1/resumo_comparativo_pymgt.csv');sumdf.to_csv(HERE/'tabela_sintetico.csv',index=False)
from redesenhar_figura01 import gerar as gerar_fluxo_mst
gerar_fluxo_mst()
# Pares definidos por mecanismo: ramificações, periodicidade e relação quadrática.
pairs=[(0,2,'Ramificações'),(0,3,'Periodicidade'),(0,5,'Relação quadrática')];cloud=dat['cloud'];fields={}
for name in METHODS:
 with np.load(R/name/'campos.npz') as a:fields[name]=a['unconditional'][0]
fig,axes=plt.subplots(3,4,figsize=(12.7,8.3))
for row,(i,j,title) in enumerate(pairs):
 allx=np.concatenate([cloud[:,i]]+[fields[n][:,i] for n in METHODS]);ally=np.concatenate([cloud[:,j]]+[fields[n][:,j] for n in METHODS]);dx=.04*np.ptp(allx);dy=.04*np.ptp(ally)
 for col,(name,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
  ax=axes[row,col];ax.scatter(cloud[:,i],cloud[:,j],s=3,c='#9ca5ad',alpha=.18,rasterized=True);ax.scatter(fields[name][:,i],fields[name][:,j],s=6,c=color,alpha=.65,rasterized=True);ax.set(xlim=(allx.min()-dx,allx.max()+dx),ylim=(ally.min()-dy,ally.max()+dy),xlabel=f'Z{i+1}',ylabel=f'Z{j+1}')
  if row==0:ax.set_title(label)
  if col==0:ax.text(-.22,.5,title,transform=ax.transAxes,rotation=90,ha='right',va='center',fontsize=9)
fig.tight_layout();finish(2,fig)
fig,axes=plt.subplots(2,3,figsize=(12.3,7.0));selected=['Z1/Z1','Z4/Z4','Z1/Z3']
for row,k in enumerate([5,9]):
 for col,par in enumerate(selected):
  ax=axes[row,col];test=k if par.split('/')[0]==par.split('/')[1] else k+1
  for name,label,color in zip(METHODS,LABELS,COLORS):
   d=pd.read_csv(R/name/f'{test:02d}_tabela.csv');d=d[d.par==par];ax.plot(d.lag,d.mediana,'o-',color=color,label=label,ms=3,lw=1.3)
  ax.plot(d.lag,d.referencia,'k--',label='Média de três verdades',lw=1.8);ax.set(xlabel='Distância (unidades da grade)',ylabel='Semivariância normalizada',title=f'{par} · '+('incondicional' if row==0 else 'condicional'))
axes[0,0].legend(fontsize=7);fig.tight_layout();finish(3,fig)
# Índices espaciais derivados, com definição explícita no texto e CSV.
rows=[]
for name in METHODS:
 rec={'metodo':name}
 for k in [5,6,9,10]:
  d=pd.read_csv(R/name/f'{k:02d}_tabela.csv');rec[f'RMSE_curvas_{k}']=float(np.sqrt(np.mean((d.mediana-d.referencia)**2)))
 rows.append(rec)
pd.DataFrame(rows).to_csv(HERE/'indices_variogramas.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
for name,label,color in zip(METHODS,LABELS,COLORS):
 d=sumdf[sumdf.metodo==name].iloc[0];axes[0].scatter(d.W1_normalizado,d.RMSE_normalizado,c=color,s=65);offset={'MST':(8,5),'FFTMA':(8,-10),'SGS':(8,5),'PPMT':(8,0)}[name];axes[0].annotate(label,(d.W1_normalizado,d.RMSE_normalizado),xytext=offset,textcoords='offset points',fontsize=8)
 t=pd.read_csv(R/name/'12_tabela.csv');c=t.groupby('nominal').cobertura.mean();axes[1].plot(c.index,c.values,'o-',color=color,label=label,ms=4)
axes[0].set(xlabel='Normalized marginal Wasserstein ↓',ylabel='Normalized predictive RMSE ↓',xlim=(.045,.205),ylim=(.94,1.025),title='');axes[1].plot([.5,.95],[.5,.95],'k--',label='Nominal calibration');axes[1].set(xlabel='Nominal coverage',ylabel='Observed coverage',title='',ylim=(.3,1));axes[1].legend(fontsize=7);fig.tight_layout();finish(4,fig)
tr=pd.read_csv(ROOT/'jura_real/dados/jura_train.csv');va=pd.read_csv(ROOT/'jura_real/dados/jura_validation.csv');sim={x:[] for x in ['lambda0_nn48','lambda1_nn48']};secondary=[]
for seed in range(20261111,20261121):
 with np.load(J/f'final_{seed}.npz') as a:
  for config in sim:
   s=a[config];sim[config].append(s)
   for j,metal in [(1,'Ni'),(2,'Zn')]:secondary.append(dict(seed=seed,config=config,metal=metal,MAE_por_realizacao=float(abs(s[:,:,j]-va[metal].to_numpy()).mean())))
sec=pd.DataFrame(secondary);sec.to_csv(HERE/'jura_secundarias.csv',index=False);sim={k:np.concatenate(v) for k,v in sim.items()};pred=[np.median(sim[k][:,:,0],axis=0) for k in sim]
fig=plt.figure(figsize=(12.2,8));gs=fig.add_gridspec(2,6,height_ratios=[1.25,1]);maxval=max(va.Cd.max(),max(x.max() for x in pred));maps=[va.Cd.to_numpy()]+pred
for col,(values,title) in enumerate(zip(maps,['Observed Cd at targets','Default MST: median of 2000 fields','Selected MST: median of 2000 fields'])):
 ax=fig.add_subplot(gs[0,2*col:2*col+2]);ax.scatter(tr.Xloc,tr.Yloc,s=7,color='#c8c8c8',label='Training');im=ax.scatter(va.Xloc,va.Yloc,c=values,s=23,cmap='viridis',vmin=0,vmax=maxval,marker='D');ax.set(title=title,xlabel='X (km)',ylabel='Y (km)',aspect='equal');fig.colorbar(im,ax=ax,shrink=.7,label='Cd (mg/kg)')
for j,metal in enumerate(['Ni','Zn']):
 ax=fig.add_subplot(gs[1,3*j:3*j+3])
 for config,label,color in [('lambda0_nn48','Default',COLORS[0]),('lambda1_nn48','Weight 1 / NN48',COLORS[1])]:
  g=sec[(sec.config==config)&(sec.metal==metal)];ax.plot(np.arange(1,11),g.MAE_por_realizacao,'o-',color=color,label=label,ms=3)
 ax.set(title=f'{metal}: discrepancy after transport',xlabel='Seed (run)',ylabel='Mean MAE per realization (mg/kg)');ax.legend(fontsize=8)
fig.tight_layout();finish(5,fig)
fig,axes=plt.subplots(1,3,figsize=(13.4,4.6));order=np.argsort(tr.Xloc.to_numpy(),kind='stable');group=np.zeros(len(tr),int)
for k,ids in enumerate(np.array_split(order,3)):group[ids]=k
axes[0].scatter(tr.Xloc,tr.Yloc,c=group,cmap='Set1',s=13);axes[0].set(title='Three outer strips',xlabel='X (km)',ylabel='Y (km)',aspect='equal')
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([],[],marker='o',ls='',color=plt.get_cmap('Set1')(k/2),label=f'Block {k+1}') for k in range(3)],fontsize=7,loc='lower right')
o=pd.read_csv(J/'metricas_outer.csv');d=pd.read_csv(J/'metricas_jura_externo.csv')
for label,color in [('MST padrão',COLORS[0]),('MST seleção interna',COLORS[1])]:
 lab={'MST padrão':'Default MST','MST seleção interna':'MST, internal selection'}[label];g=o[o.metodo==label];axes[1].plot(g.outer+1,g.MAE_mediana,'o-',label=lab,color=color);g=d[d.metodo==label];axes[2].plot(np.arange(1,11),g.MAE_mediana,'o-',label=lab,color=color,ms=3)
axes[1].set(title='',xlabel='Outer block',ylabel='MAE Cd (mg/kg)',xticks=[1,2,3]);axes[1].legend(fontsize=7);axes[2].set(title='',xlabel='Seed (run)',ylabel='MAE Cd (mg/kg)');fig.tight_layout();finish(6,fig)
# Tabelas de apoio para todas as afirmações quantitativas da redação.
external=pd.read_csv(J/'metricas_jura_externo.csv');rows=[]
for label,g in external.groupby('metodo'):
 row={'metodo':label,'cenario':'histórico, média de dez sementes'}
 for col in ['MAE_mediana','RMSE_mediana','CRPS','cobertura90','largura90']:row[col]=g[col].mean();row[col+'_sd_MC']=g[col].std()
 rows.append(row)
pd.DataFrame(rows).to_csv(HERE/'jura_medias_sementes.csv',index=False)
print('Seis figuras PNG/PDF e tabelas de apoio geradas dos resultados salvos.')
