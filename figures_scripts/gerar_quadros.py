from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_elsevier')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
HERE=Path(__file__).resolve().parent/'output';HERE.mkdir(exist_ok=True);ROOT=Path(__file__).resolve().parents[1];D=ROOT/'ppmt_pymgt_v1/benchmark/resultados'
KEYS=['MST','FFTMA','SGS','PPMT'];LABELS=['MST','FFT-MA','SGS','PPMT'];COLORS=['#226a9c','#c58128','#31816b','#92538c']
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1,'pdf.fonttype':42,'svg.fonttype':'none','axes.axisbelow':True})
with np.load(ROOT/'ppmt_pymgt_v1/benchmark/dados_congelados.npz') as z:dat={k:z[k] for k in z.files}
fields={}
for key in KEYS:
 with np.load(D/key/'campos.npz') as z:fields[key]={'u':z['unconditional'][0],'real':z['conditional'][0,0],'median':np.median(z['conditional'][0],axis=0)}
manifest=[]
def save(fig,name):
 fig.savefig(HERE/(name+'.svg'));fig.savefig(HERE/(name+'.pdf'));fig.savefig(HERE/(name+'.png'),dpi=180)
 fig.savefig(HERE/(name+'.tiff'),dpi=500,pil_kwargs={'compression':'tiff_lzw'})
 manifest.append({'file':name,'width_mm':190,'height_mm':float(fig.get_figheight()*25.4),'tiff_dpi':500,'normal_font_pt':7.5,'tick_font_pt':7,'pdf':'vector text/lines; rasterized scatter layers at native render resolution where used','panels':'a MST; b FFT-MA; c SGS; d PPMT'})
 plt.close(fig)
# Fixed mechanism pairs, same selection in every method.
pairs=[(0,2),(0,3),(0,5)]
fig=plt.figure(figsize=(190/25.4,205/25.4));outer=fig.add_gridspec(2,2,left=.10,right=.98,bottom=.07,top=.95,wspace=.34,hspace=.42)
for k,(key,label,color) in enumerate(zip(KEYS,LABELS,COLORS)):
 inner=outer[k//2,k%2].subgridspec(3,1,hspace=.55)
 for row,(i,j) in enumerate(pairs):
  ax=fig.add_subplot(inner[row]);ax.scatter(dat['cloud'][:,i],dat['cloud'][:,j],s=1.5,c='#b5bfc9',alpha=.35);ax.scatter(fields[key]['u'][:,i],fields[key]['u'][:,j],s=2.6,c=color,alpha=.7)
  xx=np.concatenate([dat['cloud'][:,i]]+[fields[x]['u'][:,i] for x in KEYS]);yy=np.concatenate([dat['cloud'][:,j]]+[fields[x]['u'][:,j] for x in KEYS]);ax.set(xlim=(xx.min()-.15,xx.max()+.15),ylim=(yy.min()-.15,yy.max()+.15),xlabel=f'Z{i+1}',ylabel=f'Z{j+1}')
  if row==0:ax.text(0,1.16,f'({chr(97+k)}) {label}',transform=ax.transAxes,fontweight='bold',fontsize=8)
save(fig,'Fig02_crossplots')
for test,pairs,name in [(9,['Z1/Z1','Z4/Z4','Z6/Z6'],'Fig03a_diretos'),(10,['Z1/Z3','Z1/Z4','Z1/Z6'],'Fig03b_cruzados')]:
 ts={key:pd.read_csv(D/key/f'{test:02d}_tabela.csv') for key in KEYS};fig=plt.figure(figsize=(190/25.4,205/25.4));outer=fig.add_gridspec(2,2,left=.10,right=.98,bottom=.07,top=.95,wspace=.34,hspace=.42)
 for k,(key,label,color) in enumerate(zip(KEYS,LABELS,COLORS)):
  inner=outer[k//2,k%2].subgridspec(3,1,hspace=.55)
  for row,pair in enumerate(pairs):
   ax=fig.add_subplot(inner[row]);d=ts[key][ts[key].par==pair];allv=pd.concat([t[t.par==pair] for t in ts.values()]);lo=min(0,allv.q05.min(),allv.referencia.min());hi=max(allv.q95.max(),allv.referencia.max());pad=.06*(hi-lo)
   ax.fill_between(d.lag,d.q05,d.q95,color=color,alpha=.18,lw=0);ax.plot(d.lag,d.mediana,'o-',c=color,ms=2);ax.plot(d.lag,d.referencia,'--',c='#263747');ax.set(xlabel='Lag distance',ylabel=pair,ylim=(lo-pad,hi+pad));ax.grid(axis='y',lw=.4,alpha=.2)
   if row==0:ax.text(0,1.16,f'({chr(97+k)}) {label}',transform=ax.transAxes,fontweight='bold',fontsize=8)
 save(fig,name)
# One pre-fixed nonlinear property, truth / realization / median in each method panel.
j=3;truth=dat['truth'][0];vv=[truth[:,j]]+[fields[k][part][:,j] for k in KEYS for part in ['real','median']];vmin=min(x.min() for x in vv);vmax=max(x.max() for x in vv)
fig=plt.figure(figsize=(190/25.4,115/25.4));outer=fig.add_gridspec(2,2,left=.055,right=.98,bottom=.10,top=.92,wspace=.28,hspace=.50)
for k,(key,label) in enumerate(zip(KEYS,LABELS)):
 inner=outer[k//2,k%2].subgridspec(1,3,wspace=.15)
 for col,(title,values) in enumerate([('Truth',truth[:,j]),('Realization',fields[key]['real'][:,j]),('Median',fields[key]['median'][:,j])]):
  ax=fig.add_subplot(inner[col]);im=ax.imshow(values.reshape(20,20),origin='lower',cmap='viridis',vmin=vmin,vmax=vmax);xy=dat['coords'][dat['anchors']];ax.scatter(xy[:,0],xy[:,1],s=4,facecolors='none',edgecolors='#ffe7a4',lw=.4);ax.set(xticks=[0,19],yticks=[0,19],title=title)
  if col==0:ax.text(0,1.50,f'({chr(97+k)}) {label}',transform=ax.transAxes,fontweight='bold',fontsize=8)
cax=fig.add_axes([.32,.035,.36,.015]);fig.colorbar(im,cax=cax,orientation='horizontal',label='Z4');save(fig,'Fig05_campos')
# Coverage plot: compact at final publication size, point estimates on a zoomed axis.
d=pd.read_csv(ROOT/'calibracao_cobertura_v1/resultados/08_comparacao_metodos.csv');d['metodo']=d.metodo.replace({'MST original':'MST (original)','MST calibrado':'MST (calibrated)'});fig,axes=plt.subplots(1,2,figsize=(190/25.4,82/25.4));fig.subplots_adjust(left=.075,right=.99,bottom=.30,top=.93,wspace=.30);colors=['#91afc5',*COLORS[:1],*COLORS[1:]];x=np.arange(5)
for ax in axes:ax.grid(axis='y',lw=.4,alpha=.25);ax.set_xticks(x);ax.set_xticklabels(d.metodo,rotation=28,ha='right')
ax=axes[0];ax.axhline(90,ls='--',c='#304757',lw=.8)
for i,row in enumerate(d.itertuples()):ax.scatter(i,row.cobertura_pct,s=28,c=colors[i]);ax.annotate(f'{row.cobertura_pct:.2f}',(i,row.cobertura_pct),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
ax.set(ylim=(75,94),yticks=[75,80,85,90],ylabel='Coverage (%)');ax.text(0,1.05,'(a)',transform=ax.transAxes,fontweight='bold');ax=axes[1];bars=ax.bar(x,d.largura_norm,color=colors,width=.6)
for bar,v in zip(bars,d.largura_norm):ax.text(bar.get_x()+bar.get_width()/2,v+.05,f'{v:.2f}',ha='center',fontsize=7)
ax.set(ylim=(0,3.1),ylabel='Normalized width');ax.text(0,1.05,'(b)',transform=ax.transAxes,fontweight='bold');save(fig,'Fig08_cobertura')
import json
(HERE/'manifesto_figuras.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Cinco composições preparadas a 190 mm: SVG/PDF vetorial e TIFF 500 dpi; fontes 7–8 pt.')
