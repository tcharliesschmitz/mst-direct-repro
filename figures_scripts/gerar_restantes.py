from gerar_quadros import *
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.text import Text
# Neutral methodological diagram, with geo-statistical choices kept in caption.
fig,ax=plt.subplots(figsize=(190/25.4,88/25.4));fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
def box(x,y,w,h,title,body):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',fc='white',ec='#4b6478',lw=.7));ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontweight='bold',fontsize=8);ax.text(x+w/2,y+h/2-.10,body,ha='center',va='center',fontsize=7.5,linespacing=1.4)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.7,color='#4b6478'))
box(.15,2.25,2.55,1.45,'Multivariate pool','Complete tuples\nAttribute combinations');box(.15,.25,2.55,1.45,'Reference','Position representation\nGuiding structure');box(4.15,1.25,2.55,1.5,'MST assignment','Multivariate compatibility\nBijection between tuples and positions');box(8.15,1.25,2.55,1.5,'Result','Tuples organized in the domain\nEmpirical distribution preserved');arrow((2.75,2.9),(4.1,2.35));arrow((2.75,1.0),(4.1,1.7));arrow((6.75,2),(8.1,2));ax.text(5.45,.70,'Cost + constraints + transport solver',ha='center',fontsize=7.5);save(fig,'Fig01_metodo')
# Re-render overview/Jura with minimum 7pt fonts at final 190mm width.
ARTICLE=Path(__file__).resolve().parent;J=ROOT/'validacao_publicacao_v1/resultados';R=D;METHODS=KEYS;LABELS=['MST','FFT-MA','SGS','PPMT'];sumdf=pd.read_csv(ROOT/'ppmt_pymgt_v1/resumo_comparativo_pymgt.csv')
base_save=save
names={4:'Fig04_previsao',5:'Fig06_jura',6:'Fig07_validacao'}
def finish(n,fig):
 fig.set_size_inches(190/25.4,({4:90,5:145,6:92}[n])/25.4)
 for ax in fig.axes:
  ax.set_title('')
  for text in ax.findobj(match=Text):text.set_fontsize(7)
 fig.tight_layout(pad=.8,w_pad=.8,h_pad=1)
 for i,ax in enumerate([x for x in fig.axes if x.get_label()!='<colorbar>' and (x.get_xlabel() or x.get_ylabel())]):
  ax.text(0,1.03,'('+chr(97+i)+')',transform=ax.transAxes,fontsize=8,fontweight='bold')
 base_save(fig,names[n])
src=(ARTICLE/'gerar_figuras.py').read_text();start=src.index('fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))');end=src.index('# Tabelas de apoio',start);exec(src[start:end],globals())
(HERE/'manifesto_figuras.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Diagrama e figuras de síntese/Jura exportados a 190 mm; sem títulos gerais dentro das imagens.')
