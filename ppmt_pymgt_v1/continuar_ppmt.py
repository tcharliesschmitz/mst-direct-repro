"""Integração do pymgt ao protocolo congelado. Não usa o PPMT simplificado.
Execução: python3 continuar_ppmt.py [caminho/do/pymgt-main.zip]
Ausência de fonte é bloqueio explícito; não produz métricas substitutas.
"""
import os,sys,json,time,hashlib,zipfile,shutil,importlib.util,io,base64,pickle,contextlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mst_pymgt')
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent;VENDOR=HERE/'vendor';BUNDLE=HERE/'benchmark';SOURCE=ROOT/'benchmark_comum_v1'
CONFIG={'target':.001,'optimiser':'original','maxiter':100};FIT_SEED=9101
if (HERE/'parada_gaussiana.json').exists():CONFIG['target']=json.loads((HERE/'parada_gaussiana.json').read_text())['target']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(state,**kwargs):
 (HERE/'status.json').write_text(json.dumps(dict(status=state,config=CONFIG,**kwargs),indent=2,ensure_ascii=False))
def find_package():
 for base in [VENDOR]:
  for candidate in base.glob('*pymgt*'):
   if candidate.is_file() and candidate.suffix=='.zip':
    dest=VENDOR/candidate.stem
    if not dest.exists():
     with zipfile.ZipFile(candidate) as z:
      for item in z.infolist():
       target=(dest/item.filename).resolve()
       if not target.is_relative_to(dest.resolve()):raise ValueError('Caminho inválido no ZIP')
      z.extractall(dest)
 for init in VENDOR.glob('**/src/pymgt/__init__.py'):
  sys.path.insert(0,str(init.parent.parent));return init.parent
 sp=importlib.util.find_spec('pymgt')
 if sp:return Path(sp.origin).parent
 raise ModuleNotFoundError('pymgt indisponível. Place pymgt-main.zip (or the extracted pymgt-main folder) in ppmt_pymgt_v1/vendor e execute novamente. Nenhum resultado do PPMT local será usado.')
def load_module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod);return mod

def setup_bundle():
 BUNDLE.mkdir(exist_ok=True)
 for file in ['motor.py','avaliacao.py','dados_congelados.npz','dados_grandes_congelados.npz','dados_sha256.txt']:
  shutil.copy2(SOURCE/file,BUNDLE/file)
 # Cópias físicas, nunca links para resultados históricos que o avaliador possa alterar.
 for name in ['MST','FFTMA','SGS']:
  dest=BUNDLE/'resultados'/name;dest.mkdir(parents=True,exist_ok=True)
  for file in ['campos.npz','variogramas.npz','campos_grandes.npz','manifesto_metodo.json','execucao.csv','execucao_grandes.csv','escala.csv']:
   shutil.copy2(SOURCE/'resultados'/name/file,dest/file)
 path=BUNDLE/'motor.py';path.write_text(path.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parents[2]'))
 with (BUNDLE/'motor.py').open('a') as f:f.write("\nLABELS['PPMT']='PPMT — pymgt externo'\n")
 sys.path.insert(0,str(ROOT/'mst_direct-2.0.0/src'));sys.path.insert(0,str(BUNDLE))
 return load_module('motor',BUNDLE/'motor.py')

def install_adapter(m,sourcepath):
 from pymgt import PPMTransform
 Base=m.Method
 class PymgtMethod(Base):
  def __init__(self,name,cloud,force_fit=False):
   if name!='PPMT':super().__init__(name,cloud);return
   from numerica import StableTables
   self.name=name;self.cloud=cloud;self.mu=cloud.mean(0);self.sd=cloud.std(0);self.geometry_seconds=0.;self.geom={};self.paths={}
   key=hashlib.sha256(cloud.tobytes()+json.dumps(CONFIG,sort_keys=True).encode()+b''.join(f.read_bytes() for f in sorted(sourcepath.glob('*.py')))).hexdigest();cache=HERE/'ajuste_pymgt.pkl'
   cached=pickle.loads(cache.read_bytes()) if cache.exists() and not force_fit else None
   if cached is not None and cached['key']==key:self.pp=cached['model'];self.fit_seconds=cached['seconds'];g=cached['g']
   else:
    old=np.random.get_state();start=time.perf_counter()
    try:
     np.random.seed(FIT_SEED);self.pp=PPMTransform(**CONFIG)
     with (HERE/('ajuste_repeticao.log' if force_fit else 'ajuste_original.log')).open('w') as log,contextlib.redirect_stdout(log):g=self.pp.fit_transform(cloud)
    finally:np.random.set_state(old)
    self.fit_seconds=time.perf_counter()-start
    if not force_fit:cache.write_bytes(pickle.dumps(dict(key=key,model=self.pp,seconds=self.fit_seconds,g=g)))
   self.tables=StableTables(self.pp);self.factor_cross=np.corrcoef(g,rowvar=False);self.training_roundtrip=float(abs(self.inverse(self.forward(cloud))-cloud).max());self.fitted_forward_agreement=float(abs(self.forward(cloud)-g).max())
   if not np.isfinite(g).all() or self.training_roundtrip>1e-6 or self.fitted_forward_agreement>1e-6:raise ValueError('pymgt: inversão ou equivalência interior ao ajuste original falhou.')
  def forward(self,x):return self.tables.forward(x) if self.name=='PPMT' else super().forward(x)
  def inverse(self,g):return self.tables.inverse(g) if self.name=='PPMT' else super().inverse(g)
 m.Method=PymgtMethod;m.LABELS['PPMT']='PPMT — pymgt, índice Friedman e busca original'
 return PymgtMethod

def gates(m):
 d=m.data();model=m.Method('PPMT',d['cloud']);hold=m.truth_transform(np.random.default_rng(20261102).normal(size=(4096,m.D)));rows=[]
 for name,x in [('treino',d['cloud']),('iid_novo',hold),('ancoras',d['truth'][:,d['anchors']].reshape(-1,m.D))]:
  y=model.forward(x);back=model.inverse(y);err=float(abs(back-x).max());rows.append(dict(amostra=name,erro_inversao_max=err,finito=bool(np.isfinite(y).all() and np.isfinite(back).all())))
 with np.load(SOURCE/'dados_grandes_congelados.npz') as big:largeanchors=big['truth100'][big['anchors100']]
 y=model.forward(largeanchors);back=model.inverse(y);rows.append(dict(amostra='ancoras_100x100',erro_inversao_max=float(abs(back-largeanchors).max()),finito=bool(np.isfinite(y).all() and np.isfinite(back).all())))
 df=pd.DataFrame(rows);df.to_csv(HERE/'auditoria_inversao.csv',index=False)
 if not df.finito.all() or df.erro_inversao_max.max()>1e-6:raise ValueError('Inversão de novos pontos falhou: examinar extrapolação antes de comparar métodos.')
 # Repetibilidade do ajuste, sem alterar a implementação da biblioteca.
 other=m.Method('PPMT',d['cloud'],force_fit=True);err=float(abs(model.forward(hold)-other.forward(hold)).max())
 if err>1e-10:raise ValueError('Ajuste pymgt não repetível com semente congelada.')
 probes=np.random.default_rng(20261103).normal(size=(128,m.D));probes/=np.linalg.norm(probes,axis=1)[:,None]
 from scipy.stats import kstest
 values=model.forward(hold);null=np.random.default_rng(20261104).normal(size=values.shape);diag=[]
 for label,z in [('pymgt',values),('Gaussianos de referência',null)]:
  for i,direction in enumerate(probes):diag.append(dict(origem=label,direcao=i,KS_normal=float(kstest(z@direction,'norm').statistic)))
 pd.DataFrame(diag).to_csv(HERE/'auditoria_gaussianidade.csv',index=False)
 (HERE/'auditoria_ajuste.json').write_text(json.dumps(dict(repetibilidade_erro_max=err,correlacao_fatores=np.corrcoef(values,rowvar=False).tolist(),iteracoes_efetivas=len(model.pp._state.iteration_steps),config=CONFIG),indent=2))
 # Integridade antes de restaurar âncoras no adaptador compartilhado.
 ga=model.forward(m.data()['truth'][0,d['anchors']]);assert abs(model.inverse(ga)-d['truth'][0,d['anchors']]).max()<1e-6
 return model

def core(m):
 dat=m.data();out=m.OUT/'PPMT';out.mkdir(exist_ok=True);start=time.perf_counter();method=m.Method('PPMT',dat['cloud']);un=[];conds=[];diag=[]
 for r,seed in enumerate(m.REAL_SEEDS):
  field,info=method.simulate(m.N,[],np.empty((0,m.D)),seed);un.append(field);diag.append(dict(scenario='incondicional',referencia=-1,realizacao=r,**info))
 for ref in range(3):
  fields=[]
  for r,seed in enumerate(m.REAL_SEEDS):
   field,info=method.simulate(m.N,dat['anchors'],dat['truth'][ref,dat['anchors']],seed+1000)
   if info['anchor_error_before_hard_imposition']>1e-6:raise ValueError('Condicionante alterado pela inversão antes da restauração.')
   fields.append(field);diag.append(dict(scenario='condicional',referencia=ref,realizacao=r,**info))
  conds.append(fields);print('PPMT pymgt referência',ref+1,'/3 concluída',flush=True)
 un=np.array(un);conds=np.array(conds);np.savez_compressed(out/'campos.npz',unconditional=un,conditional=conds);pd.DataFrame(diag).to_csv(out/'execucao.csv',index=False)
 np.savez_compressed(out/'variogramas.npz',unconditional=m.gamma_fields(un,dat),conditional=np.array([m.gamma_fields(s,dat) for s in conds]),truth=m.gamma_fields(dat['truth'],dat))
 scale=[]
 for n in [16,20,24]:
  t=time.perf_counter()
  for seed in [94001,94002,94003]:method.simulate(n,[],np.empty((0,m.D)),seed)
  dt=time.perf_counter()-t;scale.append(dict(lado=n,nos=n*n,realizacoes=3,seconds_total_generation_plus_new_geometry=dt,seconds_per_realization=dt/3))
 pd.DataFrame(scale).to_csv(out/'escala.csv',index=False)
 stats=dict(method='PPMT',label=m.LABELS['PPMT'],fit_seconds=method.fit_seconds,geometry_and_sgs_plan_seconds=method.geometry_seconds,total_seconds=time.perf_counter()-start,training_roundtrip_max=method.training_roundtrip,fitted_forward_agreement=method.fitted_forward_agreement,n_iter_ppmt=CONFIG['maxiter'],n_dirs_ppmt=None,PPMT_config=CONFIG,implementation='Ajuste pymgt original intacto; adaptador de caudas lineares robustas 1% mutuamente inversas',PPMT_tail_policy='linear_robust_1pct_mutually_inverse',iterations_actual=len(method.pp.state.iteration_steps),all_conditional_anchors_exact=True)
 (out/'manifesto_metodo.json').write_text(json.dumps(stats,indent=2));m.run_large(['PPMT'])

def notebook(m):
 import nbformat
 from IPython.core.interactiveshell import InteractiveShell
 a=load_module('avaliacao',BUNDLE/'avaliacao.py');a.prepare_tables();a._TABLES=None;shell=InteractiveShell.instance()
 # Mesmo roteiro e índices dos quatro notebooks anteriores: markdown + código por teste.
 nb=nbformat.v4.new_notebook();nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
 intro='# Benchmark sintético comum — PPMT pymgt\n\nImplementação externa pymgt, não a versão compacta anterior. Mesmo conjunto iid, três verdades independentes, 48 âncoras, 60 realizações por cenário, alcance latente prescrito 5 e dois cenários grandes. Ajuste executado pelo pymgt original intacto (índice Friedman, busca original, normal scores e sphering). Adaptador de caudas: extrapolação linear com secantes do 1% extremo em cada tabela, mutuamente inversas; necessário porque o pacote original produz NaN/Inf em novas tuplas. Equivalência no treino e inversão fora do treino auditadas. Não é execução sem adaptações numéricas. Parada calibrada pela quantil 95% de 50 amostras iid Gaussianas do mesmo tamanho, usando o índice e otimizador da biblioteca; semente 9102. Limite 100 iterações. Nenhum parâmetro foi selecionado por erros nos alvos. A tentativa inicial com target 0.001 está arquivada: apresentou falha técnica de inversão em dados novos. As quatro cópias deste comparativo usam os mesmos eixos atualizados. Comparação de configurações prescritas, não dos melhores ajustes por metodologia. Bibliografia: https://github.com/exepulveda/pymgt e https://link.springer.com/article/10.1007/s11004-023-10056-y'
 for name in m.METHODS:
  nb=nbformat.v4.new_notebook();nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'};nb.cells=[nbformat.v4.new_markdown_cell('Método desta cópia: **'+m.LABELS[name]+'**.\n\n'+intro)]
  for k,title in enumerate(a.TITLES,1):
   code=("from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path('../MST_reproducao/ppmt_pymgt_v1/benchmark').resolve()))\nimport motor, avaliacao\nMETODO="+repr(name)+"\navaliacao.render(METODO, 1)") if k==1 else f'avaliacao.render(METODO, {k})'
   cell=nbformat.v4.new_code_cell(code,execution_count=k);cell.outputs=[]
   def capture(obj):
    if hasattr(obj,'savefig'):
     b=io.BytesIO();obj.savefig(b,format='png',dpi=120);data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'Figura calculada nesta execução'}
    else:data,_=shell.display_formatter.format(obj)
    cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata={}))
   a.display=capture;a.render(name,k);nb.cells += [nbformat.v4.new_markdown_cell(f'## {k:02d}. {title}\n\n{a.NOTES[k-1]}'),cell]
  nbformat.validate(nb);nbformat.write(nb,PROJECT/'execucoes'/f'Benchmark_Pymgt_v1_{name}.ipynb')

if __name__=='__main__':
 raise SystemExit('Reproducibility package: run the PPMT comparator through ../run_all.py (table2). The original driver also rebuilt project notebooks and is not used here.')
