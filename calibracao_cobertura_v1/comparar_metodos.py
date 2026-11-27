from experimento import *
import pickle

class ExternalPPMT(motor.Method):
    """Mesmo ajuste pymgt e adaptador auditado do benchmark; leitura do cache."""
    def __init__(self,cloud):
        vendor=ROOT/'ppmt_pymgt_v1/vendor/pymgt-main/pymgt-main/src'
        sys.path.insert(0,str(vendor));sys.path.insert(0,str(ROOT/'ppmt_pymgt_v1'))
        from numerica import StableTables
        cached=pickle.loads((ROOT/'ppmt_pymgt_v1/ajuste_pymgt.pkl').read_bytes())
        self.name='PPMT';self.cloud=cloud;self.mu=cloud.mean(0);self.sd=cloud.std(0);self.geometry_seconds=0.;self.geom={};self.paths={};self.pp=cached['model'];self.fit_seconds=cached['seconds'];self.tables=StableTables(self.pp);g=cached['g'];self.factor_cross=np.corrcoef(g,rowvar=False);self.training_roundtrip=float(abs(self.inverse(self.forward(cloud))-cloud).max());self.fitted_forward_agreement=float(abs(self.forward(cloud)-g).max())
        assert self.training_roundtrip<1e-6 and self.fitted_forward_agreement<1e-6
    def forward(self,x):return self.tables.forward(x)
    def inverse(self,x):return self.tables.inverse(x)

def compare():
    with np.load(OUT/'evaluation_fields.npz') as z:evaluation={k:z[k] for k in z.files}
    factor=np.array(json.loads((OUT/'fatores_congelados.json').read_text())['factors'])
    frames=[];bytruth=[];names=['MST original','MST calibrado','FFT-MA','SGS','PPMT']
    m,t=evaluate(evaluation,factor,'Mesmas oito verdades')
    m['metodo']=m.intervalo.map({'Original':names[0],'Calibrado':names[1]});t['metodo']=t.intervalo.map({'Original':names[0],'Calibrado':names[1]});frames.append(m);bytruth.append(t)
    manifest={'evaluation_truth_seeds':evaluation['seeds'].tolist(),'fields_per_truth':60,'anchors':48,'free_nodes':352,'nominal':.9,'same_field_seeds':True,'PPMT':'pymgt original fitted cache + StableTables robust tail adapter','calibration':'Apenas MST; comparadores originais. Comparação de pipelines, não avaliação de solvers com o mesmo pós-processamento.','methods':{}}
    with threadpool_limits(limits=2):
        for name in ['FFTMA','SGS','PPMT']:
            path=OUT/('comparison_'+name+'_fields.npz');start=time.perf_counter()
            if path.exists():
                with np.load(path) as z:fields=z['fields'];assert np.array_equal(z['seeds'],evaluation['seeds'])
            else:
                method=ExternalPPMT(dat['cloud']) if name=='PPMT' else motor.Method(name,dat['cloud']);fields=[]
                for k,y in enumerate(evaluation['truth']):
                    ensemble=[]
                    for j in range(60):
                        f,_=method.simulate(20,ANCHORS,y[ANCHORS],PROTOCOL['evaluation_field_seed']+1000*k+j)
                        assert np.array_equal(f[ANCHORS],y[ANCHORS]);assert np.isfinite(f).all();ensemble.append(f)
                    fields.append(ensemble);print(name,k+1,'/8 concluído',flush=True)
                fields=np.array(fields);np.savez_compressed(path,fields=fields,seeds=evaluation['seeds'])
            data={'truth':evaluation['truth'],'fields':fields};d,t=evaluate(data,np.ones(6),'Mesmas oito verdades');d=d[d.intervalo=='Original'].copy();t=t[t.intervalo=='Original'].copy();d['metodo']='FFT-MA' if name=='FFTMA' else name;t['metodo']='FFT-MA' if name=='FFTMA' else name;frames.append(d);bytruth.append(t);manifest['methods'][name]={'field_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds_this_run':time.perf_counter()-start}
    pervar=pd.concat(frames,ignore_index=True);truth=pd.concat(bytruth,ignore_index=True);summary=pervar.groupby('metodo')[['cobertura','largura_norm','interval_score_norm','MAE_norm']].mean().loc[names].reset_index();summary['cobertura_pct']=summary.cobertura*100;summary['distancia_alvo_pp']=abs(summary.cobertura_pct-90)
    pervar.to_csv(OUT/'08_comparacao_por_propriedade.csv',index=False);truth.to_csv(OUT/'08_comparacao_por_verdade.csv',index=False);summary.to_csv(OUT/'08_comparacao_metodos.csv',index=False);(OUT/'08_manifesto_comparacao.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));return summary,pervar,truth
