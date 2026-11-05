import argparse,json,time,pickle,platform
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from data import prepare,metrics,seasonal_fit,seasonal_predict,recent_predict,HORIZONS,MEATS,feature_names

def score(m):return sum(m[str(h)]['weighted_mae']/h for h in HORIZONS)/len(HORIZONS)
def export_cpp(model,path,transform,correction):
 temp=Path(path).with_suffix('.json');model.save_model(str(temp),format='json');m=json.loads(temp.read_text());bias=m['scale_and_bias'][1];scale=m['scale_and_bias'][0];dim=len(bias)
 # CatBoost float_feature_index renvoie l'index de la feature numérique d'origine.
 fmap={i:f['flat_feature_index'] for i,f in enumerate(m['features_info']['float_features'])}
 with open(path,'w') as out:
  out.write(f'CB_OBLIVIOUS_V1\n{len(feature_names())} {dim} {len(m["oblivious_trees"])} {int(transform=="log")}\n')
  out.write(' '.join(map(str,[scale]+bias))+'\n');out.write(' '.join(map(str,correction))+'\n')
  for tree in m['oblivious_trees']:
   splits=tree.get('splits') or [];out.write(str(len(splits))+'\n')
   for sp in splits:
    assert sp['split_type']=='FloatFeature';out.write(f'{fmap[sp["float_feature_index"]]} {sp["border"]:.17g}\n')
   out.write(' '.join(f'{x:.17g}' for x in tree['leaf_values'])+'\n')
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root);results=root/'results';models=root/'models';results.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
 d=prepare(a.data,root/'cache');X,Y=d['X'],d['Y'];train=np.flatnonzero(d['train'])[::2];val=np.flatnonzero(d['val']);test=np.flatnonzero(d['test']);w=d['weights']
 print('DATA',X.shape,'TRAIN sample',len(train),'VAL',len(val),'TEST',len(test),flush=True)
 candidates={};preds={};timings={}
 table=seasonal_fit(d,d['train']);sp=seasonal_predict(d,table);rp=recent_predict(d)
 for name,pred in [('saisonnier',sp),('recent_5min',rp)]:candidates[name]=metrics(Y[val],pred[val],w[val]);preds[name]=pred
 configs=[('catboost_raw_d6','raw',6,320),('catboost_raw_d8','raw',8,360),('catboost_log_d6','log',6,320)]
 objects={}
 for name,transform,depth,iters in configs:
  t=time.perf_counter();print('FIT',name,flush=True)
  target=np.log1p(Y) if transform=='log' else Y
  # Échelle par horizon/viande : empêcher le 30 min/Hamb d'écraser le Master/1 min.
  scales=np.maximum(Y[train].mean(axis=0),1.)
  if transform=='log':scales=np.ones(15)
  model=CatBoostRegressor(loss_function='MultiRMSE',iterations=iters,depth=depth,learning_rate=.075,l2_leaf_reg=8,thread_count=4,random_seed=123,verbose=100,allow_writing_files=False)
  model.fit(X[train],target[train]/scales,sample_weight=np.minimum(w[train],3),eval_set=(X[val],target[val]/scales),early_stopping_rounds=55)
  pv=model.predict(X[val],thread_count=4)*scales
  if transform=='log':pv=np.expm1(pv)
  # Pas de recalage sur les cibles de validation : sélection seulement.
  pv=np.maximum(pv,0);candidates[name]=metrics(Y[val],pv,w[val]);objects[name]=(model,transform,scales)
  timings[name]=time.perf_counter()-t;print('VAL',name,'score',score(candidates[name]),'seconds',timings[name],flush=True)
 # Ensemble récent optionnel choisi seulement sur validation (0 exclut un mélange inutile).
 for name in list(objects):
  model,tr,scales=objects[name];p0=model.predict(X[val],thread_count=4)*scales
  if tr=='log':p0=np.expm1(p0)
  for alpha in [.15,.30]:
   label=name+f'_recent{alpha}';pv=(1-alpha)*np.maximum(p0,0)+alpha*rp[val];candidates[label]=metrics(Y[val],pv,w[val])
 best=min(candidates,key=lambda k:score(candidates[k]));print('SELECTED',best,score(candidates[best]),flush=True)
 if best in ['saisonnier','recent_5min']:raise RuntimeError('Les modèles appris ne battent pas les références sur validation ; examiner les données avant livraison.')
 base=best.split('_recent')[0];alpha=float(best.split('_recent')[1]) if '_recent' in best else 0.;model,transform,scales=objects[base]
 # Refit 12 mois, réglages/itérations sélectionnés sans le test ; purge 30 min avant janvier.
 refit=np.flatnonzero((d['split']<2)&(d['hour']>=9.5)&(d['epochs']+1800<=d['epochs'][np.flatnonzero(d['split']==2)[0]]))[::2]
 target=np.log1p(Y) if transform=='log' else Y
 final=CatBoostRegressor(loss_function='MultiRMSE',iterations=model.tree_count_,depth=model.get_param('depth'),learning_rate=.075,l2_leaf_reg=8,thread_count=4,random_seed=123,verbose=100,allow_writing_files=False)
 print('REFIT year 2025',len(refit),flush=True);final.fit(X[refit],target[refit]/scales,sample_weight=np.minimum(w[refit],3))
 final.save_model(str(models/'forecast.cbm'));final.save_model(str(models/'forecast.json'),format='json')
 # Mettre l'échelle apprise dans les feuilles/biais pour une inférence C++ directe.
 m=json.loads((models/'forecast.json').read_text());m['scale_and_bias'][1]=[b*s for b,s in zip(m['scale_and_bias'][1],scales)]
 for tree in m['oblivious_trees']:
  tree['leaf_values']=[v*scales[i%15] for i,v in enumerate(tree['leaf_values'])]
 (models/'forecast_scaled.json').write_text(json.dumps(m));scaled=CatBoostRegressor();scaled.load_model(str(models/'forecast_scaled.json'),format='json');export_cpp(scaled,models/'forecast.txt',transform,np.ones(15))
 pa=scaled.predict(X,thread_count=4)
 if transform=='log':pa=np.expm1(pa)
 pa=(1-alpha)*np.maximum(pa,0)+alpha*rp
 # Comparaisons test figées : modèle final, saisonnier 2025 et récent.
 st=seasonal_fit(d,d['split']<2);season=seasonal_predict(d,st);np.save(models/'seasonal.npy',st)
 report={'selection_validation':{k:dict(score=score(v),metrics=v) for k,v in candidates.items()},'selected':best,'transform':transform,'recent_blend':alpha,'iterations':final.tree_count_,'train_last':'2025-12-31; cibles ne dépassant pas 2026-01-01','test_first':'2026-01-01','test_last':'2026-06-30','fit_seconds':timings,'n_train_refit':len(refit),'n_test':len(test),'feature_names':feature_names(),'horizons':HORIZONS,'meats':MEATS,'scales':scales.tolist(),'test':{name:metrics(Y[test],pr[test],w[test]) for name,pr in [('modele',pa),('saisonnier',season),('recent_5min',rp)]}}
 # Segments définis par signaux publics ; surprises évaluées séparément dans le simulateur.
 for segment,mask in [('creux',(d['hour']>=14)&(d['hour']<18)),('rush',((d['hour']>=12)&(d['hour']<14))|((d['hour']>=19)&(d['hour']<21)))]:
  ids=test[mask[test]];report.setdefault('test_segments',{})[segment]={name:metrics(Y[ids],pr[ids]) for name,pr in [('modele',pa),('saisonnier',season),('recent_5min',rp)]}
 report['test_monthly']={str(month):metrics(Y[test[d['month'][test]==month]],pa[test[d['month'][test]==month]]) for month in range(1,7)}
 deployed=np.maximum.accumulate(pa.reshape(-1,5,3),axis=1).reshape(-1,15)
 report['test_deployed']={name:metrics(Y[test],pr[test],w[test]) for name,pr in [('modele',deployed),('saisonnier',season),('recent_5min',rp)]}
 report['output_postprocessing']='Cumul des besoins rendu non decroissant selon l horizon, comme dans le moteur; test contient aussi les sorties brutes pour audit.'
 (results/'forecast_metrics.json').write_text(json.dumps(report,indent=2));(models/'metadata.json').write_text(json.dumps({k:report[k] for k in ['selected','transform','recent_blend','iterations','feature_names','horizons','meats','scales']},indent=2))
 np.save(root/'cache'/'pred_model.npy',pa.astype(np.float32));np.save(root/'cache'/'pred_seasonal.npy',season.astype(np.float32));np.save(root/'cache'/'pred_recent.npy',rp.astype(np.float32))
 # Réseau complet des prédictions de test aux minutes : 15 cibles, 6 mois.
 frame=pd.DataFrame({'prediction_a':pd.to_datetime(d['epochs'][test],unit='s',utc=True).tz_convert('Europe/Paris').astype(str)})
 for j,h in enumerate(HORIZONS):
  for k,meat in enumerate(MEATS):frame[f'{meat}_prevu_{h}min']=np.round(deployed[test,3*j+k],3);frame[f'{meat}_reel_{h}min']=Y[test,3*j+k]
 frame.to_csv(results/'predictions_6_mois.csv',index=False)
 print('TEST',json.dumps(report['test']),flush=True)
if __name__=='__main__':main()
