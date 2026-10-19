"""Unit and integration tests; never execute or modify project notebooks/data."""
from pathlib import Path
from contextlib import contextmanager
import copy
import importlib.util
import inspect
import json
import shutil
import unittest
from uuid import uuid4
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('bivariado_test','3.0.0.bivariado.py')
biv=importlib.util.module_from_spec(spec);spec.loader.exec_module(biv)


@contextmanager
def scratch():
    workspace=Path.cwd().resolve()
    folder=workspace/('biv_test_'+uuid4().hex)
    folder.mkdir()  # Inherit permissions; do not change Windows ACLs.
    try:yield folder
    finally:
        target=folder.resolve()
        assert target.parent==workspace and target.name.startswith('biv_test_')
        shutil.rmtree(target)


def fixture():
    index=pd.Index(np.arange(1200)*7+13)
    values=np.arange(1200)%100
    X=pd.DataFrame({'numeric':values.astype(float),'binary':(values>=50).astype(int),
                    'category':np.tile(['a','b','c'],400)},index=index)
    y=pd.Series(((np.arange(1200)%10)<(1+2*(values>=50))).astype(int),index=index,name='indicador_default')
    metadata=pd.DataFrame({'SEGMENTO':'TRAIN','periodo_observacion':np.repeat([202501,202502,202503],400),
                           'cliente_id':np.arange(1200)+10000,'indicador_default':y},index=index)
    config=biv.parameters(POLICY_VERSION='B.2',Q_VALUES=[2,3,4],MIN_BIN_N=20,MIN_BIN_PCT=.03,MIN_GOODS=2,MIN_BADS=2,
                          MIN_MONTH_BIN_N=10,MIN_MONTH_BIN_EVENTS=1)
    return X,y,metadata,config


def fit_fixture(X=None):
    base,y,metadata,config=fixture()
    X=base if X is None else X
    initial=biv.generate_candidates(X,config)
    candidates,details,stability,registry=biv.evaluate_candidates(X,y,metadata,initial,config)
    summary,artifact=biv.select_configurations(candidates,details,registry,metadata,config)
    return X,y,metadata,config,candidates,details,stability,registry,summary,artifact


class FormulaTests(unittest.TestCase):
    def test_woe_iv_formula(self):
        s=pd.Series([0]*4+[1]*4,dtype=float)
        y=pd.Series([0,0,0,1,0,1,1,1])
        candidate=biv.learn_candidate(s,2)
        stats=biv.bin_statistics(biv.assign_bins(s,candidate),y,candidate,.5)
        np.testing.assert_allclose(stats.WoE,[np.log(.7/.3),np.log(.3/.7)])
        self.assertAlmostEqual(stats.IV_bin.sum(),.8*np.log(.7/.3))
        self.assertAlmostEqual(stats.Dist_buenos_suavizada.sum(),1)
        self.assertAlmostEqual(stats.Dist_malos_suavizada.sum(),1)

    def test_pure_bins_are_finite_nonzero(self):
        s=pd.Series([0]*4+[1]*4);y=pd.Series([0]*4+[1]*4)
        c=biv.learn_candidate(s,2);stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        self.assertTrue(stats.Bin_puro.all());self.assertTrue(np.isfinite(stats.WoE).all())
        self.assertTrue(stats.WoE.ne(0).all())
        np.testing.assert_allclose(stats.WoE,[np.log(9),-np.log(9)])

    def test_missing_is_separate_from_zero(self):
        s=pd.Series([np.nan,0.,1.,np.nan,0.,1.]);c=biv.learn_candidate(s,3)
        labels=biv.assign_bins(s,c)
        self.assertEqual(labels.eq(biv.MISSING).sum(),2)
        self.assertNotEqual(labels.iloc[0],labels.iloc[1])

    def test_reserved_real_categories_do_not_collide(self):
        s=pd.Series(['__MISSING__','x',None]);c=biv.learn_candidate(s,2)
        labels=biv.assign_bins(s,c)
        self.assertNotEqual(labels.iloc[0],biv.MISSING);self.assertEqual(labels.iloc[2],biv.MISSING)

    def test_binary_real_categories_for_all_q(self):
        s=pd.Series([0,1]*200)
        for q in range(2,11):self.assertEqual(len(biv.learn_candidate(s,q)['ordered_bins']),2)

    def test_duplicate_quantiles_no_empty_bins(self):
        s=pd.Series([0.]*990+list(map(float,range(1,101))))
        c=biv.learn_candidate(s,10)
        labels=biv.assign_bins(s,c)
        self.assertEqual(set(labels),set(c['ordered_bins']))
        self.assertTrue(np.all(np.diff(c['edges'])>0))

    def test_boundaries_and_out_of_range(self):
        s=pd.Series(np.arange(100,dtype=float));c=biv.learn_candidate(s,4)
        cut=c['edges'][1]
        labels=biv.assign_bins(pd.Series([-1e50,cut,np.nextafter(cut,np.inf),1e50]),c)
        self.assertEqual(labels.iloc[0],'B000');self.assertEqual(labels.iloc[1],'B000')
        self.assertEqual(labels.iloc[2],'B001');self.assertEqual(labels.iloc[3],'B003')

    def test_infinities_are_not_missing(self):
        s=pd.Series([0.,1.,2.,np.inf,-np.inf,np.nan]);c=biv.learn_candidate(s,2)
        labels=biv.assign_bins(s,c)
        self.assertEqual(labels.iloc[3],biv.POS_INF);self.assertEqual(labels.iloc[4],biv.NEG_INF)
        self.assertEqual(labels.iloc[5],biv.MISSING)

    def test_target_misalignment_rejected(self):
        s=pd.Series([0,1,0,1]);y=pd.Series([0,1,0,1],index=[1,2,3,4]);c=biv.learn_candidate(s,2)
        with self.assertRaises(ValueError):biv.bin_statistics(biv.assign_bins(s,c),y,c)

    def test_single_class_rejected(self):
        s=pd.Series([0,1]*10);c=biv.learn_candidate(s,2)
        with self.assertRaises(ValueError):biv.bin_statistics(biv.assign_bins(s,c),pd.Series([0]*20),c)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):biv.parameters(ALPHA=0)
        with self.assertRaises(ValueError):biv.parameters(IV_RETAIN_FRACTION=1.2)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fit=fit_fixture()
    def tearDown(self):plt.close('all')

    def test_all_requested_candidates_preserved(self):
        X,y,m,c,candidates,*_=self.fit
        raw=candidates[candidates.Origen.eq('CUANTILES')]
        self.assertEqual(len(raw),len(X.columns)*len(c['Q_VALUES']))
        self.assertFalse(candidates.Candidate_ID.duplicated().any())

    def test_constant_requires_review(self):
        X,y,m,c=fixture();X=pd.DataFrame({'constant':1.},index=X.index)
        fit=fit_fixture(X)
        self.assertEqual(fit[-2].Estado_bivariado.iloc[0],'REQUIERE_REVISION')
        self.assertEqual(fit[-2].IV_total.iloc[0],0)

    def test_all_null_requires_review(self):
        X,y,m,c=fixture();X=pd.DataFrame({'missing':np.nan},index=X.index)
        fit=fit_fixture(X)
        self.assertEqual(fit[-2].Estado_bivariado.iloc[0],'REQUIERE_REVISION')
        self.assertEqual(set(fit[5].Bin),{biv.MISSING})

    def test_new_categories_and_unseen_missing_reported(self):
        artifact=self.fit[-1]
        X=self.fit[0].copy();X['category']='never_seen';X.iloc[0,X.columns.get_loc('binary')]=np.nan
        result,report=biv.transform_woe(X,artifact)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(report.set_index('Variable').loc['category','N_categorias_nuevas'],len(X))
        self.assertEqual(report.set_index('Variable').loc['binary','N_bins_sin_estimacion_train'],1)
        self.assertTrue(result.category.eq(0).all())

    def test_raise_unseen_policy(self):
        artifact=copy.deepcopy(self.fit[-1]);artifact['variables']['category']['unseen_policy']='raise'
        X=self.fit[0].copy();X['category']='not_known'
        with self.assertRaises(ValueError):biv.transform_woe(X,artifact)

    def test_serialization_exact_transformation(self):
        X=self.fit[0];artifact=self.fit[-1]
        before,_=biv.transform_woe(X,artifact)
        with scratch() as folder:
            file=biv.save_binning(artifact,folder/'bins.json')
            # A strict JSON parser must never encounter NaN/Infinity constants.
            json.loads(file.read_text(encoding='utf-8'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
            restored=biv.load_binning(file)
            after,_=biv.transform_woe(X,restored)
            pd.testing.assert_frame_equal(before,after,check_exact=True)
            for variable in X:
                self.assertEqual(artifact['variables'][variable]['edges'],restored['variables'][variable]['edges'])

    def test_transform_preserves_index_rows_and_original(self):
        X=self.fit[0];before=X.copy();out,_=biv.transform_woe(X,self.fit[-1])
        pd.testing.assert_frame_equal(X,before,check_exact=True)
        self.assertTrue(out.index.equals(X.index));self.assertTrue(out.columns.equals(X.columns))
        self.assertEqual(len(out),len(X))

    def test_transform_reproducible_and_model_immutable(self):
        X=self.fit[0];artifact=self.fit[-1];before=copy.deepcopy(artifact)
        first,_=biv.transform_woe(X,artifact);second,_=biv.transform_woe(X,artifact)
        pd.testing.assert_frame_equal(first,second,check_exact=True);self.assertEqual(before,artifact)

    def test_no_test_oot_in_learning_api(self):
        for function in [biv.generate_candidates,biv.evaluate_candidates,biv.select_configurations]:
            self.assertFalse(any('test' in name or 'oot' in name for name in inspect.signature(function).parameters))
        X,y,m,c=fixture();m=m.assign(SEGMENTO='TEST')
        with self.assertRaises(ValueError):biv.evaluate_candidates(X,y,m,biv.generate_candidates(X,c),c)

    def test_holdout_changes_do_not_change_train_model(self):
        X=self.fit[0];artifact=copy.deepcopy(self.fit[-1]);holdout=X.copy()
        holdout['numeric']=1e100;holdout['category']='unseen'
        biv.transform_woe(holdout,artifact)
        self.assertEqual(artifact,self.fit[-1])
        # Repeated TRAIN selection must be deterministic, ignoring created_at.
        rerun=fit_fixture()[-1]
        self.assertEqual(biv._json_safe(rerun['variables']),biv._json_safe(artifact['variables']))

    def test_monotonicity_not_automatic_rejection(self):
        X,y,m,c,candidates,details,stability,registry,summary,artifact=self.fit
        altered=candidates.copy();altered['Monotonicidad']='NO_MONOTONA';altered['Calidad_ok']=True
        selected,_=biv.select_configurations(altered,details,registry,m,c)
        self.assertTrue(selected.Estado_bivariado.eq('APROBADA').all())

    def test_high_iv_not_automatic_rejection(self):
        X,y,m,c,candidates,details,stability,registry,*_=self.fit
        altered=candidates.copy();altered['IV_total']=.9;altered['Calidad_ok']=True
        selected,_=biv.select_configurations(altered,details,registry,m,c)
        self.assertTrue(selected.Estado_bivariado.eq('APROBADA').all())

    def test_simpler_configuration_within_iv_margin(self):
        X,y,m,c,candidates,details,stability,registry,*_=self.fit
        group=candidates[candidates.Variable.eq('numeric') & candidates.Origen.eq('CUANTILES')].copy()
        group['Calidad_ok']=True;group['IV_total']=[.2,.21,.22]
        selected,_=biv.select_configurations(group,details,registry,m,c)
        self.assertEqual(selected.q_solicitado.iloc[0],2)

    def test_statistics_reconcile_all_rows(self):
        X,y,m,c,candidates,details,*_=self.fit
        for key,group in details.groupby('Candidate_ID'):
            self.assertEqual(group.N.sum(),len(X));self.assertEqual(group.Malos.sum(),y.sum())
            self.assertAlmostEqual(group.Dist_clientes.sum(),1)
            self.assertAlmostEqual(group.Dist_buenos_suavizada.sum(),1)
            self.assertAlmostEqual(group.Dist_malos_suavizada.sum(),1)
            self.assertTrue(np.isfinite(group.WoE).all());self.assertTrue(group.IV_bin.ge(0).all())

    def test_monthly_rates_reconcile_bins(self):
        stability=self.fit[6];details=self.fit[5]
        sums=stability.groupby(['Candidate_ID','Bin'])[['N','Malos']].sum().sort_index()
        expected=details.set_index(['Candidate_ID','Bin'])[['N','Malos']].sort_index()
        pd.testing.assert_frame_equal(sums,expected)

    def test_missing_month_not_drawn_as_zero(self):
        stability=self.fit[6].copy();summary=self.fit[-2]
        variable=summary.Variable.iloc[0];key=summary.Candidate_ID.iloc[0]
        selected=stability[stability.Candidate_ID.eq(key) & stability.Periodo.ne('2025-02')]
        figure,axis=biv.plot_monthly_rates(variable,selected,summary)
        self.assertTrue(any(np.isnan(line.get_ydata()[1]) for line in axis.lines))

    def test_plots_do_not_write(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        summary=self.fit[-2];details=self.fit[5];selected=details[details.Candidate_ID.isin(summary.Candidate_ID)]
        with patch.object(Figure,'savefig',side_effect=AssertionError('Unexpected file write')):
            biv.plot_variable(summary.Variable.iloc[0],selected,summary)
            self.assertEqual(len(biv.plot_dashboard(selected,summary)),1)

    def test_selected_model_keeps_every_variable(self):
        self.assertEqual(set(self.fit[-1]['variables']),set(self.fit[0].columns))


class ArtifactTests(unittest.TestCase):
    def make_artifacts(self,folder):
        X,y,m,c=fixture();data=X.copy()
        data['SEGMENTO']=np.repeat(['TRAIN','TEST','OOT'],400)
        data['indicador_default']=y;data['cliente_id']=m.cliente_id;data['periodo_observacion']=m.periodo_observacion
        source=folder/'source.csv';data.to_csv(source,index=False)
        psi=folder/'psi';(psi/'reportes').mkdir(parents=True);(psi/'evidencias').mkdir()
        pd.DataFrame({'Variable':X.columns,'Estado_PSI':'APROBADA'}).to_csv(psi/'reportes/df_resumen_psi.csv',index=False)
        pd.DataFrame({'Variable':X.columns,'Filtro':'PSI mensual','Decision':'Conservar'}).to_csv(psi/'reportes/df_reporte_univariado.csv',index=False)
        (psi/'evidencias/ejecucion.json').write_text(json.dumps({'source_sha256':biv.sha256(source),'v4_preserved':True,
            'configuration':{'AUTO_DROP_PSI':False},'v4_columns':len(X.columns),'v3_columns':len(X.columns)}),encoding='utf-8')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            manifest=biv.export_v4_from_psi(source,psi,folder/'biv')
        return source,psi,manifest

    def test_persisted_load_independent_of_source(self):
        with scratch() as folder:
            source,psi,manifest=self.make_artifacts(folder)
            source.unlink()  # Only disposable synthetic fixture; never the project source.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');bundle=biv.load_v4(manifest_file=manifest)
            self.assertEqual(sum(len(X) for X in bundle['X'].values()),1200)
            self.assertFalse(bundle['manifest']['psi_recalculated'])
            self.assertFalse(bundle['manifest']['filters_recalculated'])

    def test_tamper_detection(self):
        with scratch() as folder:
            source,psi,manifest=self.make_artifacts(folder)
            file=manifest.parent/'X_train_v4.csv';file.write_bytes(file.read_bytes()+b'\n')
            with self.assertRaises(ValueError):biv.load_v4(manifest_file=manifest)

    def test_source_hash_mismatch_rejected(self):
        with scratch() as folder:
            source,psi,manifest=self.make_artifacts(folder)
            source.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaises(ValueError):biv.export_v4_from_psi(source,psi,folder/'other')

    def test_no_project_notebook_modification_on_import(self):
        paths=[Path('2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb'),Path('3.0.0.SCR_DES_ANLS_BIVARIADO.ipynb')]
        before=[biv.sha256(p) for p in paths]
        spec=importlib.util.spec_from_file_location('bivariado_runner','3.0.1.ejecutar_bivariado.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(before,[biv.sha256(p) for p in paths])




class B2Tests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def example(self, sizes=(200,200,200,200), bads=(20,80,40,120), missing_n=0, missing_bad=0):
        values, targets = [], []
        for i,(n,b) in enumerate(zip(sizes,bads)):
            values += [float(i)]*n; targets += [1]*b+[0]*(n-b)
        values += [np.nan]*missing_n; targets += [1]*missing_bad+[0]*(missing_n-missing_bad)
        index=pd.Index(np.arange(len(values))*5+7)
        s=pd.Series(values,index=index);y=pd.Series(targets,index=index,name='indicador_default')
        metadata=pd.DataFrame({'SEGMENTO':'TRAIN', 'periodo_observacion':np.resize([202501,202502,202503],len(s)),
            'cliente_id':np.arange(len(s)), 'indicador_default':y},index=index)
        config=biv.parameters(POLICY_VERSION='B.2',Q_VALUES=[len(sizes) if len(sizes)>1 else 2],MIN_BIN_N=5,MIN_GOODS=2,MIN_BADS=2,
            MIN_MONTH_BIN_N=2,MIN_MONTH_BIN_EVENTS=1)
        c=biv.learn_candidate(s,max(2,len(sizes)),config)
        c.update(kind='intervals',method='quantiles',edges=[-np.inf]+[i+.5 for i in range(len(sizes)-1)]+[np.inf])
        biv._refresh_intervals(c)
        return s,y,metadata,config,c

    def evaluate_one(self, example):
        s,y,metadata,config,c=example
        key='x::q'+str(c['q_requested'])
        registry={key:{'Variable':'x','Candidate_ID':key,'configuration':c,'Origen':'CUANTILES'}}
        config={**config,'MERGE_ADJACENT':False}
        return biv.evaluate_candidates(s.to_frame('x'),y,metadata,registry,config)

    def freeze(self,s,y,c):
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        model=copy.deepcopy(c)
        model.update(woe_by_bin=dict(zip(stats.Bin,stats.WoE)),unseen_policy='training_prior_woe_0',unseen_woe=0.,
                     bin_statistics=stats.to_dict('records'))
        return {'schema_version':2,'methodology_version':'B.2','variables':{'x':model}}

    def corrected(self,example):
        s,y,m,config,c=example
        variants,audit=biv.search_adjacent_partitions(s,y,c,config,m)
        for variant in variants:
            stats=biv.bin_statistics(biv.assign_bins(s,variant),y,variant)
            mono,_=biv.monotonicity(stats,variant,config)
            normal=stats[~stats.Es_especial]
            if mono in ['CRECIENTE','DECRECIENTE'] and normal.Dist_clientes.ge(.05-1e-12).all():
                return variant,audit
        self.fail('Expected a valid contiguous correction in this synthetic fixture')

    def test_01_missing_excluded_by_internal_code(self):
        s,y,m,config,c=self.example((200,200,200),(20,60,120),30,0)
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        stats.loc[stats.Bin.eq(biv.MISSING),'Etiqueta']='Etiqueta modificada'
        c['ordered_bins']=[biv.MISSING]+list(reversed(c['ordered_bins']))
        self.assertEqual(biv.monotonicity(stats.sample(frac=1,random_state=2),c,config)[0],'CRECIENTE')

    def test_02_missing_contributes_iv(self):
        s,y,m,config,c=self.example((200,200,200),(20,60,120),40,39)
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        missing=stats[stats.Bin.eq(biv.MISSING)]
        self.assertGreater(missing.IV_bin.sum(),0)
        self.assertAlmostEqual(stats.IV_bin.sum(),stats.loc[~stats.Es_missing,'IV_bin'].sum()+missing.IV_bin.sum())

    def test_03_missing_retained_in_woe(self):
        s,y,m,config,c=self.example((200,200,200),(20,60,120),2,0)
        artifact=self.freeze(s,y,c);result,diagnostic=biv.transform_woe(s.to_frame('x'),artifact)
        expected=artifact['variables']['x']['woe_by_bin'][biv.MISSING]
        self.assertTrue(result.loc[s.isna(),'x'].eq(expected).all())
        self.assertTrue(np.isfinite(expected));self.assertNotEqual(expected,0)

    def test_04_missing_under_five_percent_allowed(self):
        example=self.example((200,200,200),(20,60,120),2,0)
        candidates,details,*_=self.evaluate_one(example)
        self.assertTrue(candidates.Calidad_ok.iloc[0])
        self.assertTrue(candidates.Alerta_missing_escaso.iloc[0])
        missing=details[details.Bin.eq(biv.MISSING)].iloc[0]
        self.assertLess(missing.Dist_clientes,.05);self.assertTrue(missing.Exento_min_pct)

    def test_05_small_numeric_bin_merged(self):
        example=self.example((200,10,200,200),(20,2,80,120))
        corrected,_=self.corrected(example)
        stats=biv.bin_statistics(biv.assign_bins(example[0],corrected),example[1],corrected)
        self.assertTrue(stats.loc[~stats.Es_especial,'Dist_clientes'].ge(.05).all())
        self.assertGreater(len(corrected['fusion_history']),0)

    def test_06_monotonicity_fixed_by_merge(self):
        example=self.example();s,y,m,config,c=example
        self.assertEqual(biv.monotonicity(biv.bin_statistics(biv.assign_bins(s,c),y,c),c,config)[0],'NO_MONOTONA')
        corrected,_=self.corrected(example)
        self.assertIn(biv.monotonicity(biv.bin_statistics(biv.assign_bins(s,corrected),y,corrected),corrected,config)[0],['CRECIENTE','DECRECIENTE'])
        self.assertGreaterEqual(len(corrected['ordered_bins']),2)

    def test_07_rates_recomputed_from_counts(self):
        corrected,_=self.corrected(self.example((200,100,200,200),(20,40,40,120)))
        for step in corrected['fusion_history']:
            self.assertAlmostEqual(step['Tasa_after'],step['Malos_after']/step['N_after'])
            self.assertEqual(step['N_after'],step['Buenos_after']+step['Malos_after'])

    def test_08_iv_recomputed_after_merge(self):
        example=self.example();s,y,m,config,c=example;merged,_=self.corrected(example)
        before=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        after=biv.bin_statistics(biv.assign_bins(s,merged),y,merged)
        expected=((after.Dist_buenos_suavizada-after.Dist_malos_suavizada)*np.log(after.Dist_buenos_suavizada/after.Dist_malos_suavizada)).sum()
        self.assertAlmostEqual(after.IV_bin.sum(),expected)
        self.assertNotAlmostEqual(before.IV_bin.sum(),after.IV_bin.sum())
        self.assertAlmostEqual(merged['fusion_history'][-1]['IV_after'],after.IV_bin.sum())

    def test_09_woe_recomputed_after_merge(self):
        example=self.example();s,y,m,config,c=example;merged,_=self.corrected(example)
        stats=biv.bin_statistics(biv.assign_bins(s,merged),y,merged)
        good=(stats.Buenos+.5)/(stats.Buenos.sum()+.5*len(stats))
        bad=(stats.Malos+.5)/(stats.Malos.sum()+.5*len(stats))
        np.testing.assert_allclose(stats.WoE,np.log(good/bad))

    def test_10_only_adjacent_bins_merged(self):
        merged,_=self.corrected(self.example())
        for step in merged['fusion_history']:
            left=int(step['left_original_bins'][-1][1:]);right=int(step['right_original_bins'][0][1:])
            self.assertEqual(left+1,right);self.assertTrue(step['adjacent'])

    def test_11_missing_never_merged(self):
        example=self.example(missing_n=30,missing_bad=29)
        s,y,m,config,c=example;merged,_=self.corrected(example)
        before=biv.bin_statistics(biv.assign_bins(s,c),y,c).set_index('Bin').loc[biv.MISSING]
        after=biv.bin_statistics(biv.assign_bins(s,merged),y,merged).set_index('Bin').loc[biv.MISSING]
        self.assertEqual(before.N,after.N);self.assertEqual(before.Malos,after.Malos)
        self.assertTrue(all(not row['missing_merged'] for row in merged['fusion_history']))

    def test_12_binary_natural_categories(self):
        s=pd.Series([0.]*990+[1.]*10);y=pd.Series([0,1]*500)
        c=biv.learn_candidate(s,10);v,a=biv.search_adjacent_partitions(s,y,c,biv.parameters())
        self.assertEqual(v,[]);self.assertEqual(len(c['category_groups']),2)
        self.assertTrue(a['binary_or_nominal_natural'])

    def test_13_many_zeros_no_artificial_splits(self):
        s=pd.Series([0.]*990+list(map(float,range(1,11))))
        c=biv.learn_candidate(s,10)
        labels=biv.assign_bins(s,c)
        self.assertEqual(labels[s.eq(0)].nunique(),1)
        y=pd.Series([0,1]*500)
        variants,_=biv.search_adjacent_partitions(s,y,c,biv.parameters())
        for candidate in variants:
            self.assertEqual(biv.assign_bins(s,candidate)[s.eq(0)].nunique(),1)

    def test_14_nonnull_mass_infeasible(self):
        candidates,*_=self.evaluate_one(self.example((40,40),(4,16),920,460))
        self.assertFalse(candidates.Calidad_ok.iloc[0])
        self.assertIn('NO_FACTIBLE_MASA_ORDINARIA',candidates.Motivos.iloc[0])

    def test_15_one_ordinary_bin_never_approved(self):
        candidates,*_=self.evaluate_one(self.example((600,),(120,),20,19))
        self.assertFalse(candidates.Calidad_ok.iloc[0])
        self.assertIn('MENOS_DE_DOS_BINS',candidates.Motivos.iloc[0])

    def test_16_limits_domain_coverage(self):
        example=self.example();merged,_=self.corrected(example);base=example[-1]
        self.assertEqual(merged['edges'][0],-np.inf);self.assertEqual(merged['edges'][-1],np.inf)
        self.assertTrue(np.all(np.diff(merged['edges'])>0))
        self.assertTrue(set(merged['edges']).issubset(base['edges']))
        labels=biv.assign_bins(pd.Series([-1e100,1e100]),merged)
        self.assertTrue(labels.str.startswith('B').all())

    def test_17_b2_json_roundtrip(self):
        example=self.example(missing_n=3,missing_bad=0);merged,_=self.corrected(example)
        artifact=self.freeze(example[0],example[1],merged)
        with scratch() as folder:
            biv.save_binning(artifact,folder/'b2.json');loaded=biv.load_binning(folder/'b2.json')
            self.assertEqual(loaded['methodology_version'],'B.2')
            self.assertEqual(artifact['variables']['x']['fusion_history'],loaded['variables']['x']['fusion_history'])
            self.assertEqual(artifact['variables']['x']['edges'],loaded['variables']['x']['edges'])

    def test_18_transform_three_segments_and_new_clients(self):
        example=self.example(missing_n=3);merged,_=self.corrected(example)
        artifact=self.freeze(example[0],example[1],merged)
        with scratch() as folder:
            biv.save_binning(artifact,folder/'b2.json');loaded=biv.load_binning(folder/'b2.json')
            for offset in [0,10000,20000]:
                X=example[0].to_frame('x').copy();X.index=X.index+offset
                first,_=biv.transform_woe(X,artifact);second,_=biv.transform_woe(X,loaded)
                pd.testing.assert_frame_equal(first,second,check_exact=True)
            new=pd.DataFrame({'x':[-1e100,.5,1.5,1e100,np.nan]})
            first,_=biv.transform_woe(new,artifact);second,_=biv.transform_woe(new,loaded)
            pd.testing.assert_frame_equal(first,second,check_exact=True)

    def test_19_only_train_for_merges(self):
        s,y,m,config,c=self.example()
        with self.assertRaises(ValueError):
            biv.search_adjacent_partitions(s,y,c,config,m.assign(SEGMENTO='OOT'))
        self.assertFalse(any('test' in name or 'oot' in name for name in inspect.signature(biv.search_adjacent_partitions).parameters))

    def test_20_flat_numeric_trend_not_useful(self):
        candidates,details,*_=self.evaluate_one(self.example((200,200,200),(40,40,40),20,19))
        self.assertEqual(candidates.Monotonicidad.iloc[0],'NO_EVALUABLE')
        self.assertFalse(candidates.Calidad_ok.iloc[0])
        self.assertIn('DIFERENCIACION_DE_RIESGO',candidates.Motivos.iloc[0])

    def test_21_exhaustive_search_in_both_directions(self):
        s,y,m,config,c=self.example((200,200,200,200),(80,20,100,40))
        variants,audit=biv.search_adjacent_partitions(s,y,c,config,m)
        self.assertEqual(audit['partitions_examined'],2**(4-1)-2)
        directions={value['search_direction'] for value in variants}
        self.assertIn('CRECIENTE',directions);self.assertIn('DECRECIENTE',directions)
        for value in variants:
            self.assertGreaterEqual(len(value['ordered_bins']),2)

    def test_22_nominal_categories_have_no_numeric_order(self):
        s=pd.Series(['10','2','1']*200);y=pd.Series([0,1]*300)
        c=biv.learn_candidate(s,3);self.assertEqual(biv.ordinary_order(c),[])
        variants,audit=biv.search_adjacent_partitions(s,y,c,biv.parameters())
        self.assertEqual(variants,[])
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        self.assertEqual(biv.monotonicity(stats,c,biv.parameters())[0],'NO_EVALUABLE')

    def test_23_graph_missing_is_final_isolated_point(self):
        s,y,m,config,c=self.example((200,200,200),(20,60,120),3,0)
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        summary=pd.DataFrame([{'Variable':'x','IV_total':stats.IV_bin.sum(),'Monotonicidad':'CRECIENTE',
            'Estado_bivariado':'APROBADA','N_bins_efectivos':3,'N_fusiones':0}])
        figure,axis,secondary=biv.plot_variable('x',stats.assign(Variable='x'),summary)
        for line in secondary.lines:
            if len(line.get_xdata())>1:self.assertNotIn(3,line.get_xdata())
        self.assertEqual(len(secondary.lines[-1].get_xdata()),1)
        self.assertIn('Ausencia',axis.get_xticklabels()[-1].get_text())

    def test_24_two_bins_need_risk_justification(self):
        candidates,*_=self.evaluate_one(self.example((300,300),(60,61)))
        self.assertFalse(candidates.Calidad_ok.iloc[0])
        self.assertIn('DOS_BINS_SIN_JUSTIFICACION',candidates.Motivos.iloc[0])

    def test_25_five_percent_denominator_includes_missing(self):
        example=self.example((40,400,360),(4,120,180),200,20)
        s,y,m,config,c=example
        stats=biv.bin_statistics(biv.assign_bins(s,c),y,c)
        self.assertAlmostEqual(stats.iloc[0].Dist_clientes,.04)
        candidates,*_=self.evaluate_one(example)
        self.assertFalse(candidates.Cumple_5pct.iloc[0])

    def test_26_numeric_categories_merge_reproducibly(self):
        s,y,m,config,_=self.example((200,10,200,200),(20,2,80,120))
        c=biv.learn_candidate(s,4,config)  # Four numeric categories, not intervals.
        variants,_=biv.search_adjacent_partitions(s,y,c,config,m)
        self.assertTrue(variants)
        merged=next(v for v in variants if len(v['ordered_bins'])==3)
        self.assertEqual(len(merged['category_groups']),4)
        self.assertTrue(merged['fusion_history'])
        self.assertEqual(set(biv.assign_bins(s,merged)),set(merged['ordered_bins']))


    def test_27_export_cleanup_preserves_custom_and_unrequested_images(self):
        with scratch() as folder:
            (folder/'evidencias').mkdir()
            for name in ['graficos_individuales','subplots']:(folder/name).mkdir()
            old=folder/'graficos_individuales/001_old.png';old.write_bytes(b'generated')
            custom=folder/'graficos_individuales/custom.png';custom.write_bytes(b'custom')
            subplot=folder/'subplots/bivariado_1.png';subplot.write_bytes(b'generated')
            (folder/'evidencias/archivos_exportados.json').write_text(json.dumps({
                'graficos_individuales/001_old.png':'hash','subplots/bivariado_1.png':'hash'}))
            biv._remove_previous_generated_pngs(folder,['graficos_individuales'])
            self.assertFalse(old.exists());self.assertTrue(custom.exists());self.assertTrue(subplot.exists())



class B3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X,cls.y,cls.metadata,legacy=fixture()
        cls.config=biv.parameters(**{**legacy,'POLICY_VERSION':'B.3','MIN_FIT_OBS':20})
        raw=biv.generate_candidates(cls.X,cls.config)
        cls.candidates,cls.details,cls.stability,cls.registry=biv.evaluate_candidates(
            cls.X,cls.y,cls.metadata,raw,cls.config)
        cls.summary,cls.artifact=biv.select_configurations(cls.candidates,cls.details,cls.registry,cls.metadata,cls.config)

    def tearDown(self):plt.close('all')

    def bundle(self,variant=False):
        X={s:self.X.copy() for s in ['TRAIN','TEST','OOT']}
        y={s:self.y.copy() for s in X}
        if variant:
            mask=self.X.numeric.lt(50)
            X['TEST']=self.X.loc[mask].iloc[:20].copy();y['TEST']=self.y.loc[X['TEST'].index].copy()
            X['OOT']=self.X.iloc[:30].copy();y['OOT']=pd.Series(0,index=X['OOT'].index,name='indicador_default')
        metadata={s:self.metadata.loc[X[s].index].assign(SEGMENTO=s,indicador_default=y[s]) for s in X}
        return {'X':X,'y':y,'metadata':metadata}

    def fit_example(self,example,**overrides):
        s,y,m,config,c=example
        config=biv.parameters(**{**config,'POLICY_VERSION':'B.3','MIN_FIT_OBS':20,**overrides})
        key='x::q'+str(c['q_requested'])
        raw={key:{'Variable':'x','Candidate_ID':key,'configuration':c,'Origen':'CUANTILES'}}
        candidates,details,stability,registry=biv.evaluate_candidates(s.to_frame('x'),y,m,raw,config)
        summary,artifact=biv.select_configurations(candidates,details,registry,m,config)
        return candidates,details,summary,artifact

    def test_01_valid_two_bins_without_special_penalty(self):
        example=B2Tests().example((300,300),(30,90))
        _,_,summary,_=self.fit_example(example,MIN_PREFERRED_BINS=99,MIN_TWO_BIN_AUC=.99,
            MIN_TWO_BIN_IV=1.,MIN_TWO_BIN_RATE_GAP=.9,MIN_IV_ORDINARY=.99)
        self.assertEqual(summary.N_bins_efectivos.iloc[0],2)
        self.assertEqual(summary.Estado_bivariado.iloc[0],'APROBADA')
        self.assertEqual(summary.N_fusiones.iloc[0],0)

    def test_02_reject_total_iv_under_point02(self):
        _,_,summary,_=self.fit_example(B2Tests().example((300,300),(60,63)))
        self.assertLess(summary.IV_total.iloc[0],.02)
        self.assertEqual(summary.Estado_bivariado.iloc[0],'RECHAZADA')
        self.assertIn('IV_TOTAL_INFERIOR',summary.Motivos.iloc[0])

    def test_03_reject_single_numeric_bin(self):
        _,_,summary,_=self.fit_example(B2Tests().example((600,),(120,)))
        self.assertEqual(summary.Estado_bivariado.iloc[0],'RECHAZADA')
        self.assertIn('MENOS_DE_DOS_BINS',summary.Motivos.iloc[0])

    def test_04_reject_impossible_valid_monotonicity(self):
        _,_,summary,_=self.fit_example(B2Tests().example((300,300,300),(60,60,60)))
        self.assertEqual(summary.Estado_bivariado.iloc[0],'RECHAZADA')
        self.assertIn('MONOTONICIDAD_ESTRICTA_NO_RESUELTA',summary.Motivos.iloc[0])

    def test_05_five_percent_all_approved(self):
        approved=self.summary.loc[self.summary.Estado_bivariado.eq('APROBADA'),'Candidate_ID']
        bins=self.details[self.details.Candidate_ID.isin(approved)&~self.details.Es_especial]
        self.assertTrue(bins.Dist_clientes.ge(.05-1e-12).all())

    def test_06_missing_excluded_and_included_in_iv_woe(self):
        candidates,details,summary,artifact=self.fit_example(B2Tests().example((300,300),(60,63),60,50))
        self.assertEqual(summary.Estado_bivariado.iloc[0],'APROBADA')
        cfg=artifact['variables']['x'];stats=pd.DataFrame(cfg['bin_statistics'])
        self.assertIn(biv.MISSING,cfg['woe_by_bin'])
        self.assertGreater(stats.loc[stats.Bin.eq(biv.MISSING),'IV_bin'].sum(),0)
        before=biv.monotonicity(stats,cfg,artifact['parameters'])[0]
        stats.loc[stats.Bin.eq(biv.MISSING),'Tasa_default']=0.
        self.assertEqual(biv.monotonicity(stats,cfg,artifact['parameters'])[0],before)
        self.assertGreaterEqual(summary.IV_total.iloc[0],.02)

    def test_07_holdouts_no_fit_no_fusions_no_new_train_woe(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in ['learn_candidate','generate_candidates','search_adjacent_partitions',
                         'merge_adjacent','evaluate_candidates','select_configurations','bin_statistics','_counts_metrics']:
                stack.enter_context(patch.object(biv,name,side_effect=AssertionError('FIT called during evaluation')))
            stats,comparison,aggregate=biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        self.assertGreater(len(stats),0)
        self.assertEqual(set(stats.Segmento),{'TRAIN','TEST','OOT'})

    def test_08_artifact_unchanged_by_holdouts(self):
        before=json.dumps(biv._json_safe(self.artifact),sort_keys=True)
        biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        self.assertEqual(before,json.dumps(biv._json_safe(self.artifact),sort_keys=True))

    def test_09_identical_bin_order_all_samples(self):
        stats,_,_=biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        for variable,group in stats.groupby('Variable'):
            expected=group[group.Segmento.eq('TRAIN')].sort_values('Orden_bin').Bin.tolist()
            for segment in ['TEST','OOT']:
                self.assertEqual(group[group.Segmento.eq(segment)].sort_values('Orden_bin').Bin.tolist(),expected)

    def test_10_same_dashboard_variables_positions_and_scales(self):
        stats,_,_=biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        pages={s:biv.plot_sample_dashboard(stats,self.summary,self.artifact,s) for s in ['TRAIN','TEST','OOT']}
        reference=pages['TRAIN']
        for segment,figures in pages.items():
            self.assertEqual([f._biv_variables for f in figures],[f._biv_variables for f in reference])
            for a,b in zip(figures,reference):
                self.assertEqual([ax.get_ylim() for ax in a.axes if ax.get_visible()],
                                 [ax.get_ylim() for ax in b.axes if ax.get_visible()])

    def test_11_empty_bins_have_nan_rates(self):
        stats,comparison,_=biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        empty=stats[stats.Segmento.eq('TEST')&stats.N.eq(0)]
        self.assertTrue(len(empty)>0)
        self.assertTrue(empty.Tasa_default.isna().all())
        self.assertTrue(empty.Dist_clientes.eq(0).all())

    def test_12_missing_last_and_isolated_in_sample_plot(self):
        stats,_,_=biv.evaluate_frozen_samples(self.bundle(),self.artifact)
        variable=self.artifact['approved_variables'][0]
        _,axis,secondary=biv.plot_variable(variable,stats[stats.Segmento.eq('TEST')],self.summary,segmento='TEST')
        last=len(axis.get_xticklabels())-1
        self.assertIn('Ausencia',axis.get_xticklabels()[-1].get_text())
        for line in secondary.lines:
            if len(line.get_xdata())>1:self.assertNotIn(last,line.get_xdata())

    def test_13_json_exact_approved_matrices(self):
        with scratch() as folder:
            biv.save_binning(self.artifact,folder/'b3.json')
            model=biv.load_binning(folder/'b3.json')
            self.assertEqual(model['schema_version'],3)
            self.assertEqual(model['approved_variables'],self.artifact['approved_variables'])
            for segment,X in self.bundle(True)['X'].items():
                first,_=biv.transform_woe(X,self.artifact);second,_=biv.transform_woe(X,model)
                pd.testing.assert_frame_equal(first,second,check_exact=True)

    def test_14_originals_preserved_for_xgboost(self):
        bundle=self.bundle(True);before={s:X.copy() for s,X in bundle['X'].items()}
        for s,X in bundle['X'].items():biv.transform_woe(X,self.artifact)
        biv.evaluate_frozen_samples(bundle,self.artifact)
        for s,X in bundle['X'].items():pd.testing.assert_frame_equal(X,before[s],check_exact=True)

    def test_15_rejected_excluded_and_order_fixed(self):
        self.assertIn('category',self.artifact['rejected_variables'])
        for s,X in self.bundle()['X'].items():
            result,_=biv.transform_woe(X.loc[:,list(reversed(X.columns))],self.artifact)
            self.assertEqual(result.columns.tolist(),self.artifact['approved_variables'])
            self.assertNotIn('category',result.columns)

    def test_16_insufficient_data_not_rejected(self):
        X=self.X.iloc[:10];y=self.y.iloc[:10];m=self.metadata.iloc[:10]
        raw=biv.generate_candidates(X,self.config)
        r,d,st,reg=biv.evaluate_candidates(X,y,m,raw,self.config)
        summary,model=biv.select_configurations(r,d,reg,m,self.config)
        self.assertTrue(summary.Estado_bivariado.eq('NO_EVALUABLE').all())
        self.assertTrue(summary.IV_total.isna().all())
        self.assertEqual(biv.transform_woe(X,model)[0].shape,(len(X),0))

    def test_17_all_null_no_evaluable_constant_rejected(self):
        X=pd.DataFrame({'constant':1.,'null':np.nan},index=self.X.index)
        r,d,st,reg=biv.evaluate_candidates(X,self.y,self.metadata,biv.generate_candidates(X,self.config),self.config)
        summary,model=biv.select_configurations(r,d,reg,self.metadata,self.config)
        by=summary.set_index('Variable')
        self.assertEqual(by.loc['null','Estado_bivariado'],'NO_EVALUABLE')
        self.assertEqual(by.loc['constant','Estado_bivariado'],'RECHAZADA')

    def test_18_technical_error_no_evaluable(self):
        from unittest.mock import patch
        raw=biv.generate_candidates(self.X[['numeric']],self.config)
        with patch.object(biv,'assign_bins',side_effect=RuntimeError('technical fixture failure')):
            r,d,st,reg=biv.evaluate_candidates(self.X[['numeric']],self.y,self.metadata,raw,self.config)
        summary,model=biv.select_configurations(r,d,reg,self.metadata,self.config)
        self.assertEqual(summary.Estado_bivariado.iloc[0],'NO_EVALUABLE')
        self.assertTrue(summary.IV_total.isna().all())
        self.assertIn('technical fixture failure',summary.Motivos.iloc[0])

    def test_19_low_volume_warns_without_changing_edges(self):
        stats,comparison,_=biv.evaluate_frozen_samples(self.bundle(True),self.artifact)
        self.assertTrue(stats.loc[stats.Segmento.eq('TEST'),'Advertencia_volumen'].eq('BAJO_VOLUMEN').any())
        self.assertTrue(comparison.Advertencia_bajo_volumen.str.contains('BAJO_VOLUMEN').any())

    def test_20_comparison_counts_rates_and_pp_correct(self):
        bundle=self.bundle(True);stats,comparison,aggregate=biv.evaluate_frozen_samples(bundle,self.artifact)
        for (variable,segment),group in stats.groupby(['Variable','Segmento']):
            self.assertEqual(group.N.sum(),len(bundle['X'][segment]))
            self.assertEqual(group.Malos.sum(),bundle['y'][segment].sum())
            self.assertAlmostEqual(group.Dist_clientes.sum(),1)
        np.testing.assert_allclose(comparison.Diferencia_pp_TEST_TRAIN,
            100*(comparison.Tasa_malos_TEST-comparison.Tasa_malos_TRAIN),equal_nan=True)

    def test_21_new_categories_reported_without_learning(self):
        s=pd.Series(['a']*300+['b']*300)
        y=pd.Series([1]*30+[0]*270+[1]*90+[0]*210,name='indicador_default')
        m=pd.DataFrame({'SEGMENTO':'TRAIN','periodo_observacion':np.resize([202501,202502,202503],600),
                        'indicador_default':y,'cliente_id':np.arange(600)})
        X=s.to_frame('cat');config=biv.parameters(MIN_FIT_OBS=20,MIN_BIN_N=5,MIN_GOODS=2,MIN_BADS=2,
            MIN_MONTH_BIN_N=2,MIN_MONTH_BIN_EVENTS=1,Q_VALUES=[2,3])
        r,d,st,reg=biv.evaluate_candidates(X,y,m,biv.generate_candidates(X,config),config)
        summary,model=biv.select_configurations(r,d,reg,m,config)
        self.assertEqual(model['approved_variables'],['cat'])
        bundle={'X':{'TRAIN':X,'TEST':pd.DataFrame({'cat':['new','a','b']}),'OOT':X},
                'y':{'TRAIN':y,'TEST':pd.Series([1,0,1]),'OOT':y}}
        snapshot=json.dumps(biv._json_safe(model),sort_keys=True)
        stats,comparison,_=biv.evaluate_frozen_samples(bundle,model)
        self.assertEqual(stats[(stats.Segmento=='TEST')&(stats.Bin==biv.NEW)].N.iloc[0],1)
        self.assertTrue(stats[(stats.Segmento=='TRAIN')&(stats.Bin==biv.NEW)].Tasa_default.isna().all())
        self.assertEqual(snapshot,json.dumps(biv._json_safe(model),sort_keys=True))

    def test_22_no_legacy_three_bin_rules_active(self):
        active=biv.active_parameters(biv.parameters())
        self.assertEqual(active['POLICY_VERSION'],'B.3')
        self.assertEqual(active['MIN_IV_TOTAL'],.02)
        self.assertNotIn('MIN_PREFERRED_BINS',active)
        self.assertNotIn('MIN_TWO_BIN_AUC',active)

    def test_23_consider_valid_alternative_before_rejection(self):
        example=B2Tests().example((200,200,200,200),(20,80,40,120))
        candidates,details,summary,model=self.fit_example(example)
        self.assertEqual(candidates.loc[candidates.Origen.eq('CUANTILES'),'Estado_candidato'].iloc[0],'RECHAZADA')
        self.assertEqual(summary.Estado_bivariado.iloc[0],'APROBADA')
        self.assertTrue(candidates.Calidad_ok.any())


    def test_24_inactive_three_bin_parameter_does_not_restrict_b3(self):
        config=biv.parameters(POLICY_VERSION='B.3',MIN_PREFERRED_BINS=2,MIN_TWO_BIN_AUC=.1)
        self.assertEqual(config['POLICY_VERSION'],'B.3')
        self.assertNotIn('MIN_PREFERRED_BINS',biv.active_parameters(config))

    def test_25_no_forced_merge_of_valid_numeric_partition(self):
        s,y,m,legacy,c=B2Tests().example((200,200,200),(20,60,120))
        config=biv.parameters(**{**legacy,'POLICY_VERSION':'B.3'})
        result=biv.merge_adjacent(s,y,c,config)
        self.assertEqual(result['edges'],c['edges'])
        self.assertEqual(result['fusion_history'],[])


if __name__=='__main__':unittest.main(verbosity=2)

