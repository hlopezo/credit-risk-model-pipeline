"""Contratos y pruebas de selección/Logit 5.0; no modifican etapas anteriores."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import copy
import importlib.util
import json
import shutil
import unittest
import uuid
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm

spec=importlib.util.spec_from_file_location('logit_test','5.0.0.modelo_logistico.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@contextmanager
def scratch():
    folder=Path.cwd()/('logit_test_'+uuid.uuid4().hex);folder.mkdir()
    try:yield folder
    finally:
        assert folder.resolve().parent==Path.cwd().resolve() and folder.name.startswith('logit_test_')
        shutil.rmtree(folder)


def synthetic(beta=(-1.5,1.2,0),n=3000):
    rng=np.random.default_rng(239)
    X=pd.DataFrame(rng.normal(size=(n,len(beta))),columns=list('abc')[:len(beta)],index=np.arange(10000,10000+n))
    y=pd.Series(rng.binomial(1,expit(-1+X.to_numpy()@np.array(beta))),index=X.index,name='indicador_default')
    return X,y


class PolicyTests(unittest.TestCase):
    def test_01_actual_woe_convention(self):
        a=json.loads((m.BIV_ROOT/'binning_woe_train.json').read_text(encoding='utf-8'))
        c=m.validate_woe_convention(a,Path('3.0.0.bivariado.py').read_text(encoding='utf-8-sig'))
        self.assertEqual(c['expected_sign'],-1);self.assertEqual(c['default_value'],1)

    def test_02_reverse_woe_and_inconsistent_target(self):
        a={'target':'indicador_default','target_definition':{'0':'bueno','1':'malo'},
            'woe_definition':'ln(P(bin|bad)/P(bin|good)); smoothing'}
        source="stats['WoE'] = np.log(stats.Dist_malos_suavizada / stats.Dist_buenos_suavizada)"
        self.assertEqual(m.validate_woe_convention(a,source)['expected_sign'],1)
        a['target_definition']['1']='bueno'
        with self.assertRaisesRegex(ValueError,'Target'):m.validate_woe_convention(a,source)

    def test_03_iv_boundary_inclusive(self):
        iv=pd.DataFrame({'Variable':['a','b','c'],'IV_final':[.5,.500001,.2]})
        kept,report=m.filter_iv(iv,list('abc'))
        self.assertEqual(kept,['a','c']);self.assertEqual(report.iloc[1].Estado_IV,'EXCLUIDA_IV')

    def test_04_sign_and_pvalue_refits(self):
        X,y=synthetic()
        with patch.object(m,'fit_logit',wraps=m.fit_logit) as fit:
            state=m.eliminar_variables_completo(X,y,-1)
            self.assertEqual(fit.call_count,len(state.history))
        hist=state.df_iteraciones_modelo
        self.assertEqual(hist.iloc[0].Variable_eliminada,'b')
        self.assertEqual(hist.iloc[0].Criterio_eliminacion,'SIGNO')
        self.assertTrue(hist.N_variables.diff().dropna().eq(-1).all())
        self.assertEqual(state.status,'VALIDADO')
        self.assertTrue((state.model.params.drop('const')<=0).all())
        self.assertTrue((state.model.pvalues.drop('const')<=.05).all())
        self.assertEqual(sum(len(c) for c in state.coefficients),len(state.df_coeficientes_iteraciones))

    def test_05_highest_pvalue_removed(self):
        X,y=synthetic((-1.5,-.7,0))
        state=m.eliminar_variables_completo(X,y,-1,etapa='pvalue')
        history=state.df_iteraciones_modelo
        removed=history.loc[history.Criterio_eliminacion.eq('P_VALUE')]
        self.assertEqual(removed.Variable_eliminada.tolist(),['c'])
        before=state.df_coeficientes_iteraciones.query('Iteracion == 1 and Variable != "const"')
        self.assertEqual(before.sort_values('P_value').iloc[-1].Variable,'c')

    def test_06_shap_elimination_rechecks_sign_pvalues(self):
        X,y=synthetic((-2,-.15),n=4000)
        stages=[];original=m.SeleccionIterativa._candidate
        def track(state,stage):
            stages.append(tuple(state.variables));return original(state,stage)
        with patch.object(m.SeleccionIterativa,'_candidate',track):
            state=m.eliminar_variables_completo(X,y,-1,dict(P_VALUE_THRESHOLD=1,SHAP_MIN_PCT=40))
        self.assertIn('SHAP',state.df_iteraciones_modelo.Criterio_eliminacion.tolist())
        self.assertEqual(state.variables,['a']);self.assertEqual(stages,[('a','b'),('a',)])
        self.assertEqual(len(state.history),2)

    def test_07_intercept_and_negative_coefficients_not_discarded(self):
        X,y=synthetic((-1.5,-.7))
        state=m.eliminar_variables_completo(X,y,-1)
        self.assertEqual(state.variables,['a','b']);self.assertLess(state.model.params['const'],0)
        self.assertFalse(state.df_iteraciones_modelo.Variable_eliminada.notna().any())

    def test_08_last_predictor_protection(self):
        X,y=synthetic((1.5,))
        with self.assertRaisesRegex(m.ModelValidationError,'Último predictor') as cm:
            m.eliminar_variables_completo(X,y,-1)
        self.assertEqual(cm.exception.state.variables,['a'])

    def test_09_max_iterations_protection(self):
        X,y=synthetic()
        with self.assertRaisesRegex(m.ModelValidationError,'MAX_ITERACIONES'):
            m.eliminar_variables_completo(X,y,-1,dict(MAX_ITERACIONES=1))

    def test_10_singularity(self):
        X,y=synthetic((-1.5,-.7));X.b=X.a
        with self.assertRaisesRegex(m.ModelValidationError,'singular'):m.fit_logit(X,y,m.configuration())

    def test_11_nonconvergence(self):
        X,y=synthetic((-1.5,-.7))
        with self.assertRaisesRegex(m.ModelValidationError,'convergencia'):
            m.fit_logit(X,y,m.configuration(dict(FIT_MAXITER=1)))

    def test_12_perfect_separation(self):
        X=pd.DataFrame({'a':np.linspace(-3,3,200)})
        y=pd.Series((X.a>0).astype(int),name='indicador_default')
        with self.assertRaises(m.ModelValidationError):m.fit_logit(X,y,m.configuration())

    def test_13_metrics_known_values(self):
        result=m.calcular_metricas([0,0,1,1],[.1,.4,.35,.8])
        self.assertAlmostEqual(result['AUC'],.75);self.assertAlmostEqual(result['Gini'],.5)
        self.assertAlmostEqual(result['KS'],.5);self.assertAlmostEqual(result['Recall'],.5)
        self.assertAlmostEqual(result['Precision'],1);self.assertAlmostEqual(result['F1_score'],2/3)
        self.assertAlmostEqual(result['Accuracy'],.75)
        self.assertAlmostEqual(result['Brier'],(.1**2+.4**2+.65**2+.2**2)/4)
        self.assertEqual([result[k] for k in ['TN','FP','FN','TP']],[2,0,1,1])

    def test_14_undefined_metrics_explicit_nan(self):
        result=m.calcular_metricas([0,0],[.1,.2])
        for name in ['AUC','Gini','KS','Recall','Precision','F1_score']:self.assertTrue(np.isnan(result[name]))
        self.assertIn('sin ambas clases',result['Advertencias'])

    def test_15_metrics_indices_alignment(self):
        with self.assertRaisesRegex(ValueError,'índices'):
            m.calcular_metricas(pd.Series([0,1],index=[1,2]),pd.Series([.1,.9],index=[2,1]))


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=m.load_inputs();cls.kept,cls.iv=m.filter_iv(cls.inputs['iv'],cls.inputs['selection']['retained_variables'])
        cls.X=cls.inputs['X']['TRAIN'][cls.kept];cls.y=cls.inputs['y']['TRAIN']
        cls.state=m.eliminar_variables_completo(cls.X,cls.y,-1)
        cls.final={s:cls.inputs['X'][s][cls.state.variables] for s in m.SEGMENTS}

    def test_16_no_holdout_fit_and_order(self):
        with patch.object(m,'fit_logit',side_effect=AssertionError('No fit en predicción')):
            predictions=m.predict_segments(self.state.model,self.final,self.state.variables)
        for s,p in predictions.items():
            self.assertTrue(p.index.equals(self.inputs['y'][s].index));self.assertTrue(p.between(0,1).all())
        broken=dict(self.final);broken['OOT']=broken['OOT'].iloc[:,::-1]
        with self.assertRaisesRegex(ValueError,'columnas'):m.predict_segments(self.state.model,broken,self.state.variables)

    def test_17_shap_same_logit_and_reproducible(self):
        a=self.state.shap;b=m.linear_shap(self.state.model,self.X[self.state.variables],self.state.config)
        self.assertEqual(a['coefficients'],self.state.model.params.to_dict())
        pd.testing.assert_frame_equal(a['values'],b['values'],check_exact=True)
        reconstructed=a['expected_value']+a['values'].sum(axis=1)
        actual=m.design_matrix(a['sample'])@self.state.model.params
        np.testing.assert_allclose(actual,reconstructed,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(a['table'].SHAP_pct.sum(),100)

    def test_18_model_and_final_selection_consistent(self):
        self.assertEqual(self.state.model.params.index.tolist(),['const']+self.state.variables)
        self.assertEqual(len(self.state.variables),self.state.df_iteraciones_modelo.iloc[-1].N_variables)
        self.assertTrue((self.state.model.params.drop('const')<0).all())
        self.assertTrue((self.state.model.pvalues.drop('const')<=.05).all())
        self.assertTrue((self.state.shap['table'].SHAP_pct>=2).all())
        table=m.coefficient_table(self.state.model)
        np.testing.assert_array_equal(table.Coeficiente,self.state.model.params)
        np.testing.assert_array_equal(table.P_value,self.state.model.pvalues)
        self.assertIn('Logit Regression Results',self.state.model.summary().as_text())

    def test_19_serialization_exact_predictions(self):
        before=m.predict_segments(self.state.model,self.final,self.state.variables)
        with scratch() as folder:
            self.state.model.save(folder/'model.pkl');restored=sm.load(folder/'model.pkl')
            after=m.predict_segments(restored,self.final,self.state.variables)
            for s in m.SEGMENTS:np.testing.assert_allclose(before[s],after[s],rtol=1e-12,atol=1e-12)

    def test_20_inputs_do_not_recompute_prior_steps(self):
        biv=m.import_bivariado()
        names=['transform_woe','generate_candidates','evaluate_candidates','select_configurations','search_adjacent_partitions','bin_statistics']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:stack.enter_context(patch.object(biv,name,side_effect=AssertionError('Recalcular prohibido')))
            stack.enter_context(patch.object(m,'import_bivariado',return_value=biv))
            data=m.load_inputs()
        for s in m.SEGMENTS:pd.testing.assert_frame_equal(data['X'][s],self.inputs['X'][s],check_exact=True)

    def test_21_holdout_changes_do_not_select(self):
        # API de selección recibe exclusivamente X/y TRAIN; cambiar heldout no altera decisiones.
        original=copy.deepcopy(self.inputs['X'])
        original['TEST'][:]=999;original['OOT'][:]=-999
        state=m.eliminar_variables_completo(original['TRAIN'][self.kept],self.y,-1)
        self.assertEqual(state.variables,self.state.variables)
        pd.testing.assert_frame_equal(state.df_coeficientes_iteraciones,self.state.df_coeficientes_iteraciones)

    def test_22_staged_equals_complete_and_no_extra_fit(self):
        with patch.object(m,'fit_logit',wraps=m.fit_logit) as fit:
            state=m.SeleccionIterativa(self.X,self.y,-1)
            for stage in ['signo','pvalue','shap']:
                state=m.eliminar_variables_completo(self.X,self.y,-1,estado=state,etapa=stage)
            model=state.model
            self.assertIs(state.model,model)
            self.assertEqual(fit.call_count,len(state.history))
        np.testing.assert_allclose(state.model.params,self.state.model.params,rtol=1e-12,atol=1e-12)
        self.assertEqual(state.variables,self.state.variables)

    def test_23_prior_artifacts_unchanged(self):
        paths=[Path('2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb'),Path('3.0.0.SCR_DES_ANLS_BIVARIADO.ipynb'),
            Path('4.0.0.SCR_DES_ANLS_Multivar.ipynb'),m.BIV_ROOT/'binning_woe_train.json',m.MULT_ROOT/'seleccion_correlacion.json']
        before={p:m.sha256(p) for p in paths}
        m.load_inputs()
        self.assertEqual(before,{p:m.sha256(p) for p in paths})

    def test_24_signed_and_absolute_deltas(self):
        metrics=pd.DataFrame({'Segmento':list(m.SEGMENTS),'Gini':[.4,.5,.2],'KS':[.3,.4,.1]})
        d=m.segment_deltas(metrics,m.configuration())
        self.assertAlmostEqual(d.iloc[0].Diferencia_firmada,-.1)
        self.assertAlmostEqual(d.iloc[0].Diferencia_absoluta,.1)
        self.assertTrue(d.Alerta.eq('REVISAR').all())


if __name__=='__main__':unittest.main()
