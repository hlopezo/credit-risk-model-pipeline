"""Pruebas de diagnóstico 5.1: cero ajustes, trazabilidad y selección solo TRAIN."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import importlib.util
import json
import shutil
import unittest
import uuid
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('umb_test','5.1.0.evaluacion_umbrales.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)


def example():
    return pd.DataFrame({'Target':[0,0,1,1],'Probabilidad_default':[.1,.4,.35,.8]},index=[10,20,30,40])


@contextmanager
def scratch():
    folder=Path.cwd()/('umb_test_'+uuid.uuid4().hex);folder.mkdir()
    try:yield folder
    finally:
        assert folder.resolve().parent==Path.cwd().resolve() and folder.name.startswith('umb_test_')
        shutil.rmtree(folder)


class ThresholdTests(unittest.TestCase):
    def test_01_grid(self):
        grid=u.threshold_grid();self.assertEqual(len(grid),50)
        self.assertEqual(grid[0],.01);self.assertEqual(grid[-1],.50)
        np.testing.assert_allclose(np.diff(grid),.01)

    def test_02_metrics_and_confusion(self):
        m=u.threshold_metrics(example(),.5)
        self.assertEqual([m[x] for x in ['TN','FP','FN','TP']],[2,0,1,1])
        self.assertEqual(m['Recall'],.5);self.assertEqual(m['Precision'],1)
        self.assertAlmostEqual(m['F1'],2/3);self.assertEqual(m['Accuracy'],.75)
        self.assertEqual(m['Pct_clientes_alto_riesgo'],25)

    def test_03_inclusive_threshold(self):
        self.assertEqual(u.threshold_metrics(example(),.4)['Clientes_alto_riesgo'],2)

    def test_04_undefined_not_zero(self):
        result=u.threshold_metrics(example(),.99)
        self.assertTrue(np.isnan(result['Precision']));self.assertIn('sin positivos',result['Advertencias'])
        frame=example().assign(Target=0)
        self.assertTrue(np.isnan(u.threshold_metrics(frame,.5)['Recall']))

    def test_05_monotonic_recall_and_volume(self):
        grid=u.evaluate_grid(example())
        self.assertTrue(grid.Recall.diff().dropna().le(0).all())
        self.assertTrue(grid.Clientes_alto_riesgo.diff().dropna().le(0).all())
        self.assertTrue(grid[['TN','FP','FN','TP']].sum(axis=1).eq(4).all())

    def test_06_invalid_probability_and_identity(self):
        for value in [np.nan,np.inf,-.1,1.1]:
            frame=example();frame.loc[10,'Probabilidad_default']=value
            with self.assertRaises(ValueError):u.validate_predictions(frame)
        frame=example();frame.index=[1,1,2,3]
        with self.assertRaises(ValueError):u.validate_predictions(frame)

    def test_07_candidate_objectives_and_tie_break(self):
        grid=pd.DataFrame({'Umbral':[.1,.2,.3,.5],'Precision':[.2,.3,.4,.5],
            'Recall':[.9,.6,.4,.01],'F1':[.3,.4,.4,.02],'Pct_clientes_alto_riesgo':[70,30,9,1]})
        result=u.choose_candidates(grid,u.config()).set_index('Candidato')
        self.assertEqual(result.loc['F1_INTERNO','Umbral'],.3)
        self.assertEqual(result.loc['RECALL_ESCENARIO','Umbral'],.2)
        self.assertEqual(result.loc['CAPACIDAD_ESCENARIO','Umbral'],.3)

    def test_08_infeasible_candidate_explicit(self):
        grid=u.evaluate_grid(example());cfg=u.config(dict(RECALL_SCENARIO=.99,MAX_HIGH_RISK_SCENARIO=.001))
        candidates=u.choose_candidates(grid,cfg).set_index('Candidato')
        self.assertEqual(candidates.loc['CAPACIDAD_ESCENARIO','Estado'],'NO_FACTIBLE')
        self.assertTrue(np.isnan(candidates.loc['CAPACIDAD_ESCENARIO','Umbral']))

    def test_09_decile_capture_and_lift(self):
        frame=pd.DataFrame({'Target':[1,1,0,0,0,0,0,0,0,0],
            'Probabilidad_default':np.arange(10,0,-1)/10},index=range(10))
        table,assigned=u.risk_deciles({'TRAIN':frame})
        self.assertEqual(table.N.sum(),10);self.assertEqual(len(assigned),10)
        self.assertEqual(table.iloc[0].Captura_defaults_acumulada,.5)
        self.assertEqual(table.iloc[0].Lift_decil,5)
        self.assertEqual(table.iloc[-1].Captura_defaults_acumulada,1)
        self.assertEqual(table.iloc[-1].Lift_acumulado,1)

    def test_10_ties_not_ordered_using_target(self):
        frame=pd.DataFrame({'Target':[0,1]*10,'Probabilidad_default':[.1]*20},index=np.arange(20)[::-1])
        a,ids=u.risk_deciles({'TRAIN':frame});b,other=u.risk_deciles({'TRAIN':frame.assign(Target=1-frame.Target)})
        pd.testing.assert_frame_equal(ids,other)
        self.assertTrue(a.iloc[1:].Empate_probabilidad_en_frontera.all())
        self.assertTrue(a.N.max()-a.N.min()<=1)

    def test_11_empty_calibration_bins_nan(self):
        table,summary=u.calibration_diagnostics({'TRAIN':example()})
        self.assertEqual(table.N.sum(),4)
        self.assertTrue(table.loc[table.N.eq(0),'Tasa_default'].isna().all())
        self.assertAlmostEqual(summary.iloc[0].Brier,(.01+.16+.65**2+.2**2)/4)

    def test_12_probability_percentiles(self):
        summary=u.probability_summary({'TRAIN':example()}).iloc[0]
        self.assertEqual(summary.Minimo,.1);self.assertEqual(summary.Maximo,.8)
        self.assertAlmostEqual(summary.P50,.375);self.assertAlmostEqual(summary.Promedio,.4125)


class RealFrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(u.sm.Logit,'fit',side_effect=AssertionError('Reentrenamiento prohibido')):
            cls.frozen=u.load_frozen()
        cls.cfg=u.config()
        cls.valid,cls.rest,cls.manifest=u.internal_train_partition(cls.frozen['data']['TRAIN'],cls.cfg)
        cls.grid=u.evaluate_grid(cls.valid);cls.candidates=u.choose_candidates(cls.grid,cls.cfg)

    def test_13_frozen_predictions_reproduced_no_fit(self):
        self.assertEqual(len(self.frozen['features']),9)
        self.assertTrue(self.frozen['validation'].Probabilidades_originales_reproducidas.all())

    def test_14_internal_stratified_reproducible_disjoint(self):
        valid,rest,manifest=u.internal_train_partition(self.frozen['data']['TRAIN'],self.cfg)
        pd.testing.assert_frame_equal(valid,self.valid)
        self.assertFalse(set(valid.index)&set(rest.index))
        self.assertEqual(len(valid)+len(rest),len(self.frozen['data']['TRAIN']))
        self.assertLess(abs(valid.Target.mean()-rest.Target.mean()),.001)

    def test_15_candidate_selection_invariant_to_holdout(self):
        changed={s:f.copy(deep=True) for s,f in self.frozen['data'].items()}
        changed['TEST']['Probabilidad_default']=.99;changed['OOT']['Target']=0
        a=u.choose_candidates(u.evaluate_grid(self.valid),self.cfg)
        valid,_,_=u.internal_train_partition(changed['TRAIN'],self.cfg)
        b=u.choose_candidates(u.evaluate_grid(valid),self.cfg)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(a.Fuente.isin(['TRAIN_VALIDACION_INTERNA','CONFIGURACION_PREVIA']).all())

    def test_16_candidate_persistence_and_diagnostic_no_reselection(self):
        with scratch() as folder:
            partition=folder/'partition.csv';self.manifest.to_csv(partition,index_label='_row_id')
            file=folder/'thresholds.json'
            u.freeze_candidates(file,self.candidates,self.cfg,self.frozen['references'],partition)
            before=u.sha256(file);artifact,candidates=u.load_candidates(file)
            with patch.object(u,'choose_candidates',side_effect=AssertionError('No seleccionar con heldout')):
                table=u.evaluate_candidates(self.frozen['data'],candidates)
            self.assertEqual(before,u.sha256(file));self.assertEqual(len(table),12)
            self.assertIn('No es validación fuera de muestra',artifact['limitation'])
            partition.write_text('alterado',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Partición'):u.load_candidates(file)

    def test_17_full_grid_preserves_source(self):
        frame=self.frozen['data']['OOT'];copy=frame.copy(deep=True)
        grid=u.evaluate_grid(frame)
        pd.testing.assert_frame_equal(frame,copy)
        self.assertTrue(grid[['TN','FP','FN','TP']].sum(axis=1).eq(len(frame)).all())

    def test_18_brier_identical_to_phase5(self):
        _,summary=u.calibration_diagnostics(self.frozen['data'])
        old=pd.read_csv(u.SOURCE/'reportes/df_metricas_modelo.csv',float_precision='round_trip')
        np.testing.assert_allclose(summary.Brier,old.Brier,rtol=1e-12,atol=1e-12)

    def test_19_real_deciles_totals(self):
        table,ids=u.risk_deciles(self.frozen['data'])
        for s,frame in self.frozen['data'].items():
            group=table.loc[table.Segmento.eq(s)]
            self.assertEqual(len(group),10);self.assertEqual(group.N.sum(),len(frame))
            self.assertEqual(group.Defaults.sum(),frame.Target.sum())
            self.assertEqual(group.iloc[-1].Captura_defaults_acumulada,1)
            self.assertEqual(group.iloc[-1].Lift_acumulado,1)

    def test_20_original_files_and_model_params_unchanged(self):
        hashes={p:u.sha256(p) for p in self.frozen['references']}
        coef=self.frozen['model'].params.copy(deep=True)
        u.probability_summary(self.frozen['data']);u.calibration_diagnostics(self.frozen['data'])
        u.evaluate_candidates(self.frozen['data'],self.candidates)
        pd.testing.assert_series_equal(coef,self.frozen['model'].params,check_exact=True)
        self.assertEqual(hashes,{p:u.sha256(p) for p in hashes})


if __name__=='__main__':unittest.main()
