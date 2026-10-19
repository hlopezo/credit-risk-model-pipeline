"""Pruebas de tracking y privacidad sobre importación real, sin ajustar modelos."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import json
import shutil
import unittest
import uuid
import urllib.request
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
import statsmodels.api as sm
import mlflow_tracking as t
import registrar_pipeline_mlflow as r


@contextmanager
def scratch():
    folder=(t.LOCAL/'tests'/uuid.uuid4().hex).resolve();folder.mkdir(parents=True)
    try:yield folder
    finally:
        assert folder.parent==(t.LOCAL/'tests').resolve();shutil.rmtree(folder)


class PrivacyTests(unittest.TestCase):
    def test_01_reject_individual_data(self):
        with scratch() as folder:
            file=folder/'report.csv';pd.DataFrame({'Variable':['a'],'cliente_id':[123]}).to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,'individuales'):t.validate_aggregate_csv(file)
            for column in ('Target','DNI','_row_id','Probabilidad_default'):
                pd.DataFrame({'Variable':['a'],column:[1]}).to_csv(file,index=False)
                with self.assertRaises(ValueError):t.validate_aggregate_csv(file)

    def test_02_reject_original_pickle_and_matrices(self):
        with scratch() as folder:
            with self.assertRaisesRegex(ValueError,'Formato'):t.stage_file('5.MODELO_LOGISTICO/modelos/modelo_logit.pkl',folder/'bad.pkl')
            with self.assertRaisesRegex(ValueError,'matrices'):t.stage_file('5.MODELO_LOGISTICO/matrices/X_train_modelo.csv',folder/'bad.csv')

    def test_03_scrub_paths_and_secrets(self):
        value=t.scrub({'reference':str(Path('5.MODELO_LOGISTICO/configuracion_modelo.json').resolve())})
        self.assertEqual(value['reference'],'5.MODELO_LOGISTICO/configuracion_modelo.json')
        self.assertFalse(t.ABSOLUTE_PATH.search('mlflow-artifacts:/experiment/artifacts'))
        with self.assertRaises(ValueError):t.scrub({'password':'never-publish'})
        with self.assertRaises(ValueError):t.scrub({'target':[0,1]})
        self.assertEqual(t.scrub({'target':'indicador_default'})['target'],'indicador_default')

    def test_04_reject_remote_tracking(self):
        with self.assertRaisesRegex(ValueError,'local'):t.configure('https://external.example')

    def test_05_log_scalar_apis(self):
        with patch.object(mlflow,'log_param') as param,patch.object(mlflow,'log_metric') as metric:
            t.log_parameters({'threshold':.6});t.log_metrics({'train_auc':.75})
            param.assert_called_once_with('threshold',.6);metric.assert_called_once_with('train_auc',.75)


class ActualRunsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t.configure();cls.report=t.read_json(t.LOCAL/'evidencias/importacion.json');cls.client=MlflowClient()
        cls.runs=[cls.client.get_run(run) for run in cls.report['run_ids']]

    def test_06_four_finished_runs_created(self):
        active=self.client.search_runs([self.report['experiment_id']])
        self.assertEqual(len(active),4)
        self.assertEqual([x.data.tags['mlflow.runName'] for x in self.runs],list(t.PHASES))
        self.assertTrue(all(run.info.status=='FINISHED' for run in self.runs))

    def test_07_parameters_match_existing_configs(self):
        self.assertEqual(float(self.runs[2].data.params['correlation_absolute_threshold']),.6)
        self.assertEqual(float(self.runs[3].data.params['MAX_IV_MODEL']),.5)
        for run,definition in zip(self.runs,r.phase_definitions()):
            for name,value in definition['params'].items():
                expected=json.dumps(value,ensure_ascii=False,allow_nan=False) if isinstance(value,(list,dict)) else str(value)
                self.assertEqual(run.data.params[name],expected)

    def test_08_metric_correspondence(self):
        for run,definition in zip(self.runs,r.phase_definitions()):
            for name,value in definition['metrics'].items():self.assertAlmostEqual(run.data.metrics[name],float(value),places=12)
        table=r.frame(r.MODEL/'reportes/df_metricas_modelo.csv')
        for row in table.itertuples():self.assertAlmostEqual(self.runs[3].data.metrics[f'{row.Segmento.lower()}_auc'],row.AUC,places=12)

    def test_09_artifact_integrity_and_unique_names(self):
        for run in self.runs:
            prefix=run.info.artifact_uri.split('mlflow-artifacts:/',1)[1]
            root=t.LOCAL/'artifacts'/prefix
            manifest=t.read_json(root/'provenance/artifact_manifest.json')
            self.assertEqual(len(manifest),len({record['artifact_relative'] for record in manifest}))
            for record in manifest:
                self.assertEqual(t.sha256(record['source_relative']),record['source_sha256'])
                self.assertEqual(t.sha256(root/record['artifact_relative']),record['published_sha256'])

    def test_10_phase_references_and_tags(self):
        for i,run in enumerate(self.runs):
            tags=run.data.tags
            for name in ('phase','methodology','model_type','dataset_version','execution_type','source_notebook'):self.assertIn(name,tags)
            self.assertEqual(tags['execution_type'],'retrospective_import')
            self.assertEqual(tags['dataset_version'],self.report['coherence']['dataset_version'])
            if i:self.assertEqual(tags['upstream_run_id'],self.runs[i-1].info.run_id)
            self.assertEqual(tags['final_model_run_id'],self.runs[-1].info.run_id)
            self.assertFalse(any(t.ABSOLUTE_PATH.search(str(value)) for value in tags.values()))

    def test_11_registered_model_and_signature(self):
        item=self.report['model'];version=self.client.get_model_version(item['registered_name'],item['registered_version'])
        self.assertEqual(version.status,'READY');self.assertEqual(version.run_id,self.runs[-1].info.run_id)
        with t.local_tracking_session():
            model=mlflow.pyfunc.load_model(f"models:/{item['registered_name']}/{item['registered_version']}")
            self.assertEqual(model.metadata.get_input_schema().input_names(),item['input_columns'])

    def test_12_loaded_predictions_match_frozen_original_synthetic(self):
        coefficients=t.read_json(r.MODEL/'modelos/coeficientes.json');example=t.synthetic_example(coefficients)
        item=self.report['model']
        with t.local_tracking_session(),patch.object(sm.Logit,'fit',side_effect=AssertionError('No entrenar')):
            original=sm.load(r.MODEL/'modelos/modelo_logit.pkl')
            logged=mlflow.pyfunc.load_model(f"models:/{item['registered_name']}/{item['registered_version']}")
            a=original.predict(sm.add_constant(example,has_constant='add'))
            b=logged.predict(example).probabilidad_default
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_13_no_sensitive_published_artifacts(self):
        self.assertEqual(t.scan_published_tree(t.LOCAL/'artifacts')['sensitive_findings'],0)
        for file in (t.LOCAL/'artifacts').rglob('source_inventory.json'):
            inventory=t.read_json(file)
            self.assertTrue(all(not record['content_uploaded'] for record in inventory['notebooks_modules']))

    def test_14_no_previous_modifications(self):
        protected=t.read_json(t.LOCAL/'evidencias/hashes_previos.json')
        for path,digest in protected.items():self.assertEqual(t.sha256(path),digest,path)

    def test_15_idempotent_import_does_not_retrain(self):
        with patch.object(sm.Logit,'fit',side_effect=AssertionError('No entrenar')):
            result=r.import_pipeline()
        self.assertEqual(result['run_ids'],self.report['run_ids'])
        self.assertEqual(len(self.client.search_runs([self.report['experiment_id']])),4)

    def test_16_ui_health_and_html(self):
        self.assertEqual(urllib.request.urlopen(t.TRACKING_URI+'/health',timeout=5).read(),b'OK')
        html=urllib.request.urlopen(t.TRACKING_URI+'/',timeout=5).read().decode()
        self.assertIn('<html',html.lower());self.assertIn('MLflow',html)


if __name__=='__main__':unittest.main()
