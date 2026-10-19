from pathlib import Path
import hashlib,json,unittest,subprocess
import pandas as pd,numpy as np
class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/pipeline.json').exists())
        cls.ref=json.loads((cls.repo/'docs/reference/expected_results.json').read_text())
    def test_raw_authorized_unchanged(self):
        schema=json.loads((self.repo/'configs/data_schema.json').read_text());raw=self.repo/schema['file']
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),schema['sha256'])
        df=pd.read_csv(raw,sep=schema.get("separator",","));self.assertEqual(len(df),50620);self.assertEqual(df.columns.tolist(),schema['columns'])
        self.assertNotIn('SEGMENTO',df.columns)
        from segmentation_validation import require_generated_partition
        self.assertTrue(require_generated_partition(self.repo)['phase1_executed'])
        self.assertTrue(df.indicador_default.isin([0,1]).all())
    def test_metrics_match_reference_without_substitution(self):
        actual=pd.read_csv('5.MODELO_LOGISTICO/reportes/df_metricas_modelo.csv',float_precision='round_trip');expected=pd.DataFrame(self.ref['metrics'])
        np.testing.assert_allclose(actual[['AUC','Gini','KS']],expected[['AUC','Gini','KS']],rtol=1e-10,atol=1e-12)
    def test_frozen_cuts_and_selected_variables(self):
        actual=json.loads(Path('4.ANALIS_UV_BV_MV/BIVARIADO/binning_woe_train.json').read_text())
        self.assertEqual(actual['approved_variables'],self.ref['bivariate_approved'])
        for v in self.ref['woe']:self.assertEqual(actual['variables'][v]['edges'],self.ref['woe'][v]['edges'])
    def test_git_includes_raw_excludes_outputs(self):
        raw=json.loads((self.repo/'configs/pipeline.json').read_text())['paths']['raw']
        result=subprocess.run(['git','check-ignore',raw],cwd=self.repo,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        result=subprocess.run(['git','check-ignore','outputs/workspace/model.pkl'],cwd=self.repo,capture_output=True)
        self.assertEqual(result.returncode,0)
    def test_canonical_notebooks_have_no_old_outputs(self):
        for path in (self.repo/'notebooks').glob('*.ipynb'):
            nb=json.loads(path.read_text(encoding='utf-8'))
            for cell in nb['cells']:
                if cell['cell_type']=='code':self.assertFalse(cell['outputs']);self.assertIsNone(cell['execution_count'])
