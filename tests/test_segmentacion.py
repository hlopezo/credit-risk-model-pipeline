"""Contratos de la partición original y del inicio desde fuente sin SEGMENTO."""
from pathlib import Path
import ast,json,hashlib,unittest
import pandas as pd
from segmentation_validation import require_generated_partition,values_hash,rows_hash

class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/pipeline.json').exists())
        cls.cfg=json.loads((cls.repo/'configs/pipeline.json').read_text());cls.reference=json.loads((cls.repo/'configs/partition_reference.json').read_text())
        cls.frame=pd.read_csv(cls.repo/cls.cfg['paths']['runtime']/cls.cfg['paths']['segmented'])
    def test_input_is_unsegmented_original_not_precalculated(self):
        schema=json.loads((self.repo/'configs/data_schema.json').read_text());raw=pd.read_csv(self.repo/schema['file'],sep=';')
        self.assertEqual(raw.shape,(50620,47));self.assertNotIn('SEGMENTO',raw)
        self.assertFalse((self.repo/'data/raw/1.1.0.TAB_SEGM_TRAIN_TEST_OOT.csv').exists())
        self.assertEqual(hashlib.sha256((self.repo/schema['file']).read_bytes()).hexdigest(),schema['sha256'])
    def test_original_split_call_and_parameters_not_replaced(self):
        nb=json.loads((self.repo/'notebooks/1.0.0.SCR_DES_MOD_RL_SEGMT.ipynb').read_text(encoding='utf-8'))
        calls=[]
        for c in nb['cells']:
            if c['cell_type']=='code':
                calls += [n for n in ast.walk(ast.parse(''.join(c['source']))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='train_test_split']
        self.assertEqual(len(calls),1);call=calls[0]
        self.assertEqual(ast.unparse(call.args[0]),'desarrollo')
        args={k.arg:ast.unparse(k.value) for k in call.keywords}
        self.assertEqual(args,dict(test_size='0.3',random_state='123',stratify="desarrollo['indicador_default']"))
        observed=json.loads((self.repo/'outputs/executed_notebooks/phase1_split_calls.json').read_text())
        self.assertEqual(len(observed),1);self.assertEqual(observed[0]['random_state'],123)
        self.assertEqual(observed[0]['N_development'],40316)
    def test_exact_output_file_sha256(self):
        file=self.repo/self.cfg['paths']['runtime']/self.cfg['paths']['segmented']
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),self.reference['sha256'])
    def test_counts_clients_and_target_per_segment(self):
        for segment,expected in self.reference['segments'].items():
            frame=self.frame.loc[self.frame.SEGMENTO.eq(segment)]
            self.assertEqual(len(frame),expected['records']);self.assertEqual(frame.cliente_id.nunique(),expected['clients'])
            self.assertEqual({str(k):int(v) for k,v in frame.indicador_default.value_counts().items()},expected['target_counts'])
    def test_period_windows_are_original(self):
        self.assertEqual(sorted(self.frame.loc[self.frame.SEGMENTO.eq('OOT'),'periodo_observacion'].unique().tolist()),[202508,202509,202510])
        self.assertTrue(self.frame.loc[self.frame.SEGMENTO.isin(['TRAIN','TEST']),'periodo_observacion'].le(202507).all())
        for s,expected in self.reference['segments'].items():self.assertEqual(sorted(self.frame.loc[self.frame.SEGMENTO.eq(s),'periodo_observacion'].unique().tolist()),expected['periods'])
    def test_columns_types_indices_identifiers_and_all_rows(self):
        self.assertEqual(self.frame.columns.tolist(),self.reference['columns'])
        self.assertEqual({c:str(d) for c,d in self.frame.dtypes.items()},self.reference['dtypes'])
        for s,expected in self.reference['segments'].items():
            rows=self.frame.loc[self.frame.SEGMENTO.eq(s)]
            self.assertEqual(values_hash(rows.index),expected['index_sha256'])
            self.assertEqual(values_hash(rows.cliente_id),expected['identifiers_sha256'])
            self.assertEqual(values_hash(rows.indicador_default),expected['target_sha256'])
            self.assertEqual(rows_hash(rows),expected['rows_with_indices_sha256'])
    def test_downstream_requires_executed_phase1(self):
        record=require_generated_partition(self.repo)
        self.assertTrue(record['phase1_executed']);self.assertEqual(record['segmented_file_sha256'],self.reference['sha256'])
    def test_default_pipeline_starts_with_phase1(self):
        tree=ast.parse((self.repo/'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        defaults=[k.value for n in ast.walk(tree) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='default' and isinstance(k.value,ast.List)]
        self.assertTrue(any(ast.literal_eval(n)[0]=='1' for n in defaults))
    def test_phase1_notebook_executed_in_clean_namespace(self):
        notebook=json.loads((self.repo/'outputs/executed_notebooks/1.0.0.SCR_DES_MOD_RL_SEGMT.ipynb').read_text(encoding='utf-8'))
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertTrue(all(c['execution_count'] is not None for c in cells))
        self.assertFalse(any(o['output_type']=='error' for c in cells for o in c['outputs']))
