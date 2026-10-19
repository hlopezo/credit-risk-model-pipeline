import unittest
from contextlib import contextmanager


@contextmanager
def workspace_test_directory():
    # Plain inherited permissions; Python 3.14 TemporaryDirectory uses a Windows
    # private ACL unsupported by this environment. No ACL/protection changes.
    from pathlib import Path
    from uuid import uuid4
    import shutil
    workspace = Path.cwd().resolve()
    directory = workspace / ('psi_test_' + uuid4().hex)
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        resolved = directory.resolve()
        assert resolved.parent == workspace and resolved.name.startswith('psi_test_')
        shutil.rmtree(resolved)

import numpy as np
import pandas as pd
import psi_temporal as p
class PSI(unittest.TestCase):
 def calc(self,r,c):return p.calculate_psi(pd.Series(r),pd.Series(c),p.learn_bins(pd.Series(r)))
 def test_identical(self):self.assertAlmostEqual(self.calc(list(range(1000)),list(range(1000)))['PSI'],0)
 def test_different(self):self.assertGreater(self.calc([0,1]*500,[9]*1000)['PSI'],.2)
 def test_constant(self):self.assertEqual(self.calc([1]*1000,[2]*1000)['Estado_calculo'],'REFERENCIA_DEGENERADA')
 def test_binary(self):self.assertEqual(p.learn_bins(pd.Series([0,1]*500))['method'],'categories')
 def test_zeros99(self):self.assertTrue(np.isfinite(self.calc([0]*990+list(range(10)),[0]*800+list(range(200)))['PSI']))
 def test_concentrated(self):self.assertEqual(p.learn_bins(pd.Series([0]*950+list(range(1,51))))['method'],'dominant_quantiles')
 def test_all_missing(self):self.assertTrue(np.isnan(self.calc([np.nan]*1000,[1]*1000)['PSI']))
 def test_missing80(self):self.assertTrue(np.isfinite(self.calc([np.nan]*800+[0,1]*100,[np.nan]*600+[0,1]*200)['PSI']))
 def test_new_missing(self):self.assertGreater(self.calc([0,1]*500,[np.nan]*500+[0,1]*250)['PSI_missing_component'],0)
 def test_new_categories(self):self.assertEqual(self.calc(['a','b']*500,['c']*1000)['N_categorias_nuevas'],1000)
 def test_disappearing(self):self.assertGreater(self.calc(['a','b']*500,['a']*1000)['PSI'],0)
 def test_empty(self):self.assertEqual(self.calc([0,1]*500,[])['Estado_calculo'],'VOLUMEN_INSUFICIENTE')
 def test_small(self):self.assertTrue(np.isnan(self.calc([0,1]*500,[0,1])['PSI']))
 def test_duplicate_quantiles(self):self.assertLess(len(p.learn_bins(pd.Series([0]*800+list(range(200))))['edges']),7)
 def test_extremes(self):self.assertTrue(np.isfinite(self.calc(list(range(-500,500)),[-1e99,1e99]*500)['PSI']))
 def test_infinity(self):self.assertEqual(self.calc([0,1]*500,[np.inf,-np.inf]*500)['N_infinitos'],1000)
 def test_period_order(self):self.assertEqual(sorted(p.periods(pd.Series([202502,202412])).astype(str)),['2024-12','2025-02'])
 def test_gap(self):self.assertEqual(p.max_consecutive([pd.Period('2025-01'),pd.Period('2025-03')]),1)
 def test_consecutive(self):self.assertEqual(p.max_consecutive([pd.Period('2024-12'),pd.Period('2025-01')]),2)
 def test_normalization(self):
  d=self.calc([0,1]*500,[3]*1000);self.assertAlmostEqual(sum(d['p']),1);self.assertAlmostEqual(sum(d['q']),1)
 def test_collision(self):self.assertIn('VALUE:str:',p.transform(pd.Series(['__MISSING__']),p.learn_bins(pd.Series(['__MISSING__']))).iloc[0])
 def test_reproducible(self):self.assertEqual(self.calc([0,1]*500,[2]*1000)['PSI'],self.calc([0,1]*500,[2]*1000)['PSI'])
 def test_priority(self):self.assertEqual(p.classify(10,10,4,2),'CANDIDATA_DESCARTE')
 def test_coverage(self):self.assertEqual(p.classify(2,12,0,0),'NO_EVALUABLE')
 def test_boundaries(self):self.assertEqual(p.classify(12,12,1,1),'EN_OBSERVACION')

class Integration(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  parts=[]
  for seg in ['TRAIN','TEST','OOT']:
   for month in [202410,202408,202409]:
    parts.append(pd.DataFrame({'SEGMENTO':seg,'periodo_observacion':month,'cliente_id':range(220),'indicador_default':[0,1]*110,'feature':[0,1]*110}))
  cls.df=pd.concat(parts,ignore_index=True)
  cls.m={seg:cls.df.loc[cls.df.SEGMENTO==seg,['feature']].copy() for seg in ['TRAIN','TEST','OOT']}
  cls.a,cls.s,cls.r=p.run_analysis(cls.df,cls.m,export=False)
 def test_segment_consistency(self):self.assertEqual(self.a.groupby('Periodo').PSI.nunique().max(),1)
 def test_train_only_learning(self):
  df=self.df.copy();df.loc[df.SEGMENTO!='TRAIN','feature']=999
  matrices={seg:df.loc[df.SEGMENTO==seg,['feature']].copy() for seg in self.m}
  a,s,r=p.run_analysis(df,matrices,export=False)
  pd.testing.assert_frame_equal(self.r,r)
  pd.testing.assert_frame_equal(self.a[self.a.Segmento=='TRAIN'],a[a.Segmento=='TRAIN'])
 def test_rerun_identical(self):
  a,s,r=p.run_analysis(self.df,self.m,export=False);pd.testing.assert_frame_equal(a,self.a)
 def test_consolidation(self):
  a=p.consolidate(self.m['TRAIN'],[],[],self.s)
  self.assertEqual(len(a),3);self.assertFalse(a.duplicated(['Variable','Filtro']).any())
 def test_preservation_alignment(self):
  for seg,x in self.m.items():pd.testing.assert_frame_equal(x,self.df.loc[x.index,['feature']])
 def test_reject_misalignment(self):
  m={**self.m,'TEST':self.m['TEST'].set_axis(self.m['TRAIN'].index)}
  with self.assertRaises(ValueError):p.run_analysis(self.df,m,export=False)
 def test_codmes_validation(self):
  d=self.df.assign(codmes=202001)
  with self.assertRaises(ValueError):p.run_analysis(d,self.m,export=False)




class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monthly = Integration.a
        cls.summary = Integration.s
        cls.references = Integration.r

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_invalid_config_is_atomic(self):
        before = p.get_configuration()
        with self.assertRaises(ValueError):
            p.configure(alpha=0)
        self.assertEqual(before, p.get_configuration())
        with self.assertRaises(ValueError):
            p.configure(reference_n_months=0)
        self.assertEqual(before, p.get_configuration())

    def test_explicit_config_preserves_defaults(self):
        self.assertEqual(p.configure(), p.get_configuration())
        self.assertEqual(p.ALPHA, .5)

    def test_diagnostics_preserve_samples(self):
        before = {segment: matrix.copy() for segment, matrix in Integration.m.items()}
        samples, months = p.validate_samples(Integration.df, Integration.m)
        self.assertEqual(samples.N_registros.sum(), len(Integration.df))
        self.assertEqual(len(months), 9)
        for segment, matrix in Integration.m.items():
            pd.testing.assert_frame_equal(matrix, before[segment])

    def test_calculation_without_export(self):
        from unittest.mock import patch
        with patch.object(p, 'export_psi_tables', side_effect=AssertionError('Unexpected export')):
            monthly, summary, reference = p.run_analysis(Integration.df, Integration.m, export=False)
        pd.testing.assert_frame_equal(monthly, self.monthly)
        self.assertIn('bins', monthly.attrs['psi_artifacts'])
        self.assertIn('distributions', monthly.attrs['psi_artifacts'])

    def test_legacy_export_argument(self):
        from unittest.mock import patch
        with patch.object(p, 'export_psi_tables') as export:
            monthly, summary, reference = p.run_analysis(Integration.df, Integration.m, export=True)
        export.assert_called_once()
        pd.testing.assert_frame_equal(monthly, self.monthly)

    def test_dashboard_pagination_and_single_legend(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        data, summary = [], []
        for index in range(29):
            a = self.monthly.copy(); a.attrs = {}; a['Variable'] = f'feature_{index}'
            b = self.summary.copy(); b['Variable'] = f'feature_{index}'
            data.append(a); summary.append(b)
        with patch.object(Figure, 'savefig', side_effect=AssertionError('Unexpected write')):
            figures = p.plot_psi_dashboard(pd.concat(data), pd.concat(summary))
        self.assertEqual(len(figures), 3)
        self.assertEqual([sum(axis.get_visible() for axis in fig.axes) for fig in figures], [12, 12, 5])
        for figure in figures:
            self.assertEqual(len(figure.legends), 1)
            self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))

    def test_dashboard_filters_and_validation(self):
        self.assertEqual(p.plot_psi_dashboard(self.monthly, self.summary, 'NO_EVALUABLE'), [])
        self.assertEqual(len(p.plot_psi_dashboard(self.monthly, self.summary, 'APROBADA')), 1)
        with self.assertRaises(ValueError):
            p.plot_psi_dashboard(self.monthly, self.summary, por_pagina=13)
        with self.assertRaises(ValueError):
            p.plot_psi_dashboard(self.monthly, self.summary, estado='OTRO')

    def test_individual_does_not_write(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        with patch.object(Figure, 'savefig', side_effect=AssertionError('Unexpected write')):
            figure, axis = p.plot_psi_variable('feature', self.monthly, self.summary)
        self.assertEqual([line.get_label() for line in axis.lines[:3]], ['TRAIN', 'TEST', 'OOT'])
        self.assertEqual(axis.lines[3].get_ydata()[0], .10)
        self.assertEqual(axis.lines[4].get_ydata()[0], .20)

    def test_heatmap_missing_is_masked(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        data = self.monthly[~((self.monthly.Segmento == 'TEST') & (self.monthly.Periodo == '2024-09'))]
        with patch.object(Figure, 'savefig', side_effect=AssertionError('Unexpected write')):
            figure, axis = p.plot_psi_heatmap(data, 'TEST')
        self.assertTrue(np.ma.getmaskarray(axis.images[0].get_array())[0, 1])

    def test_plot_missing_month_breaks_line(self):
        data = self.monthly[~((self.monthly.Segmento == 'TRAIN') & (self.monthly.Periodo == '2024-09'))]
        figure, axis = p.plot_psi_variable('feature', data, self.summary)
        self.assertTrue(np.isnan(axis.lines[0].get_ydata()[1]))

    def test_explicit_exports_do_not_recalculate(self):
        from pathlib import Path
        from unittest.mock import patch
        with workspace_test_directory() as directory:
            root = Path(directory)
            assert root.resolve().is_relative_to(Path.cwd().resolve())
            with patch.object(p, 'calculate_psi', side_effect=AssertionError('Recalculation')):
                paths = p.export_psi_tables(self.monthly, self.summary, self.references, root=root)
                figures = p.export_psi_figures(self.monthly, self.summary, root=root)
            self.assertTrue(all(path.exists() for path in paths + figures))
            self.assertFalse((root / 'graficos_individuales').exists())
            self.assertTrue((root / 'reportes/bins.json').exists())
            self.assertEqual(len(figures), 4)  # One dashboard and three heatmaps.
            original = (root / 'reportes/df_psi_mensual.csv').read_bytes()
            p.export_psi_tables(self.monthly, self.summary, self.references, root=root)
            backups = list((root / 'evidencias').glob('respaldo_*/reportes/df_psi_mensual.csv'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_legacy_graph_wrapper(self):
        from unittest.mock import patch
        with patch.object(p, 'export_psi_figures', return_value=[]) as export:
            p.export_plots(self.monthly, self.summary)
        self.assertTrue(export.call_args.kwargs['individuales'])
        self.assertEqual(export.call_args.kwargs['estados'], ('TODAS',) + p.PSI_STATES)

    def test_auxiliary_import_does_not_execute_notebook(self):
        import importlib
        import hashlib
        from pathlib import Path
        notebook = Path('2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb')
        before = hashlib.sha256(notebook.read_bytes()).hexdigest()
        importlib.import_module('ejecutar_psi_notebook')
        importlib.import_module('informe_psi')
        self.assertEqual(before, hashlib.sha256(notebook.read_bytes()).hexdigest())

    def test_report_uses_actual_counts(self):
        from pathlib import Path
        import informe_psi
        with workspace_test_directory() as directory:
            root = Path(directory)
            assert root.resolve().is_relative_to(Path.cwd().resolve())
            report = p.consolidate(Integration.m['TRAIN'], [], [], self.summary)
            p.export_psi_tables(self.monthly, self.summary, self.references, report, root=root)
            output = informe_psi.generate(root)
            text = output.read_text(encoding='utf-8')
            self.assertIn('Variables evaluadas: **1**', text)
            self.assertIn('Evaluaciones mensuales: **9**', text)
            self.assertNotIn('29 variables', text)
            self.assertNotIn('18 de octubre de 2026', text)

    def test_notebook_has_twelve_stages_and_one_calculation(self):
        import json
        import re
        import ast
        from pathlib import Path
        notebook = json.loads(Path('2.0.0.SCR_DES_ANLS_UNV_BIV_MULT.ipynb').read_text(encoding='utf-8'))
        stages, calls = [], []
        stage = None
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown':
                match = re.match(r'## PSI\.(\d+)\b', source)
                if match:
                    stage = int(match.group(1)); stages.append(stage)
            elif stage:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'run_analysis':
                            calls.append(stage)
                        if node.func.attr in ['export_psi_tables', 'export_psi_figures']:
                            self.assertEqual(stage, 12)
        self.assertEqual(stages, list(range(1, 13)))
        self.assertEqual(calls, [3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
