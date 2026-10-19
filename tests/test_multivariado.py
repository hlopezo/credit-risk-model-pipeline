"""Pruebas determinísticas de selección 4.0 y contratos con B.3 congelado."""
from pathlib import Path
from unittest.mock import patch
import copy
import importlib.util
import json
import shutil
import unittest
import uuid
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('mult_test','4.0.0.multivariado.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def iv(names=('a','b','c')):
    return pd.DataFrame({'Variable':names,'IV_final':[.4,.3,.2][:len(names)],'Ranking_IV':range(1,len(names)+1)})


def corr(ab=.2,ac=.1,bc=.3):
    return pd.DataFrame([[1,ab,ac],[ab,1,bc],[ac,bc,1]],index=list('abc'),columns=list('abc'))


class SelectionTests(unittest.TestCase):
    def test_01_iv_order(self):
        summary,artifact=self.ranking_fixture()
        self.assertEqual(m.rank_iv(summary,artifact,['a','b','c']).Variable.tolist(),['b','a','c'])

    def ranking_fixture(self):
        summary=pd.DataFrame({'Variable':['c','a','b'],'IV_total':[.1,.2,.5],
            'Estado_bivariado':['APROBADA']*3,'N_bins_efectivos':[2]*3})
        artifact={'approved_variables':['a','b','c'],'variables':{r.Variable:{'state':'APROBADA','iv_total':r.IV_total,'effective_bins':2} for r in summary.itertuples()}}
        return summary,artifact

    def test_02_iv_tie_alphabetical(self):
        summary,artifact=self.ranking_fixture()
        summary.IV_total=.2
        for d in artifact['variables'].values():d['iv_total']=.2
        self.assertEqual(m.rank_iv(summary,artifact,['c','b','a']).Variable.tolist(),list('abc'))

    def test_03_highest_iv_priority(self):
        kept,dec,_=m.select_sequential(iv(),corr(.95,.9,.85))
        self.assertEqual(kept,['a']);self.assertTrue(dec.iloc[1:].Variable_retenida_de_referencia.eq('a').all())

    def test_04_positive(self):
        self.assertNotIn('b',m.select_sequential(iv(),corr(.61))[0])

    def test_05_negative(self):
        _,d,_=m.select_sequential(iv(),corr(-.9))
        self.assertEqual(d.iloc[1].Estado_correlacion,'ELIMINADA_CORRELACION')
        self.assertEqual(d.iloc[1].Correlacion_Pearson,-.9)

    def test_06_below_threshold(self):
        self.assertEqual(m.select_sequential(iv(),corr(.599))[0],list('abc'))

    def test_07_exact_threshold(self):
        self.assertEqual(m.select_sequential(iv(),corr(.60,-.60,.60))[0],list('abc'))

    def test_08_all_retained_and_all_references(self):
        kept,d,details=m.select_sequential(iv(),corr(.1,.7,-.9))
        self.assertEqual(kept,['a','b']);self.assertEqual(d.iloc[2].Variable_retenida_de_referencia,'b')
        self.assertEqual(set(details.Variable_retenida),{'a','b'})

    def test_09_dropped_is_not_a_blocker(self):
        self.assertEqual(m.select_sequential(iv(),corr(.8,.2,.9))[0],['a','c'])

    def test_10_nan_is_not_zero(self):
        for value in (np.nan,np.inf):
            with self.assertRaisesRegex(ValueError,'indefinida'):m.select_sequential(iv(),corr(value))

    def test_11_constant_stops(self):
        with self.assertRaisesRegex(ValueError,'constante'):
            m.train_correlation(pd.DataFrame({'a':[1,1,1],'b':[1,2,3]}),['a','b'])

    def test_12_no_final_pair_exceeds_threshold(self):
        C=corr(.9,.2,.3);kept,_,_=m.select_sequential(iv(),C)
        v=C.loc[kept,kept].abs().to_numpy()
        self.assertFalse((v[np.triu_indices(len(kept),1)]>.6).any())

    def test_13_repeatable(self):
        a=m.select_sequential(iv(),corr(.1,.8,.9));b=m.select_sequential(iv(),corr(.1,.8,.9))
        self.assertEqual(a[0],b[0]);pd.testing.assert_frame_equal(a[1],b[1]);pd.testing.assert_frame_equal(a[2],b[2])

    def test_14_mismatched_iv_stops(self):
        s,a=self.ranking_fixture()
        with self.assertRaisesRegex(ValueError,'Diferencia'):m.rank_iv(s,a,['a','b'])
        s.loc[0,'IV_total']=.999
        with self.assertRaisesRegex(ValueError,'no corresponde'):m.rank_iv(s,a,['a','b','c'])

    def test_15_threshold_parameter(self):
        self.assertEqual(m.select_sequential(iv(),corr(.7),.8)[0],list('abc'))
        with self.assertRaises(ValueError):m.select_sequential(iv(),corr(),1.1)


class RealFrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.biv=m.import_bivariado();cls.bundle=cls.biv.load_v4()
        cls.path=m.BIV_ROOT/'binning_woe_train.json';cls.artifact=cls.biv.load_binning(cls.path)
        cls.woe,_=m.frozen_transform(cls.bundle,cls.artifact,cls.biv,m.BIV_ROOT/'matrices')
        cls.ranking=m.rank_iv(pd.read_csv(m.BIV_ROOT/'reportes/df_resumen_bivariado.csv',float_precision='round_trip'),cls.artifact,cls.woe['TRAIN'].columns)
        cls.C=m.train_correlation(cls.woe['TRAIN'],cls.ranking.Variable.tolist())
        cls.kept,cls.dec,cls.details=m.select_sequential(cls.ranking,cls.C)

    def test_16_three_samples_alignment_and_indices(self):
        m.validate_inputs(self.bundle)
        final=m.apply_selection(self.woe,self.kept)
        validated=m.validate_final(self.bundle,self.woe,final,self.C,self.kept,.6)
        self.assertEqual(len(validated),3)
        for s in m.SEGMENTS:self.assertTrue(final[s].index.equals(self.bundle['X'][s].index))

    def test_17_no_fit_or_holdout_selection(self):
        snapshot=copy.deepcopy(self.artifact)
        names=['learn_candidate','generate_candidates','evaluate_candidates','select_configurations','search_adjacent_partitions','bin_statistics']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:stack.enter_context(patch.object(self.biv,name,side_effect=AssertionError('FIT prohibido')))
            matrices,_=m.frozen_transform(self.bundle,self.artifact,self.biv)
            C=m.train_correlation(matrices['TRAIN'],self.ranking.Variable.tolist())
            kept,_,_=m.select_sequential(self.ranking,C)
            altered={s:X.copy() for s,X in matrices.items()}
            altered['TEST'][:]=999;altered['OOT'][:]=-999
            with patch.object(m,'train_correlation',side_effect=AssertionError('No aprender en aplicación')):
                final=m.apply_selection(altered,kept)
            self.assertEqual(final['TEST'].columns.tolist(),self.kept)
        self.assertEqual(self.artifact,snapshot)

    def test_18_frozen_woe_exact_b3(self):
        for s in m.SEGMENTS:
            saved=pd.read_csv(m.BIV_ROOT/f'matrices/X_{s.lower()}_woe.csv',index_col=0,float_precision='round_trip')
            saved.index.name=self.woe[s].index.name
            pd.testing.assert_frame_equal(self.woe[s],saved,check_exact=True)

    def test_19_persistence_reconstruction_without_correlation(self):
        folder=Path.cwd()/('mult_test_'+uuid.uuid4().hex);folder.mkdir()
        try:
            file=folder/'selection.json'
            m.save_selection(file,self.ranking,self.dec,self.kept,.6,self.path)
            selection=m.load_selection(file,self.path)
            self.assertEqual(selection['retained_variables'],self.kept)
            with patch.object(m,'train_correlation',side_effect=AssertionError('No FIT')):
                final=m.reconstruct(self.bundle,selection,self.biv)
            for s in m.SEGMENTS:pd.testing.assert_frame_equal(final[s],self.woe[s][self.kept],check_exact=True)
            selection['woe_artifact_sha256']='alterado';file.write_text(json.dumps(selection),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash'):m.load_selection(file,self.path)
        finally:
            assert folder.resolve().parent==Path.cwd().resolve() and folder.name.startswith('mult_test_')
            shutil.rmtree(folder)

    def test_20_target_misalignment_stops(self):
        bundle=copy.deepcopy(self.bundle)
        bundle['y']['TEST'].index=bundle['y']['TEST'].index[::-1]
        with self.assertRaisesRegex(ValueError,'desalineados'):m.validate_inputs(bundle)


if __name__=='__main__':unittest.main()
