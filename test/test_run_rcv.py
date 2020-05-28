from strat.run_rcv import RCV
from strat.create_dataset import dataset, prepare_imputation
import unittest
import pandas as pd


class TestRunRCV(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.ptype = 2
        cls.include_age = ('P1', 'P2')
        cls.trts_perc = 0.5
        cls.run_rcv = RCV(cls.ptype, cls.include_age, cls.trts_perc)

    def test_prepare_cs_dataset(self):
        name_df = {'vineland1': 'toy_dataset.txt'}
        df, _ = dataset(name_df, 'toy_demoinfo.txt',
                        save=False, phenotype='autism')
        df_rid = prepare_imputation(df, 0.5)
        dict_tr, dict_ts = self.run_rcv.prepare_cs_dataset(df_rid)
        self.assertIsInstance(dict_tr, dict)
        self.assertIsInstance(dict_ts, dict)
        self.assertEqual([list(dict_tr.keys()),
                          list(dict_ts.keys())],
                         [list(self.include_age)] * 2)

    def test_gridsearch_cv(self):
        df = pd.read_csv('../data/toy_gridsearch.txt',
                         delimiter='\t',
                         header=0,
                         low_memory=False,
                         index_col='subjectkey')
        self.run_rcv.gridsearch_cv(df, (5, 7), (0.40, 0.50), (2, 4),
                                   save='toy_gsearch')

    def test_run_rcv(self):
        df = pd.read_csv('../data/toy_gridsearch.txt',
                         delimiter='\t',
                         header=0,
                         low_memory=False,
                         index_col='subjectkey')
        self.run_rcv.run_rcv(df, 0.40, 5, 2, (2, 4), scatter=True, heatmap=True)

