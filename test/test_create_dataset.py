import unittest
from strat.import_data import ReadData
import strat.utils as ut
from strat import import_data
from strat import create_levels
from strat import create_dataset
import pandas as pd


class TestCreateDataset(unittest.TestCase):
    @classmethod
    def setUp(cls):
        name_df = {'vineland1': 'toy_dataset.txt'}
        cls.instrument_file_name = name_df
        cls.pheno_file_name = '../data/toy_demoinfo.txt'
        cls.phenotype = 'autism'
        cls.readdata = ReadData(cls.instrument_file_name, cls.pheno_file_name, cls.phenotype)

    def test_data_wrangling(self):
        table_dict, demoinfo = self.readdata.data_wrangling(save=False)
        self.assertIsInstance(table_dict, dict)
        self.assertIsInstance(demoinfo, dict)
        self.assertTrue(len(demoinfo) == 9)
        self.assertEqual(table_dict['toy_dataset'].index.tolist(), ['gui1'] * 3 + [
            'gui2', 'gui3', 'gui5', 'gui7', 'gui9', 'gui14', 'gui16', 'gui17'])

    def test_read_ndar_tables(self):
        gui_index = pd.Index([''.join(['gui', str(i + 1)]) for i in range(20)])
        df = import_data._read_ndar_table('toy_dataset.txt', gui_index, '../data')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.index.tolist(), ['gui1'] * 3 + ['gui2', 'gui3'] +
                         ['gui4'] * 2 + ['gui5', 'gui6', 'gui7', 'gui8', 'gui9'] +
                         ['gui10'] * 2 + [''.join(['gui', str(i + 11)]) for i in range(10)])
        self.assertEqual(df.interview_age.tolist(), [30, 30, 60, 40, 150, 300, 300, 10, 72, 30, 40,
                                                     150, 15, 300, 10, 72, 30, 40, 150, 300, 10,
                                                     72, 30, 40])
        self.assertEqual([df.shape[0], df.shape[1]], [24, 17])

    def test_select_phenotype(self):
        guis, demoinfo = import_data._select_phenotype('../data/toy_demoinfo.txt', ut.phenotype_regex,
                                                       ut.exception_regex)
        self.assertEqual(guis.tolist(),
                         ['gui1', 'gui2', 'gui3', 'gui5', 'gui7', 'gui9', 'gui14', 'gui16', 'gui17'])
        self.assertIsInstance(demoinfo, dict)

    def test_create_levels(self):
        gui_index = pd.Index([''.join(['gui', str(i + 1)]) for i in range(20)])
        df1 = import_data._read_ndar_table('toy_dataset.txt', gui_index, '../data')
        df2 = df1.copy()
        for c1, c2 in ut.vineland_mergecol.items():
            df2.rename(columns={c1: c2}, inplace=True)
        double_df_p1 = create_levels.vineland_levels({'vineland1': df1, 'vineland2': df2},
                                                     ut.fixed_col, ut.vineland_col_names,
                                                     ut.vineland_mergecol, ptype=1)
        double_df_p2 = create_levels.vineland_levels({'vineland1': df1, 'vineland2': df2},
                                                     ut.fixed_col, ut.vineland_col_names,
                                                     ut.vineland_mergecol, ptype=2)
        self.assertEqual([double_df_p1.shape[0], double_df_p1.shape[1]], [20, 18])
        self.assertEqual([double_df_p2.shape[0], double_df_p2.shape[1]], [20, 18])
        self.assertEqual(double_df_p1.interview_period.loc[['gui5', 'gui6']].tolist(), ['P1', 'P2'])
        self.assertEqual(double_df_p2.interview_period.loc[['gui5', 'gui6']].tolist(), ['P1', 'P1'])
        self.assertEqual(double_df_p1.columns.tolist(), double_df_p2.columns.tolist())

    def test_drop_obs(self):
        df = pd.DataFrame({'a': [None, 0, 1, 2], 'b': [None, 0, 0, 1]})
        subj = create_levels._drop_obs(df)
        self.assertEqual(subj, [0, 1])

    def test_generate_age_bins(self):
        iage = pd.Series([30, 72, 156, 203, 204])
        page1 = create_levels._generate_age_bins(iage)
        page2 = create_levels._generate_age_bins(iage, ptype=2)
        self.assertEqual(list(page1), ['P1', 'P2', 'P3', 'P4', 'P5'])
        self.assertEqual(list(page2), ['P1', 'P1', 'P2', 'P3', 'P3'])

    def test_dataset(self):
        name_df = {'vineland1': 'toy_dataset.txt',
                   'vineland2': 'toy_dataset.txt'}
        df, demodict = create_dataset.dataset(name_df, '../data/toy_demoinfo.txt',
                                              ptype=1,
                                              save=False, phenotype='autism')
        self.assertEqual(sorted(['gui2', 'gui3', 'gui5', 'gui7', 'gui9', 'gui14', 'gui16',
                                 'gui17']), df.index.tolist())
        self.assertIsInstance(demodict, dict)
        self.assertEqual(list(demodict.keys()), sorted(['gui2', 'gui3', 'gui5', 'gui7', 'gui9', 'gui14', 'gui16',
                                                        'gui17']))

    def test_prepare_imputation(self):
        name_df = {'vineland1': 'toy_dataset.txt',
                   'vineland2': 'toy_dataset.txt'}
        df, _ = create_dataset.dataset(name_df, '../data/toy_demoinfo.txt',
                                       ptype=1,
                                       save=False, phenotype='autism')
        df_rid1 = create_dataset.prepare_imputation(df, 0.05)
        df_rid2 = create_dataset.prepare_imputation(df, 0.40)
        self.assertEqual(df_rid1.index.tolist(), sorted(['gui3', 'gui7', 'gui9', 'gui14', 'gui16',
                                                         'gui17']))
        self.assertEqual(df_rid2.index.tolist(), sorted(['gui2', 'gui3', 'gui7', 'gui5', 'gui9', 'gui14', 'gui16',
                                                         'gui17']))

    def test_check_na_perc(self):
        name_df = {'vineland1': 'toy_dataset.txt',
                   'vineland2': 'toy_dataset.txt'}
        df, _ = create_dataset.dataset(name_df, '../data/toy_demoinfo.txt',
                                       ptype=1,
                                       save=False, phenotype='autism')
        na_feat, na_subj = create_dataset._check_na_perc(df)
        self.assertEqual(list(na_feat.keys()),
                         ['receptive_vscore', 'expressive_vscore', 'written_vscore',
                          'personal_vscore', 'domestic_vscore',
                          'community_vscore', 'interprltn_vscore', 'playleis_vscore', 'copingskill_vscore',
                          'communicationdomain_totalb', 'livingskillsdomain_totalb', 'socializationdomain_totalb',
                          'composite_totalb'])
        self.assertEqual(list(na_subj.keys()),
                         [idx for idx in range(len(na_subj.keys()))])
