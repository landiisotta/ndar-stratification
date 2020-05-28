import numpy as np
import sys
import pandas as pd
from strat.create_dataset import prepare_imputation, _impute, _check_na_perc
import strat.utils as ut
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from umap import UMAP
from strat.visualization import _scatter_plot, plot_metrics, plot_miss_heat
import logging
import re
import os
import csv

sys.path.append('/Users/ilandi/PycharmProjects/relative_validation_clustering/')
from reval.best_nclust_cv import FindBestClustCV


class RCV:
    """
    Class initialized with the period type (either 1 or 2, see `create_levels:_generate_age_bins` method),
    the tuple of interview periods to consider for the RCV procedure, and the proportion of observations
    to be included in the test set.

    :param ptype: possible values 1 or 2
    :type ptype: int
    :param include_age: strings of the inerview periods to consider
    :type include_age: tuple
    :param trts_perc: proportion of subjects in test set
    :type trts_perc: float
    """

    def __init__(self, ptype, include_age, trts_perc):
        self.ptype = ptype
        self.include_age = include_age
        self.trts_perc = trts_perc

    def prepare_cs_dataset(self, df):
        """
        Function that takes as input a longitudinal, long-format dataset
        that has already be prepared for imputation with `create_dataset:prepare_imputation`
        funtion.
        It returns a dictionary with keys the interview period to consider
        and values train and test datasets stratified according to
        the count of missing features (if possible). Duplicates are dropped
        and the entries with the higher percentage of available info are retained.

        :param df: longitudinal dataset
        :type df: pandas dataframe
        :return: train dictionary divided by interview period, test dictionary
        :rtype: dict, dict
        """

        # Remove duplicates and retain the row with the
        # lowest missing info
        mask = df.reset_index().duplicated(['subjectkey', 'interview_period'], keep=False)
        dfdup = df.loc[mask.tolist()].copy()
        gui_list = np.unique(dfdup.index)
        mask_drop = []
        for idx in gui_list:
            cou = dfdup.loc[idx].countna.tolist()
            tmp = [False] * dfdup.loc[idx].shape[0]
            tmp[cou.index(min(cou))] = True
            mask_drop.extend(tmp)
        df_out = pd.concat([dfdup.loc[mask_drop], df.loc[(~mask).tolist()]])
        df_out.sort_values(['subjectkey', 'interview_period'], inplace=True)

        # Create a dictionary with interview period as keys
        dict_df = {p: df_out.loc[df_out.interview_period == p]
                   for p in self.include_age}

        # Build train/test sets, stratify by NA=1 or notNA=0
        tr_dict, ts_dict = {}, {}
        for k in dict_df.keys():
            logging.info(f'Number of subjects at {k}: {dict_df[k].shape[0]}')
            try:
                idx_tr, idx_ts = train_test_split(dict_df[k].index,
                                                  stratify=dict_df[k][['hasna']],
                                                  test_size=self.trts_perc,
                                                  random_state=42)
            except ValueError:
                idx_tr, idx_ts = train_test_split(dict_df[k].index,
                                                  test_size=self.trts_perc,
                                                  random_state=42)
            tr_dict[k] = dict_df[k].loc[idx_tr].sort_values(['subjectkey',
                                                             'interview_period'])
            ts_dict[k] = dict_df[k].loc[idx_ts].sort_values(['subjectkey',
                                                             'interview_period'])
            logging.info(f'Number of subjects in training set: {tr_dict[k].shape[0]}')
            logging.info(f'Number of subjects in test set: {ts_dict[k].shape[0]}')
        return tr_dict, ts_dict

    def gridsearch_cv(self, df, n_neigh, na_perc, cl_range, cv_fold, save=None):
        """
        This function can be performed to decide which percentage of missing information
        to allow, and what's the best number of neighbors to consider both for the KNNImputer
        and the KNNClassifier. It takes as input the dataset as output from the function
        `create_dataset:dataset`.

        """
        subdomain_feat = [c for c in df.columns if re.search('vscore', c) and not re.search('written', c)]
        domain_feat = [c for c in df.columns if re.search('totalb', c) and not re.search('composite', c)]
        if save is not None:
            with(open(os.path.join(ut.out_folder, f'{save}.csv'), 'w')) as f:
                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                wr.writerow(['na_perc_thrs', 'n_neigh', 'period', 'feat_lev',
                             'N', 'nclust', 'val_acc', 'val_ci', 'test_acc'])
        transformer = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
        for nap in na_perc:
            dict_tr, dict_ts = self.prepare_cs_dataset(prepare_imputation(df, nap))
            for n in n_neigh:
                impute = KNNImputer(n_neighbors=n)
                dict_imp = {p: _impute(dict_tr[p], dict_ts[p], impute)
                            for p in self.include_age}
                knn = KNeighborsClassifier(n_neighbors=n)
                clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')

                relval = FindBestClustCV(s=knn, c=clust, nfold=cv_fold, nclust_range=cl_range,
                                         nrand=100)
                # Run the model
                for p, tup in dict_imp.items():
                    X_tr = transformer.fit_transform(tup[0][subdomain_feat])
                    X_ts = transformer.transform(tup[1][subdomain_feat])
                    metric, ncl, cv_scores = relval.best_nclust(X_tr)
                    out = relval.evaluate(X_tr, X_ts, ncl)
                    ci = (1 - (metric['val'][ncl][1][0] + metric['val'][ncl][1][1]),
                          1 - (metric['val'][ncl][1][0] - metric['val'][ncl][1][1]))
                    if save is not None:
                        with open(os.path.join(ut.out_folder, f'{save}.csv'), 'a') as f:
                            wr = csv.writer(f, delimiter=';', lineterminator='\n')
                            wr.writerow([nap, n, p, 'subdomain', (X_tr.shape[0], X_ts.shape[0]),
                                         ncl, 1 - metric['val'][ncl][0],
                                         ci, out.test_acc])
                    X_tr = transformer.fit_transform(tup[0][domain_feat])
                    X_ts = transformer.transform(tup[1][domain_feat])
                    metric, ncl, cv_scores = relval.best_nclust(X_tr)
                    out = relval.evaluate(X_tr, X_ts, ncl)
                    ci = (1 - (metric['val'][ncl][1][0] + metric['val'][ncl][1][1]),
                          1 - (metric['val'][ncl][1][0] - metric['val'][ncl][1][1]))
                    if save is not None:
                        with open(os.path.join(ut.out_folder, f'{save}.csv'), 'a') as f:
                            wr = csv.writer(f, delimiter=';', lineterminator='\n')
                            wr.writerow([nap, n, p, 'domain', (X_tr.shape[0], X_ts.shape[0]),
                                         ncl, 1 - metric['val'][ncl][0],
                                         ci, out.test_acc])

    def run_rcv(self, df, n_perc, n_neigh, cv_fold, cl_range, scatter=False, heatmap=False):
        flatui = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                  "#7f7f7f", "#bcbd22", "#17becf", "#8c564b", "#a55194"]

        subdomain_feat = [c for c in df.columns if re.search('vscore', c) and not re.search('written', c)]
        domain_feat = [c for c in df.columns if re.search('totalb', c) and not re.search('composite', c)]

        dict_tr, dict_ts = self.prepare_cs_dataset(prepare_imputation(df, n_perc))

        transformer = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
        impute = KNNImputer(n_neighbors=n_neigh)
        dict_imp = {p: _impute(dict_tr[p], dict_ts[p], impute)
                    for p in self.include_age}
        knn = KNeighborsClassifier(n_neighbors=n_neigh)
        clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')

        relval = FindBestClustCV(s=knn, c=clust, nfold=cv_fold, nclust_range=cl_range,
                                 nrand=100)
        # Run the model
        for p, tup in dict_imp.items():
            X_tr = transformer.fit_transform(tup[0][subdomain_feat])
            X_ts = transformer.transform(tup[1][subdomain_feat])
            metric, ncl, cv_scores = relval.best_nclust(X_tr)
            out = relval.evaluate(X_tr, X_ts, ncl)
            logging.info(f"Best number of clusters: {ncl}")
            logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")
            dict_imp[p][0]['cluster_subdomain'], dict_imp[p][1]['cluster_subdomain'] = out.train_cllab, out.test_cllab
            plot_metrics(metric,
                         f'UMAP preprocessed dataset, RCV misclassification performance at {p}, level subdomain')
            if scatter:
                _scatter_plot(X_tr,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][0].index, out.train_cllab)],
                              flatui,
                              10, 15,
                              {str(ncl): '-'.join(['cluster', str(ncl + 1)]) for ncl in
                               sorted(np.unique(out.train_cllab))},
                              title=f'Subgroups of UMAP preprocessed Vineland TRAINING '
                                    f'dataset (period: {p} -- level: subdomain)')

                _scatter_plot(X_ts,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][1].index, out[2])],
                              flatui,
                              10, 15, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(out[2]))},
                              title=f'Subgroups of UMAP preprocessed Vineland TEST '
                                    f'dataset (period: {p} -- level: subdomain)')
            if heatmap:
                dict_ts[p]['cluster'] = out.test_cllab
                feat = []
                values = []
                cl_labels = np.repeat(sorted(dict_ts[p].cluster.unique().astype(str)), len(subdomain_feat))
                for lab in np.unique(sorted(out.test_cllab)):
                    na_feat, _ = _check_na_perc(dict_ts[p].loc[dict_ts[p].cluster == lab][subdomain_feat])
                    feat.extend(list(na_feat.keys()))
                    values.extend(list(na_feat.values()))
                plot_miss_heat(dict_ts[p], cl_labels, feat, values, period=p, hierarchy='subdomain')

            X_tr = transformer.fit_transform(tup[0][domain_feat])
            X_ts = transformer.transform(tup[1][domain_feat])
            metric, ncl, cv_scores = relval.best_nclust(X_tr)
            plot_metrics(metric,
                         f'UMAP preprocessed dataset, RCV misclassification performance at {p}, level domain')
            out = relval.evaluate(X_tr, X_ts, ncl)
            logging.info(f"Best number of clusters: {ncl}")
            logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")
            dict_imp[p][0]['cluster_domain'], dict_imp[p][1]['cluster_domain'] = out.train_cllab, out.test_cllab
            if scatter:
                _scatter_plot(X_tr,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][0].index, out.train_cllab)],
                              flatui,
                              10, 15,
                              {str(ncl): '-'.join(['cluster', str(ncl + 1)]) for ncl in
                               sorted(np.unique(out.train_cllab))},
                              title=f'Subgroups of UMAP preprocessed Vineland TRAINING '
                                    f'dataset (period: {p} -- level: domain)')

                _scatter_plot(X_ts,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][1].index, out[2])],
                              flatui,
                              10, 15, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(out[2]))},
                              title=f'Subgroups of UMAP preprocessed Vineland TEST '
                                    f'dataset (period: {p} -- level: domain)')
            if heatmap:
                dict_ts[p]['cluster'] = out.test_cllab
                feat = []
                values = []
                cl_labels = np.repeat(sorted(dict_ts[p].cluster.unique().astype(str)), len(domain_feat))
                for lab in np.unique(sorted(out.test_cllab)):
                    na_feat, _ = _check_na_perc(dict_ts[p].loc[dict_ts[p].cluster == lab][domain_feat])
                    feat.extend(list(na_feat.keys()))
                    values.extend(list(na_feat.values()))
                plot_miss_heat(dict_ts[p], cl_labels, feat, values, period=p, hierarchy='subdomain')
