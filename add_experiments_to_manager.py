from manager_df import ManagerDf
import os
import pickle
import numpy as np


def add_liu_attack(manager, nkw_list, dataset_list, nqr_list, kw_random=False):

    if kw_random:
        tt_mode = 'split_rand_past5-50'
    else:
        tt_mode = 'split_past5-50'
    query_dist = 'poiss'
    def_name = 'none'
    def_params = ()
    att_alg = 'unco'
    att_name = 'liu'
    att_params = ()
    for nkw in nkw_list:
        for dataset in dataset_list:
            for nqr in nqr_list:
                experiment_params = {'dataset': dataset, 'nkw': nkw, 'tt_mode': tt_mode,
                                     'query_number_dist': query_dist, 'query_params': (nqr,),
                                     'def_name': def_name, 'def_params': def_params,
                                     'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}

                print("nkw={:d}, {:s}, nqr={:d} ".format(nkw, dataset, nqr), end='')
                manager.initialize_or_add_runs(experiment_params, 30)
    return


def add_hung_all_defs(manager, nkw_list, dataset_list=('enron_db', 'lucene_db'), nqr_list=(5,), alpha_list=(0.5,),
                      tt_mode_list=('split_past5-50',),
                      dumb_attack=True,
                      def_pairs=[('none', ()), ('ppyy', (1.0,)), ('ppyy', (0.2,)), ('ppyy', (0.1,))]
                                + [('clrz', (0.999, 0.01)), ('clrz', (0.999, 0.05)), ('clrz', (0.999, 0.1))]
                                + [('sealvol', (2,)), ('sealvol', (3,)), ('sealvol', (4,))]
                      ):
    att_alg = 'hung'
    query_dist = 'poiss'
    att_false = [('ccl', (np.round(alpha, 2), False)) for alpha in alpha_list]
    att_true = [('ccl', (np.round(alpha, 2), True)) for alpha in alpha_list]
    for tt_mode in tt_mode_list:
        for nkw in nkw_list:
            for dataset in dataset_list:
                for nqr in nqr_list:
                    for def_name, def_params in def_pairs:
                        if def_name == 'none' or dumb_attack == False:
                            att_list = att_false
                        else:
                            att_list = att_false + att_true
                        for att_name, att_params in att_list:
                            experiment_params = {'dataset': dataset, 'nkw': nkw, 'tt_mode': tt_mode,
                                                 'query_number_dist': query_dist, 'query_params': (nqr,),
                                                 'def_name': def_name, 'def_params': def_params,
                                                 'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}
                            print("nkw={:d}, {:s}, nqr={:d}, {:s} {:s}, at_p={:s} {:s}  ".format(nkw, dataset, nqr, def_name, str(def_params), str(att_params), tt_mode), end='')
                            manager.initialize_or_add_runs(experiment_params, 30)
    return


def add_hung_vs_seal_rand(manager, nkw_list, dataset_list, nqr_list, alpha_list):
    tt_mode = 'split_rand_past5-50'
    def_pairs = [('none', ()), ('sealvol', (2,)), ('sealvol', (3,)), ('sealvol', (4,))]
    att_alg = 'hung'
    query_dist = 'poiss'
    att_false = [('ccl', (np.round(alpha, 2), False)) for alpha in alpha_list]
    att_true = [('ccl', (np.round(alpha, 2), True)) for alpha in alpha_list]
    for nkw in nkw_list:
        for dataset in dataset_list:
            for nqr in nqr_list:
                for def_name, def_params in def_pairs:
                    if def_name == 'none':
                        att_list = att_false
                    else:
                        att_list = att_false + att_true
                    for att_name, att_params in att_list:
                        experiment_params = {'dataset': dataset, 'nkw': nkw, 'tt_mode': tt_mode,
                                             'query_number_dist': query_dist, 'query_params': (nqr,),
                                             'def_name': def_name, 'def_params': def_params,
                                             'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}
                        print("nkw={:d}, {:s}, nqr={:d} ".format(nkw, dataset, nqr), end='')
                        manager.initialize_or_add_runs(experiment_params, 30)
    return


def add_hung_vs_basic(manager, nkw_list, dataset_list, nqr_list, alpha_list, offset_list=(5,), query_dist='poiss', kw_random=False):
    att_alg = 'hung'
    def_name = 'none'
    def_params = ()
    att_false = [('ccl', (np.round(alpha, 2), False)) for alpha in alpha_list]
    if kw_random:
        tt_mode_list = ['split_rand_past{:d}-50'.format(offset) for offset in offset_list]
    else:
        tt_mode_list = ['split_past{:d}-50'.format(offset) for offset in offset_list]
    for nkw in nkw_list:
        for tt_mode in tt_mode_list:
            for dataset in dataset_list:
                for nqr in nqr_list:
                    if query_dist == 'poiss':
                        query_params = (nqr,)
                    elif query_dist == 'each':
                        query_params = ()
                    for att_name, att_params in att_false:
                        experiment_params = {'dataset': dataset, 'nkw': nkw, 'tt_mode': tt_mode,
                                             'query_number_dist': query_dist, 'query_params': query_params,
                                             'def_name': def_name, 'def_params': def_params,
                                             'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}
                        print("nkw={:d}, {:s}, {:s}, {:s} {:s}, at_p={:s} {:s}  ".format(nkw, dataset, str(query_params), def_name, str(def_params), str(att_params), tt_mode), end='')
                        manager.initialize_or_add_runs(experiment_params, 30)
    return


def add_graphm_vs_basic(manager, nkw_list, dataset_list, nqr_list, alpha_list, query_dist='poiss', kw_random=False):

    if kw_random:
        tt_mode = 'split_rand_past5-50'
    else:
        tt_mode = 'split_past5-50'
    def_name = 'none'
    def_params = ()
    att_alg = 'graphm'
    att_list = [('pw', (np.round(alpha, 2), False)) for alpha in alpha_list]

    for nkw in nkw_list:
        for dataset in dataset_list:
            for nqr in nqr_list:
                if query_dist == 'poiss':
                    query_params = (nqr,)
                elif query_dist == 'each':
                    query_params = ()
                for att_name, att_params in att_list:
                    experiment_params = {'dataset': dataset, 'nkw': nkw, 'tt_mode': tt_mode,
                                         'query_number_dist': query_dist, 'query_params': query_params,
                                         'def_name': def_name, 'def_params': def_params,
                                         'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}
                    print("nkw={:d}, {:s}, {:s}, par={:s} ".format(nkw, dataset, str(query_params), str(att_params)), end='')
                    manager.initialize_or_add_runs(experiment_params, 30)
    return


if __name__ == "__main__":

    manager_filename = 'manager_df_data.pkl'
    if not os.path.exists(manager_filename):
        manager = ManagerDf()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    # Offset experiment
    add_hung_vs_basic(manager, [100, 500, 1000], ['enron_db'], [5, 100], alpha_list=[1.0], offset_list=[0, 5, 10, 20, 50, 100, 200], kw_random=True)

    # SAP vs Alpha
    add_hung_vs_basic(manager, [100, 500, 1000, 3000], ['enron_db', 'lucene_db'], [5],
                      alpha_list=[i / 10 for i in range(11)], kw_random=True)

    # Attack comparison
    add_hung_vs_basic(manager, [500], ['enron_db', 'lucene_db'], [5, 100, 500], alpha_list=[0.5], kw_random=True)
    add_liu_attack(manager, [500], ['enron_db', 'lucene_db'], [5, 100, 500], kw_random=True)
    add_graphm_vs_basic(manager, nkw_list=[500], dataset_list=['enron_db', 'lucene_db'], nqr_list=[5, 100, 500],
                        alpha_list=[i / 10 for i in range(11)], kw_random=True)

    # Attack vs defenses
    add_hung_all_defs(manager, nkw_list=[100, 500, 1000, 3000], alpha_list=(0.5,), tt_mode_list=('split_rand_past5-50',), dumb_attack=True)
    add_hung_all_defs(manager, nkw_list=[100, 500, 1000, 3000], alpha_list=(0.5,), tt_mode_list=('split_rand_past5-50',), dumb_attack=False)
    add_hung_vs_basic(manager, [100, 500, 1000, 3000], ['enron_db', 'lucene_db'], [5], alpha_list=[1.0], offset_list=[5], kw_random=True)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)
