import os
import numpy as np
import pickle
import time
from attacks import GraphmAttack, HungAttack, UncoAttack
from defenses import Defense


def load_pro_dataset(pro_dataset_path):
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))

    with open(pro_dataset_path, "rb") as f:
        dataset, keyword_dict = pickle.load(f)

    return dataset, keyword_dict


def generate_experiment_id_and_subfolder(experiment_path):
    """Given a path, finds an id that does not exist in there and creates a results_id.temp file and a exp_id subfolder"""

    # Create subfolder if it does not exist
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Choose a experiment number for a subsubfolder:
    exp_number = np.random.randint(1000000)
    tries = 0
    while tries <= 10 and (os.path.exists(os.path.join(experiment_path, 'results_{:d}.pkl'.format(exp_number)))
                           or os.path.exists(os.path.join(experiment_path, 'results_{:d}.temp'.format(exp_number)))):
        exp_number = np.random.randint(1000000)
        tries += 1
    if tries == 100:
        print("Could not find a file that didn't exist, aborting...")
        return -1
    else:
        with open(os.path.join(experiment_path, 'results_{:d}.temp'.format(exp_number)), 'w') as f:
            pass
        return exp_number


def generate_keyword_queries(trend_matrix_norm, query_number_dist, query_params):

    n_kw, n_weeks = trend_matrix_norm.shape
    real_queries = []
    if query_number_dist == 'multi':  # Multinomial distribution
        n_qr = query_params[0]
        for i_week in range(n_weeks):
            query_pmf = trend_matrix_norm[:, i_week]
            real_queries_this_week = list(np.random.choice(list(range(n_kw)), n_qr, p=query_pmf))
            real_queries.append(real_queries_this_week)
    if query_number_dist == 'poiss':  # Poisson distribution
        n_qr = query_params[0]
        for i_week in range(n_weeks):
            n_qr_this_week = np.random.poisson(n_qr)
            query_pmf = trend_matrix_norm[:, i_week]
            real_queries_this_week = list(np.random.choice(list(range(n_kw)), n_qr_this_week, p=query_pmf))
            real_queries.append(real_queries_this_week)
    elif query_number_dist == 'each':  # One query of each keyword per week
        for i_week in range(n_weeks):
            queries = list(range(trend_matrix_norm.shape[0]))
            np.random.shuffle(queries)
            real_queries.append(queries)
    else:
        raise ValueError("Query params not recognized")
    return real_queries


def run_single_experiment(pro_dataset_path, experiments_path_ext, exp_params, seed, debug_mode=False):
    """
    parameter_dict has fields: init_dataset,
    Runs a single graph matching experiment
    :return: Nothing. It saves the results to results.pkl file. In debug mode it returns the accuracy_dict.
        The results contain a dictionary with the accuracy of each attack and the experiment parameters.
    """

    def _generate_exp_number(experiments_path_ext, seed):
        """
        Generates an experiment number for this run, and creates a results_{}.temp file to save that number for this experiment.
        :param experiments_path_ext: path to create the experiment folder exp_id
        :param seed: seed used for this run
        :return exp_number: experiment number
        """
        exp_number = int((seed % 1e3) * 1000000) + int(time.time()*1e12 % 1e3) * 1000 + int((os.getpid()) % 1e3)
        tries = 0
        while tries <= 10 and (os.path.exists(os.path.join(experiments_path_ext, 'results_{:06d}.pkl'.format(exp_number)))
                               or os.path.exists(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))):
            exp_number = int((seed % 1e3) * 10000000) + int(time.time()*1e12 % 1e3) * 10000 + int((os.getpid()) % 1e4)
            tries += 1
        if tries == 10:
            print("Could not find a file that didn't exist, skipping...")
            return -1
        else:
            with open(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)), 'w') as f:
                pass
            if debug_mode:
                print('Running experiment to generate file results_{:06d}.pkl'.format(exp_number))
            return exp_number

    def _generate_train_test_data(dataset_name, n_keywords, tt_mode):

        dataset, keyword_dict = load_pro_dataset(os.path.join(pro_dataset_path, dataset_name + '.pkl'))

        if len(tt_mode.split('_')) == 2:
            mode_ds, mode_freq = tt_mode.split('_')
            mode_kw = 'top'
        elif len(tt_mode.split('_')) == 3:
            mode_ds, mode_kw, mode_freq = tt_mode.split('_')
        else:
            raise ValueError("Wrong format for tt_mode: {:s}".format(tt_mode))
        assert mode_ds in ('same', 'split')
        assert mode_kw in ('top', 'rand')

        if mode_kw == 'top':
            chosen_keywords = sorted(keyword_dict.keys(), key=lambda x: keyword_dict[x]['count'], reverse=True)[:n_keywords]
        else:  # mode_kw == 'rand':
            keywords = list(keyword_dict.keys())
            permutation = np.random.permutation(len(keywords))
            chosen_keywords = [keywords[idx] for idx in permutation[:n_keywords]]

        trend_matrix = np.array([keyword_dict[kw]['trend'] for kw in chosen_keywords])
        trend_matrix_norm = trend_matrix.copy()
        for i_col in range(trend_matrix_norm.shape[1]):
            if sum(trend_matrix_norm[:, i_col]) == 0:
                print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
                trend_matrix_norm[:, i_col] = 1 / n_keywords
            else:
                trend_matrix_norm[:, i_col] = trend_matrix_norm[:, i_col] / sum(trend_matrix_norm[:, i_col])

        if mode_ds == 'same':
            permutation = np.random.permutation(len(dataset))
            dataset_half = [dataset[i] for i in permutation[int(len(dataset) / 2):]]
            data_adv = dataset_half
            data_cli = dataset_half
        elif mode_ds == 'split':
            permutation = np.random.permutation(len(dataset))
            data_adv = [dataset[i] for i in permutation[int(len(dataset) / 2):]]
            data_cli = [dataset[i] for i in permutation[:int(len(dataset) / 2)]]
        else:
            raise ValueError('Unknown dataset tt mode {:d}'.format(mode_ds))

        if mode_freq.startswith('same'):
            n_weeks = int(mode_freq[4:])
            assert n_weeks > 0
            freq_adv = trend_matrix_norm[:, -n_weeks:]
            freq_cli = trend_matrix_norm[:, -n_weeks:]
            freq_real = trend_matrix_norm[:, -n_weeks:]
        elif mode_freq.startswith('past'):
            offset, n_weeks = [int(val) for val in mode_freq[4:].split('-')]
            if offset == 0:
                freq_adv = trend_matrix_norm[:, -n_weeks:]
                freq_cli = trend_matrix_norm[:, -n_weeks:]
                freq_real = trend_matrix_norm[:, -n_weeks:]
            else:
                freq_adv = trend_matrix_norm[:, -offset-n_weeks:-offset]
                freq_cli = trend_matrix_norm[:, -offset-n_weeks:-offset]
                freq_real = trend_matrix_norm[:, -n_weeks:]
        elif mode_freq.startswith('randn'):
            scaling, n_weeks = [int(val) for val in mode_freq[5:].split('-')]
            assert n_weeks > 0
            freq_real = trend_matrix_norm[:, -n_weeks:]
            trend_matrix_noisy = np.copy(trend_matrix_norm[:, -n_weeks:])
            for row in trend_matrix_noisy:
                row += np.random.normal(loc=0, scale=scaling * np.std(row), size=len(row))
            trend_matrix_noisy = np.abs(trend_matrix_noisy)
            freq_adv = trend_matrix_noisy
            freq_cli = freq_adv

        else:
            raise ValueError('Unknown frequencies tt mode {:d}'.format(mode_freq))

        full_data_adv = {'dataset': data_adv,
                         'keywords': chosen_keywords,
                         'frequencies': freq_adv}
        full_data_client = {'dataset': data_cli,
                            'keywords': chosen_keywords,
                            'frequencies': freq_cli}
        return full_data_adv, full_data_client, freq_real

    exp_number = _generate_exp_number(experiments_path_ext, seed)
    if exp_number == -1:
        return

    np.random.seed(seed)

    full_data_adv, full_data_client, trend_real_norm = _generate_train_test_data(exp_params['dataset'], exp_params['nkw'], exp_params['tt_mode'])

    if exp_params['att_alg'] == 'graphm':
        attack = GraphmAttack(experiments_path_ext, exp_number, full_data_adv, exp_params['def_name'], exp_params['def_params'],
                              exp_params['query_number_dist'], exp_params['query_params'])
    elif exp_params['att_alg'] == 'hung':
        attack = HungAttack(experiments_path_ext, exp_number, full_data_adv, exp_params['def_name'], exp_params['def_params'],
                            exp_params['query_number_dist'], exp_params['query_params'])
    elif exp_params['att_alg'] == 'unco':
        attack = UncoAttack(experiments_path_ext, exp_number, full_data_adv, exp_params['def_name'], exp_params['def_params'],
                            exp_params['query_number_dist'], exp_params['query_params'])
    else:
        raise ValueError('Unknown attack algorithm: {:s}'.format(exp_params['att_alg']))
    defense = Defense(full_data_client, exp_params['def_name'], exp_params['def_params'])
    real_queries = generate_keyword_queries(trend_real_norm, exp_params['query_number_dist'], exp_params['query_params'])
    attack.process_initial_information(n_docs_test=defense.get_dataset_size_for_adversary())
    traces, bw_overhead = defense.generate_query_traces(real_queries)
    attack.process_traces(traces)

    query_predictions_for_each_obs, query_predictions_for_each_tag = attack.run_attack(exp_params['att_name'], exp_params['att_params'])

    if query_predictions_for_each_obs is None:
        print("query_predictions was none for exp_number={:d}, aborting".format(exp_number))
        return

    flat_real = [kw for week_kws in real_queries for kw in week_kws]
    flat_pred = [kw for week_kws in query_predictions_for_each_obs for kw in week_kws]
    accuracy = np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(flat_real, flat_pred)]))
    if debug_mode:
        print("For {:s} {}  bw_overhead = {:.3f}, time_attack = {:.3f} secs, accuracy = {:.3f}"
              .format(exp_params['att_name'], exp_params['att_params'], bw_overhead, attack.time_info['time_attack'], accuracy))
        print(" ")
        return accuracy
    else:
        results_filename = 'results_{:06d}.pkl'.format(exp_number)
        with open(os.path.join(experiments_path_ext, results_filename), 'wb') as f:
            res_dict = {'seed': seed, 'accuracy': accuracy}
            time_info = attack.return_time_info()
            res_dict.update(time_info)
            res_dict['bw_overhead'] = bw_overhead
            pickle.dump((exp_params, res_dict), f)
            # print('Saved {:s} (elapsed {:.0f} secs)'.format(results_filename, time.time() - time_init))
    os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))

