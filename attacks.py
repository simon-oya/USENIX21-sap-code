import os
import numpy as np
from collections import Counter
import stat
import subprocess
from scipy.optimize import linear_sum_assignment as hungarian
import time
import utils


GRAPHM_PATH = '/graphm-0.52/bin'


FROM_FUNC_NAME_TO_TIME_NAME = {
    '__init__': 'time_init',
    'process_traces': 'time_process_traces',
    'run_attack': 'time_attack',
    '_build_cost_vol': '_time_cost_vol',
    '_build_cost_freq': '_time_cost_freq',
    '_build_score_vol': '_time_score_vol',
    '_build_score_freq': '_time_score_freq',
    '_build_adj_train_co': '_time_adj_train',
    '_build_adj_test_co': '_time_adj_test',
    '_run_hungarian_attack_given_matrix': '_time_algorithm',
    '_run_graphm_attack_given_matrices': '_time_algorithm',
    '_run_unconstrained_attack_given_matrix': '_time_algorithm'
}


def _timeit(method):
    def timed(self, *args, **kw):
        ts = time.time()
        result = method(self, *args, **kw)
        te = time.time()
        if method.__name__ not in FROM_FUNC_NAME_TO_TIME_NAME:
            raise ValueError("Cannot measure time for method with name '{:s}' since it is not in the list".format(method.__name__))
        else:
            self.time_info[FROM_FUNC_NAME_TO_TIME_NAME[method.__name__]] = np.round((te - ts), 3)
        return result
    return timed


class Attack:

    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        """

        :param results_dir: directory to save the results of the attack
        :param experiment_id: integer to use in filenames
        :param training_data: dictionary with training information
        :param def_name: name of the target defense
        :param def_params: tuple of parameters that characterize the defense
        :param query_dist: distribution from which the queries are generated
        :param query_params: parameters used for query generation
        """

        # Initial information
        self.training_dataset = training_data['dataset']
        self.keywords = training_data['keywords']
        self.frequencies = training_data['frequencies']
        self.n_keywords = len(training_data['keywords'])
        self.n_docs_train = len(training_data['dataset'])

        self.def_name = def_name
        self.def_params = def_params
        self.query_dist = query_dist

        self.results_dir = results_dir

        # Information that the attack uses later (initialize)
        self.n_docs_test = -1
        self.tag_traces = []
        self.tag_info = []

        self.time_info = {}

        return

    def process_initial_information(self, n_docs_test):
        self.n_docs_test = n_docs_test

    @_timeit
    def process_traces(self, traces):
        """
        Assigns tags to each query received, and creates a dictionary with info about each tag.
        Depends on defense name
        :param traces:  for no defense and CLRZ: [ week1, week2, ...] where week1=[trace1, trace2, ...] and trace1 = [doc1, doc2, ...]
                        for PPYY and SEAL: [ week1, week2, ...] where week1=[(id1, vol1), (id2, vol2), ...] where id's are explicit SP leakage
        :return: It assigns self.tag_traces and self.tag_info
        """

        def _process_traces_with_search_pattern_leakage_given_access_pattern(traces):
            """tag_info is a dict [tag] -> AP (list of doc ids)"""
            tag_traces = []
            seen_tuples = {}
            tag_info = {}
            count = 0
            for week in traces:
                weekly_tags = []
                for trace in week:
                    obs_sorted = tuple(sorted(trace))
                    if obs_sorted not in seen_tuples:
                        seen_tuples[obs_sorted] = count
                        tag_info[count] = obs_sorted
                        count += 1
                    weekly_tags.append(seen_tuples[obs_sorted])
                tag_traces.append(weekly_tags)
            return tag_traces, tag_info

        def _process_traces_with_search_pattern_leakage_given_volume(traces):
            """tag_info is a dict [tag] -> response volume"""
            tag_traces = []
            seen_ids = {}
            tag_info = {}
            count = 0
            for week in traces:
                weekly_tags = []
                for id, vol in week:
                    if id not in seen_ids:
                        seen_ids[id] = count
                        tag_info[count] = vol
                        count += 1
                    weekly_tags.append(seen_ids[id])
                tag_traces.append(weekly_tags)
            return tag_traces, tag_info

        if self.def_name in ('none', 'clrz'):
            tag_traces, tag_info = _process_traces_with_search_pattern_leakage_given_access_pattern(traces)
        elif self.def_name in ('ppyy', 'sealvol'):
            tag_traces, tag_info = _process_traces_with_search_pattern_leakage_given_volume(traces)
        else:
            raise ValueError("def name {:s} not recognized".format(self.def_name))

        self.tag_traces = tag_traces
        self.tag_info = tag_info
        return

    @_timeit
    def run_attack(self, att_name, att_params):
        query_predictions_for_each_obs = [[], ]
        query_predictions_for_each_tag = {}
        return query_predictions_for_each_obs, query_predictions_for_each_tag

    def return_time_info(self):
        return self.time_info.copy()


class GraphmAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

        database_matrix = np.zeros((self.n_docs_train, self.n_keywords))
        for i_doc, doc in enumerate(self.training_dataset):
            for keyword in doc:
                if keyword in self.keywords:
                    i_kw = self.keywords.index(keyword)
                    database_matrix[i_doc, i_kw] = 1
        self.binary_database_matrix = database_matrix
        self.results_subdir = os.path.join(results_dir, 'exp_{:06d}'.format(experiment_id))

    @_timeit
    def run_attack(self, att_name, att_params):

        # Run the attack to get query_predictions_for_each_tag
        alpha, dumb_flag = att_params
        m_matrix = self._build_adj_train_co(dumb_flag)
        m_prime_matrix = self._build_adj_test_co(dumb_flag)
        np.fill_diagonal(m_matrix, 0)
        c_matrix = self._build_score_vol(dumb_flag)
        np.fill_diagonal(m_prime_matrix, 0)

        query_predictions_for_each_tag = self._run_graphm_attack_given_matrices(self.results_subdir, list(self.tag_info), m_matrix,
                                                                                m_prime_matrix, c_matrix, alpha)

        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_graphm_attack_given_matrices(self, folder_path, tag_list, m_matrix, m_prime_matrix, c_matrix, alpha, clean_after_attack=True):
        """Runs the attack given the actual matrices already. All the attacks use this, but they provide different matrices.
        Returns the dictionary of predictions for each tag"""

        os.makedirs(folder_path)  # We create and destroy the subdir in this function

        with open(os.path.join(folder_path, 'graph_1'), 'wb') as f:
            utils.write_matrix_to_file_ascii(f, m_matrix)

        with open(os.path.join(folder_path, 'graph_2'), 'wb') as f:
            utils.write_matrix_to_file_ascii(f, m_prime_matrix)

        if alpha > 0:
            with open(os.path.join(folder_path, 'c_matrix'), 'wb') as f:
                utils.write_matrix_to_file_ascii(f, c_matrix)

        with open(os.path.join(folder_path, 'config.txt'), 'w') as f:
            f.write(utils.return_config_text(['PATH'], alpha, os.path.relpath(folder_path, '.'), 'graphm_output'))

        test_script_path = os.path.join(folder_path, 'run_script')
        with open(test_script_path, 'w') as f:
            f.write("#!/bin/sh\n")
            f.write("{:s}/graphm {:s}/config.txt\n".format(os.path.relpath(GRAPHM_PATH, ''), os.path.relpath(folder_path, '.')))
        st = os.stat(test_script_path)
        os.chmod(test_script_path, st.st_mode | stat.S_IEXEC)

        # RUN THE ATTACK
        subprocess.run([os.path.join(folder_path, "run_script")], capture_output=True)

        results = []
        with open(os.path.relpath(folder_path, '.') + '/graphm_output', 'r') as f:
            while f.readline() != "Permutations:\n":
                pass
            f.readline()  # This is the line with the attack names (only PATH, in theory)
            for line in f:
                results.append(int(line)-1)  # Line should be a single integer now

        # COMPUTE PREDICTIONS
        # A result = is a list, where the i-th value (j) means that the i-th training keyword is the j-th testing keyword.
        # This following code reverts that, so that query_predictions_for_each_obs[attack] is a vector that contains the indices of the training
        # keyword for each testing keyword.
        query_predictions_for_each_tag = {}
        for tag in tag_list:
            query_predictions_for_each_tag[tag] = results.index(tag)

        if clean_after_attack:
            os.remove(os.path.join(folder_path, 'graph_1'))
            os.remove(os.path.join(folder_path, 'graph_2'))
            if alpha > 0:
                os.remove(os.path.join(folder_path, 'c_matrix'))
            os.remove(os.path.join(folder_path, 'config.txt'))
            os.remove(os.path.join(folder_path, 'run_script'))
            os.remove(os.path.relpath(folder_path, '.') + '/graphm_output')
            os.rmdir(folder_path)

        return query_predictions_for_each_tag

    @_timeit
    def _build_adj_train_co(self, dumb_flag):
        if dumb_flag or self.def_name in ('none',):
            adj_train_co = np.matmul(self.binary_database_matrix.T, self.binary_database_matrix) / self.n_docs_train
        # elif self.def_name in ('clrz', 'osse'):
        #     tpr, fpr = self.def_params
        #     common_elements = np.matmul(self.binary_database_matrix.T, self.binary_database_matrix)
        #     common_not_elements = np.matmul((1 - self.binary_database_matrix).T, 1 - self.binary_database_matrix)
        #     adj_matrix_train = common_elements * tpr * (tpr - fpr) + common_not_elements * fpr * (fpr - tpr) + self.n_docs_train * tpr * fpr
        #     np.fill_diagonal(adj_matrix_train, np.diag(common_elements) * tpr + np.diag(common_not_elements) * fpr)
        #     adj_train_co = adj_matrix_train / self.n_docs_train
        else:
            raise ValueError('Def name {:s} not recognized'.format(self.def_name))
        return adj_train_co

    @_timeit
    def _build_adj_test_co(self, dumb_flag):
        if dumb_flag or self.def_name in ('none', 'clrz'):
            database_matrix = np.zeros((self.n_docs_test, len(self.tag_info)))
            for tag in self.tag_info:
                for doc_id in self.tag_info[tag]:
                    database_matrix[doc_id, tag] = 1
            adj_train_co = np.matmul(database_matrix.T, database_matrix) / self.n_docs_test
        else:
            raise ValueError('Def name {:s} not recognized'.format(self.def_name))
        return adj_train_co

    # Score matrices for Graphm
    @_timeit
    def _build_score_vol(self, dumb_flag):
        """ Computes the C matrix based on volume values and the Binomial distribution
                :return: C matrix, of dimensions (n_kw_train x n_tags_test)
                """
        if dumb_flag or self.def_name in ('none',):
            # Computing keyword frequency in the training set
            keyword_counter_train = Counter([kw for document in self.training_dataset for kw in document])
            kw_prob_train = [keyword_counter_train[kw] / self.n_docs_train for kw in self.keywords]
            # Computing keyword frequency in the testing set
            kw_counts_test = [len(self.tag_info[tag]) for tag in self.tag_info]
            score_vol = np.exp(utils.compute_log_binomial_probability_matrix(self.n_docs_test, kw_prob_train, kw_counts_test))
        else:
            raise ValueError('def name {:s} not recognized'.format(self.def_name))
        return score_vol


class HungAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

    @_timeit
    def run_attack(self, att_name, att_params):

        alpha, dumb_flag = att_params
        if alpha == 1:
            c_matrix = self._build_cost_freq(dumb_flag)
        elif alpha == 0:
            c_matrix = self._build_cost_vol(dumb_flag)
        else:
            cost_vol = self._build_cost_vol(dumb_flag)
            cost_freq = self._build_cost_freq(dumb_flag)
            c_matrix = cost_freq * alpha + cost_vol * (1 - alpha)

        query_predictions_for_each_tag = self._run_hungarian_attack_given_matrix(c_matrix)
        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_hungarian_attack_given_matrix(self, c_matrix):
        """Runs the Hungarian algorithm with the given cost matrix
        :param c_matrix: cost matrix, (n_keywords x n_tags)"""

        row_ind, col_ind = hungarian(c_matrix)

        query_predictions_for_each_tag = {}
        for tag, keyword in zip(col_ind, row_ind):
            query_predictions_for_each_tag[tag] = keyword

        return query_predictions_for_each_tag

    # Cost matrices for Hungarian
    @_timeit
    def _build_cost_vol(self, dumb_flag):
        keyword_counter_train = Counter([kw for document in self.training_dataset for kw in document])
        kw_probs_train = [keyword_counter_train[kw] / self.n_docs_train for kw in self.keywords]

        # Computing the counts on the test set (volume)
        if self.def_name in ('ppyy', 'sealvol'):
            kw_counts_test = [self.tag_info[tag] for tag in self.tag_info]
        else:
            kw_counts_test = [len(self.tag_info[tag]) for tag in self.tag_info]

        # Computing the cost matrix
        if dumb_flag or self.def_name in ('none',):
            log_prob_matrix = utils.compute_log_binomial_probability_matrix(self.n_docs_test, kw_probs_train, kw_counts_test)
        elif self.def_name in ('clrz',):
            tpr, fpr = self.def_params
            kw_probs_train_mod = [prob * (tpr - fpr) + fpr for prob in kw_probs_train]
            log_prob_matrix = utils.compute_log_binomial_probability_matrix(self.n_docs_test, kw_probs_train_mod, kw_counts_test)
        elif self.def_name in ('ppyy',):
            epsilon = self.def_params[0]
            lap_mean = 2 / epsilon * (64 * np.log(2) + np.log(len(self.keywords)))
            lap_scale = 2 / epsilon
            log_prob_matrix = utils.compute_log_binomial_plus_laplacian_probability_matrix(self.n_docs_test, kw_probs_train, kw_counts_test, lap_mean, lap_scale)
        elif self.def_name in ('sealvol',):
            x = self.def_params[0]
            log_prob_matrix = utils.compute_log_binomial_with_power_rounding(int(self.n_docs_test / x), kw_probs_train, kw_counts_test, x)
        else:
            raise ValueError('def name {:s} not recognized'.format(self.def_name))

        cost_vol = - log_prob_matrix
        return cost_vol

    @_timeit
    def _build_cost_freq(self, dumb_flag):
        if dumb_flag or self.def_name in ('none', 'osse', 'clrz', 'ppyy', 'debug', 'sealvol'):
            trends_tags = utils.build_trend_matrix(self.tag_traces, len(self.tag_info))
            nq_per_week = [len(trace) for trace in self.tag_traces]
            log_c_matrix = np.zeros((len(self.frequencies), len(trends_tags)))
            if self.query_dist in ('multi', 'poiss'):
                for i_week, nq in enumerate(nq_per_week):
                    probabilities = self.frequencies[:, i_week].copy()
                    probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100
                    log_c_matrix += (nq * trends_tags[:, i_week]) * np.log(np.array([probabilities]).T)
            else:
                raise ValueError("Unrecognized query distribution '{:s}".format(self.query_dist))
            return -log_c_matrix
        else:
            raise ValueError('def name {:s} not recognized'.format(self.def_name))


class UncoAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

    @_timeit
    def run_attack(self, att_name, att_params):
        # Run the attack
        c_matrix = self._build_cost_freq()

        query_predictions_for_each_tag = self._run_unconstrained_attack_given_matrix(c_matrix)
        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_unconstrained_attack_given_matrix(self, c_matrix):
        query_predictions_for_each_tag = {}
        for tag in range(c_matrix.shape[1]):
            keyword = np.argmin(c_matrix[:, tag])
            query_predictions_for_each_tag[tag] = keyword
        return query_predictions_for_each_tag

    @_timeit
    def _build_cost_freq(self):
        trends_tags = utils.build_trend_matrix(self.tag_traces, len(self.tag_info))
        cost_freq = np.array([[np.linalg.norm(trend1 - trend2) for trend2 in trends_tags] for trend1 in self.frequencies])
        return cost_freq



