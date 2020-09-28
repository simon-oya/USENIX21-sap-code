import numpy as np
from collections import Counter
import scipy.stats


def print_matrix(matrix, precision=2):
    for row in matrix:
        for el in row:
            print("{val:.{pre}f} ".format(pre=precision, val=el), end="")
        print("")
    print("")


def write_matrix_to_file_ascii(file, matrix):
    for row in matrix:
        row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
        file.write(row_str.encode('ascii'))


def return_config_text(algorithms_list, alpha, relpath_experiments, out_filename):
    """relpath_experiments: path from where we run graphm to where the graph files are"""

    config_text = """//*********************GRAPHS**********************************
//graph_1,graph_2 are graph adjacency matrices,
//C_matrix is the matrix of local similarities  between vertices of graph_1 and graph_2.
//If graph_1 is NxN and graph_2 is MxM then C_matrix should be NxM
graph_1={relpath:s}/graph_1 s
graph_2={relpath:s}/graph_2 s
C_matrix={relpath:s}/c_matrix s
//*******************ALGORITHMS********************************
//used algorithms and what should be used as initial solution in corresponding algorithms
algo={alg:s} s
algo_init_sol={init:s} s
solution_file=solution_im.txt s
//coeficient of linear combination between (1-alpha_ldh)*||graph_1-P*graph_2*P^T||^2_F +alpha_ldh*C_matrix
alpha_ldh={alpha:.6f} d
cdesc_matrix=A c
cscore_matrix=A c
//**************PARAMETERS SECTION*****************************
hungarian_max=10000 d
algo_fw_xeps=0.01 d
algo_fw_feps=0.01 d
//0 - just add a set of isolated nodes to the smallest graph, 1 - double size
dummy_nodes=0 i
// fill for dummy nodes (0.5 - these nodes will be connected with all other by edges of weight 0.5(min_weight+max_weight))
dummy_nodes_fill=0 d
// fill for linear matrix C, usually that's the minimum (dummy_nodes_c_coef=0),
// but may be the maximum (dummy_nodes_c_coef=1)
dummy_nodes_c_coef=0.01 d

qcvqcc_lambda_M=10 d
qcvqcc_lambda_min=1e-5 d


//0 - all matching are possible, 1-only matching with positive local similarity are possible
blast_match=0 i
blast_match_proj=0 i


//****************OUTPUT***************************************
//output file and its format
exp_out_file={relpath:s}/{out:s} s
exp_out_format=Parameters Compact Permutation s
//other
debugprint=0 i
debugprint_file=debug.txt s
verbose_mode=1 i
//verbose file may be a file or just a screen:cout
verbose_file=cout s
""".format(alg=" ".join(alg for alg in algorithms_list), init=" ".join("unif" for _ in algorithms_list),
           out=out_filename, alpha=alpha, relpath=relpath_experiments)
    return config_text


def _log_binomial(n, beta):
    """Computes an approximation of log(binom(n, n*alpha)) for alpha < 1"""
    if beta == 0 or beta == 1:
        return 0
    elif beta < 0 or beta > 1:
        raise ValueError("beta cannot be negative or greater than 1 ({})".format(beta))
    else:
        entropy = -beta * np.log(beta) - (1 - beta) * np.log(1 - beta)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * beta * (1 - beta))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix


def _vol_prob_matrix(count_rows, counts_cols, prob_fun):
    return np.array([[prob_fun(count1, count2) for count2 in counts_cols] for count1 in count_rows])


def compute_log_binomial_plus_laplacian_probability_matrix(ntrials, probabilities, observations, lap_mean, lap_scale):

    def prob_laplacian_rounded_up_is_x(mu, b, x):
        if x <= mu:
            return 0.5 * (np.exp((x - mu) / b) - np.exp((x - 1 - mu) / b))
        elif mu < x < mu + 1:
            return 1 - 0.5 * (np.exp(-(x - mu) / b) + np.exp((x - 1 - mu) / b))
        else:
            return 0.5 * (np.exp(-(x - 1 - mu) / b) - np.exp(-(x - mu) / b))

    pmf_discrete_laplacian = [prob_laplacian_rounded_up_is_x(lap_mean, lap_scale, x) for x in range(int(2*lap_mean) + 1)]
    log_matrix = np.zeros((len(probabilities), len(observations)))
    for i_row, prob in enumerate(probabilities):
        pmf_binomial = scipy.stats.binom(ntrials, prob).pmf(range(ntrials))
        pmf_sum = np.convolve(pmf_binomial, pmf_discrete_laplacian)
        log_matrix[i_row, :] = [np.log(pmf_sum[obs]) if pmf_sum[obs] > 0 else np.nan_to_num(-np.inf) for obs in observations]
    return log_matrix


def compute_log_binomial_with_power_rounding(ntrials, probabilities, observations, x):
    log_matrix = np.zeros((len(probabilities), len(observations)))
    round_limits = [0] + [x ** i for i in range(int(np.ceil(np.log(ntrials) / np.log(x))) + 1)]
    for i_row, prob in enumerate(probabilities):
        pmf_binomial = scipy.stats.binom(ntrials, prob).pmf(range(ntrials))
        pmf_rounded_dict = {round_limits[i]: sum(pmf_binomial[round_limits[i - 1] + 1:round_limits[i] + 1]) for i in range(1, len(round_limits))}
        log_matrix[i_row, :] = [np.log(pmf_rounded_dict[obs]) if pmf_rounded_dict[obs] > 0 else np.nan_to_num(-np.inf) for obs in observations]
    return log_matrix


def traces_to_binary(traces_flattened, n_docs_test):
    # TODO: do this more efficiently
    binary_traces = np.zeros((len(traces_flattened), n_docs_test))
    for i_trace, trace in enumerate(traces_flattened):
        for doc_id in trace:
            binary_traces[i_trace, doc_id] = 1
    return binary_traces


def build_trend_matrix(traces, n_tags):
    n_weeks = len(traces)
    tag_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_tags in enumerate(traces):
        if len(weekly_tags) > 0:
            counter = Counter(weekly_tags)
            for key in counter:
                tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
    return tag_trend_matrix