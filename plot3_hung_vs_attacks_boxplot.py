from manager_df import ManagerDf
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from matplotlib.lines import Line2D
import os

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


if __name__ == "__main__":

    results_file = 'manager_df_data.pkl'
    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    with open(results_file, 'rb') as f:
        manager = pickle.load(f)

    def_name = 'none'
    def_params = ()
    query_dist = 'poiss'

    rand_kw = True

    pp = {'enron_db': 'Enron',
          'lucene_db': 'Lucene',
          'enron_all_db': 'Enron*',
          'graphm': r'$\mathtt{graphm}$',
          'hung': r'$\mathtt{sap}$',
          'unco': r'$\mathtt{freq}$'}

    block_list = [('enron_db', 5), ('enron_db', 100), ('enron_db', 500), ('lucene_db', 5), ('lucene_db', 100), ('lucene_db', 500)]

    att_list = [('hung', 'ccl'), ('graphm', 'pw'), ('unco', 'liu')]
    nkw = 500

    if not rand_kw:
        tt_mode = 'split_past5-50'
        filename = "attcomp_box_nkw{:d}.pdf".format(nkw)
    else:
        tt_mode = 'split_rand_past5-50'
        filename = "randkw_attcomp_box_nkw{:d}.pdf".format(nkw)

    boxes_vals = []
    boxes_time_vals = []
    xvals = []
    boxcolors = []

    for i_block, (dataset_name, nqr) in enumerate(block_list):
        query_params = (nqr,)
        for i_att, (att_alg, att_name) in enumerate(att_list):

            if att_alg == 'hung':
                exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                            'query_number_dist': query_dist, 'query_params': query_params,
                            'def_name': def_name, 'def_params': def_params,
                            'att_alg': att_alg, 'att_name': att_name, 'att_params': (0.5, False)}
                accuracy_vals, time_vals, _ = manager.get_accuracy_time_and_overhead(exp_dict)
            elif att_alg == 'unco':
                exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                            'query_number_dist': query_dist, 'query_params': query_params,
                            'def_name': def_name, 'def_params': def_params,
                            'att_alg': att_alg, 'att_name': att_name, 'att_params': ()}
                accuracy_vals, time_vals, _ = manager.get_accuracy_time_and_overhead(exp_dict)
            elif att_alg == 'graphm':
                alpha_list = [np.round(alpha, 2) for alpha in np.linspace(0, 1, 11)]
                current_best_vals = [-1]
                current_time_vals = [-1]
                for alpha in alpha_list:
                    exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                                'query_number_dist': query_dist, 'query_params': query_params,
                                'def_name': def_name, 'def_params': def_params,
                                'att_alg': att_alg, 'att_name': att_name, 'att_params': (alpha, False)}
                    accuracy_vals, time_vals, _ = manager.get_accuracy_time_and_overhead(exp_dict)
                    if len(accuracy_vals) > 0:
                        if np.mean(current_best_vals) < np.mean(accuracy_vals):
                            current_best_vals = accuracy_vals
                            current_time_vals = time_vals
                accuracy_vals = current_best_vals
                time_vals = current_time_vals
            else:
                raise ValueError("Wrong attack alg {:s}".format(att_alg))

            boxes_vals.append(accuracy_vals)
            boxes_time_vals.append(time_vals)
            xvals.append(i_block * (len(block_list) + 0) + i_att)
            boxcolors.append('C{:d}'.format(i_att))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    box = ax1.boxplot(boxes_vals, positions=xvals, patch_artist=True)
    for patch, color in zip(box['boxes'], boxcolors):
        patch.set_facecolor(color)
    for item in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box[item], color='k')
    lgd = ax1.legend(box['boxes'][:len(att_list)] + [Line2D([0], [0], linestyle='', marker='x', color='r')],
               ['{:s} accuracy'.format(pp[att_alg]) for att_alg, _ in att_list] + ['running time'],
               ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.05))

    xtick_labels = ['{:s}\n '.format(pp[dataset_name]) + '$\\bar{\\eta}$' + '={:d}'.format(nqr) for dataset_name, nqr in block_list]
    xtick_positions = []
    for i in range(len(block_list)):
        pos = 0.5*(xvals[i*len(att_list)] + xvals[(i+1)*len(att_list)-1])
        xtick_positions.append(pos)
    plt.xticks(xtick_positions, xtick_labels, fontsize=12)
    ax1.set_ylim([-0.02, 1.01])
    ax1.set_ylabel('Attack Accuracy', fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(xvals, [np.mean(times) for times in boxes_time_vals], 'rx')
    ax2.set_ylabel('Running Time (seconds)', color='r', fontsize=12)
    ax2.set_yscale('log')

    for i in range(len(block_list) - 1):
        pos = 0.5*(xvals[(i+1)*len(att_list)-1] + xvals[(i+1)*len(att_list)])
        ax1.plot([pos, pos], [0, 1], 'k-', alpha=0.2)

    plt.savefig(plots_path + '/' + filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()
