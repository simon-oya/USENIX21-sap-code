from manager_df import ManagerDf
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
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

    styles = ['.-', 'x--', '^:']  # One per attack

    dataset_list = ['enron_db', 'lucene_db']
    nqr_list = [5]
    att_list = [('hung', 'ccl')]
    nkw_list = [100, 500, 1000, 3000]

    rand_kw = True

    pp = {'enron_db': 'Enron',
          'lucene_db': 'Lucene',
          'enron_all_db': 'Enron*',
          'graphm': r'$\mathtt{graphm}$',
          'hung': r'$\mathtt{hung}$',
          'unco': r'$\mathtt{freq}$'}
    for nqr in nqr_list:
        if not rand_kw:
            filename = "hungalpha_{:d}.pdf".format(nqr)
            tt_mode = 'split_past5-50'
        else:
            filename = "randkw_hungalpha_{:d}.pdf".format(nqr)
            tt_mode = 'split_rand_past5-50'

        fig, axes = plt.subplots(ncols=2, figsize=(6, 4))
        for i_db, dataset_name in enumerate(dataset_list):

            query_params = (nqr,)

            acc_to_plot = []
            acc_lo_to_plot = []
            acc_hi_to_plot = []
            time_to_plot = []
            x_to_plot = []
            styles_to_plot = []
            colors_to_plot = []
            legends_to_plot = []

            for i_att, (att_alg, att_name) in enumerate(att_list):

                for i_nkw, nkw in enumerate(nkw_list):

                    acc_list = []
                    acc_lo_list = []
                    acc_hi_list = []
                    time_list = []
                    x_list = []

                    alpha_list = [np.round(alpha, 2) for alpha in np.linspace(0, 1, 11)]

                    for alpha in alpha_list:
                        exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                                    'query_number_dist': query_dist, 'query_params': query_params,
                                    'def_name': def_name, 'def_params': def_params,
                                    'att_alg': att_alg, 'att_name': att_name, 'att_params': (alpha, False)}
                        accuracy_vals, time_vals, _ = manager.get_accuracy_time_and_overhead(exp_dict)
                        if len(accuracy_vals) > 0:
                            x_list.append(alpha)
                            avg_acc, lo_acc, hi_acc = mean_confidence_interval(accuracy_vals)
                            acc_list.append(avg_acc)
                            acc_lo_list.append(lo_acc)
                            acc_hi_list.append(hi_acc)
                            time_list.append(np.mean(time_vals))
                        else:
                            print("Empty! att={:s}, nkw={:d}, alpha={:.3f}".format(att_name, nkw, alpha))

                        if alpha == 0.5 and dataset_name == 'lucene_db':
                            print('For alpha=0.5 and lucene, n={:d}, acc={:.2f}'.format(nkw, avg_acc*100))

                    acc_to_plot.append(acc_list)
                    acc_lo_to_plot.append(acc_lo_list)
                    acc_hi_to_plot.append(acc_hi_list)
                    time_to_plot.append(time_list)
                    x_to_plot.append(x_list)
                    styles_to_plot.append(styles[i_att])
                    colors_to_plot.append('C{:d}'.format(i_nkw))
                    legends_to_plot.append('$n$={:d}'.format(nkw))

            for yvals, ylo, yhi, xvals, style, color, legend in \
                    zip(acc_to_plot, acc_lo_to_plot, acc_hi_to_plot, x_to_plot, styles_to_plot, colors_to_plot, legends_to_plot):
                if len(yvals) > 0:
                    axes[i_db].plot(xvals, yvals, style, color=color, label=legend)
                    axes[i_db].fill_between(xvals, ylo, yhi, color=color, alpha=0.2)
            if i_db == 0:
                axes[i_db].legend(loc='upper left')
                axes[i_db].set_ylabel("Attack Accuracy", fontsize=12)
            axes[i_db].set_title(pp[dataset_name])
            axes[i_db].set_xlabel("$\\alpha$", fontsize=12)
            axes[i_db].set_ylim([-0.05, 1.05])

        plt.savefig(plots_path + '/' + filename)

    plt.show()





