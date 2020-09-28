from manager_df import ManagerDf
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import warnings
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import os


if __name__ == "__main__":

    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    with open('manager_df_data.pkl', 'rb') as f:
        manager = pickle.load(f)

    plot_dictionary_list = []
    styles = ['-', '--']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    title_list = ['clrz', 'ppyy', 'sealvol']

    rand_kw = True

    pp = {'enron_db': 'Enron',
          'lucene_db': 'Lucene',
          'enron_all_db': 'Enron*',
          'graphm': r'$\mathtt{graphm}$',
          'hung': r'$\mathtt{sap}$',
          'unco': r'$\mathtt{freq}$'}

    for dataset_name in ['enron_db', 'lucene_db']:
        for main_title in title_list:
            if main_title == 'ppyy':
                def_list = [('none', ()), ('ppyy', (1,)), ('ppyy', (.2,)), ('ppyy', (.1,))]
                nqr_list = [5]
            elif main_title == 'clrz':
                def_list = [('none', ()), ('clrz', (0.999, 0.01)), ('clrz', (0.999, 0.05)), ('clrz', (0.999, 0.1))]
                nqr_list = [5]
            elif main_title == 'sealvol':
                def_list = [('none', ()), ('sealvol', (2,)), ('sealvol', (3,)), ('sealvol', (4,))]
                nqr_list = [5]
            else:
                raise ValueError("wrong def name")

            alpha_list = [0.5]
            nkw_list = [100, 500, 1000, 3000]

            results_to_plot_accuracy = []
            results_to_plot_times = []
            att_name = 'ccl'
            att_alg = 'hung'

            for iqr, nqr in enumerate(nqr_list):

                if not rand_kw:
                    filename = "defperf_{:s}_{:s}_{:d}.pdf".format(dataset_name, main_title, nqr)
                else:
                    filename = "randkw_defperf_{:s}_{:s}_{:d}.pdf".format(dataset_name, main_title, nqr)


                # For boxplots
                yvalues = []
                ydumbvals = []
                yalpha1vals = []
                bwvals = []
                xvalues = []
                xlabels = []
                colors = []
                for i_def, (def_name, def_params) in enumerate(def_list):
                    for i_nkw, nkw in enumerate(nkw_list):
                        if nkw == 3000 or not rand_kw:
                            tt_mode = 'split_past5-50'
                        else:
                            tt_mode = 'split_rand_past5-50'
                        current_y_mean = -1
                        candidate_y_vals = []
                        candidate_bw_oh = -1
                        current_dumb_mean = -1
                        mean_y_alpha1 = -1
                        for alpha in alpha_list:
                            # For no dumb defense
                            exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                                        'query_number_dist': 'poiss', 'query_params': (nqr,),
                                        'def_name': def_name, 'def_params': def_params,
                                        'att_alg': att_alg, 'att_name': att_name, 'att_params': (alpha, False)}
                            accuracy_vals, _, oh = manager.get_accuracy_time_and_overhead(exp_dict)
                            if len(accuracy_vals) > 0:
                                if np.mean(accuracy_vals) > current_y_mean:
                                    current_y_mean = np.mean(accuracy_vals)
                                    candidate_y_vals = accuracy_vals
                                    if len(oh) == 0:
                                        candidate_bw_oh = 1
                                    else:
                                        candidate_bw_oh = np.mean(oh)
                            else:
                                print("this is empty!")
                            # For dumb defense
                            if def_name == 'none':
                                accuracy_vals_dumb = accuracy_vals
                            else:
                                exp_dict['att_params'] = (alpha, True)
                                accuracy_vals_dumb = manager.get_accuracy_time_and_overhead(exp_dict)[0]
                            if len(accuracy_vals_dumb) > 0:
                                if np.mean(accuracy_vals_dumb) > current_dumb_mean:
                                    current_dumb_mean = np.mean(accuracy_vals_dumb)

                        # For just frequency info (alpha=1), NO DEFENSE (IT DOES NOT MATTER)
                        exp_dict = {'dataset': dataset_name, 'nkw': nkw, 'tt_mode': tt_mode,
                                    'query_number_dist': 'poiss', 'query_params': (nqr,),
                                    'def_name': 'none', 'def_params': (),
                                    'att_alg': att_alg, 'att_name': att_name, 'att_params': (1, False)}
                        accuracy_vals, _, oh = manager.get_accuracy_time_and_overhead(exp_dict)
                        if len(accuracy_vals) > 0:
                            mean_y_alpha1 = np.mean(accuracy_vals)

                        xvalues.append(i_def * (len(nkw_list) + 1) + i_nkw)
                        xlabels.append(str(def_params) + ' ' + str(nkw))
                        yvalues.append(candidate_y_vals)
                        ydumbvals.append(current_dumb_mean)
                        yalpha1vals.append(mean_y_alpha1)
                        bwvals.append(candidate_bw_oh)
                        colors.append('C{:d}'.format(i_nkw))

                fig, ax1 = plt.subplots(figsize=(6, 4))
                box = ax1.boxplot(yvalues, positions=xvalues, patch_artist=True)
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                for item in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(box[item], color='k')
                if dataset_name == 'enron_db':
                    legend_elements = box['boxes'][:len(nkw_list)]
                    legend_labels = ['$n$={:d}'.format(nkw) for nkw in nkw_list]
                    legend1 = Legend(ax1, legend_elements, legend_labels, title='$\\mathtt{sap}$ accuracy', loc='upper left')
                    ax1.add_artist(legend1)

                    legend_elements = [Line2D([0], [0], color='b', linestyle=':', marker='v'), Line2D([0], [0], color='k', linestyle=':', marker='v'),
                                       Line2D([0], [0], color='r', linestyle='', marker='x')]
                    legend_labels = ['freq only $\\mathtt{sap}$', 'naive $\mathtt{sap}$', 'bandwith overhead']
                    legend2 = Legend(ax1, legend_elements, legend_labels, loc='upper center')
                    ax1.add_artist(legend2)

                xtick_positions = []
                for i in range(len(def_list)):
                    mid_pos = 0.5 * (xvalues[i*len(nkw_list)] + xvalues[(i+1)*len(nkw_list) - 1])
                    xtick_positions.append(mid_pos)
                    ax1.plot(xvalues[i*len(nkw_list):(i+1)*len(nkw_list)], ydumbvals[i*len(nkw_list):(i+1)*len(nkw_list)], 'kv:')
                    ax1.plot(xvalues[i*len(nkw_list):(i+1)*len(nkw_list)], yalpha1vals[i*len(nkw_list):(i+1)*len(nkw_list)], 'bv:')
                if main_title == 'ppyy':
                    xtick_labels = ['no defense'] + ['$\\epsilon={:.1f}$'.format(defense[1][0]) for defense in def_list[1:]]
                elif main_title == 'clrz':
                    xtick_labels = ['no defense'] + ['$\\mathtt{FPR}$' + '={:.3f}'.format(defense[1][1]) for defense in def_list[1:]]
                elif main_title == 'sealvol':
                    xtick_labels = ['no defense'] + ['$x$={:d}'.format(defense[1][0]) for defense in def_list[1:]]
                plt.xticks(xtick_positions, xtick_labels, fontsize=12)
                ax1.set_ylim([-0.01, 1.01])
                ax1.set_ylabel('Attack Accuracy', fontsize=12)

                ax2 = ax1.twinx()
                ax2.plot(xvalues, [(oh-1)*100 for oh in bwvals], 'rx')
                ax2.set_ylabel('Bandwidth Overhead (%)', color='r', fontsize=12)
                ax2.set_ylim([-3, 603])
                # plt.title('{:s}  {:s}  {:s}  nqr={:d}'.format(dataset_name, main_title, tt_mode, nqr))

                plt.savefig(plots_path + '/' + filename)
                plt.show()