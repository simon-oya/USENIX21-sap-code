from manager_df import ManagerDf
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":

    results_file = 'manager_df_data.pkl'
    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    with open(results_file, 'rb') as f:
        manager = pickle.load(f)

    rand_kw = True
    offset_list = [0, 5, 10, 20, 50, 100, 200]
    if not rand_kw:
        filename = "hungoffset.pdf"
        tt_mode_list = ['split_past{:d}-50'.format(offset) for offset in offset_list]
        nqr_list = [5, 100, 250]

    else:
        filename = "randkw_hungoffset.pdf"
        tt_mode_list = ['split_rand_past{:d}-50'.format(offset) for offset in offset_list]
        nqr_list = [5, 100]

    styles = ['.-', 'x--', '^:']
    def_name = 'none'
    def_params = ()
    query_dist = 'poiss'
    att_alg = 'hung'
    att_name = 'ccl'
    att_params = (1.0, False)
    nkw_list = [100, 500, 1000]


    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    for i_nqr, nqr in enumerate(nqr_list):
        for i_nkw, nkw in enumerate(nkw_list):
            yvalues = []
            for i_tt, tt_mode in enumerate(tt_mode_list):
                # For just frequency info
                exp_dict = {'dataset': 'enron_db', 'nkw': nkw, 'tt_mode': tt_mode,
                            'query_number_dist': query_dist, 'query_params': (nqr,),
                            'def_name': def_name, 'def_params': def_params,
                            'att_alg': att_alg, 'att_name': att_name, 'att_params': att_params}
                accuracy_vals, _, _ = manager.get_accuracy_time_and_overhead(exp_dict)
                if len(accuracy_vals) > 0:
                    yvalues.append(np.mean(accuracy_vals))
                else:
                    yvalues.append(np.nan)

            ax1.plot(list(range(len(offset_list))), yvalues, styles[i_nqr], color='C{:d}'.format(i_nkw),
                     label='$n$={:d}'.format(nkw) + ' $\\bar{\\eta}=' + '{:d}$'.format(nqr))
    ax1.legend(ncol=len(nqr_list), loc='upper right')
    plt.xticks(list(range(len(offset_list))), offset_list, fontsize=12)
    ax1.set_ylim([0, 1.01])
    ax1.set_ylabel('Attack Accuracy', fontsize=12)
    ax1.set_xlabel("Adversary's frequency information offset (weeks)", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + '/' + filename)
    plt.show()


