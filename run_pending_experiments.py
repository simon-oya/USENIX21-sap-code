import numpy as np
import os
import time
from experiment import run_single_experiment
import multiprocessing
from functools import partial
import pickle
from datetime import datetime
import pytz


def print_exp_to_run(d, n_runs):
    print('********************************************************')
    print('Setting: {:s} {:s} nkw={:d}'.format(d['dataset'], d['tt_mode'], d['nkw']))
    print('Query: {:s} {:s}'.format(d['query_number_dist'], str(d['query_params'])))
    print('Def: {:s} {:s}'.format(d['def_name'], str(d['def_params'])))
    print('Att: {:s} {:s}, {:s}'.format(d['att_alg'], d['att_name'], str(d['att_params'])))
    print("* Number of runs: {:d}".format(n_runs), flush=True)


if __name__ == "__main__":

    time_init = time.time()
    PRO_DATASETS_PATH = 'datasets_pro'
    EXPERIMENTS_PATH = 'results'

    tz_ON = pytz.timezone('Canada/Eastern')

    # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(30)

    subdirectories = os.scandir(EXPERIMENTS_PATH)
    for file in sorted(os.scandir(EXPERIMENTS_PATH), key=lambda e: e.name):
        if file.name.startswith('todo_') and file.name.endswith('.pkl'):
            with open(file, 'rb') as f:
                parameter_dict, seeds_to_run = pickle.load(f)
            os.remove(file)
            print("Read and deleted {:s}".format(file.name))
            print_exp_to_run(parameter_dict, len(seeds_to_run))
            time_init_this_exp = time.time()

            # Let's create the folder here so that we avoid concurrency issues
            EXTENSION = '{:s}_{:d}_{:s}_{:s}_{:s}'.format(parameter_dict['dataset'], parameter_dict['nkw'], parameter_dict['tt_mode'],
                                                        parameter_dict['def_name'], parameter_dict['att_name'])
            EXPERIMENTS_PATH_EXT = os.path.join(EXPERIMENTS_PATH, EXTENSION)
            if not os.path.exists(EXPERIMENTS_PATH_EXT):
                os.makedirs(EXPERIMENTS_PATH_EXT)

            partial_function = partial(run_single_experiment, PRO_DATASETS_PATH, EXPERIMENTS_PATH_EXT, parameter_dict)
            pool.map(partial_function, seeds_to_run)

            now = datetime.now(tz_ON)
            print("****[{:s}] Done experiments, elapsed time {:.0f} (total {:.0f})".format(now.strftime("%H:%M:%S"), time.time() - time_init_this_exp,
                                                                                           time.time() - time_init), flush=True)
