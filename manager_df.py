import numpy as np
import os
import pickle
import pandas as pd


def print_console_options():
    print("""
--------------------------------------
[pa] Prints ALL experiments
[pr] Prints REMAINING experiments
[pp <int>] Print a PARTICULAR experiment given an index
[p <col=value> ...] Print experiments that match the column values
[w] Write experiments to run to a file
[eat] Eat pickles        
[reset <int> <int>] Reset target runs and results of experiments between two indices (included)
[remove <int> <int>] Removes experiments and results between two indices (included)
[e] Exit""")


exp_params_column = ['dataset', 'nkw', 'tt_mode', 'query_number_dist', 'query_params',
                     'def_name', 'def_params', 'att_alg', 'att_name', 'att_params',
                     'target_runs', 'res_pointer']


class ManagerDf:

    def __init__(self):
        """
        self.experiments is dataframe with the experiment description
        self.results is a dictionary with the results stored in a dataframe format (seed, accuracy, time_attack, time....)
        """
        self.experiments = pd.DataFrame(columns=exp_params_column)
        self.results = {}

    def _get_new_pointer(self):
        new_pointer = len(self.results)
        if new_pointer in self.results:
            new_pointer = 0
            while new_pointer in self.results:
                new_pointer += 1
        return new_pointer

    def _find_indices(self, parameter_dict):
        if len(self.experiments.index) == 0:
            return []
        mask = pd.Series([True]*len(self.experiments.index))
        for key, value in parameter_dict.items():
            mask &= self.experiments[key] == value
        if any(mask):
            return self.experiments[mask].index
        else:
            return []

    def _find_pointer(self, param_dict):
        indices = self._find_indices(param_dict)
        if len(indices) == 1:  # Probably error because this is a list?
            return self.experiments.at[indices[0], 'res_pointer']
        else:
            return -1

    def _create_new_experiment(self, param_dict, target_runs):
        dataframe_row = param_dict.copy()
        indices = self._find_indices(dataframe_row)
        if len(indices) == 1:
            print("WARNING! Experiment already existed, cannot create")
            return -1
        elif len(indices) > 1:
            print("WARNING! Experiment was duplicated, there is something wrong here")
            return -1
        else:
            dataframe_row['target_runs'] = target_runs
            new_pointer = self._get_new_pointer()
            dataframe_row['res_pointer'] = new_pointer
            self.experiments = self.experiments.append(dataframe_row, ignore_index=True)
            self.results[new_pointer] = pd.DataFrame(columns=['seed', 'accuracy', 'time_attack', 'bw_overhead'])
            return new_pointer

    def _add_results(self, param_dict, res_dict):
        indices = self._find_indices(param_dict)
        if len(indices) > 1:
            print("We found multiple experiments that match the query, cannot add results")
            return -1
        elif len(indices) == 0:
            print("Experiment did not exist, creating it")
            pointer = self._create_new_experiment(param_dict, [])
        else:  # len(index) = 1
            index = indices[0]
            pointer = self.experiments.loc[index]['res_pointer']

        if res_dict['seed'] not in self.results[pointer]['seed'].values:
            self.results[pointer] = self.results[pointer].append(res_dict, ignore_index=True)
        else:
            print("Cannot add experiment because this seed is already registered")

    def import_experiments_from_old_manager_list(self, experiment_list):
        for i, experiment in enumerate(experiment_list[:1]):
            self._create_new_experiment(experiment['params'], len(experiment['target_seeds']))
            df_dict = experiment['df'].to_dict('index')
            for key in df_dict:
                self._add_results(experiment['params'], df_dict[key])
            print("Done adding results for experiment {:d}/{:d}".format(i, len(experiment_list)))

    # Called from plot
    def get_accuracy_time_and_overhead(self, param_dict):
        pointer = self._find_pointer(param_dict)
        if pointer >= 0:
            return list(self.results[pointer]['accuracy'].values), list(self.results[pointer]['time_attack'].values), list(self.results[pointer]['bw_overhead'].values)
        else:
            return [], [], []

    def initialize_or_add_runs(self, param_dict, target_runs):
        indices = self._find_indices(param_dict)
        if len(indices) == 1:
            index = indices[0]
            if self.experiments.loc[index]['target_runs'] >= target_runs:
                print("Experiment already existed and had more target runs, ignoring")
            else:
                print("Experiment already existed, increasing target runs")
                self.experiments.loc[index]['target_runs'] = target_runs
        elif len(indices) == 0:
            self._create_new_experiment(param_dict, target_runs)
            print("Created experiment")
        pass

    def print_all(self):
        results_table_rows = [[self.results[pointer]["accuracy"].mean(), self.results[pointer]["time_attack"].mean(), len(self.results[pointer].index)]
                              for pointer in self.experiments['res_pointer']]
        results_table = pd.DataFrame(np.array(results_table_rows), columns=['acc', 'time_att', 'runs'])
        print(pd.concat([self.experiments, results_table], axis=1).to_string())

    def print_results_given_indices(self, indices):
        smaller_df = self.experiments.loc[indices].copy()
        results_table_rows = [
            [self.results[pointer]["accuracy"].mean(), self.results[pointer]["time_attack"].mean(), len(self.results[pointer].index)]
            for pointer in smaller_df['res_pointer']]
        results_df = pd.DataFrame(np.array(results_table_rows), index=indices, columns=['acc', 'time_att', 'runs'])
        concat_df = pd.concat([smaller_df, results_df], axis=1)
        print(concat_df.to_string())

    def print_given_dict(self, exp_dict):
        print(exp_dict)
        indices = self._find_indices(exp_dict)
        self.print_results_given_indices(indices)

    def print_pending_experiments(self):
        unfinished_indices = []
        for index, row in self.experiments.iterrows():
            pointer = row['res_pointer']
            if len(self.results[pointer]["accuracy"]) < row['target_runs']:
                unfinished_indices.append(index)
        if len(unfinished_indices) > 1:
            self.print_results_given_indices(unfinished_indices)
        else:
            print("Nothing unfinished!")

    def print_results_table_given_index(self, index):
        pointer = self.experiments.loc[index]['res_pointer']
        print(self.results[pointer])

    def write_pending_experiments_request(self, experiments_path):
        for index, row in self.experiments.iterrows():
            seeds_to_run = []
            pointer = row['res_pointer']
            for seed in range(row['target_runs']):
                if seed not in self.results[pointer]['seed'].values:
                    seeds_to_run.append(seed)
            if len(seeds_to_run) > 0:
                with open(os.path.join(experiments_path, 'todo_{:04d}.pkl'.format(index)), 'wb') as f:
                    exp_dict = row.to_dict()
                    del exp_dict['target_runs']
                    del exp_dict['res_pointer']
                    pickle.dump((exp_dict, seeds_to_run), f)
                    print("Created todo_{:04d}.pkl".format(index))

    def write_pending_experiments_request_range(self, experiments_path, minidx, maxidx):
        for index, row in self.experiments.iterrows():
            if index >= minidx and index <= maxidx:
                seeds_to_run = []
                pointer = row['res_pointer']
                for seed in range(row['target_runs']):
                    if seed not in self.results[pointer]['seed'].values:
                        seeds_to_run.append(seed)
                if len(seeds_to_run) > 0:
                    with open(os.path.join(experiments_path, 'todo_{:04d}.pkl'.format(index)), 'wb') as f:
                        exp_dict = row.to_dict()
                        del exp_dict['target_runs']
                        del exp_dict['res_pointer']
                        pickle.dump((exp_dict, seeds_to_run), f)
                        print("Created todo_{:04d}.pkl".format(index))

    def eat_pickles(self, experiments_path):
        """eats pickles in given directory and all subdirectories"""
        count = 0
        subdirectories = os.scandir(experiments_path)
        for subdir in subdirectories:
            if subdir.is_dir():
                for file in os.scandir(subdir):
                    if file.name.startswith('results') and file.name.endswith('.pkl'):
                        with open(file, 'rb') as f:
                            exp_dict, res_dict = pickle.load(f)
                            self._add_results(exp_dict, res_dict)
                        os.remove(file)
                        count += 1
                try:
                    os.rmdir(subdir)  # Can only delete it if it's empty
                except OSError:
                    print("Dir not empty, not removing")
        if count > 0:
            print("Yum x{:d}".format(count))
        else:
            print("Nothing to eat")
        return count

    def reset_results(self, exp_dict):
        print(exp_dict)
        indices = self._find_indices(exp_dict)
        response = input("We found {:d} indices, type reset to continue!".format(len(indices)))
        if response == 'reset':
            self.reset_results_given_indices(indices)
        else:
            print("Aborting")

    def reset_results_given_indices(self, indices):
        for index in indices:
            pointer = self.experiments.loc[index]['res_pointer']
            self.results[pointer] = pd.DataFrame(columns=['seed', 'accuracy', 'time_attack', 'bw_overhead'])
            print('reseted {:d}'.format(index))

    def remove_experiments_between_indices(self, start, end):
        # self.print_pending_experiments()
        for index in range(start, end+1):
            pointer = self.experiments.loc[index]['res_pointer']
            del self.results[pointer]
            self.experiments.drop(index=index, inplace=True)
        print("Deleted!")


if __name__ == "__main__":

    EXPERIMENTS_PATH = 'results'
    manager_data_filename = 'manager_df_data.pkl'

    if not os.path.exists(EXPERIMENTS_PATH):
        os.makedirs(EXPERIMENTS_PATH)

    if not os.path.exists(manager_data_filename):
        manager = ManagerDf()
    else:
        with open(manager_data_filename, 'rb') as f:
            manager = pickle.load(f)

    while True:
        print_console_options()
        print(exp_params_column)
        choice = input("Enter your option: ").lower().split(' ')
        if choice[0] in ('e',):
            print("Saving ResultsManager...")
            with open(manager_data_filename, 'wb') as f:
                pickle.dump(manager, f)
            break
        elif choice[0] == 'p':
            exp_dict = {}
            for vals in choice[1:]:
                if len(vals.split('=')) == 2:
                    key, val = vals.split('=')
                    if key in exp_params_column:
                        exp_dict[key] = eval(val)

            manager.print_given_dict(exp_dict)
        elif choice[0] in ('pa',):
            manager.print_all()
        elif choice[0] in ('w',):
            if len(choice) > 1:
                manager.write_pending_experiments_request_range(EXPERIMENTS_PATH, int(choice[1]), int(choice[2]))
            else:
                manager.write_pending_experiments_request(EXPERIMENTS_PATH)
        elif choice[0] in ('eat',):
            count = manager.eat_pickles(EXPERIMENTS_PATH)
            if count > 0:
                with open(manager_data_filename, 'wb') as f:
                    pickle.dump(manager, f)
                print('Saved manager')
        elif choice[0] in ('pr',):
            manager.print_pending_experiments()
        elif choice[0] in ('reset',):
            if len(choice) == 1:
                print("must provide key-val pairs")
            else:
                exp_dict = {}
                for vals in choice[1:]:
                    if len(vals.split('=')) == 2:
                        key, val = vals.split('=')
                        if key in exp_params_column:
                            exp_dict[key] = eval(val)
                manager.reset_results(exp_dict)
        elif choice[0] in ('pp',):
            if len(choice) == 2:
                index = int(choice[1])
                manager.print_results_table_given_index(index)
        elif choice[0] in ('remove',):
            start = int(choice[1])
            end = int(choice[2])
            input("Going to REMOVE experiments from index {:d} to {:d}, type 'remove' to proceed: ".format(start, end))
            manager.remove_experiments_between_indices(start, end)
        else:
            print("Unrecognized command")

    print("Bye")
