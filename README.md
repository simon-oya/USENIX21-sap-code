# README
This document explains the basic structure of our code and the steps that need to be followed to reproduce our results.
Each realization of our experiments is initialized with a seed (from 0 to 29) so that running this code should generate **exactly** the plots in our paper
(except for the running times). 

We are working on getting a new version of our code with documented functions and classes, and a more thorough description of the code functionalities.
Current docstrings in the code might refer to old versions of the functions/classes.
Note that the original code evaluated other attack variations, query distributions, and theoretical defenses. Many design decisions that were taken in the initial design of the code are suboptimal in this version.

## Summary of files

Our code consists of two main entities: a results manager, that stores experiment information and results, and the experiment runner, that produces the results.
Our basic workflow consists of
1) adding experiments to the manager 
2) telling the manager to generate ``todo`` files with the experiment information
3) run the experiments specified in the ``todo`` files
4) load the results with the manager

The basic files in our code are:
* ``manager_df.py``: implements the ManagerDf class that stores the parameters of each experiment 
(each experiment is a pandas DataFrame row) and the experiment results
(the results of each experiment are stored in an independent DataFrame).
* ``add_experiments_to_manager.py``: loads the manager with the experiments that we run in the paper.
* ``experiment.py``: provides the function ``run_single_experiment`` that runs an experiment and saves the results in a pickle file.
* ``attacks.py``: implements the different attacks we evaluate in our paper.
Each attack is an instantation of an Attack class (this object-oriented approach made sense in early stages of our research,
 where we could run many attacks on the same class instance to save computational time.
 In the current approach, ``run_single_experiment`` only runs a single attack per run, so
 the object-oriented approach feels a bit weird).
* ``defenses.py``: implements the defenses we consider in the paper.
The only goal of this class is to generate the adversary observations given the sequence of real keywords.
* ``utils.py``: provides access to different functions that we use in the code (e.g., to print matrices, write the configuration file of the graphm attack, or compute probabilities of observed query volumes given auxiliary information).
* ``run_pending_experiments.py``: reads ``todo_X.pkl`` files and runs the experiments. 
It runs many instances of the ``run_single_experiment`` function from ``experiments.py`` in parallel within each experiment, with a different seed each time.
Different experiments are run sequentially.

### The manager:
The ManagerDf class has two attributes: ``self.experiments`` and ``self.results``.

1) ``self.experiments`` is a pandas DataFrame (a table) where each column represents an experiment attribute,
and each row is an experiment that we want to run. The columns are the following:

    * ``'dataset'``: dataset name, it can be ``'enron_db'``, or ``'lucene_db'``.
    * ``'nkw'``: number of keywords to take.
    * ``'tt_mode'``: train/test separation mode. For the experiments in the paper, we use ``'split_rand_past5-50'``.
    This means that we split the dataset into training and testing sets, we select keywords randomly, the frequency information
    for the adversary has an offset of 5 weeks, and the observation time is 50 weeks.
    * ``'query_number_dist'``: for the experiments in the paper, we use ``poiss`` for Poisson query generation.
    * ``'query_params'``: tuple with query generation parameters. For ``poiss``, this is just the average number
    of queries per week, e.g., ``(nqr_per_week,)``.
    * ``'def_name'``: for the paper, it can be ``'none'``, ``'clrz'``, ``'ppyy'``, or ``'sealvol'``.
    * ``'def_params'``: tuple with the defense parameters. For the defenses above, the tuples are
    respectively ``()``, ``(TPR, FPR)``, ``(epsilon,)``, ``(x,)``.
    * ``'att_alg''``: algorithm used to solve the attack. For the paper, ``'hung'`` (Hungarian algorithm), ``'graphm'`` (graph matching) or ``'unco'`` (unconstrained 
    assignment, for the frequency attack by Liu et al.).
    * ``'att_name'``: name of the attack that we run. In the paper, we use a single attack per algorithm,
    but in previous versions of the code we had different variations of the attacks.
    For the paper, the attack names for the algorithms above are ``'ccl'``, ``'pw'``, and ``'liu'``.
    * ``'att_params'``: parameters of the attack. This is an empty tuple for ``'liu'``, and
    otherwise it is the tuple ``(alpha, naive)`` where ``alpha`` is the alpha of the convex
    combination in the objective function of the attack, and ``naive`` is a boolean that is
    ``True`` when we want to evaluate the naive attack that is unaware of the defense,
    and ``False`` otherwise.
    * ``'target_runs'``: number of repetitions of a experiment that we want to run (30 in the paper).
    * ``'res_pointer'``: integer pointing at where the results of the experiment will be stored.
    
2) ``self.results`` is a dictionary that maps the previous ``res_pointer`` values to
    dataframes that store the results. These dataframes contain one row per run and have the columns:
     * ``'seed'``: random seed of this run, there cannot be
    repeated seeds.
     * ``'accuracy'``: query recovery accuracy of the attack.
     * ``'time_attack'``: time of running the attack (only solving the attack problem, not query
     generation, initialization, etc.)
     * ``'bw_overhead'``: bandwidth overhead of this run
     
 
### Datasets:
The processed datasets are under the ``datasets_pro`` folder.
Each dataset is a pickle file, that contains two variables: ``dataset, keyword_dict`` 
(for more information on how to load them, check ``load_pro_dataset`` in ``experiment.py``).
1) ``dataset``: is a list (dataset) of lists (documents). Each document is a list of the keywords (strings) associated to that document.
2) ``keyword_dict``: is a dictionary whose keys are the keywords (strings). The values are dictionaries with two keys:
 
    a) ``'count'``, that maps to the number of times that keyword appears in the dataset
 (overall popularity, just used to select the 3000 most popular keywords) and
 
    b) ``'trend'``, that maps to a numpy array with the trends of this keyword (one value per week, 260 weeks computed from Google Trends).
 
### Trace structure:
The real keywords of the user are generated in ``experiment.py`` using the ``generate_keyword_queries`` function.
The trace is a list (trace) of lists (weekly queries). The weekly queries are lists of *integers* (an integer *i* represents the *i*th most popular keyword in the dataset).
The defense receives this trace of real keywords and generates the observation trace.

1) When the defense name is ``'none'`` or ``'clrz'``, the observation trace is a list (trace) of lists (weekly observations) of lists (id's of the documents returned in this query).
Our attack only uses the length of this final list (the response volume), but graphm uses the whole access pattern to compute co-occurrence information.
(We implemented graphm against CLRZ as well, even though we do not show the results in the paper since this attack provides less accuracy than ours.)

2) When the defense name is ``'ppyy'`` or ``'sealvol'``, the observation trace is a list (trace) of lists (weekly observations) of tuples. The tuples are simply ``(tag_id, volume)``.
The adversary uses these tuples to compute the observed frequency and volume information. We do not care about the access pattern structure, since our attack only uses the frequency and volume.




## Quick instructions to generate the paper results
Make sure that you compile the graphm binary, more info here: <http://projects.cbio.mines-paristech.fr/graphm/>.
The path to this binary should be assigned to the ``GRAPHM_PATH`` variable in ``attacks.py``.

1) Run ``add_experiments_to_manager.py``.
This creates a ``manager_df_data.pkl`` file with a ManagerDf object and adds the experiments that we run in the paper to the manager.

2) Run ``manager_df.py`` to open the manager.
(Optional: Type ``pr`` and press return in the console to visualize the dataframe with all the pending experiments in the manger.)
Type ``w`` and press return to create ``todo_X.pkl`` files that contain the experiments to run.
Type ``e`` and press return to exit the manager.

3) Run ``run_pending_experiments.py``.
This scrip reads the ``todo_X.pkl`` files, performs the experiments (sequentially for each file, in parallel within each file), and saves result files ``results_X.temp``.
Note that this **might take long**. One can run less experiments by adding less experiments to the manager in step 1.
It is also possible to run the other steps while experiments are running.
(This process keeps saving results as they are done. If it dies before it finishes, run step 4, then step 2, and then 3 again.)

4) Run ``manager_df.py`` and type ``eat`` in the console to read and delete the result files.
The results will be loaded in the manager.
The full experiment table can be shown by typing ``pa`` (print all) in the console.
Close the manager typing ``e`` in the console.

5) Run the plot scripts to plot the results.
