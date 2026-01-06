import os.path
import pickle
import numpy as np
import copy


def load_accuracies(all_paths, n_runs=5, n_epochs=300, val_steps=10, context_unaware=False):
    """ loads all accuracies into a dictionary, val_steps should be set to the same as val_frequency during training
    """
    result_dict = {'train_acc': [], 'val_acc': [], 'test_acc': [],
                   'train_message_lengths': [], 'val_message_lengths': [],
                   'cu_train_acc': [], 'cu_val_acc': [], 'cu_test_acc': [],
                   'cu_train_message_lengths': [], 'cu_val_message_lengths': [],
                  }

    for path_idx, path in enumerate(all_paths):

        train_accs = []
        val_accs = []
        test_accs = []
        train_message_lengths = []
        val_message_lengths = []
        cu_train_accs = []
        cu_val_accs = []
        cu_test_accs = []
        cu_train_message_lengths = []
        cu_val_message_lengths = []

        # prepare paths
        context_aware_path = "context_aware"
        context_unaware_path = "context_unaware"

        file_name = "loss_and_metrics"
        file_extension = "pkl"

        for run in range(n_runs):

            run_path = str(run)

            # context-aware (standard)
            if not context_unaware:
                file_path = f"{path}/{context_aware_path}/{run_path}/{file_name}.{file_extension}"
                data = pickle.load(open(file_path, 'rb'))
                # train and validation accuracy
                lists = sorted(data['metrics_train0'].items())
                _, train_acc = zip(*lists)
                train_accs.append(train_acc)
                lists = sorted(data['metrics_test0'].items())
                _, val_acc = zip(*lists)
                if (len(val_acc) > n_epochs // val_steps):  # old: we had some runs where we set val freq to 5 instead of 10
                    val_acc = val_acc[::2]
                val_accs.append(val_acc)
                test_accs.append(data['final_test_acc'])
                # message lengths
                lists = sorted(data['metrics_train1'].items())
                _, train_message_length = zip(*lists)
                lists = sorted(data['metrics_test1'].items())
                _, val_message_length = zip(*lists)
                train_message_lengths.append(train_message_length)
                val_message_lengths.append(val_message_length)

            # context-unaware
            elif context_unaware:
                file_path = f"{path}/{context_unaware_path}/{run_path}/{file_name}.{file_extension}"
                cu_data = pickle.load(open(file_path, 'rb'))
                # accuracies
                lists = sorted(cu_data['metrics_train0'].items())
                _, cu_train_acc = zip(*lists)
                if (len(cu_train_acc) != n_epochs):
                    print(path, run, len(cu_train_acc))
                    raise ValueError(
                        "The stored results don't match the parameters given to this function. "
                        "Check the number of epochs in the above mentioned runs.")
                cu_train_accs.append(cu_train_acc)
                lists = sorted(cu_data['metrics_test0'].items())
                _, cu_val_acc = zip(*lists)
                # for troubleshooting in case the stored results don't match the parameters given to this function
                if (len(cu_val_acc) != n_epochs // val_steps):
                    print(context_unaware_path, len(cu_val_acc))
                    raise ValueError(
                        "The stored results don't match the parameters given to this function. "
                        "Check the above mentioned files for number of epochs and validation steps.")
                if (len(cu_val_acc) > n_epochs // val_steps):
                    cu_val_acc = cu_val_acc[::2]
                cu_val_accs.append(cu_val_acc)
                cu_test_accs.append(cu_data['final_test_acc'])
                # message lengths
                lists = sorted(cu_data['metrics_train1'].items())
                _, cu_train_message_length = zip(*lists)
                lists = sorted(cu_data['metrics_test1'].items())
                _, cu_val_message_length = zip(*lists)
                cu_train_message_lengths.append(cu_train_message_length)
                cu_val_message_lengths.append(cu_val_message_length)

        if not context_unaware:
            result_dict['train_acc'].append(train_accs)
            result_dict['val_acc'].append(val_accs)
            result_dict['test_acc'].append(test_accs)
            result_dict['train_message_lengths'].append(train_message_lengths)
            result_dict['val_message_lengths'].append(val_message_lengths)
        elif context_unaware:
            result_dict['cu_train_acc'].append(cu_train_accs)
            result_dict['cu_val_acc'].append(cu_val_accs)
            result_dict['cu_test_acc'].append(cu_test_accs)
            result_dict['cu_train_message_lengths'].append(cu_train_message_lengths)
            result_dict['cu_val_message_lengths'].append(cu_val_message_lengths)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict

def load_entropies(all_paths, n_runs=5, context_unaware=False, verbose=False):
    """ loads all entropy scores into a dictionary"""

    if context_unaware:
           setting = 'context_unaware'
    else:
        setting = 'context_aware'

    result_dict = {'NMI': [], 'effectiveness': [], 'consistency': [],
                   'NMI_context_dep': [], 'effectiveness_context_dep': [], 'consistency_context_dep': [],
                   'NMI_concept_x_context': [], 'effectiveness_concept_x_context': [],
                   'consistency_concept_x_context': []}

    for path_idx, path in enumerate(all_paths):

        NMIs, effectiveness_scores, consistency_scores = [], [], []
        NMIs_context_dep, effectiveness_scores_context_dep, consistency_scores_context_dep = [], [], []
        NMIs_conc_x_cont, effectiveness_conc_x_cont, consistency_conc_x_cont = [], [], []


        for run in range(n_runs):
            standard_path = path + '/' + setting + '/' + str(run) + '/'
            data = pickle.load(open(standard_path + 'entropy_scores' + '.pkl', 'rb'))
            if verbose:
                print("Entropy scores loaded from:", standard_path + 'entropy_scores' + '.pkl')
            NMIs.append(data['normalized_mutual_info'])
            effectiveness_scores.append(data['effectiveness'])
            consistency_scores.append(data['consistency'])
            NMIs_context_dep.append(data['normalized_mutual_info_context_dep'])
            effectiveness_scores_context_dep.append(data['effectiveness_context_dep'])
            consistency_scores_context_dep.append(data['consistency_context_dep'])
            NMIs_conc_x_cont.append(data['normalized_mutual_info_concept_x_context'])
            effectiveness_conc_x_cont.append(data['effectiveness_concept_x_context'])
            consistency_conc_x_cont.append(data['consistency_concept_x_context'])

        result_dict['NMI'].append(NMIs)
        result_dict['consistency'].append(consistency_scores)
        result_dict['effectiveness'].append(effectiveness_scores)
        result_dict['NMI_context_dep'].append(NMIs_context_dep)
        result_dict['consistency_context_dep'].append(consistency_scores_context_dep)
        result_dict['effectiveness_context_dep'].append(effectiveness_scores_context_dep)
        result_dict['NMI_concept_x_context'].append(NMIs_conc_x_cont)
        result_dict['consistency_concept_x_context'].append(consistency_conc_x_cont)
        result_dict['effectiveness_concept_x_context'].append(effectiveness_conc_x_cont)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict
