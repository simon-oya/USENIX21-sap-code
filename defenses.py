import numpy as np


class Defense:

    def __init__(self, client_data, def_name, def_params):
        """
        :param client_data: dictionary with keys 'dataset', 'keywords', 'frequencies' (trend matrix)
        :param defense_parameters:
        """

        self.dataset = client_data['dataset']
        self.n_docs = len(client_data['dataset'])
        self.keywords = client_data['keywords']
        self.frequencies = client_data['frequencies']
        self.def_name = def_name
        self.def_params = def_params

        return

    def generate_query_traces(self, kw_id_traces):
        """
        Generates the query traces, that contain weekly lists of queries, and each query is a list of document ids
        :return traces: List of weekly_traces, which are lists of traces, which are lists of document ids
        :return bw_overhead: Documents received divided by the actual number of documents
        """
        traces = []  # List of weekly_traces, which are lists of traces, which are lists of document ids
        inverted_index = {}
        for kw_id in range(len(self.keywords)):
            inverted_index[kw_id] = [doc_id for doc_id, doc_kws in enumerate(self.dataset) if self.keywords[kw_id] in doc_kws]

        if self.def_name == 'none':
            for weekly_kw_trace in kw_id_traces:
                weekly_access_patterns = []
                for kw_id in weekly_kw_trace:
                    weekly_access_patterns.append(inverted_index[kw_id])
                traces.append(weekly_access_patterns)

            bw_overhead = 1

        elif self.def_name.lower() == 'clrz':

            tpr, fpr = self.def_params
            obf_inverted_index = {}
            for kw_id in range(len(self.keywords)):
                coin_flips = np.random.rand(len(self.dataset))
                obf_inverted_index[kw_id] = [doc_id for doc_id, doc_kws in enumerate(self.dataset) if
                                             (self.keywords[kw_id] in doc_kws and coin_flips[doc_id] < tpr) or
                                             (self.keywords[kw_id] not in doc_kws and coin_flips[doc_id] < fpr)]

            ndocs_retrieved = 0
            ndocs_real = 0
            for weekly_kw_trace in kw_id_traces:
                weekly_access_patterns = []
                for kw_id in weekly_kw_trace:
                    weekly_access_patterns.append(obf_inverted_index[kw_id])
                    ndocs_retrieved += len(obf_inverted_index[kw_id])
                    ndocs_real += len(inverted_index[kw_id])
                traces.append(weekly_access_patterns)

            bw_overhead = ndocs_retrieved / ndocs_real

        elif self.def_name.lower() == 'ppyy':
            # The traces are just tags and volume pairs. That's the only leakage
            epsilon = self.def_params[0]
            laplacian_decay = 2 / epsilon
            laplacian_constant = 2 / epsilon * (64 * np.log(2) + np.log(len(self.keywords)))
            map_kw_id_to_tag_and_volume = {}
            ndocs_retrieved = 0
            ndocs_real = 0
            count = 0
            for kw_id in list(set([kw_id for weekly_kw_trace in kw_id_traces for kw_id in weekly_kw_trace])):
                true_volume = len([doc_id for doc_id, doc_kws in enumerate(self.dataset) if self.keywords[kw_id] in doc_kws])
                dp_volume = int(np.min((true_volume + np.ceil(np.random.laplace(laplacian_constant, laplacian_decay)), len(self.dataset))))
                map_kw_id_to_tag_and_volume[kw_id] = (count, dp_volume)
                count += 1
                ndocs_retrieved += dp_volume
                ndocs_real += true_volume

                # print("For '{:s}': {:d}->{:d}".format(self.keywords[kw_id], true_volume, dp_volume))
            traces = [[map_kw_id_to_tag_and_volume[kw_id] for kw_id in weekly_trace] for weekly_trace in kw_id_traces]

            bw_overhead = ndocs_retrieved / ndocs_real

        elif self.def_name.lower() == 'sealvol':
            x = self.def_params[0]

            map_kw_id_to_tag_and_volume = {}
            ndocs_retrieved = 0
            ndocs_real = 0
            count = 0
            for kw_id in list(set([kw_id for weekly_kw_trace in kw_id_traces for kw_id in weekly_kw_trace])):
                true_volume = len([doc_id for doc_id, doc_kws in enumerate(self.dataset) if self.keywords[kw_id] in doc_kws])
                obf_volume = x ** int(np.ceil(np.log(true_volume) / np.log(x)))
                map_kw_id_to_tag_and_volume[kw_id] = (count, obf_volume)
                # print("{:d} -> {:d}".format(true_volume, obf_volume))
                count += 1
                ndocs_retrieved += obf_volume
                ndocs_real += true_volume

            traces = [[map_kw_id_to_tag_and_volume[kw_id] for kw_id in weekly_trace] for weekly_trace in kw_id_traces]

            bw_overhead = ndocs_retrieved / ndocs_real

        else:
            raise ValueError("Unrecognized defense name: {:s}".format(self.def_name))

        return traces, bw_overhead

    def get_dataset_size_for_adversary(self):
        if self.def_name in ('none', 'clrz', 'osse', 'ppyy'):
            return len(self.dataset)
        elif self.def_name in ('sealvol',):
            x = self.def_params[0]
            return len(self.dataset) * x
        else:
            raise ValueError("Unrecognized defense name: {:s}".format(self.def_name))
