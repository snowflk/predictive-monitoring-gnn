from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as F
import torch as T
import pandas as pd
import numpy as np
import os
import json
import random


# BPIC12_URL = 'https://www.win.tue.nl/bpi/lib/exe/fetch.php?media=2012:financial_log.xes.gz'
# BPIC12_URL = 'https://data.4tu.nl/ndownloader/files/24027287'


def events_to_graph(df, num_classes):
    """
    Create features for the events
    :param event: a DataFrame containing N input events
    :return:
        type node features: a N x D matrix,
        attr node featutes:
        edge indices: 2 x Num edges matrix

    """
    days_of_week = T.tensor(df['DayOfWeek'].to_numpy()).long()

    event_onehots = F.one_hot(T.tensor([i for i in range(num_classes + 2)]).long(), num_classes=num_classes + 2)
    dow_onehots = F.one_hot(days_of_week, num_classes=7)

    event_to_id = {}
    edge_exists = {}
    edge_index = []
    type_node_features = []
    attr_node_features = []
    # First event
    first_event_type = df.iloc[0]['ActivityNum']
    event_to_id[first_event_type] = 0
    type_node_features.append(event_onehots[first_event_type])
    curr_id = 1
    # Create edges of flow type
    for i in range(1, len(df)):
        event_type = df.iloc[i]['ActivityNum']
        prev_event_type = df.iloc[i - 1]['ActivityNum']
        if event_type not in event_to_id:
            event_to_id[event_type] = curr_id
            type_node_features.append(event_onehots[event_type])
            curr_id += 1
        edge_key = f"{prev_event_type}-{event_type}"
        backward_edge_key = f"{event_type}-{prev_event_type}"
        if (edge_key not in edge_exists) or (backward_edge_key not in edge_key):
            edge_index.append([event_to_id[prev_event_type], event_to_id[event_type]])
            edge_index.append([event_to_id[event_type], event_to_id[prev_event_type]])
            edge_exists[edge_key] = True
            edge_exists[backward_edge_key] = True

    # Create start/end node
    event_to_id[num_classes] = curr_id
    event_to_id[num_classes + 1] = curr_id + 1
    type_node_features.append(event_onehots[num_classes])
    type_node_features.append(event_onehots[num_classes + 1])
    edge_index.append([event_to_id[first_event_type], event_to_id[num_classes]])
    edge_index.append([event_to_id[num_classes], event_to_id[first_event_type]])
    last_event_type = df.iloc[-1]['ActivityNum']
    edge_index.append([event_to_id[last_event_type], event_to_id[num_classes + 1]])
    edge_index.append([event_to_id[num_classes + 1], event_to_id[last_event_type]])
    curr_id += 2

    # Create edges of attr type
    for i in range(len(df)):
        event_type = df.iloc[i]['ActivityNum']
        type_node_idx = event_to_id[event_type]
        edge_index.append([curr_id, type_node_idx])
        edge_index.append([type_node_idx, curr_id])
        curr_id += 1
        attr_node_features.append(
            T.cat([
                T.tensor([df['Duration'].iloc[i]]),
                dow_onehots[i]], dim=0)
        )

    return T.stack(type_node_features), \
           T.stack(attr_node_features), \
           T.tensor(edge_index).long().t().contiguous()


class MixedData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class BPIC12Dataset(InMemoryDataset):
    def __init__(self, root, filename='bpi12_w.csv', train_split=0.7, train_mode=True):
        self.filename = filename
        self.filename_no_ext = os.path.splitext(filename)[0]
        self.train_split = train_split
        self.train_mode = train_mode
        self.mode = 'train' if train_mode is True else 'val'
        super(BPIC12Dataset, self).__init__(root=root, pre_filter=None, pre_transform=None)
        self.data, self.slices = T.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.filename_no_ext}_{self.mode}_processed.pt']

    def process(self):
        # Read the event log
        df = pd.read_csv(os.path.join(self.root, self.filename))
        df['ActivityNum'] = -1
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df['CompleteTime'] = pd.to_datetime(df['CompleteTime'])
        df = df.assign(ActivityNum=df['ActivityID'].astype('category').cat.codes)
        # Store categorical ID to json
        activity_num_map = {}
        for aid in df['ActivityNum'].unique():
            name = df[df['ActivityNum'] == aid].iloc[0]['ActivityName']
            activity_num_map[int(aid)] = name
        with open(os.path.join(self.processed_dir, f'{self.filename_no_ext}_activity_nummap.json'), 'w') as f:
            json.dump(activity_num_map, f)

        # Make dataset
        case_indices = df['CaseID'].unique()
        num_classes = df['ActivityNum'].nunique()

        data_list = []

        # Normalize
        df['Duration'] /= 60  # normalize to minutes
        df['AmountReq'] /= df['AmountReq'].max()  # simple normalization
        df['DayOfWeek'] = df['CompleteTime'].dt.dayofweek

        random.shuffle(case_indices)
        if self.train_mode:
            n_cases = int(self.train_split * len(case_indices))
            case_indices = case_indices[:n_cases]
        else:
            n_cases = len(case_indices) - int(self.train_split * len(case_indices))
            case_indices = case_indices[-n_cases:]

        for idx, case_id in enumerate(case_indices):
            print(f"Processing {idx+1}/{len(case_indices)}")
            df_trace_full = df[df['CaseID'] == case_id]
            if len(df_trace_full) < 3:
                continue
            for j in range(3, len(df_trace_full)):
                df_trace = df_trace_full.iloc[:j]
                x_events = df_trace.iloc[:-1]
                y_class = df_trace.iloc[-1]['ActivityNum']
                y_onehot = F.one_hot(T.tensor([y_class]).long(), num_classes=num_classes)
                y_time = np.busday_count(df_trace.iloc[-2]['CompleteTime'].date(),
                                         df_trace.iloc[-1]['StartTime'].date())
                type_nodes, attr_nodes, edge_index = events_to_graph(x_events, num_classes)

                global_features = T.tensor([
                    df_trace.iloc[0]['AmountReq'],
                    np.busday_count(df_trace.iloc[0]['StartTime'].date(), df_trace.iloc[-1]['CompleteTime'].date())
                ])
                entry = MixedData(edge_index=edge_index,
                                  type_nodes=type_nodes,
                                  attr_nodes=attr_nodes,
                                  n_type_nodes=type_nodes.shape[0],
                                  n_attr_nodes=attr_nodes.shape[0],
                                  global_features=global_features,
                                  num_nodes=type_nodes.shape[0] + attr_nodes.shape[0],
                                  y=y_onehot,
                                  y_time=T.tensor(y_time))

                data_list.append(entry)
        data, slices = self.collate(data_list)
        T.save((data, slices), self.processed_paths[0])


BPIC12Dataset('./data')
