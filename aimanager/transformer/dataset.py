import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, df, input_columns, target_column, query_columns, pad_idx=-1):
        sequences, targets, queries = self.create_sequences(df, input_columns, target_column, query_columns, pad_idx)

        self.data = sequences[:, :-1]
        self.targets = targets[:, 1:]
        self.queries = queries[:, 1:]
        self.mask = torch.ones_like(self.targets)
        self.contribution_mask = torch.ones_like(self.targets)
        self.punishment_mask = torch.ones_like(self.targets)
        self.contribution_mask[:, 1::2] = 0
        self.punishment_mask[:, ::2] = 0
        self.mask[self.targets == pad_idx] = 0
        self.contribution_mask[self.targets == pad_idx] = 0
        self.punishment_mask[self.targets == pad_idx] = 0

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        q = self.queries[index]  # Query tensor
        mask = self.mask[index]
        contribution_mask = self.contribution_mask[index]
        punishment_mask = self.punishment_mask[index]
        return x, y, q, mask, contribution_mask, punishment_mask

    def __len__(self):
        return len(self.data)

    def create_sequences(self, df, input_columns, target_column, query_columns, pad_idx=-1):
        sequences = []
        targets = []
        queries = []

        for perm_group in df['perm_group'].unique():
            perm_group_data = df[df['perm_group'] == perm_group]

            for episode_id in perm_group_data['episode_id'].unique():
                episode_data = perm_group_data[perm_group_data['episode_id'] == episode_id]

                sequence = torch.tensor(episode_data[input_columns].astype('float64').values, dtype=torch.float32)
                target = torch.tensor(episode_data[target_column].values, dtype=torch.int64)
                query = torch.tensor(episode_data[query_columns].astype('float64').values, dtype=torch.float32)

                # Add a padding of zeros at the start of each sequence
                sequence_start = torch.zeros((1, sequence.shape[1]), dtype=torch.float32)
                sequence = torch.cat((sequence_start, sequence), dim=0)
                target_start = torch.full((1,), pad_idx, dtype=torch.int64)
                target = torch.cat((target_start, target), dim=0)
                query_start = torch.zeros((1, query.shape[1]), dtype=torch.float32)
                query = torch.cat((query_start, query), dim=0)

                sequences.append(sequence)
                targets.append(target)
                queries.append(query)

        # Pad the sequences so they all have the same length
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        padded_queries = pad_sequence(queries, batch_first=True, padding_value=0)

        # Add zero

        return padded_sequences, padded_targets, padded_queries
