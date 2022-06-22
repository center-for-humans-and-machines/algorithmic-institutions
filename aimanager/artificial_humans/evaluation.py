
import pandas as pd
import torch as th
import os
from aimanager.artificial_humans.metrics import create_metrics, create_confusion_matrix, calc_log_loss
from aimanager.utils.array_to_df import add_labels, using_multiindex
from aimanager.utils.utils import make_dir
from torch_geometric.data import Batch


class Evaluator:
    def __init__(self):
        self.metrics = []
        self.confusion_matrix = []
        self.synthetic_predicitions = []

    # def set_data(self, train, test, syn, syn_df):
    #     self.data = {
    #         'train': train,
    #         'test': test,
    #         'syn': syn,
    #         'syn_df': syn_df
    #     }

    def set_labels(self, **labels):
        self.labels = labels

    def eval_set(self, model, data, calc_confusion=True, **add_labels):
        y_pred, y_pred_proba = model.predict(data)

        data = Batch.from_data_list(data)

        mask = data['mask']
        y_true = th.masked_select(data['y'], mask)
        y_true = y_true.detach().cpu().numpy()
        n_levels = y_pred_proba.shape[-1]

        y_pred_proba_ = th.masked_select(y_pred_proba, mask.unsqueeze(-1))
        y_pred_proba_ = y_pred_proba_.reshape(-1, n_levels)

        y_pred_proba_ = y_pred_proba_.detach().cpu().numpy()

        self.metrics.append(calc_log_loss(
            y_true, y_pred_proba_, n_levels=n_levels, **add_labels, **self.labels))

        if calc_confusion:
            self.confusion_matrix.append(create_confusion_matrix(
                data['y'], y_pred_proba, mask, **add_labels, **self.labels))

        strategies = ['greedy', 'sampling']
        for strategy in strategies:
            if strategy == 'greedy':
                pass
            elif strategy == 'sampling':
                shape = y_pred_proba.shape
                y_pred = th.multinomial(y_pred_proba.reshape(-1, shape[-1]), 1)
                y_pred = y_pred.squeeze(-1).reshape(shape[:-1])
            else:
                raise ValueError(f"Unknown strategy {strategy}")

            y_pred = th.masked_select(y_pred, mask)
            y_pred = y_pred.detach().cpu().numpy()

            self.metrics += create_metrics(y_true, y_pred, strategy=strategy, **add_labels, **self.labels)


    def eval_syn(self, model, data, data_df):
        y_pred, y_pred_proba = model.predict(data)
        y_pred = y_pred.detach().cpu().numpy()
        y_pred_proba = y_pred_proba.detach().cpu().numpy()
        proba_df = using_multiindex(
            y_pred_proba, ['idx', 'round_number', 'contribution']).rename(columns={'value': 'proba'})
        proba_df['exp_contribution'] = proba_df['contribution'] * proba_df['proba']
        exp_con_df = proba_df.groupby(['idx', 'round_number'])['exp_contribution'].sum().reset_index()
        # pred_df = using_multiindex(
        #     y_pred, ['idx', 'round_number']).rename(columns={'value': 'pred_contribution'})

        exp_con_df = data_df.merge(exp_con_df)
        exp_con_df = add_labels(exp_con_df, self.labels)
        self.synthetic_predicitions.append(exp_con_df)

    def add_loss(self, loss):
        self.metrics.append(dict(name='loss', value=loss, **self.labels))

    def save(self, output_path, labels):
        make_dir(output_path)
        self._save_metric(self.metrics, 'metrics.parquet', output_path, labels)
        self._save_metric(pd.concat(self.confusion_matrix), 'confusion_matrix.parquet', output_path, labels)
        self._save_metric(pd.concat(self.synthetic_predicitions), 'synthetic_predicitions.parquet', output_path, labels)


    @staticmethod
    def _save_metric(rec, filename, output_path, labels):
        df = pd.DataFrame(rec)
        df = add_labels(df, labels)
        df.to_parquet(os.path.join(output_path, filename))

    eval_sync = eval_syn # backport typo