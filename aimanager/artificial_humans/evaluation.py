
import pandas as pd
import torch as th
import os
from aimanager.artificial_humans.metrics import create_metrics, create_confusion_matrix
from aimanager.utils.array_to_df import add_labels, using_multiindex
from aimanager.utils.utils import make_dir


class Evaluator:
    def __init__(self):
        self.metrics = []
        self.confusion_matrix = []
        self.synthetic_predicitions = []

    def set_data(self, train, test, syn):
        self.data = {
            'train': train,
            'test': test,
            'syn': syn
        }

    def set_labels(self, **labels):
        self.labels = labels

    def eval_set(self, model, set_name):
        y_pred, y_pred_proba = model.predict(**self.data[set_name])

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

            mask = self.data[set_name]['valid']
            y_true = th.masked_select(self.data[set_name]['contributions'], mask)
            y_pred = th.masked_select(y_pred, mask)
            y_true = y_true.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            self.metrics += create_metrics(y_true, y_pred, set=set_name, strategy=strategy, **self.labels)
            self.confusion_matrix += create_confusion_matrix(
                y_true, y_pred, set=set_name, strategy=strategy, **self.labels)

    def eval_sync(self, model):
        y_pred, y_pred_proba = model.predict(**self.data['syn'])
        y_pred = y_pred.detach().cpu().numpy()
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba.detach().cpu().numpy()
            proba_df = using_multiindex(
                y_pred_proba, ['prev_contribution', 'prev_punishment', 'dummy', 'contribution']).rename(columns={'value': 'proba'})
            pred_df = using_multiindex(
                y_pred, ['prev_contribution', 'prev_punishment', 'dummy']).rename(columns={'value': 'pred_contribution'})
            pred_df = pred_df.merge(proba_df)
            pred_df['predicted'] = pred_df['contribution'] == pred_df['pred_contribution']
            pred_df = pred_df.drop(columns=['pred_contribution', 'dummy'])
        else:
            raise NotImplementedError("Needs y_pred_proba.")
        pred_df = add_labels(pred_df, {'set': 'train', **self.labels})
        self.synthetic_predicitions += pred_df.to_dict('records')

    def add_loss(self, loss):
        self.metrics.append(dict(name='loss', value=loss, **self.labels))

    def save(self, output_path, labels):
        make_dir(output_path)
        self._save_metric(self.metrics, 'metrics.parquet', output_path, labels)
        self._save_metric(self.confusion_matrix, 'confusion_matrix.parquet', output_path, labels)
        self._save_metric(self.synthetic_predicitions, 'synthetic_predicitions.parquet', output_path, labels)


    @staticmethod
    def _save_metric(rec, filename, output_path, labels):
        df = pd.DataFrame(rec)
        df = add_labels(df, labels)
        df.to_parquet(os.path.join(output_path, filename))