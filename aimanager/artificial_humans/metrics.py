from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss, confusion_matrix
from aimanager.utils.array_to_df import using_multiindex, add_labels


def create_metrics(y_true, y_pred,  **labels):
    accuracy = accuracy_score(y_true, y_pred)
    mabserr = mean_absolute_error(y_true, y_pred)
    metrics = [
    {
        'name': 'mean_absolute_error',
        'value': mabserr,
        **labels
    },
    {
        'name': 'accuracy',
        'value': accuracy,
        **labels
    }
    ]
    return metrics


def calc_log_loss(y_true, y_pred_proba, n_levels, **labels):
    ll =  log_loss(y_true, y_pred_proba, labels=list(range(n_levels)))
    return {
        'name': 'log_loss',
        'value': ll,
        **labels
    }


def create_confusion_matrix(y_true, y_pred_proba, mask, y_name, **labels):
    y_pred_proba = y_pred_proba.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    proba_df = using_multiindex(
        y_pred_proba, ['idx', 'round_number', f'pred_{y_name}']).rename(columns={'value': 'proba'})
    mask_df = using_multiindex(
        mask, ['idx', 'round_number']).rename(columns={'value': 'valid'})
    y_df = using_multiindex(
        y_true, ['idx', 'round_number']).rename(columns={'value': f'true_{y_name}'})


    df = proba_df.merge(mask_df).merge(y_df)
    df = add_labels(df, labels)
    return df