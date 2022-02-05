from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss, confusion_matrix
from aimanager.utils.array_to_df import using_multiindex, add_labels


def create_metrics(y_true, y_true_ord, y_pred, y_pred_ordinal, y_pred_proba, *, n_contributions, ordinal_y, **labels):
    accuracy = accuracy_score(y_true, y_pred)
    mabserr = mean_absolute_error(y_true, y_pred)
    if ordinal_y:
        logloss = log_loss(y_true_ord.reshape(-1), y_pred_proba.reshape(-1,2))
    else:
        logloss = log_loss(y_true, y_pred_proba, labels=range(n_contributions))
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
    },
    {
        'name': 'log_loss',
        'value': logloss,
        **labels
    },
    ]
    return metrics



def create_confusion_matrix(y_true, y_pred, **labels):
    # first dimension is true, second dimension is predicted
    cm = confusion_matrix(y_true, y_pred)
    cm_df = using_multiindex(cm, columns=['y_true','y_pred'])
    cm_df = add_labels(cm_df, labels)
    return cm_df.to_dict(orient='records')