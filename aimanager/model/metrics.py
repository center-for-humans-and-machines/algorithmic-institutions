from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss, confusion_matrix
from aimanager.utils.array_to_df import using_multiindex, add_labels


def create_metrics(y_true, y_pred, **labels):
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



def create_confusion_matrix(y_true, y_pred, **labels):
    # first dimension is true, second dimension is predicted
    cm = confusion_matrix(y_true, y_pred)
    cm_df = using_multiindex(cm, columns=['y_true','y_pred'])
    cm_df = add_labels(cm_df, labels)
    return cm_df.to_dict(orient='records')