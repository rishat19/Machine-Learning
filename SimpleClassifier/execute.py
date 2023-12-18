import numpy as np
import plotly.graph_objects as go
from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier


def prepare_data():
    dataset = Sportsmanheight()()
    confidence = Classifier()(dataset['height'])
    gt = dataset['class']
    indices = confidence.argsort()[::-1]
    confidence = confidence[indices]
    gt = gt[indices]
    return confidence, gt


def find_tp_fp_tn_fn(confidence: np.ndarray, gt: np.ndarray):
    indexes = np.unique(confidence, return_index=True)[1]
    unique_confidence = np.array([confidence[index] for index in sorted(indexes)])
    unique_confidence = np.append(unique_confidence, 0)
    tp = np.zeros(unique_confidence.size)
    fp = np.zeros(unique_confidence.size)
    tn = np.zeros(unique_confidence.size)
    fn = np.zeros(unique_confidence.size)
    for i in range(unique_confidence.size):
        predictions = np.zeros(gt.size)
        predictions[confidence >= unique_confidence[i]] = 1
        tp[i] = np.count_nonzero((predictions == 1) & (gt == 1))
        fp[i] = np.count_nonzero((predictions == 1) & (gt == 0))
        tn[i] = np.count_nonzero((predictions == 0) & (gt == 0))
        fn[i] = np.count_nonzero((predictions == 0) & (gt == 1))
    return tp, fp, tn, fn, unique_confidence


def calculate_metrics(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    f_score[0] = 0
    alpha = fp / (tn + fp)
    beta = fn / (tp + fn)
    return accuracy, precision, recall, f_score, alpha, beta


def make_steps(precision: np.ndarray):
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    return precision


def visualise_pr_curve(precision: np.ndarray, recall: np.ndarray, accuracy: np.ndarray, f_score: np.ndarray,
                       confidence: np.ndarray, plot_title=''):
    custom = []
    for i in range(accuracy.size):
        custom.append('Accuracy:' + str(accuracy[i]) + '<br>F score: ' + str(f_score[i]) + '<br>Confidence: ' +
                      str(confidence[i]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall,
                             y=precision,
                             customdata=custom,
                             mode='lines',
                             name='PR curve'))
    fig.update_layout(title=plot_title,
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    fig.update_traces(hoverinfo='all',
                      hovertemplate='Recall: %{x}<br>Precision: %{y}<br>%{customdata}')
    fig.show()


def visualise_roc_curve(alpha: np.ndarray, beta: np.ndarray, plot_title=''):
    inverse_beta = 1 - beta
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alpha,
                             y=inverse_beta,
                             customdata=beta,
                             mode='lines',
                             name='ROC curve'))
    fig.update_layout(title=plot_title,
                      xaxis_title='Alpha',
                      yaxis_title='1 - Beta')
    fig.update_traces(hoverinfo='all',
                      hovertemplate='Alpha: %{x}<br>Beta: %{customdata}')
    fig.show()


def calculate_area(x: np.ndarray, y: np.ndarray):
    return np.trapz(y, x)


if __name__ == '__main__':
    confidence_arr, gt_arr = prepare_data()
    tp_arr, fp_arr, tn_arr, fn_arr, confidence_arr = find_tp_fp_tn_fn(confidence_arr, gt_arr)
    accuracy_arr, precision_arr, recall_arr, f_score_arr, alpha_arr, beta_arr = calculate_metrics(tp_arr, fp_arr,
                                                                                                  tn_arr, fn_arr)

    recall_arr = np.append(recall_arr, 1)
    precision_arr = np.append(precision_arr, 0)
    accuracy_arr = np.append(accuracy_arr, 0.5)
    f_score_arr = np.append(f_score_arr, 0)
    confidence_arr = np.append(confidence_arr, 0)

    precision_arr = make_steps(precision_arr)

    visualise_pr_curve(precision=precision_arr,
                       recall=recall_arr,
                       accuracy=accuracy_arr,
                       f_score=f_score_arr,
                       confidence=confidence_arr,
                       plot_title=f'PR curve. Area = {round(calculate_area(recall_arr, precision_arr), 4)}')

    visualise_roc_curve(alpha=alpha_arr,
                        beta=beta_arr,
                        plot_title=f'ROC curve. Area = {round(calculate_area(alpha_arr, 1 - beta_arr), 4)}')
