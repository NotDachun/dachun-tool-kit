
import evaluate
import numpy as np

def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

def metric_fn(metrics=["f1", "accuracy", "recall", "precision"], average="binary", threshold=0.5):
    """
    Generates a function comment for the given function body.

    Args:
        metrics (list, optional): A list of metrics to be combined. Defaults to ["f1", "accuracy", "recall", "precision"].
        average (str, optional): The averaging method. Defaults to "binary".
            Can be one of the other values: "micro", "macro", "weighted", None
        threshold (float, optional): The threshold for the predictions. Defaults to 0.5.

    Returns:
        function: The compute_metrics function.
    """
    metric = evaluate.combine(metrics)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if average == "binary":
            # apply sigmoid
            predictions = sigmoid(predictions)
        else:
            predictions = softmax(predictions)
        
        # apply threshold
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        
        return metric.compute(predictions=predictions, references=labels, average=average)
    
    return compute_metrics