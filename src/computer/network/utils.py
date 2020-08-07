"""
Script containing helper functions
"""
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import io
import torch.utils.tensorboard.writer as Writer
import torch


def generate_confusion_matrix(targets: torch.Tensor, predictions: torch.Tensor, writer: Writer.SummaryWriter = None) -> None:
    """Generates a confusion matrix and adds it to the summary writer or shows the plot

    Args:
        targets (torch.Tensor): Tensor of targets
        predictions (torch.Tensor): Tensor of predictions
        writer (Writer.SummaryWriter, optional): Summarywriter. Defaults to None.
    """    
    data = {'targets': targets, 'predictions': predictions}
    df = pd.DataFrame(data)
    confusion_matrix = pd.crosstab(df['targets'], df['predictions'], rownames=['Actual'], colnames=['Predicted'], margins = True)

    sn.heatmap(confusion_matrix, fmt="d", annot=True)

    if writer is not None:
        figure = plt.gcf()
        writer.add_figure('Test/confusion matrix', figure)
    
    plt.show()