import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from os.path import join
import math
import sys
import numpy as np 
from laplace import Laplace
from netcal.metrics import ECE 
import torch.distributions as dists
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from lora_vit import create_lora_vit_model
from utils import print_trainable_parameters
from dataloaders import create_food_data_loaders


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()
    

def fit_laplace_and_compute_predictions(model_path, config, hessian_structure="kron"):

    assert hessian_structure in {"kron", "full"}

    lora_model, transform = create_lora_vit_model(**config)
    lora_model.load_state_dict(torch.load(model_path))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    lora_model.to(device)
    lora_model.eval() 

    # Make sure C and D are not trainable before fitting Laplace
    for name, module in lora_model.named_modules():
        if 'C' in name or 'D' in name or 'linear1' in name: 
            module.weight.requires_grad = False

    print_trainable_parameters(lora_model, detailed=True)

    trainloader, testloader = create_food_data_loaders(transform, **config)

    la = Laplace(
        lora_model,
        likelihood="classification",
        subset_of_weights="all",
        hessian_structure=hessian_structure,
    )

    print(f'Fitting Laplace with hessian_structure {hessian_structure} ...')
    la.fit(trainloader) # TODO use train data (possibly a smaller sample to make it faster, say 100 images)
    
    print('Optimizing prior precision ...')
    la.optimize_prior_precision(prior_structure='layerwise') # prior_structure='layerwise'
    print('Fitting Laplace - Done')

    probs_laplace = predict(testloader, la, laplace=True)
    probs_baseline = predict(testloader, lora_model, laplace=False)
    targets = torch.cat([y for x, y in testloader], dim=0).cpu()

    return probs_laplace, probs_baseline, targets


def eval_metrics(probs, targets, name): 
    
    acc_laplace = (probs.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs).log_prob(targets).mean()
    
    print(
        f"[{name}] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )
    

def plot_calibration_curve_multiclass(y_true, y_pred, n_bins=10):
    """
    Plot calibration curves (reliability diagrams) for a multi-class classification problem.

    Parameters:
    - y_true: array-like of shape (n_samples,), true class labels.
    - y_pred: array-like of shape (n_samples, n_classes), predicted probabilities for each class.
    - n_bins: int, number of bins to use for calibration curve.
    """
    classes = np.unique(y_true)
    plt.figure(figsize=(10, 7))

    for cls in classes:
        # Create a binary version of y_true for the current class
        y_true_binary = (y_true == cls).astype(int)
        y_prob = y_pred[:, cls]

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=n_bins, strategy='uniform')

        # Plot calibration curve for the current class
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {cls}')

    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

    # Add labels and title
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid()
    plt.show()
