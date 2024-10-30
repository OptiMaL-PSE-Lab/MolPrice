# pyright: reportAttributeAccessIssue=false
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import datashader as ds
from datashader.mpl_ext import dsshow
from datashader.transfer_functions import dynspread


def plot_parity(logits: np.ndarray, labels: np.ndarray, r2score: float):
    "Plot parity plot of model predictions"
    # use sns.jointplot to plot the parity plot
    labels, logits = labels.flatten(), logits.flatten()
    ax1 = sns.jointplot(
        x=labels,
        y=logits,
        color="k",
        marginal_kws={"bins": 30, "fill": True, "color": "k", "alpha": 0.5},
        joint_kws={
            "s": 20,  # size of scatter points
            "alpha": 0.5,  # transparency of scatter points
            "color": "red",
            "edgecolors": "black",
        },
    )

    ax1.ax_joint.cla()
    df = pd.DataFrame(dict(x=labels, y=logits))
    ds_plot = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        norm="linear",
        ax=ax1.ax_joint,
        aspect="auto",
        shade_hook=dynspread,
    )
    ax1.ax_joint.plot(
        [min(labels), max(labels)], [min(labels), max(labels)], color="black"
    )
    ax1.set_axis_labels(ylabel="Predicted values", xlabel="True values")
    # limist axes
    ax1.ax_joint.set_xlim(min(labels), max(labels))
    ax1.ax_joint.set_ylim(min(labels), max(labels))

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1)
    # add r2 score as text to plot
    ax1.ax_joint.text(
        0.05,
        0.95,
        f"$R^2$: {r2score:.3f}",
        fontsize=10,
        transform=ax1.ax_joint.transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    return ax1.figure


def plot_dist_overlap(preds: dict[str, np.ndarray]):
    """Plot overlapping distribution of prices in multiple datasets"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), layout="constrained")
    default_colour = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]
    for i, (key, prices) in enumerate(preds.items()):
        weights = np.ones_like(prices) * 100 / len(prices)
        ax.hist(
            prices,
            bins=100,
            weights=weights,
            range=(1, 20),
            label=key,
            color=default_colour[i],
            alpha=0.4,
            lw=1,
            edgecolor="black",
        )  # type:ignore

    if len(preds.keys()) == 2:
        preds_keys = list(preds.keys())
        mcc, threshold = _mcc_calculator(preds[preds_keys[0]], preds[preds_keys[1]])
        hs_pred, es_pred = preds[preds_keys[0]], preds[preds_keys[1]]
        # preds_keys[0] has labels 1 and preds_keys[1] has labels 0
        pos_true = np.where(hs_pred >= threshold)[0]
        neg_true = np.where(es_pred < threshold)[0]
        pos_false = np.where(hs_pred < threshold)[0]
        neg_false = np.where(es_pred >= threshold)[0]
        # calculate roc auc score using labels
        roc_auc = roc_auc_score(
            np.concatenate([np.ones(len(hs_pred)), np.zeros(len(es_pred))]),
            np.concatenate([hs_pred, es_pred]),
        )
        print(f"ROC AUC: {roc_auc:.3f}")
        # calculate accuracy 
        accuracy = (len(pos_true) + len(neg_true)) / (len(pos_true) + len(neg_true) + len(pos_false) + len(neg_false))
        f1_score = 2 * len(pos_true) / (2 * len(pos_true) + len(pos_false) + len(neg_false))
        print(f"F1 Score: {f1_score:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        ax.text(
            0.95,
            0.95,
            f"$MCC$: {mcc:.3f}",
            fontsize=12,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # draw vertical line at optimal threshold
        ax.axvline(x=threshold, color="black", linestyle="--", label="Threshold")

    ax.set_xlabel(r"log Price $(\$/mmol)$", fontsize=13)
    ax.set_ylabel(r"Frequency (%)", fontsize=13)
    ax.legend(fontsize=12, loc="upper left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig


def _mcc_calculator(hs_preds: np.ndarray, es_preds: np.ndarray):
    y_pred = np.concatenate([es_preds, hs_preds])
    y_true = np.concatenate([np.zeros(len(es_preds)), np.ones(len(hs_preds))])

    thresholds = np.linspace(1, np.max(y_pred), 50)

    # Step 3: Find the optimal threshold that maximizes MCC
    mcc_scores = []
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        mcc_scores.append(mcc)

    # Convert to numpy array for easier indexing
    mcc_scores = np.array(mcc_scores)
    optimal_idx = np.argmax(mcc_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_mcc = mcc_scores[optimal_idx]
    return optimal_mcc, optimal_threshold
