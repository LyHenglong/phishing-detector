"""
data_split.py
-------------
Domain-grouped train/val/test splitting.

Random row-level splitting leaks information here: many rows in the raw
dataset are different paths on the same abused domain (free-hosting / IPFS
gateways, Google Docs, etc. used to host phishing pages), sharing near-
identical domain-level features. A row-level split lets the same domain land
in both train and test, so the model partly memorizes domains instead of
learning transferable URL patterns, inflating test metrics. Splitting by
domain group removes that leakage and measures generalization to domains
never seen during training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def grouped_train_val_test_split(X, y, groups, test_size=0.15, val_size=0.15 / 0.85, random_state=42):
    """
    Split (X, y) into train/val/test with no group appearing in more than one
    split, stratified by each group's majority label so class balance stays
    consistent across splits (domains here are almost always label-pure, so
    a group's majority label is effectively "its" label).

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Row indices into the original X / y / groups arrays.
    """
    groups = pd.Series(groups)
    domain_label = pd.Series(y).groupby(groups.values).mean().round().astype(int)
    unique_domains = domain_label.index.values
    domain_y = domain_label.values

    train_val_domains, test_domains = train_test_split(
        unique_domains, test_size=test_size, stratify=domain_y, random_state=random_state)

    train_val_y = domain_label.loc[train_val_domains].values
    train_domains, val_domains = train_test_split(
        train_val_domains, test_size=val_size, stratify=train_val_y, random_state=random_state)

    train_idx = np.flatnonzero(groups.isin(train_domains).values)
    val_idx   = np.flatnonzero(groups.isin(val_domains).values)
    test_idx  = np.flatnonzero(groups.isin(test_domains).values)

    g_train, g_val, g_test = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    leaked = (g_train & g_val) | (g_train & g_test) | (g_val & g_test)
    if leaked:
        raise RuntimeError(f"Domain leakage across splits: {leaked}")

    return train_idx, val_idx, test_idx
