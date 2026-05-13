"""
Created on Fri May 13 12:00:00 2026

@author: Andrei V. Ermolaev
"""

import numpy as np
from numpy.linalg import pinv
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def elm_classifier(
    Ispec_train,
    Ispec_test,
    YTrain,
    YTest,
    normalize=False,
    lambda_reg=None,
    scan_lambda=False,
    n_lambda=100,
    lambda_min=1e-10,
    lambda_max=1e2,
    objective='mse',
    PCvar=0.95,
    validation=False,
    permutation=False,
    val_size=1000,
    verbose=True
):
    """
    Extreme Learning Machine (ELM) classifier based on the hidden-layer representation
    uses linear / Ridge regression to train the system. It also performs optional estimation 
    of the number of principal components required to explain a specified variance level 
    and computation of the Hotelling trace (Fisher LDA) criterion to quantify class separability.

    Parameters
    ----------
    Ispec_train : ndarray (n_samples_train, n_features)
        Training input features.

    Ispec_test : ndarray (n_samples_test, n_features)
        Test input features.

    YTrain : ndarray (n_samples_train,)
        Training labels (integers).

    YTest : ndarray (n_samples_test,)
        Test labels (integers).

    normalize : {False, 'std', 'area'}, optional
        Normalization strategy:
        - False : no normalization
        - 'std' : standardization (zero mean, unit variance)
        - 'area': divide by mean area under the hidden-layer state

    lambda_reg : float or None
        Regularization strength (ridge). If None and scan_lambda=False,
        pseudo-inverse solution is used.

    scan_lambda : bool
        If True, performs log-scale search for optimal lambda for a given loss function (see below).

    n_lambda : int
        Number of lambda values to test during scan.

    lambda_min, lambda_max : float
        Range for lambda scan.

    objective : {'mse', 'ce'}
        Optimization objective:
        - 'mse': mean squared error
        - 'ce' : cross-entropy

    PCvar : float or None
        If not None, computes number of PCA components explaining this variance.

    validation : bool
        Whether to split training data into train/validation.

    permutation : bool
        Whether to randomly shuffle train and test sets.

    val_size : int
        Size of validation set (taken from end of training set).

    verbose : bool
        If True, prints performance metrics.

    Returns
    -------
    results : dict
        Dictionary containing all the necessary metrics and model diagnostics.
    """

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------
    def softmax(x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(y_true, y_prob):
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_prob), axis=1))

    # ---------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------
    if objective not in ['mse', 'ce']:
        raise ValueError("objective must be 'mse' or 'ce'")

    if normalize not in [False, 'std', 'area']:
        raise ValueError("normalize must be False, 'std', or 'area'")

    # ---------------------------------------------------------
    # Optional permutation
    # ---------------------------------------------------------
    if permutation:
        perm_train = np.random.permutation(len(Ispec_train))
        Ispec_train, YTrain = Ispec_train[perm_train], YTrain[perm_train]

        perm_test = np.random.permutation(len(Ispec_test))
        Ispec_test, YTest = Ispec_test[perm_test], YTest[perm_test]

    # ---------------------------------------------------------
    # Validation split
    # ---------------------------------------------------------
    if validation:
        if val_size >= len(Ispec_train):
            raise ValueError("val_size must be smaller than training set")

        split_idx = len(Ispec_train) - val_size

        H_val = Ispec_train[split_idx:]
        Y_val = YTrain[split_idx:]

        Ispec_train = Ispec_train[:split_idx]
        YTrain = YTrain[:split_idx]
    else:
        H_val, Y_val = None, None

    # ---------------------------------------------------------
    # One-hot encoding (* for older versions use sparse=False instead of sparse_output=False *)
    # ---------------------------------------------------------
#     encoder = OneHotEncoder(sparse=False)
    encoder = OneHotEncoder(sparse_output=False)

    YTrain_encoded = encoder.fit_transform(YTrain.reshape(-1, 1))
    YTest_encoded = encoder.transform(YTest.reshape(-1, 1))

    if validation:
        YVal_encoded = encoder.transform(Y_val.reshape(-1, 1))

    # ---------------------------------------------------------
    # Hidden layer (identity mapping in this implementation)
    # ---------------------------------------------------------
    H_train = Ispec_train.copy()
    H_test = Ispec_test.copy()
    if validation:
        H_val = H_val.copy()

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    if normalize == 'std':
        eps = 1e-12
        mu = np.mean(H_train, axis=0, keepdims=True)
        sigma = np.std(H_train, axis=0, keepdims=True)

        H_train = (H_train - mu) / (sigma + eps)
        H_test = (H_test - mu) / (sigma + eps)
        if validation:
            H_val = (H_val - mu) / (sigma + eps)

    elif normalize == 'area':
        eps = 1e-12
        mean_area = np.mean(np.sum(H_train, axis=1))

        H_train /= (mean_area + eps)
        H_test /= (mean_area + eps)
        if validation:
            H_val /= (mean_area + eps)

    n_features = H_train.shape[1]

    # ---------------------------------------------------------
    # PCA diagnostic (how many components needed to describe a specified variance level?)
    # ---------------------------------------------------------
    N_comp = None
    if PCvar is not None:
        pca = PCA(PCvar)
        pca.fit(H_train)
        N_comp = pca.n_components_

    # ---------------------------------------------------------
    # Lambda selection / training
    # ---------------------------------------------------------
    if scan_lambda:
        lambda_values = np.logspace(np.log10(lambda_min),
                                   np.log10(lambda_max),
                                   n_lambda)

        best_score = np.inf
        output_weights = None
        lambda_used = None

        HtH = H_train.T @ H_train
        HtY = H_train.T @ YTrain_encoded

        for lam in lambda_values:
            W = np.linalg.solve(HtH + lam * np.eye(n_features), HtY)

            X_eval = H_val if validation else H_test
            Y_eval = YVal_encoded if validation else YTest_encoded

            Y_pred = X_eval @ W

            if objective == 'mse':
                score = np.mean((Y_eval - Y_pred) ** 2)
            else:
                score = cross_entropy(Y_eval, softmax(Y_pred))

            if score < best_score:
                best_score = score
                output_weights = W
                lambda_used = lam

    else:
        if not lambda_reg:
            output_weights = pinv(H_train) @ YTrain_encoded
            lambda_used = 0.0
        else:
            output_weights = np.linalg.solve(
                H_train.T @ H_train + lambda_reg * np.eye(n_features),
                H_train.T @ YTrain_encoded
            )
            lambda_used = lambda_reg

    # ---------------------------------------------------------
    # Predictions and metrics
    # ---------------------------------------------------------
    def evaluate(H, Y_true, Y_encoded):
        Y_pred = H @ output_weights
        probs = softmax(Y_pred)

        acc = np.mean(np.argmax(Y_pred, axis=1) == Y_true)
        mse = np.mean((Y_encoded - Y_pred) ** 2)
        ce = cross_entropy(Y_encoded, probs)

        return acc, mse, ce

    train_accuracy, train_mse, train_ce = evaluate(H_train, YTrain, YTrain_encoded)
    test_accuracy, test_mse, test_ce = evaluate(H_test, YTest, YTest_encoded)

    # ---------------------------------------------------------
    # Hotelling trace (Fisher LDA) criterion computation
    # ---------------------------------------------------------
    eps = 1e-8
    X = np.vstack((H_train, H_test))
    Y_all = np.concatenate((YTrain, YTest))

    classes = np.unique(Y_all)
    mu_global = np.mean(X, axis=0)

    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    for c in classes:
        Xc = X[Y_all == c]
        mu_c = np.mean(Xc, axis=0)

        S_W += (Xc - mu_c).T @ (Xc - mu_c)

        diff = (mu_c - mu_global).reshape(-1, 1)
        S_B += Xc.shape[0] * (diff @ diff.T)

    fisher_value = np.trace(
        np.linalg.solve(S_W + eps * np.eye(n_features), S_B)
    )

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    if verbose:
        print(f"Features: {n_features}")
        print(f"Lambda: {lambda_used:.3e}")
        print(f"Train Acc: {train_accuracy * 100:.2f}%")
        print(f"Test Acc: {test_accuracy * 100:.2f}%")
        print(f"Train CE: {train_ce:.6f}")
        print(f"Test CE: {test_ce:.6f}")
        print(f"Train MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        if N_comp is not None:
            print(f"PCA components ({PCvar:.2f} var): {N_comp}")
        print(f"Fisher: {fisher_value:.6f}")

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    results = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_crossent": train_ce,
        "test_crossent": test_ce,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "lambda_used": lambda_used,
        "fisher_criterion": fisher_value,
        "S_W": S_W,
        "S_B": S_B,
    }

    if N_comp is not None:
        results["n_components"] = N_comp

    return results