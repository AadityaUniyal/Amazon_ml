"""
Training Module for Matching Classifier (Role B).
Trains LightGBM classifier on hard-negative and positive candidate pairs.
"""
import os
import pickle
import numpy as np
import lightgbm as lgb

def train_matching_model(X_train, y_train, sample_weight=None, model_params=None, save_path=None):
    if model_params is None:
        model_params = {
            'n_estimators': 400,
            'learning_rate': 0.06,
            'num_leaves': 45,
            'max_depth': 8,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    if sample_weight is None:
        # Asymmetric cost-sensitive loss: penalize negative examples (False Positives) 4x harder
        # Directly aligning GBDT loss with Macro F0.5 evaluation metric (beta=0.5 -> 1/beta^2 = 4.0)
        sample_weight = np.where(np.array(y_train) == 1, 1.0, 4.0)
    
    clf = lgb.LGBMClassifier(**model_params)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
            
    return clf

def load_matching_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
