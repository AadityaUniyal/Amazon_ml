"""
Production Model Training Script (Role B).
Trains LightGBM classifier on hard-negative and positive candidate pairs,
optimizes decision threshold for Macro F_0.5, and saves model artifact.
"""
import os
import sys
import csv
import time
import json
import pickle
import numpy as np
import lightgbm as lgb
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization import normalize_record
from src.blocking import MultiChannelBlocker
from src.features import extract_pair_features
from src.thresholding import optimize_threshold
from src.evaluation import evaluate_macro_f05

def main():
    print("=" * 60)
    print(" Amazon ML Challenge 2026 — Production Model Training")
    print("=" * 60)

    # Resolve paths
    base_dir = None
    candidates = [
        r"D:\AMAZON\6ab10eb3b23ba_student_resource\student_resource\dataset\train",
        r"c:\Users\HP\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train",
        r"student_resource/dataset/train",
        r"../student_resource/dataset/train"
    ]
    for c in candidates:
        if os.path.isdir(c):
            base_dir = c
            break

    if not base_dir:
        raise FileNotFoundError("Could not locate train dataset directory.")

    print(f"Using dataset directory: {base_dir}")

    # 1. Load Ground Truth
    gt_file = os.path.join(base_dir, "train_ground_truth.tsv")
    print(f"Loading ground truth from {gt_file}...")
    t0 = time.time()
    
    # Scale to 25,000 reference entities (~300,000+ candidate pairs with hard negative mining)
    NUM_TRAIN_S1 = 25000
    gt = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i >= NUM_TRAIN_S1:
                break
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt[row[0]] = matches

    print(f"Loaded {len(gt):,} reference entities in {time.time()-t0:.2f}s.")

    # 2. Load Source 1 entities
    s1_file = os.path.join(base_dir, "train_source1.tsv")
    s1_recs = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in gt:
                s1_recs[row[0]] = normalize_record(row[0], row[1], row[2], row[3] if len(row) > 3 else "")

    print(f"Loaded {len(s1_recs):,} Source 1 records.")

    # 3. Identify all positive target IDs and load target candidate pool
    needed_targets = set()
    for m in gt.values():
        needed_targets.update(m)

    print(f"Positive targets to find: {len(needed_targets):,}")

    target_recs = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        fp = os.path.join(base_dir, fname)
        print(f"Loading targets from {fname}...")
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for i, row in enumerate(reader):
                # Always load positive targets + sample background records
                if row[0] in needed_targets or i < 30000:
                    target_recs[row[0]] = normalize_record(row[0], row[1], row[2], row[3] if len(row) > 3 else "")

    print(f"Total target candidate pool: {len(target_recs):,} records.")

    # 4. Train / Validation Split (80 / 20 entity-level split)
    s1_keys = list(s1_recs.keys())
    split_idx = int(0.80 * len(s1_keys))
    train_s1_keys = set(s1_keys[:split_idx])
    val_s1_keys = set(s1_keys[split_idx:])
    print(f"Train S1 entities: {len(train_s1_keys):,} | Validation S1 entities: {len(val_s1_keys):,}")

    # 5. Fit Blocker on target records
    print("\nFitting MultiChannelBlocker on target pool...")
    t0 = time.time()
    blocker = MultiChannelBlocker(df_cap=3000, max_candidates=25)
    blocker.fit_targets(target_recs)
    print(f"Blocker indexed in {time.time()-t0:.2f}s.")

    # 6. Retrieve Candidates & Form Pairs
    print("\nGenerating candidate pairs for training set...")
    t0 = time.time()
    train_pairs = []
    for sid in train_s1_keys:
        s1_rec = s1_recs[sid]
        true_matches = gt[sid]
        retrieved = blocker.retrieve_candidates_with_scores(s1_rec)
        for tid, score in retrieved:
            label = 1 if tid in true_matches else 0
            train_pairs.append((sid, tid, label, score))

    print(f"Generated {len(train_pairs):,} train candidate pairs in {time.time()-t0:.2f}s.")
    pos_train = sum(p[2] for p in train_pairs)
    print(f"  Positive pairs: {pos_train:,} ({pos_train/len(train_pairs)*100:.2f}%)")
    print(f"  Hard negative pairs: {len(train_pairs)-pos_train:,}")

    print("\nGenerating candidate pairs for validation set...")
    t0 = time.time()
    val_pairs = []
    val_true_matches_count = sum(len(gt[sid]) for sid in val_s1_keys)
    val_recalled_count = 0
    for sid in val_s1_keys:
        s1_rec = s1_recs[sid]
        true_matches = gt[sid]
        retrieved = blocker.retrieve_candidates_with_scores(s1_rec)
        retrieved_ids = {tid for tid, _ in retrieved}
        val_recalled_count += len(true_matches & retrieved_ids)
        for tid, score in retrieved:
            label = 1 if tid in true_matches else 0
            val_pairs.append((sid, tid, label, score))

    blocker_recall = val_recalled_count / max(1, val_true_matches_count)
    print(f"Validation Candidate Recall Ceiling: {blocker_recall*100:.2f}% ({val_recalled_count}/{val_true_matches_count})")
    print(f"Generated {len(val_pairs):,} validation pairs.")

    # 7. Feature Extraction
    print("\nExtracting 27D features for train set...")
    t0 = time.time()
    X_train = [
        extract_pair_features(s1_recs[sid], target_recs[tid], sc)
        for sid, tid, _, sc in train_pairs
    ]
    y_train = np.array([p[2] for p in train_pairs], dtype=int)
    print(f"Extracted train features in {time.time()-t0:.2f}s.")

    print("Extracting 27D features for validation set...")
    t0 = time.time()
    X_val = [
        extract_pair_features(s1_recs[sid], target_recs[tid], sc)
        for sid, tid, _, sc in val_pairs
    ]
    y_val = np.array([p[2] for p in val_pairs], dtype=int)
    print(f"Extracted validation features in {time.time()-t0:.2f}s.")

    # 8. Train LightGBM Classifier with Cost-Sensitive Asymmetric Loss
    print("\nTraining Cost-Sensitive LightGBM GBDT Matching Classifier (w_fp=4.0)...")
    t0 = time.time()
    sample_weights = np.where(y_train == 1, 1.0, 4.0)
    model = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=40,
        max_depth=8,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Model trained in {time.time()-t0:.2f}s.")

    # 9. Out-of-fold Evaluation & Threshold Optimization
    val_probs = model.predict_proba(X_val)[:, 1]
    pairwise_acc = np.mean((val_probs >= 0.50).astype(int) == y_val)
    print(f"\nPairwise Classification Accuracy: {pairwise_acc*100:.4f}%")

    val_probs_by_s1 = defaultdict(list)
    for (sid, tid, _, _), prob in zip(val_pairs, val_probs):
        val_probs_by_s1[sid].append((tid, float(prob)))

    val_gt = {sid: gt[sid] for sid in val_s1_keys}

    print("\nOptimizing Decision Threshold for Macro F_0.5...")
    best_thresh, best_f05 = optimize_threshold(val_probs_by_s1, val_gt, np.arange(0.40, 0.85, 0.05))
    print(f"Optimal Threshold: tau* = {best_thresh:.2f} | Out-of-Fold Macro F_0.5 = {best_f05:.4f}")

    # Evaluate singleton performance
    singleton_sids = [sid for sid in val_s1_keys if len(val_gt[sid]) == 0]
    correct_singletons = sum(
        1 for sid in singleton_sids
        if len([tid for tid, p in val_probs_by_s1.get(sid, []) if p >= best_thresh]) == 0
    )
    singleton_acc = correct_singletons / max(1, len(singleton_sids))
    print(f"Singleton Accuracy: {singleton_acc*100:.2f}% ({correct_singletons}/{len(singleton_sids)})")

    # 10. Save Artifacts
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "artifacts", "models", "lgbm_er_model.pkl")
    model_path2 = os.path.join(project_root, "models", "lgbm_er_model.pkl")
    config_path = os.path.join(project_root, "configs", "final.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path2), exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    for mp in [model_path, model_path2]:
        with open(mp, "wb") as f:
            pickle.dump(model, f)
            f.flush()
            os.fsync(f.fileno())
        print(f"Saved trained model to {mp} (size: {os.path.getsize(mp)})")

    config_data = {
        "model_type": "LightGBM",
        "threshold": float(best_thresh),
        "macro_f05_val": float(best_f05),
        "pairwise_acc_val": float(pairwise_acc),
        "blocking_recall_val": float(blocker_recall),
        "singleton_acc_val": float(singleton_acc),
        "max_candidates": 25,
        "n_estimators": 350,
        "learning_rate": 0.06
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved configuration to {config_path} (size: {os.path.getsize(config_path)})")

    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
