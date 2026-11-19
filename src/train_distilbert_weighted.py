
from pathlib import Path
import json
import random
import time
import gc

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tqdm.auto import tqdm


# ============================================================
# STAGE 9.2 - CLASS-WEIGHTED DISTILBERT
# ============================================================

SEED = 42
MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 96
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
N_SPLITS = 4

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "distilbert_weighted_predictions_v1.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "distilbert_weighted_metrics_v1.json"
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. DATASET
# ============================================================

class TweetDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.labels = np.asarray(labels, dtype=np.int64)

        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.encodings[
                "input_ids"
            ][index],
            "attention_mask": self.encodings[
                "attention_mask"
            ][index],
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


# ============================================================
# 3. LOAD HUMAN-VERIFIED LABELS
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"},
    )

    required = {
        "post_id",
        "text",
        "final_incident_target",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs.")

    target = pd.to_numeric(
        df["final_incident_target"],
        errors="coerce",
    )

    invalid = (
        target.notna()
        & ~target.isin([0, 1])
    )

    if invalid.any():
        raise ValueError(
            "Invalid target values."
        )

    df["target"] = target

    # Only definite verified labels.
    df = df[
        df["target"].isin([0, 1])
    ].copy()

    if df["text"].isna().any():
        raise ValueError("Missing tweet text.")

    df["text"] = df["text"].astype(str)

    if df["text"].str.strip().eq("").any():
        raise ValueError("Empty tweet text.")

    df["target"] = df["target"].astype(int)

    df = df.reset_index(drop=True)

    counts = df["target"].value_counts()

    if len(counts) != 2 or counts.min() < N_SPLITS:
        raise ValueError(
            "Insufficient examples for 4-fold CV."
        )

    print("\nVerified training examples:", len(df))
    print("\nClass distribution:")
    print(counts.sort_index().to_string())

    return df


# ============================================================
# 4. CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(labels, device):

    counts = np.bincount(
        labels,
        minlength=2,
    )

    if np.any(counts == 0):
        raise ValueError(
            "Training fold must contain both classes."
        )

    total = len(labels)

    weights = total / (2.0 * counts)

    print("\nTraining class counts:", counts.tolist())
    print("Class weights:", weights.tolist())

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# 5. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
):

    model.train()
    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:

        optimizer.zero_grad(set_to_none=True)

        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch[
                "attention_mask"
            ].to(device),
        }

        labels = batch["labels"].to(device)

        outputs = model(**inputs)

        # Weighted loss replaces default loss.
        loss = criterion(
            outputs.logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============================================================
# 6. PREDICT VALIDATION PROBABILITIES
# ============================================================

@torch.no_grad()
def predict_probabilities(model, loader, device):

    model.eval()
    probabilities = []

    for batch in loader:

        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch[
                "attention_mask"
            ].to(device),
        }

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1,
        )[:, 1]

        probabilities.extend(
            probs.cpu().numpy().tolist()
        )

    return np.asarray(probabilities)


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def get_metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist()
        ),
    }


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 9.2 - CLASS-WEIGHTED DISTILBERT")
    print("=" * 65)

    # Preserve existing experiment outputs.
    for path in [
        PREDICTIONS_FILE,
        METRICS_FILE,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    df = load_data()

    texts = df["text"].tolist()
    labels = df["target"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof_probs = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    oof_folds = np.full(
        len(df),
        -1,
        dtype=int,
    )

    fold_metrics = []

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(texts, labels),
        start=1,
    ):

        print("\n" + "=" * 65)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 65)

        set_seed(SEED + fold)

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_texts = [
            texts[i] for i in train_idx
        ]

        val_texts = [
            texts[i] for i in val_idx
        ]

        print("Training size:", len(train_idx))
        print("Validation size:", len(val_idx))

        train_dataset = TweetDataset(
            train_texts,
            train_labels,
            tokenizer,
        )

        val_dataset = TweetDataset(
            val_texts,
            val_labels,
            tokenizer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # Compute weights using training fold only.
        class_weights = calculate_class_weights(
            train_labels,
            device,
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        # Initialize a fresh model for this fold.
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_NAME,
                num_labels=2,
            )
        )

        model.to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        total_steps = (
            len(train_loader) * EPOCHS
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        for epoch in range(1, EPOCHS + 1):

            print(
                f"\nFold {fold} - Epoch {epoch}/{EPOCHS}"
            )

            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
            )

            print(
                f"Average weighted loss: {loss:.4f}"
            )

        val_probs = predict_probabilities(
            model,
            val_loader,
            device,
        )

        val_preds = (
            val_probs >= 0.50
        ).astype(int)

        oof_probs[val_idx] = val_probs
        oof_folds[val_idx] = fold

        metrics = get_metrics(
            val_labels,
            val_preds,
        )

        metrics["fold"] = fold
        metrics["train_size"] = len(train_idx)
        metrics["validation_size"] = len(val_idx)
        metrics["class_weights"] = (
            class_weights.cpu().tolist()
        )

        fold_metrics.append(metrics)

        print("\nFold metrics:")
        print(json.dumps(metrics, indent=2))

        # Release model memory.
        del model
        del optimizer
        del scheduler
        del criterion
        del class_weights
        del train_loader
        del val_loader

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # FINAL OUT-OF-FOLD EVALUATION
    # --------------------------------------------------------

    if np.isnan(oof_probs).any():
        raise RuntimeError(
            "Missing out-of-fold probabilities."
        )

    if (oof_folds < 1).any():
        raise RuntimeError(
            "Missing validation fold assignments."
        )

    predictions = (
        oof_probs >= 0.50
    ).astype(int)

    overall = get_metrics(
        labels,
        predictions,
    )

    print("\n" + "=" * 65)
    print("WEIGHTED DISTILBERT - FINAL RESULTS")
    print("=" * 65)

    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[
                "Non-event",
                "Actual incident",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nPositive precision:", overall["precision"])
    print("Positive recall:", overall["recall"])
    print("Positive F1:", overall["f1"])
    print("Accuracy:", overall["accuracy"])

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    output = df[
        ["post_id", "text", "target"]
    ].copy()

    output["cv_fold"] = oof_folds
    output["predicted_target"] = predictions
    output["positive_probability"] = oof_probs
    output["correct"] = (
        output["target"]
        == output["predicted_target"]
    )

    elapsed = time.time() - start_time

    summary = {
        "stage": "9.2",
        "model": MODEL_NAME,
        "training_policy": "verified_only",
        "loss": "class_weighted_cross_entropy",
        "class_weight_method": "balanced_per_training_fold",
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "n_splits": N_SPLITS,
        "decision_threshold": 0.50,
        "device": str(device),
        "verified_training_examples": len(df),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "overall_metrics": overall,
        "fold_metrics": fold_metrics,
        "training_seconds": round(elapsed, 2),
        "pseudo_labels_used": False,
        "human_labels_modified": False,
        "evaluation_note": (
            "Exploratory stratified out-of-fold "
            "evaluation, not an independent test set."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nOUTPUT FILES:")
    print(PREDICTIONS_FILE)
    print(METRICS_FILE)

    print("\nTraining seconds:", round(elapsed, 2))

    print("\nVerified labels were NOT modified.")
    print("Pseudo-labels were NOT used.")
    print("STAGE 9.2 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
