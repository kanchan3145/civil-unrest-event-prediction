
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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
# STAGE 9 - DISTILBERT VERIFIED-ONLY BASELINE
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
    OUTPUT_DIR / "distilbert_verified_predictions_v1.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "distilbert_verified_metrics_v1.json"
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
# 2. DATASET CLASS
# ============================================================

class TweetDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


# ============================================================
# 3. LOAD VERIFIED DATA
# ============================================================

def load_verified_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Verified dataset not found: {INPUT_FILE}"
        )

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
        raise ValueError("Missing post IDs detected.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs detected.")

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
            "Invalid verified target values."
        )

    df["target"] = target

    # Exclude uncertain labels.
    train_df = df[
        df["target"].isin([0, 1])
    ].copy()

    train_df["target"] = (
        train_df["target"].astype(int)
    )

    if train_df["text"].isna().any():
        raise ValueError(
            "Missing text in definite training examples."
        )

    train_df["text"] = (
        train_df["text"].astype(str)
    )

    if train_df["text"].str.strip().eq("").any():
        raise ValueError(
            "Empty text in definite training examples."
        )

    counts = train_df["target"].value_counts()

    if len(counts) != 2:
        raise ValueError(
            "Both binary classes are required."
        )

    if counts.min() < N_SPLITS:
        raise ValueError(
            "Not enough examples for stratified CV."
        )

    print("\nVerified dataset:", len(df))
    print("Definite examples:", len(train_df))
    print("Excluded uncertain:", len(df) - len(train_df))

    print("\nClass distribution:")
    print(counts.sort_index().to_string())

    return train_df.reset_index(drop=True)


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
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

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / max(len(loader), 1)


# ============================================================
# 5. VALIDATE MODEL
# ============================================================

@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    device,
):

    model.eval()

    probabilities = []

    for batch in loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1,
        )

        probabilities.extend(
            probs[:, 1].cpu().numpy().tolist()
        )

    return np.asarray(probabilities)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

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
# 7. MAIN CROSS-VALIDATION EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 9 - DISTILBERT VERIFIED-ONLY TRAINING")
    print("=" * 65)

    # Protect previous experiment results.
    for path in [PREDICTIONS_FILE, METRICS_FILE]:

        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cpu":
        print(
            "CPU training may take a considerable time."
        )

    df = load_verified_data()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    texts = df["text"].tolist()

    labels = df["target"].to_numpy()

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    # Out-of-fold predictions:
    # Each tweet is predicted by a model that
    # was not trained on that tweet.
    oof_probabilities = np.full(
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

        # Reset the random seed for reproducibility.
        set_seed(SEED + fold)

        train_texts = [
            texts[i] for i in train_idx
        ]

        val_texts = [
            texts[i] for i in val_idx
        ]

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        print("Training examples:", len(train_idx))
        print("Validation examples:", len(val_idx))
        print(
            "Training positives:",
            int(train_labels.sum()),
        )
        print(
            "Validation positives:",
            int(val_labels.sum()),
        )

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

        # A fresh pretrained model for every fold.
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
                device,
            )

            print(
                f"Average training loss: {loss:.4f}"
            )

        # Predict held-out examples.
        val_probs = predict_probabilities(
            model,
            val_loader,
            device,
        )

        val_preds = (
            val_probs >= 0.50
        ).astype(int)

        oof_probabilities[val_idx] = val_probs

        oof_folds[val_idx] = fold

        metrics = calculate_metrics(
            val_labels,
            val_preds,
        )

        metrics["fold"] = fold
        metrics["train_size"] = len(train_idx)
        metrics["validation_size"] = len(val_idx)

        fold_metrics.append(metrics)

        print("\nFold metrics:")
        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        # Release memory between folds.
        del model
        del optimizer
        del scheduler
        del train_loader
        del val_loader

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # FINAL OUT-OF-FOLD RESULTS
    # --------------------------------------------------------

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            "Some tweets have no out-of-fold prediction."
        )

    if (oof_folds < 1).any():
        raise RuntimeError(
            "Some tweets have no validation fold."
        )

    predictions = (
        oof_probabilities >= 0.50
    ).astype(int)

    overall_metrics = calculate_metrics(
        labels,
        predictions,
    )

    print("\n" + "=" * 65)
    print("DISTILBERT OUT-OF-FOLD RESULTS")
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

    print("\nPositive precision:",
          round(overall_metrics["precision"], 4))

    print("Positive recall:",
          round(overall_metrics["recall"], 4))

    print("Positive F1:",
          round(overall_metrics["f1"], 4))

    print("Accuracy:",
          round(overall_metrics["accuracy"], 4))

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    output = df[
        ["post_id", "text", "target"]
    ].copy()

    output["cv_fold"] = oof_folds

    output["predicted_target"] = predictions

    output["positive_probability"] = (
        oof_probabilities
    )

    output["correct"] = (
        output["target"]
        == output["predicted_target"]
    )

    elapsed_seconds = time.time() - start_time

    summary = {
        "stage": "9",
        "model": MODEL_NAME,
        "training_source": str(INPUT_FILE),
        "training_policy": "human_verified_only",
        "random_seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "n_splits": N_SPLITS,
        "decision_threshold": 0.50,
        "device": str(device),
        "total_verified_definite": len(df),
        "positive_count": int(labels.sum()),
        "negative_count": int(
            (labels == 0).sum()
        ),
        "out_of_fold_metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "training_seconds": round(
            elapsed_seconds,
            2,
        ),
        "pseudo_labels_used": False,
        "human_labels_modified": False,
        "note": (
            "Exploratory 4-fold cross-validation. "
            "No independent test set was evaluated."
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

    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)

    print(PREDICTIONS_FILE)
    print(METRICS_FILE)

    print(
        "\nTraining time (seconds):",
        round(elapsed_seconds, 2),
    )

    print("\nOriginal human labels were NOT modified.")
    print("No pseudo-labels were used.")
    print("STAGE 9 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
