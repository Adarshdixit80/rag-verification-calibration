"""Verifier Model Training Pipeline for RAG Verification Calibration.

Target Publication: IEEE Conference / Transactions on AI / NLP.
Architecture: DeBERTa-v3-base (microsoft/deberta-v3-base) for binary sequence classification.

This module implements:
1. Pretrained DeBERTa-v3 loading for binary fact verification (SUPPORTED vs. REFUTED).
2. Data preparation & cross-encoder pair tokenization: [CLS] Evidence [SEP] Answer [SEP].
3. Stratified 80/20 train/validation partitioning with random seed control (seed=42).
4. HuggingFace Trainer configuration with comprehensive evaluation metrics (Accuracy, F1, Precision, Recall).
5. Model checkpointing, saving the top-performing model to results/model_checkpoint/.
6. Standalone verification / forward-pass sanity check for local validation before GPU training.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import transformers

# Sanitize environment variables on Windows to prevent ValueError (>32767 chars)
for _env_k in list(os.environ.keys()):
    if len(os.environ[_env_k]) > 30000:
        os.environ[_env_k] = os.environ[_env_k][:30000]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("train_verifier")

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "microsoft/deberta-v3-base"
DEFAULT_CHECKPOINT_DIR = "results/model_checkpoint"
DEFAULT_OUTPUT_DIR = "results/checkpoints"
DEFAULT_LOGS_DIR = "results/raw_logs"

LABEL2ID: Dict[str, int] = {"REFUTED": 0, "SUPPORTED": 1}
ID2LABEL: Dict[int, str] = {0: "REFUTED", 1: "SUPPORTED"}

REQUIRED_COLUMNS = [
    "question_or_claim",
    "evidence_or_context",
    "answer",
    "label",
]


# ---------------------------------------------------------------------------
# 1. Model & Tokenizer Loader
# ---------------------------------------------------------------------------

def load_verifier_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_ID,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pretrained DeBERTa-v3-base model and tokenizer for binary verification.

    Args:
        model_name: HuggingFace model hub identifier.
        num_labels: Number of classification targets (2 for binary verification).
        device: Target execution device (CPU or CUDA).

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading tokenizer from: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading sequence classification model from: %s (num_labels=%d)", model_name, num_labels)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    if device is not None:
        model.to(device)

    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Data Preparation & Tokenization
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    dataset_name: str = "",
) -> Dataset:
    """Preprocess and tokenize a DataFrame into a Hugging Face Dataset.

    The sequence-pair cross-encoder format pairs the evidence passage (premise)
    with the candidate answer/claim (hypothesis):
        [CLS] Evidence Context [SEP] Candidate Answer / Claim [SEP]

    Args:
        df: Input DataFrame containing required verification columns.
        tokenizer: PreTrained Hugging Face tokenizer instance.
        max_length: Maximum sequence truncation length.
        dataset_name: Optional name tag (e.g. 'Train', 'Validation') for logging clarity.

    Returns:
        Hugging Face Dataset with input_ids, attention_mask, and labels tensors.

    Raises:
        ValueError: If mandatory schema columns are missing.
    """
    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        # Check for column aliases
        alias_map = {
            "question_or_claim": ["question/claim", "question", "claim"],
            "evidence_or_context": ["evidence/context", "evidence", "context"],
            "answer": ["response"],
            "label": ["labels"],
        }
        for req in missing_cols:
            found = False
            for alias in alias_map.get(req, []):
                if alias in df.columns:
                    df[req] = df[alias]
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"DataFrame is missing required column: '{req}'. "
                    f"Present columns: {list(df.columns)}"
                )

    # Extract sequences
    # Text A (Premise): Retrieved evidence context
    # Text B (Hypothesis): Candidate answer or claim
    evidence_texts: list[str] = [str(x).strip() for x in df["evidence_or_context"]]
    answer_texts: list[str] = []
    for _, row in df.iterrows():
        ans = str(row.get("answer", "")).strip()
        qc = str(row.get("question_or_claim", "")).strip()
        # Fall back to question/claim if answer string is empty
        answer_texts.append(ans if ans else qc)

    labels: list[int] = [int(lbl) for lbl in df["label"]]

    name_label = f" ({dataset_name})" if dataset_name else ""
    logger.info("Tokenizing %d sequence pairs%s with max_length=%d...", len(df), name_label, max_length)
    tokenized = tokenizer(
        text=evidence_texts,
        text_pair=answer_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # Return Python lists for Dataset conversion
    )

    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

    if "token_type_ids" in tokenized:
        dataset_dict["token_type_ids"] = tokenized["token_type_ids"]

    hf_dataset = Dataset.from_dict(dataset_dict)
    logger.info("Prepared%s Dataset with %d examples.", name_label, len(hf_dataset))
    return hf_dataset


# ---------------------------------------------------------------------------
# 3. Evaluation Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute verification evaluation metrics for HuggingFace Trainer.

    Metrics calculated:
    - Accuracy: Overall classification correctness.
    - F1 Score (binary): Harmonic mean of precision and recall for positive (SUPPORTED) class.
    - F1 Score (macro): Unweighted mean of F1 across both classes (REFUTED and SUPPORTED),
      penalizing majority-class collapse.
    - Precision (binary): Ratio of true supported claims among all claims flagged as supported.
    - Recall (binary): Ratio of supported claims correctly verified.
    - Pred Positive Ratio: Fraction of validation predictions predicted as Class 1.

    Args:
        eval_pred: EvalPrediction namedtuple containing (predictions, label_ids).

    Returns:
        Dictionary of computed metric values.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, predictions)
    f1_bin = f1_score(labels, predictions, average="binary", zero_division=0)
    f1_mac = f1_score(labels, predictions, average="macro", zero_division=0)
    prec_bin = precision_score(labels, predictions, average="binary", zero_division=0)
    rec_bin = recall_score(labels, predictions, average="binary", zero_division=0)
    pct_pred_1 = float(np.mean(predictions == 1))

    return {
        "accuracy": float(acc),
        "f1": float(f1_bin),
        "f1_macro": float(f1_mac),
        "precision": float(prec_bin),
        "recall": float(rec_bin),
        "pred_positive_ratio": pct_pred_1,
    }


# ---------------------------------------------------------------------------
# 4. Custom Weighted Trainer (Handling Class Imbalance)
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Custom HuggingFace Trainer with class-weighted CrossEntropyLoss.

    Mitigates majority-class collapse when fine-tuning on imbalanced verification
    datasets (e.g. 72% SUPPORTED vs. 28% REFUTED) by scaling loss inversely proportional
    to class frequencies: weight_i = total_samples / (num_classes * count_i).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize WeightedTrainer with optional precomputed class weights tensor."""
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self._logged_compute_loss: bool = False
        self._loss_call_count: int = 0

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Union[torch.Tensor, int]] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute class-weighted cross-entropy loss against ground-truth labels.

        Compatible with transformers >= 5.x Trainer signature:
        (self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs).
        """
        self._loss_call_count += 1

        # Debug print: confirms compute_loss override is executed during training
        if not self._logged_compute_loss:
            print(
                f"\n[WeightedTrainer.compute_loss] OVERRIDE VERIFIED ACTIVE during training!\n"
                f"  Class weights tensor: {self.class_weights}\n"
                f"  num_items_in_batch: {num_items_in_batch}\n"
            )
            self._logged_compute_loss = True
        elif self._loss_call_count <= 5 or self._loss_call_count % 50 == 0:
            print(
                f"[WeightedTrainer.compute_loss] Step/Call #{self._loss_call_count}: "
                f"compute_loss() executed | class_weights={self.class_weights.tolist() if self.class_weights is not None else None}"
            )

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        num_classes = logits.shape[-1]
        if labels is not None and self.class_weights is not None:
            # Move class weights tensor to the active model device and dtype
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
        elif labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
        else:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss

        # Ensure model output object contains the weighted loss
        if isinstance(outputs, dict):
            outputs["loss"] = loss
        elif hasattr(outputs, "loss"):
            outputs.loss = loss

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# 5. Training Pipeline Setup
# ---------------------------------------------------------------------------

def setup_training(
    df: Optional[pd.DataFrame] = None,
    model_name: str = DEFAULT_MODEL_ID,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    best_model_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    val_split: float = 0.2,
    seed: int = 42,
    max_length: int = 512,
    logging_steps: int = 10,
    train_df: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> Tuple[WeightedTrainer, PreTrainedModel, PreTrainedTokenizerBase]:
    """Construct and configure the HuggingFace Trainer instance for verification fine-tuning.

    Args:
        df: Input DataFrame with verification training examples (or train_df).
        model_name: Pretrained model hub path.
        output_dir: Intermediate checkpoints directory.
        best_model_dir: Directory where top-performing model will be persisted.
        batch_size: Per-device batch size for training and evaluation.
        epochs: Number of complete training epochs.
        learning_rate: Initial peak learning rate for AdamW optimizer.
        weight_decay: L2 regularization penalty.
        warmup_steps: Number of linear learning rate warmup steps.
        val_split: Proportion of data allocated for validation (default 0.20 = 80/20 split).
        seed: Random seed for deterministic shuffling and reproducibility.
        max_length: Maximum sequence truncation length.
        logging_steps: Frequency of training loss logging.
        train_df: Optional alias for df.

    Returns:
        Tuple of (Trainer, Model, Tokenizer).
    """
    # 0. Handle DataFrame input parameter and print length at the very start of setup_training()
    if df is None and train_df is not None:
        df = train_df
    elif df is not None and train_df is None:
        train_df = df

    input_count = len(train_df) if train_df is not None else (len(df) if df is not None else 0)
    print(f"\n[setup_training] len(train_df) at start of setup_training: {input_count} samples (before train/val split)")
    logger.info("setup_training received %d samples in train_df before train/val split.", input_count)

    set_seed(seed)
    output_path = Path(output_dir).resolve()
    best_model_path = Path(best_model_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path.mkdir(parents=True, exist_ok=True)

    # 1. Train / Validation Split (80 / 20)
    logger.info("Partitioning dataset into 80/20 train/validation split (val_split=%.2f, seed=%d)...", val_split, seed)
    stratify_col = df["label"] if df["label"].nunique() > 1 else None
    train_split_df, val_split_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=stratify_col,
    )
    logger.info("Split sizes: %d training samples, %d validation samples.", len(train_split_df), len(val_split_df))
    print(f"[setup_training] Train/Val split: {len(train_split_df)} train samples, {len(val_split_df)} validation samples.")

    # 2. Compute Class Weights for Imbalance Mitigation
    train_labels = np.array(train_split_df["label"].tolist())
    unique_classes = np.unique(train_labels)
    if len(unique_classes) == 2:
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels,
        )
    else:
        class_weights_arr = np.ones(2, dtype=np.float32)

    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

    # Verification print of computed class weights
    print(
        f"[WeightedTrainer] Computed Class Weights (balanced): "
        f"Class 0 (REFUTED): {class_weights_arr[0]:.4f}, "
        f"Class 1 (SUPPORTED): {class_weights_arr[1]:.4f}"
    )
    logger.info("Class weights tensor: %s", class_weights_tensor)

    # 3. Model & Tokenizer Initialization
    model, tokenizer = load_verifier_model_and_tokenizer(model_name=model_name)
    # Explicitly move model to GPU if CUDA is available before creating Trainer
    model = model.to('cuda') if torch.cuda.is_available() else model

    # 4. Tokenize Datasets
    train_dataset = prepare_data(train_split_df, tokenizer, max_length=max_length, dataset_name="Train")
    val_dataset = prepare_data(val_split_df, tokenizer, max_length=max_length, dataset_name="Validation")

    # 5. Training Arguments
    # Dynamically select 'eval_strategy' (transformers >= 4.41) or 'evaluation_strategy' (< 4.41)
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())
    eval_param_key = "eval_strategy" if "eval_strategy" in valid_params else "evaluation_strategy"

    # Precision & Stability Configuration for DeBERTa-v3:
    # DeBERTa-v3 is known to produce NaN gradients under fp16 mixed precision.
    # Check if bf16 is supported on the GPU; if so, enable bf16, otherwise run in standard fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        logger.info("GPU supports bf16: using bf16=True for numerical stability (fp16=False).")
    else:
        logger.info("bf16 not supported or CUDA unavailable: running with fp16=False, bf16=False.")

    # Only include well-established, universally compatible parameters
    # Note: explicitly ensuring no CPU-forcing parameters (like no_cuda=True or use_cpu=True)
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": 1.0,
        "fp16": False,
        "bf16": use_bf16,
        eval_param_key: "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "seed": seed,
    }

    # Merge any extra valid TrainingArguments parameters from kwargs
    for k, v in kwargs.items():
        if k in valid_params:
            training_kwargs[k] = v

    # Ensure no parameters force CPU usage
    for cpu_flag in ("no_cuda", "use_cpu"):
        training_kwargs.pop(cpu_flag, None)

    try:
        training_args = TrainingArguments(**training_kwargs)
    except Exception as exc:  # Catch TypeError or version incompatibilities and report exact signature
        valid_keys = sorted(list(sig.parameters.keys()))
        version_str = getattr(transformers, "__version__", "unknown")
        print(f"\n[ERROR] TrainingArguments initialization failed: {exc}")
        print(f"[DEBUG] Installed transformers version: {version_str}")
        print(f"[DEBUG] Valid parameters accepted by TrainingArguments ({len(valid_keys)} total):")
        for param_name in valid_keys:
            print(f"  - {param_name}")
        logger.error(
            "TrainingArguments initialization failed: %s. Transformers version: %s. Valid parameters: %s",
            exc,
            version_str,
            valid_keys,
        )
        raise exc

    # 6. Trainer Construction (WeightedTrainer)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    logger.info("WeightedTrainer successfully configured and initialized with class weights.")
    print(f"Model is on device: {next(model.parameters()).device}")
    return trainer, model, tokenizer


def train_verifier(
    df: pd.DataFrame,
    best_model_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute complete training cycle and persist best model checkpoint.

    NOTE: GPU acceleration (e.g. Google Colab) is strongly advised for full runs.

    Args:
        df: Training DataFrame.
        best_model_dir: Target directory to save the best model weights.
        **kwargs: Arguments forwarded to setup_training.

    Returns:
        Evaluation metrics summary dictionary.
    """
    trainer, model, tokenizer = setup_training(df, best_model_dir=best_model_dir, **kwargs)

    logger.info("Beginning DeBERTa-v3 verifier model training...")
    train_result = trainer.train()

    logger.info("Training complete. Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info("Final Validation Metrics: %s", eval_metrics)

    # Validation Predictions & Logits Distribution Inspection
    logger.info("Analyzing validation predictions and logits distribution...")
    val_pred_output = trainer.predict(trainer.eval_dataset)
    val_logits = val_pred_output.predictions
    val_labels = val_pred_output.label_ids
    val_preds = np.argmax(val_logits, axis=-1)

    print("\n" + "=" * 80)
    print("VALIDATION LOGITS & PREDICTIONS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Total Validation Samples Evaluated: {len(val_labels)}")
    print(
        f"Ground Truth Distribution : Class 0 (REFUTED): {np.sum(val_labels == 0):4d} ({np.mean(val_labels == 0):.2%}) | "
        f"Class 1 (SUPPORTED): {np.sum(val_labels == 1):4d} ({np.mean(val_labels == 1):.2%})"
    )
    print(
        f"Predicted Distribution    : Class 0 (REFUTED): {np.sum(val_preds == 0):4d} ({np.mean(val_preds == 0):.2%}) | "
        f"Class 1 (SUPPORTED): {np.sum(val_preds == 1):4d} ({np.mean(val_preds == 1):.2%})"
    )

    if np.sum(val_preds == 0) == 0:
        print("  >> [STATUS] MAJORITY-CLASS COLLAPSE: Model predicted ONLY Class 1 (SUPPORTED) for all validation samples.")
    elif np.sum(val_preds == 1) == 0:
        print("  >> [STATUS] MINORITY-CLASS COLLAPSE: Model predicted ONLY Class 0 (REFUTED) for all validation samples.")
    else:
        print(f"  >> [STATUS] BALANCED MIX PREDICTED: Model predicted {np.sum(val_preds == 0)} REFUTED and {np.sum(val_preds == 1)} SUPPORTED.")

    print("\nSample Validation Batch Predictions (First 10 samples with raw logits):")
    val_probs = torch.softmax(torch.tensor(val_logits[:10]), dim=-1).numpy()
    for idx in range(min(10, len(val_labels))):
        gt = val_labels[idx]
        pred = val_preds[idx]
        is_correct = "CORRECT" if pred == gt else "INCORRECT"
        print(
            f"  Sample {idx:02d} | True: {ID2LABEL.get(gt, str(gt)):<9} | "
            f"Pred: {ID2LABEL.get(pred, str(pred)):<9} [{is_correct:<9}] | "
            f"Logits: [0: {val_logits[idx][0]:+.4f}, 1: {val_logits[idx][1]:+.4f}] | "
            f"P(0)={val_probs[idx][0]:.4f}, P(1)={val_probs[idx][1]:.4f}"
        )
    print("=" * 80 + "\n")

    best_path = Path(best_model_dir).resolve()
    logger.info("Saving best model and tokenizer to: %s", best_path)
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    return {
        "train_result": train_result,
        "eval_metrics": eval_metrics,
        "best_model_dir": str(best_path),
        "val_logits": val_logits,
        "val_preds": val_preds,
        "val_labels": val_labels,
    }


# ---------------------------------------------------------------------------
# 5. Architecture Sanity Check (Forward Pass Validation)
# ---------------------------------------------------------------------------

def run_sanity_check() -> bool:
    """Validate model loading, tokenization, and forward pass on 3 dummy samples.

    Ensures zero errors in tensor dimensions, token type assignments, and loss computation
    prior to submitting GPU jobs.

    Returns:
        True if all sanity checks pass successfully.
    """
    print("=" * 80)
    print("DEBERTA-V3 VERIFIER SANITY CHECK (Forward Pass Validation)")
    print("Purpose: Verify model initialization & forward pass without full GPU training")
    print(f"Target Architecture: {DEFAULT_MODEL_ID}")
    print("=" * 80)

    # 1. Create 3 representative test samples
    dummy_data = [
        {
            "question_or_claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "evidence_or_context": "Nikolaj Coster-Waldau is a Danish actor who worked with the Fox Broadcasting Company.",
            "answer": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "label": 1,  # SUPPORTED
        },
        {
            "question_or_claim": "Nikolaj Coster-Waldau worked with Warner Bros. Discovery.",
            "evidence_or_context": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company, not Warner Bros.",
            "answer": "Nikolaj Coster-Waldau was hired by Warner Bros. Discovery.",
            "label": 0,  # REFUTED (Contradiction)
        },
        {
            "question_or_claim": "When did the Danish actor land his first film role?",
            "evidence_or_context": "Photosynthesis in marine phytoplankton accounts for approximately half of Earth's oxygen.",
            "answer": "He landed his breakthrough role in 1994.",
            "label": 0,  # REFUTED (Irrelevant / Unsupported)
        },
    ]
    dummy_df = pd.DataFrame(dummy_data)
    print(f"\n[1] Created {len(dummy_df)} test verification examples.")

    # 2. Load Tokenizer & Model
    print("\n[2] Loading DeBERTa-v3 Tokenizer & Model...")
    model, tokenizer = load_verifier_model_and_tokenizer(model_name=DEFAULT_MODEL_ID)
    model.eval()

    # 3. Test prepare_data tokenization
    print("\n[3] Testing prepare_data tokenization...")
    dataset = prepare_data(dummy_df, tokenizer, max_length=128)
    print(f"    Dataset length: {len(dataset)}")
    print(f"    Feature columns: {list(dataset.features.keys())}")
    print(f"    Input IDs shape: {len(dataset['input_ids'])} sequences")

    # 4. Construct a batch for a single forward pass
    print("\n[4] Executing single forward pass...")
    input_ids = torch.tensor(dataset["input_ids"])
    attention_mask = torch.tensor(dataset["attention_mask"])
    labels = torch.tensor(dataset["labels"])

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = outputs.loss.item()
    logits = outputs.logits
    probs = torch.softmax(logits.float(), dim=-1)

    print(f"    Forward Pass Loss: {loss:.4f} (Valid Scalar Float: {not np.isnan(loss)})")
    print(f"    Logits Tensor Shape: {list(logits.shape)} (Expected: [{len(dummy_df)}, 2])")
    print("    Sample Prediction Probs (REFUTED vs. SUPPORTED):")
    for i, p in enumerate(probs):
        refuted_prob = p[0].item()
        supported_prob = p[1].item()
        true_lbl = ID2LABEL[labels[i].item()]
        print(f"      Sample {i} (Ground Truth: {true_lbl:<9}): P(REFUTED)={refuted_prob:.3f}, P(SUPPORTED)={supported_prob:.3f}")

    # 5. Validate assertions
    assert not np.isnan(loss), "Sanity check failed: Loss is NaN"
    assert logits.shape == (len(dummy_df), 2), f"Unexpected logits shape: {logits.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(len(dummy_df), dtype=torch.float32), atol=1e-4), "Softmax probabilities do not sum to 1.0"

    print("\n[5] Metrics Computation Test:")
    dummy_eval_pred = EvalPrediction(
        predictions=logits.float().numpy(),
        label_ids=labels.numpy(),
    )
    metrics = compute_metrics(dummy_eval_pred)
    for k, v in metrics.items():
        print(f"    {k:<20}: {v:.4f}")

    print("\n[6] Testing WeightedTrainer compute_loss() override execution...")
    dummy_weights = torch.tensor([1.8182, 0.6897], dtype=torch.float32)
    sanity_out = Path("./tmp_sanity_check").resolve()
    sanity_out.mkdir(parents=True, exist_ok=True)
    try:
        training_args = TrainingArguments(
            output_dir=str(sanity_out),
            per_device_train_batch_size=len(dummy_df),
            use_cpu=True,
        )
        weighted_trainer = WeightedTrainer(
            model=model,
            args=training_args,
            class_weights=dummy_weights,
        )
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        dummy_loss, dummy_outputs = weighted_trainer.compute_loss(
            model=model,
            inputs=dummy_inputs,
            return_outputs=True,
        )
        print(f"    Weighted Loss Value: {dummy_loss.item():.4f}")
        assert dummy_loss is not None and not torch.isnan(dummy_loss), "Weighted loss is NaN"
        print("    WeightedTrainer compute_loss() override successfully verified.")
    finally:
        import shutil
        shutil.rmtree(str(sanity_out), ignore_errors=True)

    print("\n" + "=" * 80)
    print("SANITY CHECK PASSED: Architecture, Tokenization, and Forward Pass Validated!")
    print("Status: Model structure is ready for Colab GPU execution.")
    print("=" * 80)
    return True


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBERTa-v3 Verifier Training for RAG Calibration")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        default=True,
        help="Run lightweight model & forward pass sanity check without full training (default: True)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute full training pipeline (recommended only with GPU)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max samples to load from dataset for training (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    args, unknown = parser.parse_known_args()

    if args.train:
        logger.info("Full training requested via CLI flag --train.")
        from src.data_loader import load_verification_dataset
        df_train = load_verification_dataset("ragtruth", max_samples=args.max_samples)
        train_verifier(
            df_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
    else:
        # Default behavior: Validate architecture and single forward pass
        run_sanity_check()
