"""Data loader module for RAG verification calibration study.

This module provides utilities to download and load:
1. RAGTruth: A word-level hallucination corpus for RAG systems (ParticleMedia/RAGTruth).
2. FEVER: Fact Extraction and VERification dataset (fever, v1.0) via HuggingFace datasets.

Both datasets are processed into pandas DataFrames with a consistent schema:
- question_or_claim (str): The query prompt or factual assertion to verify.
- evidence_or_context (str): The retrieved passages or reference knowledge context.
- answer (str): The model output or candidate statement being evaluated.
- label (int): Binary truth/faithfulness label (1 = SUPPORTED/faithful, 0 = REFUTED/hallucinated).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("data_loader")

# Constants
RAGTRUTH_REPO_URL = "https://github.com/ParticleMedia/RAGTruth.git"
RAGTRUTH_RAW_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
RAGTRUTH_FILES = ("source_info.jsonl", "response.jsonl")

# Standard column schema across all verification datasets
CORE_SCHEMA_COLUMNS = [
    "question_or_claim",
    "evidence_or_context",
    "answer",
    "label",
]


def download_ragtruth(
    dest_dir: Union[str, Path] = "data/ragtruth",
    method: str = "download",
    force: bool = False,
) -> Path:
    """Download or clone the RAGTruth dataset from the official GitHub repository.

    Args:
        dest_dir: Destination directory path where files will be stored.
        method: Download approach. Options:
            - 'download': Fetch source_info.jsonl and response.jsonl directly via HTTPS (recommended, fast).
            - 'clone': Use git clone to pull the entire repository.
            - 'auto': Try HTTPS download first, fall back to git clone if needed.
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory containing RAGTruth data files.

    Raises:
        RuntimeError: If downloading or cloning fails.
    """
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    files_present = all((dest_path / f).exists() for f in RAGTRUTH_FILES)
    if files_present and not force:
        logger.info("RAGTruth data files already exist in %s. Skipping download.", dest_path)
        return dest_path

    if method in ("download", "auto"):
        try:
            logger.info("Downloading RAGTruth dataset files from GitHub raw...")
            for filename in RAGTRUTH_FILES:
                file_dest = dest_path / filename
                if file_dest.exists() and not force:
                    continue
                file_url = f"{RAGTRUTH_RAW_BASE}/{filename}"
                logger.info("Fetching %s -> %s", file_url, file_dest)
                req = urllib.request.Request(
                    file_url,
                    headers={"User-Agent": "RAG-Verification-Research/1.0"}
                )
                with urllib.request.urlopen(req) as response, open(file_dest, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                logger.info(
                    "Successfully downloaded %s (%.2f MB)",
                    filename,
                    file_dest.stat().st_size / (1024 * 1024),
                )
            return dest_path
        except (urllib.error.URLError, OSError) as exc:
            if method == "auto":
                logger.warning("Direct HTTPS download failed (%s). Falling back to git clone...", exc)
            else:
                raise RuntimeError(f"Failed to download RAGTruth dataset via HTTPS: {exc}") from exc

    # Clone via git
    if method in ("clone", "auto"):
        try:
            clone_target = dest_path / "_git_clone"
            logger.info("Cloning RAGTruth repository from %s into %s...", RAGTRUTH_REPO_URL, clone_target)
            if clone_target.exists():
                shutil.rmtree(clone_target, ignore_errors=True)

            cmd = ["git", "clone", "--depth", "1", RAGTRUTH_REPO_URL, str(clone_target)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Move dataset files from cloned repo to dest_path
            dataset_dir = clone_target / "dataset"
            if not dataset_dir.exists():
                raise FileNotFoundError(f"dataset/ folder not found inside cloned repo at {clone_target}")

            for filename in RAGTRUTH_FILES:
                src_file = dataset_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, dest_path / filename)

            # Clean up temporary git clone folder to save space
            shutil.rmtree(clone_target, ignore_errors=True)
            logger.info("Successfully copied dataset files to %s and cleaned up clone.", dest_path)
            return dest_path
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"Failed to clone RAGTruth repository: {exc}") from exc

    raise ValueError(f"Invalid download method '{method}'. Choose 'download', 'clone', or 'auto'.")


def load_ragtruth(
    data_dir: Union[str, Path] = "data/ragtruth",
    split: Optional[str] = None,
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    max_samples: Optional[int] = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the RAGTruth dataset and map it into the standard verification schema.

    In RAGTruth:
    - For QA: question comes from source_info['question'], evidence from source_info['passages'].
    - For Summary: question is the task prompt, evidence is the article text in source_info.
    - For Data2txt: question is the prompt, evidence is formatted attributes from source_info dict.
    - Answer: model generated text from response['response'].
    - Label: 1 (SUPPORTED/faithful) if response has 0 hallucination spans, 0 (HALLUCINATED) if > 0.

    Args:
        data_dir: Path to directory containing source_info.jsonl and response.jsonl.
        split: Optional filter for split ('train' or 'test').
        task_type: Optional filter for task type ('QA', 'Summary', 'Data2txt').
        model: Optional filter for LLM model name (e.g., 'gpt-4-0613', 'mistral-7B-instruct').
        max_samples: Optional maximum number of rows to return.
        download_if_missing: If True, automatically downloads files if not present.

    Returns:
        pd.DataFrame: Formatted DataFrame adhering to CORE_SCHEMA_COLUMNS.

    Raises:
        FileNotFoundError: If data files do not exist and download_if_missing is False.
    """
    path = Path(data_dir).resolve()
    source_file = path / "source_info.jsonl"
    resp_file = path / "response.jsonl"

    if (not source_file.exists() or not resp_file.exists()) and download_if_missing:
        download_ragtruth(dest_dir=path, method="auto")

    if not source_file.exists():
        raise FileNotFoundError(f"RAGTruth source file not found at {source_file}")
    if not resp_file.exists():
        raise FileNotFoundError(f"RAGTruth response file not found at {resp_file}")

    # 1. Load source contexts into memory keyed by source_id
    logger.info("Loading RAGTruth source context items from %s...", source_file)
    sources: Dict[str, Dict[str, Any]] = {}
    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sources[str(item["source_id"])] = item

    # 2. Parse responses and merge with corresponding sources
    logger.info("Loading RAGTruth responses from %s...", resp_file)
    records: List[Dict[str, Any]] = []

    with open(resp_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            resp = json.loads(line)

            # Apply filters early for efficiency
            if split and resp.get("split") != split:
                continue
            if model and resp.get("model") != model:
                continue

            source_id = str(resp.get("source_id"))
            source_data = sources.get(source_id)
            if not source_data:
                logger.warning("Source id %s not found in sources mapping. Skipping.", source_id)
                continue

            current_task = source_data.get("task_type", "")
            if task_type and current_task != task_type:
                continue

            # Extract question/claim and evidence/context based on task_type
            raw_source_info = source_data.get("source_info", "")
            prompt_text = source_data.get("prompt", "")

            if current_task == "QA":
                if isinstance(raw_source_info, dict):
                    question_claim = raw_source_info.get("question", prompt_text)
                    evidence_context = raw_source_info.get("passages", "")
                else:
                    question_claim = prompt_text
                    evidence_context = str(raw_source_info)
            elif current_task == "Summary":
                question_claim = prompt_text if prompt_text else "Summarize the provided document."
                evidence_context = raw_source_info if isinstance(raw_source_info, str) else str(raw_source_info)
            elif current_task == "Data2txt":
                question_claim = prompt_text if prompt_text else "Describe the entity based on provided data."
                if isinstance(raw_source_info, dict):
                    # Format dict nicely as readable attributes
                    evidence_context = json.dumps(raw_source_info, indent=2)
                else:
                    evidence_context = str(raw_source_info)
            else:
                question_claim = prompt_text
                evidence_context = str(raw_source_info)

            answer_text = resp.get("response", "")
            hallucination_spans = resp.get("labels", [])
            # 1 = Supported (faithful, 0 hallucination spans), 0 = Refuted/Hallucinated
            is_faithful = 1 if len(hallucination_spans) == 0 else 0
            raw_label = "SUPPORTED" if is_faithful == 1 else "HALLUCINATED"

            record = {
                # Core standardized schema columns
                "question_or_claim": str(question_claim).strip(),
                "evidence_or_context": str(evidence_context).strip(),
                "answer": str(answer_text).strip(),
                "label": int(is_faithful),
                # Convenience aliases for research scripts
                "question/claim": str(question_claim).strip(),
                "evidence/context": str(evidence_context).strip(),
                "question": str(question_claim).strip(),
                "evidence": str(evidence_context).strip(),
                # Dataset-specific metadata
                "raw_label": raw_label,
                "split": resp.get("split", ""),
                "task_type": current_task,
                "model": resp.get("model", ""),
                "temperature": resp.get("temperature", None),
                "quality": resp.get("quality", ""),
                "hallucination_spans": hallucination_spans,
                "num_hallucinations": len(hallucination_spans),
                "source_id": source_id,
                "response_id": resp.get("id", ""),
                "dataset": "ragtruth",
            }
            records.append(record)

            if max_samples and len(records) >= max_samples:
                break

    df = pd.DataFrame(records)
    logger.info("Loaded RAGTruth DataFrame with %d rows. Columns: %s", len(df), list(df.columns))
    return df


def load_fever(
    dataset_name: str = "fever",
    subset: str = "v1.0",
    split: str = "train",
    include_nei: bool = True,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load the FEVER dataset using HuggingFace datasets library into the standard schema.

    In FEVER:
    - question_or_claim: The claim statement under verification.
    - evidence_or_context: Wikipedia page title and sentence index citation (or empty if NEI).
    - answer: The claim statement being evaluated.
    - label: 1 if 'SUPPORTS', 0 if 'REFUTES' or 'NOT ENOUGH INFO'.

    Args:
        dataset_name: HuggingFace dataset name (defaults to 'fever').
        subset: Configuration subset (defaults to 'v1.0').
        split: Dataset split ('train', 'labelled_dev', 'paper_dev', 'paper_test', etc.).
        include_nei: If False, filters out 'NOT ENOUGH INFO' examples.
        max_samples: Optional maximum number of rows to return.
        cache_dir: Optional custom HuggingFace cache directory.

    Returns:
        pd.DataFrame: Formatted DataFrame adhering to CORE_SCHEMA_COLUMNS.

    Raises:
        ImportError: If huggingface datasets is not installed.
        RuntimeError: If loading from HuggingFace Hub fails.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required to load FEVER. "
            "Please install it using: pip install \"datasets<4.0.0\""
        ) from exc

    logger.info("Loading FEVER dataset from HuggingFace (name='%s', subset='%s', split='%s')...", dataset_name, subset, split)

    # Note: Modern Hugging Face Hub addresses datasets via 'fever/fever' or 'fever'.
    # We attempt both candidates to ensure high compatibility across environments.
    dataset = None
    last_err: Optional[Exception] = None
    candidate_names = [dataset_name]
    if "/" not in dataset_name:
        candidate_names.append(f"{dataset_name}/{dataset_name}")

    for candidate in candidate_names:
        try:
            logger.info("Attempting to load FEVER with identifier '%s'...", candidate)
            dataset = load_dataset(
                candidate,
                subset,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            break
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            logger.warning("Failed to load FEVER with '%s': %s", candidate, exc)

    if dataset is None:
        raise RuntimeError(
            f"Failed to load FEVER dataset from HuggingFace Hub. Error: {last_err}. "
            "Note: HuggingFace 'datasets<4.0.0' is recommended for the legacy FEVER loading script."
        ) from last_err

    # Convert to pandas DataFrame
    logger.info("Converting HuggingFace dataset to pandas DataFrame...")
    raw_df = pd.DataFrame(dataset)

    # Filter out NOT ENOUGH INFO if requested
    if not include_nei:
        raw_df = raw_df[raw_df["label"] != "NOT ENOUGH INFO"].reset_index(drop=True)

    if max_samples:
        raw_df = raw_df.head(max_samples)

    records: List[Dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        claim_text = str(row.get("claim", "")).strip()
        raw_label = str(row.get("label", "")).strip().upper()

        # Format evidence citation
        wiki_url = row.get("evidence_wiki_url")
        sentence_id = row.get("evidence_sentence_id")

        if wiki_url and pd.notna(wiki_url) and str(wiki_url).strip() and str(wiki_url) != "None":
            clean_title = str(wiki_url).replace("_", " ").strip()
            if sentence_id is not None and pd.notna(sentence_id) and int(sentence_id) >= 0:
                evidence_text = f"Wikipedia: {clean_title} (Sentence {int(sentence_id)})"
            else:
                evidence_text = f"Wikipedia: {clean_title}"
        else:
            evidence_text = "No evidence provided."

        # Map label: 1 = SUPPORTS, 0 = REFUTES / NOT ENOUGH INFO
        binary_label = 1 if raw_label == "SUPPORTS" else 0

        record = {
            # Core standardized schema columns
            "question_or_claim": claim_text,
            "evidence_or_context": evidence_text,
            "answer": claim_text,
            "label": int(binary_label),
            # Convenience aliases for research scripts
            "question/claim": claim_text,
            "evidence/context": evidence_text,
            "question": claim_text,
            "evidence": evidence_text,
            # Dataset-specific metadata
            "raw_label": raw_label,
            "claim_id": row.get("id"),
            "evidence_id": row.get("evidence_id"),
            "evidence_wiki_url": wiki_url,
            "evidence_sentence_id": sentence_id,
            "dataset": "fever",
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("Loaded FEVER DataFrame with %d rows. Columns: %s", len(df), list(df.columns))
    return df


def load_verification_dataset(
    name: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience factory function to load a verification dataset by name.

    Args:
        name: Name of dataset ('ragtruth' or 'fever').
        **kwargs: Additional parameters passed to load_ragtruth or load_fever.

    Returns:
        pd.DataFrame: Formatted DataFrame adhering to CORE_SCHEMA_COLUMNS.

    Raises:
        ValueError: If dataset name is unrecognized.
    """
    normalized = name.strip().lower()
    if normalized in ("ragtruth", "rag_truth"):
        return load_ragtruth(**kwargs)
    elif normalized == "fever":
        return load_fever(**kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: 'ragtruth', 'fever'")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing src/data_loader.py")
    print("=" * 70)

    # 1. Test RAGTruth Loader
    print("\n--- 1. Testing RAGTruth Loader (5 samples) ---")
    df_ragtruth = load_ragtruth(max_samples=5)
    print(f"RAGTruth shape: {df_ragtruth.shape}")
    print("Columns:", list(df_ragtruth.columns))
    print("\nSample Row 0:")
    for col in CORE_SCHEMA_COLUMNS + ["raw_label", "task_type", "model"]:
        val = str(df_ragtruth.iloc[0][col])
        if len(val) > 120:
            val = val[:117] + "..."
        print(f"  {col}: {val}")

    # 2. Test FEVER Loader
    print("\n--- 2. Testing FEVER Loader (5 samples) ---")
    df_fever = load_fever(split="train", max_samples=5)
    print(f"FEVER shape: {df_fever.shape}")
    print("Columns:", list(df_fever.columns))
    print("\nSample Row 0:")
    for col in CORE_SCHEMA_COLUMNS + ["raw_label"]:
        val = str(df_fever.iloc[0][col])
        if len(val) > 120:
            val = val[:117] + "..."
        print(f"  {col}: {val}")

    # 3. Check Schema Consistency
    print("\n--- 3. Schema Consistency Check ---")
    ragtruth_has_core = all(c in df_ragtruth.columns for c in CORE_SCHEMA_COLUMNS)
    fever_has_core = all(c in df_fever.columns for c in CORE_SCHEMA_COLUMNS)
    print(f"RAGTruth contains all core columns: {ragtruth_has_core}")
    print(f"FEVER contains all core columns:    {fever_has_core}")
    print("\nData loader verification complete!")
