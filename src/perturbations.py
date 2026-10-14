"""Evidence Perturbation Engine for RAG Verification Calibration.

This module implements controlled evidence perturbation operators to evaluate the
robustness, calibration error, and reliability of Retrieval-Augmented Generation (RAG)
verification systems under out-of-distribution (OOD) and corrupted evidence distributions.

Operators:
1. delete_evidence: Incomplete evidence scenarios (sentence truncation/pruning).
2. contradict_evidence: Factual distortion via key entity/date/numerical swaps.
3. make_irrelevant: Distractor injection via unrelated cross-domain assertions.

Designed for reproducible peer-reviewed research (IEEE conference publication standards).
All operators accept an explicit `seed` parameter for deterministic replication.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("perturbations")

# ---------------------------------------------------------------------------
# Lexical & Domain Knowledge Banks for Rule-Based Perturbations
# ---------------------------------------------------------------------------

# Pool of neutral, cross-domain factual distractors for `make_irrelevant`
DISTRACTOR_SENTENCES: List[str] = [
    "Photosynthesis in marine phytoplankton accounts for approximately half of Earth's oxygen production.",
    "The Kuiper Belt extends from the orbit of Neptune to approximately fifty astronomical units from the Sun.",
    "Basaltic lava flows typically exhibit low viscosity due to their relatively low silica content.",
    "Gothic cathedrals often employ ribbed vaults and flying buttresses to distribute structural loads.",
    "The Mariana Trench reaches a maximum known depth of approximately eleven kilometers at the Challenger Deep.",
    "Certain cephalopods can rapidly change skin texture and coloration using specialized chromatophores.",
    "Glacial erratics are large boulders transported and deposited by moving ice sheets far from their native bedrock.",
    "Titanium alloys are widely utilized in aerospace engineering due to their high strength-to-weight ratio.",
    "Geothermal energy harnesses subterranean hydrothermal reservoirs to drive steam turbines.",
    "The rings of Saturn consist predominantly of water ice particles ranging in size from micrometers to meters.",
    "Dendrochronology is the scientific method of dating tree rings to analyze past environmental conditions.",
    "Bioluminescence in deep-sea organisms is predominantly produced by luciferin-luciferase enzymatic reactions.",
]

# Entity candidate banks for plausible entity substitution in `contradict_evidence`
PERSON_CANDIDATES: List[str] = [
    "Mads Mikkelsen",
    "Pedro Pascal",
    "Viggo Mortensen",
    "Benedict Cumberbatch",
    "Javier Bardem",
    "Willem Dafoe",
    "Michael Fassbender",
    "Christoph Waltz",
    "Stellan Skarsgard",
]

ORGANIZATION_CANDIDATES: List[str] = [
    "Warner Bros. Pictures",
    "Paramount Global",
    "NBCUniversal Media",
    "Sony Pictures Entertainment",
    "Columbia Broadcasting System",
    "Walt Disney Studios",
    "Metro-Goldwyn-Mayer",
]

NATIONALITY_CANDIDATES: Dict[str, List[str]] = {
    "Danish": ["Swedish", "Norwegian", "German", "French", "Dutch"],
    "Swedish": ["Danish", "Finnish", "Norwegian", "German"],
    "Norwegian": ["Danish", "Swedish", "Icelandic", "British"],
    "German": ["French", "Austrian", "Polish", "Italian"],
    "French": ["Italian", "Spanish", "German", "Belgian"],
    "Dutch": ["Belgian", "German", "Danish", "French"],
    "American": ["British", "Canadian", "Australian", "Irish"],
    "British": ["American", "French", "German", "Scottish"],
    "Israeli": ["Jordanian", "Egyptian", "Lebanese", "Cypriot"],
    "Palestinian": ["Syrian", "Jordanian", "Lebanese", "Turkish"],
    "Italian": ["Spanish", "French", "Greek", "Portuguese"],
    "Spanish": ["Portuguese", "Italian", "French", "Mexican"],
    "Japanese": ["South Korean", "Chinese", "Taiwanese", "Singaporean"],
}

MONTH_CANDIDATES: List[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

NUMBER_WORDS_MAP: Dict[str, str] = {
    "one": "five",
    "two": "seven",
    "three": "eight",
    "four": "nine",
    "five": "one",
    "six": "two",
    "seven": "eleven",
    "eight": "three",
    "nine": "four",
    "ten": "twenty",
    "fifteen": "forty",
    "twenty": "sixty",
    "seventy": "twenty",
    "hundred": "thousand",
}


# ---------------------------------------------------------------------------
# Sentence Segmentation & Helper Utilities
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Segment raw text into individual sentences using a regex boundary detector.

    This tokenizer handles standard punctuation boundaries (period, exclamation point,
    question mark) while maintaining sentence integrity without requiring heavy
    external tokenizers or downloaded model weights.

    Args:
        text: Input string to segment.

    Returns:
        List of segmented sentence strings with leading/trailing whitespace stripped.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    # Split on terminal punctuation followed by whitespace and a capitalized letter or quote
    segments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", cleaned)
    return [s.strip() for s in segments if s.strip()]


def join_sentences(sentences: Sequence[str]) -> str:
    """Recombine a sequence of sentences into a coherent text block.

    Args:
        sentences: Ordered sequence of sentence strings.

    Returns:
        Combined text string separated by single spaces.
    """
    return " ".join(s.strip() for s in sentences if s.strip())


# ---------------------------------------------------------------------------
# Perturbation Operator 1: Delete Evidence (Evidence Pruning)
# ---------------------------------------------------------------------------

def delete_evidence(
    text: str,
    severity: str = "medium",
    seed: Optional[int] = None,
) -> str:
    """Prune or truncate evidence sentences at varying severity levels.

    Simulates retrieval degradation where relevant context sentences are omitted,
    causing the verifier to evaluate claims against incomplete evidence.

    Severity Ratios:
    - 'low': Remove exactly 1 sentence.
    - 'medium': Remove approximately 30% of sentences (at least 1).
    - 'high': Remove approximately 60% of sentences (at least 1; up to all sentences).

    Args:
        text: The source context / evidence document string.
        severity: Degradation level ('low', 'medium', 'high').
        seed: Optional integer for deterministic PRNG state (reproducibility).

    Returns:
        Perturbed text with selected sentences removed.

    Raises:
        ValueError: If severity is not one of ('low', 'medium', 'high').
    """
    sentences = split_sentences(text)
    n_sents = len(sentences)

    if n_sents == 0:
        return ""
    if n_sents == 1:
        # Pruning the only available sentence yields empty evidence
        return ""

    rng = random.Random(seed)
    sev = severity.strip().lower()

    if sev == "low":
        k_delete = 1
    elif sev == "medium":
        k_delete = max(1, int(round(n_sents * 0.30)))
    elif sev == "high":
        k_delete = max(1, int(math.ceil(n_sents * 0.60)))
    else:
        raise ValueError(f"Unknown severity '{severity}'. Supported options: 'low', 'medium', 'high'.")

    # Clamp deletion count to total sentences
    k_delete = min(k_delete, n_sents)

    # Randomly select unique indices to delete while preserving relative ordering of survivors
    indices_to_delete = set(rng.sample(range(n_sents), k_delete))
    surviving_sentences = [s for idx, s in enumerate(sentences) if idx not in indices_to_delete]

    return join_sentences(surviving_sentences)


# ---------------------------------------------------------------------------
# Perturbation Operator 2: Contradict Evidence (Factual Distortion)
# ---------------------------------------------------------------------------

def _extract_fact_spans(text: str) -> List[Tuple[int, int, str, str]]:
    """Identify non-overlapping key factual spans (entities, dates, numbers) in text.

    Args:
        text: Input string to scan for facts.

    Returns:
        List of tuples: (start_offset, end_offset, fact_type, matched_text)
    """
    spans: List[Tuple[int, int, str, str]] = []

    def _has_overlap(start: int, end: int) -> bool:
        return any(s[0] < end and start < s[1] for s in spans)

    # 1. Dates & 4-digit years (e.g., 1994, 2021)
    for m in re.finditer(r"\b(1[7-9]\d\d|20\d\d)\b", text):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "year", m.group()))

    # 2. Months
    month_pattern = r"\b(?:" + "|".join(MONTH_CANDIDATES) + r")\b"
    for m in re.finditer(month_pattern, text):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "month", m.group()))

    # 3. Organizations (capitalized phrases ending with corporate/media suffixes)
    org_pattern = (
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\s+"
        r"(?:Company|Broadcasting Company|Broadcasting|Corp|Corporation|"
        r"Network|Studios|House|Court|Agency|University|Institute|LLC|Inc|Press)\b"
    )
    for m in re.finditer(org_pattern, text):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "organization", m.group()))

    # 4. Nationalities / Demonyms
    for m in re.finditer(r"\b[A-Z][a-z]+(?:ish|ch|an|ese|i)\b", text):
        word = m.group()
        if word in NATIONALITY_CANDIDATES and not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "nationality", word))

    # 5. Named Persons (multi-word capitalized names not overlapping with previous spans)
    person_pattern = r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?\b"
    for m in re.finditer(person_pattern, text):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "person", m.group()))

    # 6. General numbers & percentages
    for m in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "number", m.group()))

    # 7. Spelled-out number words
    num_words_pat = r"\b(?:" + "|".join(re.escape(w) for w in NUMBER_WORDS_MAP.keys()) + r")\b"
    for m in re.finditer(num_words_pat, text, flags=re.IGNORECASE):
        if not _has_overlap(m.start(), m.end()):
            spans.append((m.start(), m.end(), "number_word", m.group()))

    # Sort spans in ascending chronological order of appearance
    spans.sort(key=lambda s: s[0])
    return spans


def _generate_fact_replacement(
    orig_text: str,
    fact_type: str,
    rng: random.Random,
) -> str:
    """Generate a plausible but contradictory factual replacement for a detected span.

    Args:
        orig_text: Original text token to be substituted.
        fact_type: Category identifier ('year', 'month', 'person', 'organization', etc.).
        rng: Local PRNG instance for reproducible random selection.

    Returns:
        Contradictory replacement string.
    """
    if fact_type == "year":
        # Displace the year by a plausible offset (e.g. +18 or -15 years)
        offsets = [-30, -20, -15, 12, 18, 24]
        try:
            yr = int(orig_text)
            new_yr = yr + rng.choice(offsets)
            return str(new_yr)
        except ValueError:
            return "2018"

    if fact_type == "month":
        cands = [m for m in MONTH_CANDIDATES if m != orig_text]
        return rng.choice(cands)

    if fact_type == "organization":
        cands = [org for org in ORGANIZATION_CANDIDATES if org != orig_text]
        return rng.choice(cands)

    if fact_type == "nationality":
        cands = NATIONALITY_CANDIDATES.get(orig_text, ["Swedish", "Norwegian", "French", "German"])
        return rng.choice(cands)

    if fact_type == "person":
        cands = [p for p in PERSON_CANDIDATES if p != orig_text]
        return rng.choice(cands)

    if fact_type == "number":
        try:
            is_pct = orig_text.endswith("%")
            val_str = orig_text.rstrip("%")
            val = float(val_str)
            factor = rng.choice([0.2, 0.5, 2.0, 3.5])
            new_val = int(round(val * factor)) if val.is_integer() else round(val * factor, 1)
            if new_val == val:
                new_val = int(val + 5)
            return f"{new_val}%" if is_pct else str(new_val)
        except ValueError:
            return "42"

    if fact_type == "number_word":
        lower = orig_text.lower()
        rep = NUMBER_WORDS_MAP.get(lower, "seven")
        return rep.capitalize() if orig_text.istitle() else rep

    return "Alternative Source"


def contradict_evidence(
    text: str,
    severity: str = "medium",
    method: str = "rule",
    seed: Optional[int] = None,
) -> str:
    """Introduce factual contradictions by identifying and corrupting key factual assertions.

    Detects named entities (persons, organizations, nationalities), dates, and numbers,
    replacing them with plausible but strictly counterfactual substitutions.

    Severity Scaling:
    - 'low': Perturbs exactly 1 key factual anchor.
    - 'medium': Perturbs 2 factual anchors (or 50% of available facts).
    - 'high': Perturbs multiple/all identified factual anchors (or inverts polarity).

    Optional LLM Mode:
    If `method='llm'` is specified, the function queries the Gemini API using the
    `GEMINI_API_KEY` environment variable. If not set or if `method='rule'` (default),
    a deterministic, reproducible rule-based entity substitution engine is executed.

    Args:
        text: Source evidence text to corrupt.
        severity: Degree of factual corruption ('low', 'medium', 'high').
        method: Substitution approach ('rule' for deterministic regex, 'llm' for Gemini API).
        seed: Optional integer for deterministic PRNG state (reproducibility).

    Returns:
        Perturbed text with contradictory factual assertions.

    Raises:
        ValueError: If severity is unrecognized or if LLM mode is requested without API key.
    """
    sev = severity.strip().lower()
    if sev not in ("low", "medium", "high"):
        raise ValueError(f"Unknown severity '{severity}'. Supported options: 'low', 'medium', 'high'.")

    # Optional LLM path if requested by user configuration
    if method == "llm":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is missing. "
                "Set it via: export GEMINI_API_KEY='your-key-here' or use method='rule'."
            )
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            prompt = (
                f"Rewrite the following text to introduce subtle but contradictory factual claims "
                f"(severity: {sev}). Keep the overall structure and grammar identical, but change key facts "
                f"like names, dates, or numbers:\n\n{text}\n\nRewritten text:"
            )
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as exc:
            logger.warning("Gemini LLM perturbation failed (%s). Falling back to rule-based engine.", exc)

    # Deterministic Rule-Based & Regex Engine
    rng = random.Random(seed)
    spans = _extract_fact_spans(text)

    # Determine number of facts to perturb based on severity
    if sev == "low":
        k_facts = 1
    elif sev == "medium":
        k_facts = max(1, min(2, len(spans)))
    else:  # high
        k_facts = max(1, len(spans))

    if spans:
        # Sample non-overlapping fact spans
        chosen_spans = rng.sample(spans, min(k_facts, len(spans)))
        # Sort in reverse order of start offset so string modifications do not alter subsequent offsets
        chosen_spans.sort(key=lambda s: s[0], reverse=True)

        corrupted = text
        for start, end, fact_type, orig_val in chosen_spans:
            replacement = _generate_fact_replacement(orig_val, fact_type, rng)
            corrupted = corrupted[:start] + replacement + corrupted[end:]
        return corrupted

    # Fallback if no named entities or numbers detected: invert common verb polarities / negate
    negation_rules = [
        (r"\bworked with\b", "never worked with"),
        (r"\bwas\b", "was not"),
        (r"\bwere\b", "were not"),
        (r"\bhas\b", "has not"),
        (r"\bhave\b", "have not"),
        (r"\bis\b", "is not"),
        (r"\blanded\b", "failed to obtain"),
        (r"\bappeared\b", "did not appear"),
        (r"\brevealed\b", "disproved"),
    ]
    corrupted = text
    for pattern, replacement in negation_rules:
        if re.search(pattern, corrupted, flags=re.IGNORECASE):
            corrupted = re.sub(pattern, replacement, corrupted, count=1, flags=re.IGNORECASE)
            break

    return corrupted


# ---------------------------------------------------------------------------
# Perturbation Operator 3: Make Irrelevant (Distractor Injection)
# ---------------------------------------------------------------------------

def make_irrelevant(
    text: str,
    severity: str = "medium",
    seed: Optional[int] = None,
) -> str:
    """Inject cross-domain unrelated distractor sentences into the evidence passage.

    Simulates retrieval distraction (retriever noise) by replacing portions of the
    authentic context with completely irrelevant, neutral statements from unrelated fields.

    Severity Ratios:
    - 'low': Replace approximately 30% of sentences with distractors (at least 1 sentence).
    - 'medium': Replace approximately 50% of sentences with distractors.
    - 'high': Replace approximately 70% of sentences with distractors (or all sentences).

    Args:
        text: Source evidence text.
        severity: Noise ratio level ('low' = 30%, 'medium' = 50%, 'high' = 70%).
        seed: Optional integer for deterministic PRNG state (reproducibility).

    Returns:
        Perturbed text containing injected distractor sentences.

    Raises:
        ValueError: If severity is unrecognized.
    """
    sentences = split_sentences(text)
    n_sents = len(sentences)

    if n_sents == 0:
        return ""

    rng = random.Random(seed)
    sev = severity.strip().lower()

    if sev == "low":
        ratio = 0.30
    elif sev == "medium":
        ratio = 0.50
    elif sev == "high":
        ratio = 0.70
    else:
        raise ValueError(f"Unknown severity '{severity}'. Supported options: 'low', 'medium', 'high'.")

    # Determine replacement count
    k_replace = max(1, min(n_sents, int(math.ceil(n_sents * ratio))))

    # Select random indices to replace
    indices_to_replace = set(rng.sample(range(n_sents), k_replace))

    # Sample non-repeating distractor sentences from the knowledge pool
    distractor_pool = rng.sample(DISTRACTOR_SENTENCES, min(len(DISTRACTOR_SENTENCES), k_replace))

    result_sentences: List[str] = []
    distractor_idx = 0
    for i, s in enumerate(sentences):
        if i in indices_to_replace:
            result_sentences.append(distractor_pool[distractor_idx])
            distractor_idx += 1
        else:
            result_sentences.append(s)

    return join_sentences(result_sentences)


# ---------------------------------------------------------------------------
# Self-Test & Reproducibility Verification Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEED = 42
    SAMPLE_TEXT = (
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. "
        "He then landed a role in a Danish film in 1994."
    )
    SEVERITY_LEVELS = ["low", "medium", "high"]

    print("=" * 80)
    print("RAG EVIDENCE PERTURBATION ENGINE -- REPRODUCIBILITY TEST SUITE")
    print("Target: IEEE Conference / Journal Submission")
    print(f"Random Seed: {SEED}")
    print("=" * 80)
    print(f"\n[ORIGINAL BENCHMARK TEXT]\n\"{SAMPLE_TEXT}\"\n")

    # 1. Test delete_evidence
    print("=" * 80)
    print("1. OPERATOR: delete_evidence (Context Pruning)")
    print("=" * 80)
    for sev in SEVERITY_LEVELS:
        out = delete_evidence(SAMPLE_TEXT, severity=sev, seed=SEED)
        display_out = f'"{out}"' if out else "[EMPTY STRING -- 100% PRUNED]"
        print(f"[{sev.upper():<6}] -> {display_out}")

    # 2. Test contradict_evidence
    print("\n" + "=" * 80)
    print("2. OPERATOR: contradict_evidence (Factual Distortion via Entity/Date Swapping)")
    print("=" * 80)
    for sev in SEVERITY_LEVELS:
        out = contradict_evidence(SAMPLE_TEXT, severity=sev, seed=SEED)
        print(f"[{sev.upper():<6}] -> \"{out}\"")

    # 3. Test make_irrelevant
    print("\n" + "=" * 80)
    print("3. OPERATOR: make_irrelevant (Distractor & Noise Injection)")
    print("=" * 80)
    for sev in SEVERITY_LEVELS:
        out = make_irrelevant(SAMPLE_TEXT, severity=sev, seed=SEED)
        print(f"[{sev.upper():<6}] -> \"{out}\"")

    # 4. Multi-sentence scaling demonstration (5-sentence paragraph)
    print("\n" + "=" * 80)
    print("4. OPERATOR SCALING ON MULTI-SENTENCE PASSAGE (5 Sentences)")
    print("=" * 80)
    PARAGRAPH = (
        "Anne Frank died in a Nazi concentration camp at the age of 15. "
        "Her diary became one of the most widely read books in the world. "
        "The Bergen-Belsen camp was liberated by British forces in April 1945. "
        "New research indicates the sisters passed away earlier than recorded. "
        "Historical archives continue to preserve critical Holocaust testimonies."
    )
    print(f"Original 5-Sentence Passage:\n\"{PARAGRAPH}\"\n")
    for op_name, op_func in [
        ("delete_evidence", delete_evidence),
        ("contradict_evidence", contradict_evidence),
        ("make_irrelevant", make_irrelevant),
    ]:
        print(f"--- Operator: {op_name} ---")
        for sev in SEVERITY_LEVELS:
            res = op_func(PARAGRAPH, severity=sev, seed=SEED)
            print(f"  [{sev.upper():<6}] -> \"{res}\"")

    # 5. Peer-Review Reproducibility Check
    print("\n" + "=" * 80)
    print("5. PEER-REVIEW REPRODUCIBILITY VERIFICATION (Idempotence Across Independent Runs)")
    print("=" * 80)
    run1 = contradict_evidence(SAMPLE_TEXT, severity="high", seed=SEED)
    run2 = contradict_evidence(SAMPLE_TEXT, severity="high", seed=SEED)
    is_identical = (run1 == run2)
    print(f"Run 1 (seed={SEED}): \"{run1}\"")
    print(f"Run 2 (seed={SEED}): \"{run2}\"")
    print(f"Exact Match Verification: {is_identical} (Deterministic & Reproducible)")
    print("=" * 80)
