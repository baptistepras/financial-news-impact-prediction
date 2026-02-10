"""
Local LLM-as-a-judge for finance-style impact summaries.

This script reads a CSV containing a financial news article and a model-generated impact summary per row.
A local instruction-tuned causal LLM evaluates each summary against its source text across several
dimensions (accuracy, issuer grounding, numeric fidelity, coverage, conciseness, professionalism).
The judge is instructed to output strict JSON for easy parsing. The script aggregates per-category
statistics (mean, 5th and 95th percentiles) and extracts best/worst tail examples, then writes a
single JSON report to disk.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

DEFAULT_CATEGORIES = [
    "accuracy",
    "issuer_grounding",
    "numeric_fidelity",
    "coverage",
    "conciseness",
    "anti_filler_professionalism",
]


def _safe_strip(s: Any) -> str:
    """Convert values to a clean string without changing semantics."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first complete JSON object from text using brace balancing."""
    text = _safe_strip(text)
    if not text:
        return None

    # Fast path: direct JSON dict
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1].strip()
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None

    # No balanced object found (likely truncated output)
    return None


def clamp_score(x: Any, lo: float = 0.0, hi: float = 5.0) -> float:
    """Clamp a score to a fixed range while preserving NaN for missing values."""
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if np.isnan(v):
        return v
    return max(lo, min(hi, v))


def percentile(vals: List[float], p: float) -> float:
    """Compute a percentile while ignoring NaNs."""
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def mean(vals: List[float]) -> float:
    """Compute the mean while ignoring NaNs."""
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def get_tail_examples(
    rows: List[Dict[str, Any]],
    category: str,
    fraction: float,
    mode: str,
) -> List[Dict[str, Any]]:
    """Select a fraction of best or worst examples for a given category."""
    scored: List[tuple[float, Dict[str, Any]]] = []
    for r in rows:
        s = r.get("scores", {}).get(category, float("nan"))
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        scored.append((float(s), r))

    if not scored:
        return []

    scored.sort(key=lambda t: t[0])
    k = max(1, int(np.ceil(len(scored) * fraction)))

    if mode == "worst":
        chosen = scored[:k]
    else:
        chosen = scored[-k:]
        chosen.reverse()

    # Keep the payload light so the output JSON stays readable.
    out: List[Dict[str, Any]] = []
    for score_val, r in chosen:
        out.append(
            {
                "row_id": r.get("row_id"),
                "score": score_val,
                "key_issues": r.get("key_issues", [])[:6],
                "one_line_summary": (r.get("one_line_summary", "") or "")[:400],
            }
        )
    return out


def build_judge_prompt(source_text: str, summary: str) -> str:
    """Build a strict evaluation prompt that requires a JSON-only response."""
    return f"""
You are a STRICT financial disclosure auditor and evaluator.

You will be given:
(1) SOURCE: the original disclosure text
(2) SUMMARY: an "impact summary" produced by a model

Your job: evaluate factual reliability and usefulness for an institutional investor.
Be conservative: if a claim is not clearly supported by SOURCE, treat it as unsupported.

IMPORTANT CONSTRAINTS:
- DO NOT reward creativity. Reward fidelity, specificity, correct attribution, and brevity.
- Penalize vague filler and generic investor-advice language.
- Penalize missing key numbers when SOURCE contains them (amounts, durations, percentages, dates).
- Penalize entity confusion (wrong company/party performing an action, or "the company" ambiguity).
- If the summary uses hedges like "could", "may", "likely" without explicit SOURCE support, penalize accuracy.

SCORING:
Give a 0–5 score (integer) for each category:
1) accuracy:
   5 = no unsupported claims; correct attributions; no hallucinated facts
   3 = mostly correct but a few weak/unsupported statements or hedges
   1 = multiple unsupported/incorrect claims or clear hallucinations
2) issuer_grounding:
   5 = consistently attributes actions to the correct main issuer/party
   3 = some ambiguity ("the company") or occasional confusion
   1 = frequent confusion between issuer vs third parties
3) numeric_fidelity:
   5 = key numbers/dates/terms from SOURCE preserved when present
   3 = some numbers omitted but no invented numbers
   1 = omits most key numbers OR invents/changes numbers
4) coverage:
   5 = captures the key event + consequences/impact
   3 = captures event but misses important impact/risk or context
   1 = misses the main point
5) conciseness:
   5 = 2–4 sentences, dense with info, no repetition
   3 = slightly verbose or mild repetition
   1 = very verbose, repetitive, or rambling
6) anti_filler_professionalism:
   5 = professional analyst tone; no clichés; no investor advice
   3 = minor filler/clichés
   1 = heavy filler, generic phrases, investor advice, hype

OUTPUT REQUIREMENTS (STRICT):
Return JSON ONLY (no markdown, no commentary) with EXACT keys:
{{
  "scores": {{
    "accuracy": <int 0-5>,
    "issuer_grounding": <int 0-5>,
    "numeric_fidelity": <int 0-5>,
    "coverage": <int 0-5>,
    "conciseness": <int 0-5>,
    "anti_filler_professionalism": <int 0-5>
  }},
  "key_issues": [<short strings, max 8 items>],
  "unsupported_claims": [<short strings, max 8 items>],
  "missing_key_numbers": [<short strings, max 8 items>],
  "entity_confusions": [<short strings, max 8 items>],
  "one_line_summary": "<one sentence describing quality>"
}}

SOURCE:
{source_text}

SUMMARY:
{summary}
""".strip()


@dataclass
class JudgeConfig:
    model_name: str
    max_new_tokens: int
    temperature: float
    do_sample: bool
    device_map: str
    torch_dtype: str
    max_source_chars: int


def load_model_and_tokenizer(cfg: JudgeConfig) -> tuple[Any, Any]:
    """Load a local causal LM and tokenizer using Hugging Face Transformers."""
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    dtype = None
    if cfg.torch_dtype.lower() == "float16":
        dtype = torch.float16
    elif cfg.torch_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
    elif cfg.torch_dtype.lower() == "float32":
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )
    model.eval()
    return tok, model


def generate_judgment(
    tok: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    """Generate a judge response with deterministic decoding settings."""
    inputs = tok(prompt, return_tensors="pt")

    # With device_map="auto", we just place the prompt on the first parameter device.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # Some causal models include the prompt in the decoded output; strip it if present.
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()

    return text.strip()


def validate_and_normalize_judge_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Validate expected keys and normalize scores and lists into a consistent structure."""
    out: Dict[str, Any] = dict(obj) if isinstance(obj, dict) else {}

    scores = out.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    norm_scores: Dict[str, float] = {}
    for k in DEFAULT_CATEGORIES:
        v = scores.get(k, float("nan"))
        vv = clamp_score(v)
        if not np.isnan(vv):
            vv = float(int(round(vv)))
        norm_scores[k] = vv

    out["scores"] = norm_scores

    for list_key in ["key_issues", "unsupported_claims", "missing_key_numbers", "entity_confusions"]:
        v = out.get(list_key, [])
        if not isinstance(v, list):
            v = []
        out[list_key] = [str(x)[:240] for x in v][:8]

    out["one_line_summary"] = _safe_strip(out.get("one_line_summary", ""))[:600]
    return out


def main() -> None:
    """Run batch judging over a CSV file and write per-row results plus aggregate statistics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="pred.csv")
    ap.add_argument("--out_json", default="judge_results.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max_new_tokens", type=int, default=420)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_source_chars", type=int, default=9000, help="truncate SOURCE to limit prompt size")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns or "pred_summary" not in df.columns:
        raise ValueError("CSV must contain columns: 'text' and 'pred_summary'")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    cfg = JudgeConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_source_chars=args.max_source_chars,
    )

    tok, model = load_model_and_tokenizer(cfg)

    results_rows: List[Dict[str, Any]] = []
    n = len(df)

    for i, row in df.iterrows():
        source = _safe_strip(row["text"])
        summ = _safe_strip(row["pred_summary"])

        # Truncate source to keep the prompt manageable for local models.
        if len(source) > cfg.max_source_chars:
            source_for_prompt = source[: cfg.max_source_chars] + "\n[TRUNCATED]"
        else:
            source_for_prompt = source

        prompt = build_judge_prompt(source_for_prompt, summ)
        raw = generate_judgment(
            tok=tok,
            model=model,
            prompt=prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
        )

        obj = extract_first_json_object(raw)
        if obj is None:
            judge_obj = {
                "scores": {k: float("nan") for k in DEFAULT_CATEGORIES},
                "key_issues": ["PARSE_ERROR: model did not return valid JSON"],
                "unsupported_claims": [],
                "missing_key_numbers": [],
                "entity_confusions": [],
                "one_line_summary": "Parse error: no valid JSON extracted.",
                "_raw_output": raw[:4000],
            }
        else:
            judge_obj = validate_and_normalize_judge_json(obj)
            judge_obj["_raw_output"] = raw[:2000]

        results_rows.append(
            {
                "row_id": int(i),
                "scores": judge_obj["scores"],
                "key_issues": judge_obj.get("key_issues", []),
                "unsupported_claims": judge_obj.get("unsupported_claims", []),
                "missing_key_numbers": judge_obj.get("missing_key_numbers", []),
                "entity_confusions": judge_obj.get("entity_confusions", []),
                "one_line_summary": judge_obj.get("one_line_summary", ""),
                "_raw_output": judge_obj.get("_raw_output", ""),
            }
        )

    stats: Dict[str, Dict[str, Any]] = {"mean": {}, "p5": {}, "p95": {}, "worst_5pct": {}, "best_5pct": {}}
    for cat in DEFAULT_CATEGORIES:
        vals = [r["scores"].get(cat, float("nan")) for r in results_rows]
        stats["mean"][cat] = mean(vals)
        stats["p5"][cat] = percentile(vals, 5)
        stats["p95"][cat] = percentile(vals, 95)
        stats["worst_5pct"][cat] = get_tail_examples(results_rows, cat, 0.05, "worst")
        stats["best_5pct"][cat] = get_tail_examples(results_rows, cat, 0.05, "best")

    out = {
        "config": {
            "model": cfg.model_name,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "do_sample": cfg.do_sample,
            "device_map": cfg.device_map,
            "torch_dtype": cfg.torch_dtype,
            "max_source_chars": cfg.max_source_chars,
            "categories": DEFAULT_CATEGORIES,
            "n_rows": len(results_rows),
        },
        "per_row": results_rows,
        "summary": stats,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)