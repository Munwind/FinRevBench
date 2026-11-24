# FinRevBench v1

FinRevBench tests financial question answering over supplied SEC EDGAR disclosures when figures are revised, withdrawn from reliance, or restated. Every question comes with an information cutoff. Each question supplies its sources with their filing dates. The model must answer with what was available and still supportable at that cutoff.

The benchmark evaluates question answering with supplied text evidence. The full dataset is a single evaluation set.

## Contents

| Track | Questions | Notes |
|---|---:|---|
| Temporal QA | 1,818 | 909 PRE/POST pairs. Each pair holds the financial target fixed and has distinct cutoffs and reference values. |
| HFKR Source State | 1,085 | supported value at cutoff (338, of which 125 null), historically reported value (320), reporting-state sequence (258), withdrawal scope (169) |
| HFKR Numerical | 717 | signed change (251), relative revision (251), liabilities-to-assets ratio (143), ratio revision (72) |
| **Total** | **3,620** | 320 issuer groups, 264 HFKR reporting events, 640 distinct SEC filings |

## Files

```
data/finrevbench.jsonl        model inputs: question_id, track, case_id, issuer_group_id, event_id|side, messages
data/sources.jsonl            per question: source label "[n]" -> SEC accession, CIK, form, filed date, URL
scripts/run_model.py          runs a model through any OpenAI-compatible endpoint
scripts/score.py              scorer for use with a private reference file
```

All scripts need only Python 3.9+ and the standard library.

## How to run

**1. Generate predictions.** `run_model.py` sends each row's `messages` (one system and one user message) unchanged. It writes one `{"question_id", "raw_output"}` row per question. The standard profile sends temperature 0 by default and an output budget of 128 tokens for Temporal and 512 for HFKR. Use a separate output file for each model and configuration. The runner skips question IDs already present in that file.

```bash
# hosted API
export OPENAI_API_KEY=...
python -I scripts/run_model.py --model MODEL_NAME --output my_model.jsonl

# GPT-5 mini with the paper's request settings
python -I scripts/run_model.py --model gpt-5-mini-2025-08-07 --profile gpt5-mini --output gpt5_mini.jsonl

# local server (vLLM, llama.cpp `llama-server`, ...)
python -I scripts/run_model.py --model my-model --base-url http://localhost:8000/v1 --output my_model.jsonl

# smoke test on the first 5 questions
python -I scripts/run_model.py --model my-model --base-url http://localhost:8000/v1 --output /tmp/smoke.jsonl --limit 5
```

The GPT-5 mini profile sends `reasoning_effort="minimal"` and `max_completion_tokens=1024`, leaving temperature at the provider default. The released runner sends requests sequentially. For your own runner, send only the supplied `messages` to the model. Use `sources.jsonl` to look up filing provenance.

**2. Score.** Scoring requires the private `gold.jsonl` reference file held by the benchmark maintainers.

```bash
python -I scripts/score.py my_model.jsonl --gold /path/to/gold.jsonl --parse fence
```

The scorer prints a JSON summary per track.

- `--parse strict` requires the raw output to be exactly one JSON object.
- `--parse fence` first removes one surrounding Markdown code fence.
- A missing or malformed output counts as wrong.
- Always report which parsing policy you used.
- The paper used `strict` for Llama 3.1 8B and the four Qwen models, and `fence` for Gemma, Ministral, GPT-4o mini and GPT-5 mini.

## Scoring

- **Temporal QA:** an answer is correct when the number exactly matches the reference at the requested scale; there is no tolerance. Reported metrics are PRE, POST, pair accuracy (both endpoints correct), delta accuracy and issuer-macro accuracy.
- **HFKR:** each question is scored on four flags.
  - *answer*: all requested fields are exact; derived values are rounded half away from zero at the stated precision.
  - *evidence*: for every role, the citations contain at least one complete approved sufficient set. Any additional citations must belong to an approved set for that role.
  - *temporal validity*: every cited source was filed on or before the role's cutoff.
  - *joint*: all three of the above hold.
- **Aggregation:** results are given per family and as an event-macro joint score (the mean within each event, then the mean over events).

## Source data

The supplied evidence contains excerpts from public SEC EDGAR filings. Each source label resolves to an accession in `data/sources.jsonl`. The form identifies the filing and the URL identifies a document or filing index within that accession. [SEC EDGAR reuse policy](https://www.sec.gov/about/webmaster-frequently-asked-questions)

