"""Run a model on FinRevBench through any OpenAI-compatible Chat Completions endpoint
(OpenAI, vLLM, llama.cpp server, ...). Standard library only.

Usage:
  python run_model.py --model MODEL --base-url http://localhost:8000/v1 --output predictions.jsonl
  (API key, if needed, is read from the OPENAI_API_KEY environment variable.)

Each row's `messages` are sent unchanged; the raw text is saved as {"question_id", "raw_output"}.
Re-running with the same --output resumes: questions already in the file are skipped.
"""
import argparse, json, os, time, urllib.error, urllib.request
from itertools import islice
from pathlib import Path

# Output budgets: answers are short JSON objects.
MAX_TOKENS = {"temporal": 128, "hfkr_source_state": 512, "hfkr_numerical": 512}


def request_body(model, question, profile, temperature):
    body = {"model": model, "messages": question["messages"]}
    if profile == "gpt5-mini":
        return {**body, "reasoning_effort": "minimal", "max_completion_tokens": 1024}
    return {**body, "temperature": 0.0 if temperature is None else temperature,
            "max_tokens": MAX_TOKENS[question["track"]]}


def chat(base_url, key, body, retries=5):
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())["choices"][0]["message"].get("content") or ""
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {error}", flush=True)
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/finrevbench.jsonl"))
    ap.add_argument("--profile", choices=("standard", "gpt5-mini"), default="standard",
                    help="gpt5-mini uses the paper's Chat Completions settings")
    ap.add_argument("--temperature", type=float, help="sampling temperature for the standard profile (default 0)")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    a = ap.parse_args()
    if a.profile == "gpt5-mini" and a.temperature is not None:
        ap.error("--profile gpt5-mini leaves temperature at the provider default")
    if a.limit is not None and a.limit < 0:
        ap.error("--limit must be nonnegative")
    out = Path(a.output)
    done = {json.loads(l)["question_id"] for l in out.open()} if out.exists() else set()
    key = os.environ.get("OPENAI_API_KEY")
    with open(a.data, encoding="utf-8") as source, out.open("a", encoding="utf-8") as sink:
        questions = (json.loads(line) for line in source if line.strip())
        for i, q in enumerate(islice(questions, a.limit), 1):
            if q["question_id"] in done:
                continue
            text = chat(a.base_url, key, request_body(a.model, q, a.profile, a.temperature))
            sink.write(json.dumps({"question_id": q["question_id"], "raw_output": text}, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"{i}: {q['question_id']}", flush=True)
