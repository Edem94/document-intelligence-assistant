## 7. Evaluation: run every question in eval_set.json through the real RAG chain and score it
#
#   python -m eval.run_eval                # full run with LLM judge
#   python -m eval.run_eval --no-judge     # deterministic checks only (cheaper, faster)
#   python -m eval.run_eval --only compose-define-multi k8s-restart-pod
#
# Case fields: question, expected_sources (retrieval hit = any one retrieved), answerable,
# must_include (case-insensitive substrings; a nested list means "any of these"),
# must_match (optional regexes on the answer), code_order (commands that must appear in the
# answer's code blocks in this order, e.g. install -> start -> verify), partial (docs only partly answer it, so an
# "I don't have enough information to answer completely" opening is fine), notes.

import argparse
import json
import os
import re
import time
from datetime import datetime

from langchain_anthropic import ChatAnthropic

from app.config import ANTHROPIC_API_KEY, BASE_DIR
from app.rag_chain import format_docs, get_qa_chain

EVAL_DIR = os.path.join(BASE_DIR, "eval")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# "Refused" for an out-of-scope question: any honest "the docs don't cover this" phrasing
REFUSAL_RE = re.compile(
    r"(don't|do not) have enough information|not enough information|"
    r"(doesn't|does not|don't|do not) (cover|contain|include|provide|mention)|not (covered|included|mentioned)",
    re.IGNORECASE,
)
# A full refusal on a question the docs DO answer (the prompt's wording for "not covered at all")
HARD_REFUSAL_RE = re.compile(r"(don't|do not) have enough information", re.IGNORECASE)

JUDGE_PROMPT = """You are grading a RAG assistant's answer. Use ONLY the context to judge facts.

Question: {question}

Context the assistant was given:
{context}

Assistant's answer:
{answer}

Score 1-5:
- faithfulness: are all claims, commands and config in the answer supported by the context? (5 = everything supported, 1 = mostly invented)
- completeness: does it fully answer the question, as far as the context allows? (5 = fully, 1 = misses the point)
List any unsupported claims briefly.

Reply with JSON only: {{"faithfulness": <1-5>, "completeness": <1-5>, "unsupported_claims": ["..."]}}"""


def check_includes(answer: str, must_include: list) -> list[str]:
    """Return the missing items. An item is a string, or a list of alternatives (any one is enough)."""
    text = answer.lower()
    missing = []
    for item in must_include:
        options = item if isinstance(item, list) else [item]
        if not any(opt.lower() in text for opt in options):
            missing.append(" | ".join(options))
    return missing


def invented_code_lines(answer: str, docs) -> list[str]:
    """Lines inside the answer's code blocks that don't appear in the retrieved context.

    Catches merged or edited examples (e.g. an invented `image: redis`), which an LLM judge
    tends to wave through. Indentation is ignored. Config lines must match a whole context line
    (so `image: redis` doesn't pass because `image: redis:alpine` exists); shell commands only need
    to appear somewhere, since docs often mention them inline without a "$ " prompt. Lines that are
    only an elision marker ("# ...", "...") are allowed. A shell command may also differ from a docs
    command in placeholder values only (image name/tag, port numbers) - see command_shape().
    """
    context_text = "\n".join(d.page_content for d in docs)
    context_lines = {line.strip() for line in context_text.splitlines()}
    context_shapes = {command_shape(line.removeprefix("$ ")) for line in context_lines if line}
    invented = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", answer, re.DOTALL):
        for line in (l.strip() for l in block.splitlines()):
            if not line or line in context_lines:
                continue
            if re.fullmatch(r"(#|//)?\s*\.\.\..*", line):  # an explicit "# ..." elision isn't invented config
                continue
            # Shell commands, with or without a "$ " prompt: the docs often write "$ docker compose up"
            # (or inline in a sentence, without a prompt). "Command-like" = starts with a lowercase word
            # and a space, so YAML keys ("image: x") and Dockerfile lines ("EXPOSE 80") never match here.
            command = line.removeprefix("$ ")
            is_command = line.startswith("$ ") or re.match(r"[a-z][\w.-]* ", command)
            if f"$ {command}" in context_lines or (is_command and command in context_text):
                continue
            if is_command and command_shape(command) in context_shapes:
                continue
            invented.append(line)
    return invented


def command_shape(command: str) -> str:
    """A command with its placeholder values blanked: `docker run -p 127.0.0.1:5000:5000 my-app:latest`
    and `docker run -p 127.0.0.1:8000:8000 test:latest` have the same shape. Commands, subcommands and
    flags are kept, so adding a flag or a different command still counts as invented."""
    command = re.sub(r"(-t|--tag)\s+\S+", r"\1 IMG", command.strip())
    command = re.sub(r"\d+", "N", command)
    command = re.sub(r"\b[\w./-]+:[\w.-]+\b", "IMG", command)  # image:tag (also N.N.N.N:N:N port maps)
    return re.sub(r"\s+", " ", command)


def undisclosed_dependencies(answer: str) -> list[str]:
    """Dependencies used in code examples that the rest of the answer never explains.

    - ${VARS} need the answer to mention where they come from (.env / environment variables)
    - an included file (include: - path: ./infra.yaml) must be named again outside its include line
    - features that silently do nothing without a flag: `develop: watch` needs `--watch`
      (or `docker compose watch`), service `profiles:` need `--profile`
    """
    code = "\n".join(re.findall(r"```[^\n]*\n(.*?)```", answer, re.DOTALL))
    problems = []
    variables = sorted(set(re.findall(r"\$\{(\w+)", code)))
    if variables and not re.search(r"\.env\b|environment variable", answer, re.IGNORECASE):
        problems.append("unexplained variables: " + ", ".join(variables))
    for path in set(re.findall(r"path:\s*\.?/?([\w./-]+\.ya?ml)", code)):
        if answer.count(path.split("/")[-1]) < 2:
            problems.append(f"included file not shown or explained: {path}")
    if re.search(r"(?m)^\s*develop:\s*$", code) and re.search(r"(?m)^\s*watch:\s*$", code) \
            and not re.search(r"--watch|docker compose watch", answer):
        problems.append("develop.watch shown but never says to run with --watch")
    if re.search(r"(?m)^\s+profiles:", code) and "--profile" not in answer:
        problems.append("profiles shown but never says to run with --profile")
    return problems


def code_order_problems(answer: str, steps: list) -> list[str]:
    """Check that the answer's commands appear in the order a user must run them.

    `steps` is a list of items (a string, or a list of alternatives); each must occur inside the
    answer's code blocks, and the first occurrence of each must come after the previous step's.
    Only code is checked, so a summary sentence like "run `minikube start`" up front is fine.
    """
    code = "\n".join(re.findall(r"```[^\n]*\n(.*?)```", answer, re.DOTALL)).lower()
    problems, last_pos, last_name = [], -1, None
    for step in steps:
        options = step if isinstance(step, list) else [step]
        positions = [code.find(opt.lower()) for opt in options if opt.lower() in code]
        name = " | ".join(options)
        if not positions:
            problems.append(f"step missing from code: {name}")
            continue
        if min(positions) < last_pos:
            problems.append(f"'{name}' comes before '{last_name}'")
        last_pos, last_name = min(positions), name
    return problems


# Wording that exposes the RAG plumbing to the user ("The context provides..."). Docker's own term
# "build context" is fine, so only phrasings that refer to the retrieved text are matched.
CONTEXT_LEAK_RE = re.compile(
    r"\b(provided|given|retrieved) context\b|"
    r"\bthe context (provides|provided|covers|includes|contains|mentions|shows|only|does not|doesn't)\b|"
    r"\b(in|from) the context\b",
    re.IGNORECASE,
)


def broken_links(answer: str) -> list[str]:
    """Markdown links that aren't absolute http(s) URLs - in the UI they resolve against this app and 404."""
    prose = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    return [url for url in re.findall(r"\]\(\s*<?([^)\s>]*)", prose) if not url.startswith(("http://", "https://"))]


def judge(judge_llm, case: dict, answer: str, docs) -> dict:
    reply = judge_llm.invoke(JUDGE_PROMPT.format(question=case["question"], context=format_docs(docs), answer=answer))
    match = re.search(r"\{.*\}", reply.content, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"error": reply.content[:200]}
    except json.JSONDecodeError:
        return {"error": reply.content[:200]}


def run_case(chain, judge_llm, case: dict) -> dict:
    start = time.time()
    result = chain.invoke({"question": case["question"]})
    latency = time.time() - start
    answer, docs = result["answer"], result["source_documents"]
    retrieved = sorted({d.metadata.get("source", "unknown") for d in docs})

    row = {"id": case["id"], "answerable": case["answerable"], "latency_s": round(latency, 2),
           "retrieved_sources": retrieved, "answer": answer, "broken_links": broken_links(answer),
           "context_leak": bool(CONTEXT_LEAK_RE.search(answer))}
    if case["answerable"]:
        expected = case["expected_sources"]
        row["retrieval_hit"] = any(s in retrieved for s in expected)
        row["missing_keywords"] = check_includes(answer, case["must_include"])
        # Regex checks for structure that substrings can't express, e.g. "a redis service is defined"
        row["missing_keywords"] += [f"/{p}/" for p in case.get("must_match", []) if not re.search(p, answer)]
        row["missing_keywords"] += code_order_problems(answer, case.get("code_order", []))
        # On a "partial" case, "I don't have enough information to answer this completely" is the honest
        # opening the prompt asks for - as long as it then delivers what the docs do cover (must_include)
        row["false_refusal"] = bool(HARD_REFUSAL_RE.search(answer)) and not case.get("partial")
        row["invented_code"] = invented_code_lines(answer, docs)
        row["undisclosed_deps"] = undisclosed_dependencies(answer)
        row["passed"] = (row["retrieval_hit"] and not row["missing_keywords"] and not row["false_refusal"]
                         and not row["invented_code"] and not row["undisclosed_deps"] and not row["broken_links"]
                         and not row["context_leak"])
    else:
        row["refused"] = bool(REFUSAL_RE.search(answer))
        row["passed"] = row["refused"] and not row["broken_links"] and not row["context_leak"]
    if judge_llm and case["answerable"]:
        row["judge"] = judge(judge_llm, case, answer, docs)
    return row


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    oos = [r for r in rows if not r["answerable"]]
    pct = lambda n, d: round(100 * n / d, 1) if d else None
    judged = [r["judge"] for r in answerable if "faithfulness" in r.get("judge", {})]
    avg = lambda key: round(sum(j[key] for j in judged) / len(judged), 2) if judged else None
    return {
        "cases": len(rows),
        "pass_rate_pct": pct(sum(r["passed"] for r in rows), len(rows)),
        "retrieval_hit_rate_pct": pct(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "keyword_pass_rate_pct": pct(sum(not r["missing_keywords"] for r in answerable), len(answerable)),
        "false_refusals": sum(r["false_refusal"] for r in answerable),
        "answers_with_invented_code": sum(bool(r["invented_code"]) for r in answerable),
        "answers_with_undisclosed_deps": sum(bool(r["undisclosed_deps"]) for r in answerable),
        "answers_with_broken_links": sum(bool(r["broken_links"]) for r in rows),
        "answers_mentioning_context": sum(r["context_leak"] for r in rows),
        "out_of_scope_refusal_rate_pct": pct(sum(r["refused"] for r in oos), len(oos)),
        "avg_faithfulness": avg("faithfulness"),
        "avg_completeness": avg("completeness"),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
    }


def print_report(rows: list[dict], summary: dict):
    print(f"\n{'id':28} {'pass':5} {'retr':5} {'faith':5} {'compl':5} {'sec':>5}  problems")
    for r in rows:
        j = r.get("judge", {})
        problems = []
        if r.get("missing_keywords"):
            problems.append("missing: " + ", ".join(r["missing_keywords"]))
        if r.get("false_refusal"):
            problems.append("refused an answerable question")
        if r["context_leak"]:
            problems.append('says "the context" instead of "the docs"')
        if r["broken_links"]:
            problems.append("broken links: " + ", ".join(r["broken_links"][:3]))
        if r.get("undisclosed_deps"):
            problems.append("; ".join(r["undisclosed_deps"]))
        if r.get("invented_code"):
            problems.append("code not in docs: " + "; ".join(r["invented_code"][:3]))
        if not r["answerable"] and not r["refused"]:
            problems.append("answered an out-of-scope question")
        if r["answerable"] and not r["retrieval_hit"]:
            problems.append("expected source not retrieved")
        mark = lambda v: "-" if v is None else ("ok" if v else "FAIL")
        print(f"{r['id']:28} {mark(r['passed']):5} {mark(r.get('retrieval_hit')):5} "
              f"{str(j.get('faithfulness', '-')):5} {str(j.get('completeness', '-')):5} {r['latency_s']:5.1f}  {'; '.join(problems)}")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:32} {value}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against eval/eval_set.json")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge scoring")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these case ids")
    args = parser.parse_args()

    with open(os.path.join(EVAL_DIR, "eval_set.json"), encoding="utf-8") as f:
        cases = json.load(f)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    chain = get_qa_chain()
    judge_llm = None if args.no_judge else ChatAnthropic(
        model="claude-haiku-4-5-20251001", api_key=ANTHROPIC_API_KEY, temperature=0, max_tokens=400)

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        rows.append(run_case(chain, judge_llm, case))

    summary = summarize(rows)
    print_report(rows, summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nFull results (answers included): {out_path}")


if __name__ == "__main__":
    main()
