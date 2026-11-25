"""GSM8K correctness eval for the SDAR engine, with a per-token commit trace.

This is the GSM8K setting in SDAR's OpenCompass config
(``gsm8k_0shot_v2_gen_17d799``):

- prompt: ``{question}\\nPlease reason step by step, and put your final answer within \\boxed{}.``
- one user turn, chat template, greedy
- prediction: ``math_postprocess_v2`` (boxed answer, else the "answer is" sentence)
- reference: the number after ``####``, commas removed
- match: ``MATHEvaluator`` version v2 (``is_equiv``)

The checkpoint is the b32 chat model, so the block stays 32. Their published
greedy run uses a static schedule with one denoising step per token in the
block (``denoising_steps = block_length``, ``threshold = 1``). That is
``--steps 32 --strategy static`` here. The score is on the first 100 test
problems, not the full 1,319-item set.

Each completion records the diffusion decision that produced every token: the
denoising step that committed it, the probability of that token at that step,
and the runner-up. Answer-span tokens are the ones inside the boxed answer
the grader read.
"""

import argparse
import json
import os
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Decoded end markers. The engine also stops when these token ids are committed.
STOPS = ("<|im_end|>", "<|endoftext|>")
NULL = "NULL"


def last_boxed_only_string(string):
    """OpenCompass ``last_boxed_only_string``. The last ``\\boxed`` or ``\\fbox``."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    depth = 0
    while i < len(string):
        if string[i] == "{":
            depth += 1
        if string[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(text):
    left = "\\boxed{"
    if not text.startswith(left) or not text.endswith("}"):
        return None
    return text[len(left) : -1]


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed = last_boxed_only_string(pred_str)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match(r"^\{(.*)\}$", answer)
        if match:
            answer = match.group(1)
    return answer


def normalize_final_answer(final_answer):
    """OpenCompass ``normalize_final_answer``."""
    substitutions = (
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        ("\\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    )
    removed = (
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        "\\mathrm{th}",
        "^\\circ",
        "^{\\circ}",
        "\\;",
        ",\\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    )
    for before, after in substitutions:
        final_answer = final_answer.replace(before, after)
    for expr in removed:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    if "\n" in final_answer or "\r" in final_answer or "\f" in final_answer:
        raise ValueError("newline survived normalization")
    found = re.findall(r"finalansweris(.*)", final_answer)
    if found:
        final_answer = found[-1]
    found = re.findall(r"answer?is:?(.*)", final_answer)
    if found:
        final_answer = found[-1]
    found = re.findall(r"oxed\{(.*?)\}", final_answer)
    if found:
        final_answer = found[-1]
    found = re.findall(r"\$(.*?)\$", final_answer)
    if found:
        final_answer = found[-1]
    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def math_postprocess_v2(text):
    """OpenCompass ``math_postprocess_v2``."""
    boxed = extract_boxed_answer(text, strip_double_curly_brace=True)
    if boxed:
        return boxed
    for sentence in text.split("."):
        if re.search(r"final answer|answer is", sentence.lower()):
            return normalize_final_answer(sentence)
    return normalize_final_answer(text.split(".")[0])


def _fix_fracs(string):
    parts = string.split("\\frac")
    new_str = parts[0]
    if len(parts) == 1:
        return string
    for substr in parts[1:]:
        new_str += "\\frac"
        if len(substr) > 0 and substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        a = substr[0]
        b = substr[1]
        if b != "{":
            rest = substr[2:] if len(substr) > 2 else ""
            new_str += "{" + a + "}{" + b + "}" + rest
        else:
            rest = substr[2:] if len(substr) > 2 else ""
            new_str += "{" + a + "}" + b + rest
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        if string != f"{a}/{b}":
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except ValueError:
        return string


def _fix_sqrt_v2(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _strip_string_v2(string):
    """OpenCompass ``MATHEvaluator._strip_string_v2``."""
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    stripped = re.sub(r"\\text{.*?}$", "", string).strip()
    if stripped != "" and stripped != string:
        string = stripped
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt_v2(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def is_equiv(pred, gold):
    """OpenCompass ``MATHEvaluator.is_equiv`` for version v2."""
    if pred is None or gold is None:
        return False
    try:
        left = _strip_string_v2(pred)
        right = _strip_string_v2(gold)
        if left == right:
            return True
        if normalize_final_answer(left) == normalize_final_answer(right):
            return True
    except Exception:
        pass
    try:
        if normalize_final_answer(str(pred)) == normalize_final_answer(str(gold)):
            return True
    except Exception:
        pass
    return str(pred) == str(gold)


def gold_number(answer):
    """OpenCompass ``gsm8k_dataset_postprocess``."""
    return answer.split("#### ")[1].replace(",", "")


def grade(gold, completion):
    """One GSM8K item under the OpenCompass v2 math evaluator."""
    reference = gold_number(gold)
    try:
        prediction = math_postprocess_v2(completion)
    except Exception:
        prediction = NULL
    if prediction is None or prediction == "":
        prediction = NULL
    return {
        "prediction": prediction,
        "gold_norm": reference,
        "correct": bool(prediction != NULL and is_equiv(prediction, reference)),
    }


def _check_grader():
    """Cases the OpenCompass postprocessor and v2 equivalence must agree on."""
    gold = "Janet sells 16 - 3 - 4 = 9 eggs.\nShe makes 9 * 2 = 18.\n#### 18"
    boxed = grade(gold, "Working.\nSo \\boxed{18}")
    if not boxed["correct"] or boxed["prediction"] != "18":
        raise RuntimeError(f"boxed answer failed: {boxed}")
    comma = grade("Total.\n#### 1,000", "\\boxed{1,000}")
    if not comma["correct"] or comma["gold_norm"] != "1000":
        raise RuntimeError(f"comma gold failed: {comma}")
    sentence = grade(gold, "Therefore the answer is 18.")
    if not sentence["correct"]:
        raise RuntimeError(f"answer-is sentence failed: {sentence}")
    wrong = grade(gold, "\\boxed{17}")
    if wrong["correct"]:
        raise RuntimeError(f"wrong box should fail: {wrong}")
    missing = grade(gold, "I cannot solve this")
    if missing["correct"]:
        raise RuntimeError(f"missing answer should fail: {missing}")
    first_box = grade(gold, "\\boxed{17} then \\boxed{18}")
    if first_box["prediction"] != "18" or not first_box["correct"]:
        raise RuntimeError(f"last box should win: {first_box}")


def build_prompt(question):
    """OpenCompass gsm8k 0-shot v2 user turn."""
    return question + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def cut_at_stops(text):
    cut = len(text)
    for stop in STOPS:
        found = text.find(stop)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def token_records(tokenizer, generation, stop_ids):
    """One record per generated token, with the piece it adds to the decode.

    Prefix decoding keeps piece boundaries aligned with ``tokenizer.decode``
    of the whole completion, including the spaces a byte-pair token owns.
    """
    records = []
    ids = []
    for block_index, block in enumerate(generation.blocks):
        trace = getattr(block, "trace", None)
        for index, token_id in enumerate(block.token_ids):
            if token_id in stop_ids:
                return records, ids
            piece_record = {
                "block": block_index,
                "index": index,
                "token_id": token_id,
                "commit_step": block.commit_step[index],
            }
            if trace is not None:
                confidence = trace["confidence"][index]
                runner_p = trace["runner_up_p"][index]
                piece_record["confidence"] = round(confidence, 4)
                piece_record["runner_up_id"] = trace["runner_up_id"][index]
                piece_record["runner_up_p"] = round(runner_p, 4)
                piece_record["margin"] = round(confidence - runner_p, 4)
            records.append(piece_record)
            ids.append(token_id)
    return records, ids


def attach_pieces(tokenizer, records, ids):
    """Fill ``piece`` / character offsets. Returns the decoded completion."""
    previous = ""
    for end, record in enumerate(records, start=1):
        current = tokenizer.decode(ids[:end])
        record["piece"] = current[len(previous):]
        record["start"] = len(previous)
        record["end"] = len(current)
        previous = current
    return previous


def runner_up_text(tokenizer, record):
    if "runner_up_id" not in record:
        return ""
    return tokenizer.decode([record["runner_up_id"]])


def answer_span(text):
    """Character span of the answer ``math_postprocess_v2`` reads.

    A boxed answer is the span inside the last ``\\boxed{...}``. Otherwise it
    is the extracted string if that string still occurs in the completion.
    """
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        inner = remove_boxed(boxed)
        if inner is not None:
            at = text.rfind(boxed)
            start = at + len("\\boxed{")
            return (start, start + len(inner)), "boxed"
    try:
        extracted = math_postprocess_v2(text)
    except Exception:
        return None, ""
    if not extracted:
        return None, ""
    at = text.rfind(extracted)
    if at < 0:
        return None, ""
    return (at, at + len(extracted)), "postprocess"


def overlapping(records, span):
    if span is None:
        return []
    start, end = span
    return [
        record
        for record in records
        if record["start"] < end and record["end"] > start
    ]


def trace_summary(records, answer_tokens, steps, which):
    """One sentence on how the answer tokens were committed relative to the rest."""
    if not answer_tokens:
        return "No number could be extracted, so the completion has no answer span to trace."
    answer_step = sum(token["commit_step"] for token in answer_tokens) / len(answer_tokens)
    answer_ids = {(token["block"], token["index"]) for token in answer_tokens}
    others = [
        token
        for token in records
        if (token["block"], token["index"]) not in answer_ids and token["piece"].strip()
    ]
    bits = []
    for token in answer_tokens:
        alternative = token.get("runner_up", "")
        bits.append(
            f"{token['piece']!r} committed at step {token['commit_step']} of {steps} "
            f"in block {token['block']} (p={token['confidence']:.3f}, "
            f"runner-up {alternative!r} p={token['runner_up_p']:.3f})"
        )
    sentence = f"The {which} span is " + "; ".join(bits) + "."
    if others and "confidence" in answer_tokens[0]:
        other_step = sum(token["commit_step"] for token in others) / len(others)
        if answer_step + 0.25 < other_step:
            sentence += (
                " Those tokens were committed earlier than the rest of the completion,"
                " so they were among the positions the model was most sure about"
                " while the surrounding text was still masked."
            )
        elif answer_step > other_step + 0.25:
            sentence += (
                " Those tokens were committed later than the rest of the completion,"
                " after the model had already filled the positions it was more sure of."
            )
        else:
            sentence += (
                " Those tokens were committed on a similar step to the rest of the completion."
            )
    return sentence


def unmasking(records, block_index):
    """Commit order inside one block, which is the denoising trace for that block."""
    group = [token for token in records if token["block"] == block_index]
    lines = []
    for step in sorted({token["commit_step"] for token in group}):
        committed = sorted(
            (token for token in group if token["commit_step"] == step),
            key=lambda token: token["index"],
        )
        shown = ", ".join(
            f"{token['piece']!r}@{token['index']} p={token.get('confidence', 0):.2f}"
            for token in committed
        )
        lines.append(f"step {step}: {shown}")
    return lines


def public_token(token):
    """Drop character offsets from the saved record. The piece is enough."""
    keep = (
        "block",
        "index",
        "token_id",
        "piece",
        "commit_step",
        "confidence",
        "runner_up_id",
        "runner_up",
        "runner_up_p",
        "margin",
    )
    return {key: token[key] for key in keep if key in token}


def score_row(row):
    """How clear this row's answer decision is. Higher is a sharper trace."""
    tokens = row.get("answer_trace") or []
    if not tokens or "confidence" not in tokens[0]:
        return None
    confidence = min(token["confidence"] for token in tokens)
    step = sum(token["commit_step"] for token in tokens) / len(tokens)
    return confidence, -step


def write_summary(path, rows, protocol):
    graded = [row for row in rows if "correct" in row]
    count = len(graded)
    hits = sum(row["correct"] for row in graded)

    def mean(values):
        return sum(values) / len(values) if values else None

    def answer_stats(correct):
        chosen = [
            row
            for row in graded
            if row["correct"] == correct and row.get("answer_trace")
        ]
        confidences = [
            token["confidence"]
            for row in chosen
            for token in row["answer_trace"]
            if "confidence" in token
        ]
        steps = [
            token["commit_step"]
            for row in chosen
            for token in row["answer_trace"]
        ]
        return {
            "n": len(chosen),
            "mean_answer_confidence": mean(confidences),
            "mean_answer_commit_step": mean(steps),
        }

    correct_rows = [row for row in graded if row["correct"] and score_row(row)]
    wrong = [
        row
        for row in graded
        if not row["correct"] and row.get("prediction") != NULL and score_row(row)
    ]
    clearest = max(correct_rows, key=score_row) if correct_rows else None
    confident_wrong = max(wrong, key=score_row) if wrong else None

    def feature(row):
        if row is None:
            return None
        return {
            "index": row["index"],
            "question": row["question"],
            "gold_norm": row["gold_norm"],
            "correct": row["correct"],
            "prediction": row["prediction"],
            "completion": row["completion"],
            "trace_summary": row["trace_summary"],
            "answer_trace": row["answer_trace"],
            "answer_block_unmasking": row["answer_block_unmasking"],
        }

    summary = {
        "protocol": protocol,
        "n": count,
        "accuracy": hits / count if count else None,
        "correct": hits,
        "mean_prefill_s": mean([row["prefill_s"] for row in graded]),
        "mean_decode_s": mean([row["decode_s"] for row in graded]),
        "correct_answers": answer_stats(True),
        "incorrect_answers": answer_stats(False),
        "clearest_correct_trace": feature(clearest),
        "most_confident_incorrect_trace": feature(confident_wrong),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def load_problems(limit):
    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]")
    return [
        {"index": index, "question": row["question"], "answer": row["answer"]}
        for index, row in enumerate(test)
    ]


def run(args):
    _check_grader()
    problems = load_problems(args.limit)
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    summary_path = os.path.join(args.out, "summary.json")
    done = {}
    if os.path.exists(results_path) and not args.overwrite:
        with open(results_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[row["index"]] = row
    pending = [problem for problem in problems if problem["index"] not in done]
    print(
        f"problems {len(problems)}  done {len(done)}  pending {len(pending)}  "
        f"steps {args.steps}  strategy {args.strategy}  threshold {args.threshold}",
        flush=True,
    )
    protocol = {
        "benchmark": "gsm8k",
        "harness": "OpenCompass gsm8k_0shot_v2_gen_17d799",
        "evaluator": "MATHEvaluator v2 is_equiv after math_postprocess_v2",
        "reference": "gsm8k_dataset_postprocess",
        "split": f"test[:{args.limit}]",
        "fewshot": 0,
        "prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        "decoding": "greedy",
        "steps": args.steps,
        "strategy": args.strategy,
        "threshold": args.threshold,
        "block_size": 32,
        "max_new_tokens": args.max_new_tokens,
        "metric": "accuracy",
    }
    if not pending:
        summary = write_summary(summary_path, list(done.values()), protocol)
        accuracy = summary["accuracy"]
        print(f"n {summary['n']}  accuracy {accuracy:.3f}", flush=True)
        return

    import torch
    from transformers import AutoTokenizer

    from run_forward import MODEL, load_model
    from sdar_engine import SDAREngine

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    engine = SDAREngine(model, tokenizer)
    append = open(results_path, "w" if args.overwrite else "a", encoding="utf-8")
    try:
        for problem in pending:
            prompt = build_prompt(problem["question"])
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
            print(
                f"index {problem['index']}  prompt_tokens {len(prompt_ids)}",
                flush=True,
            )

            def on_block(result, block):
                print(
                    f"  block {len(result.blocks)} passes {block.passes}",
                    flush=True,
                )

            generation = engine.generate(
                prompt_ids,
                steps=args.steps,
                strategy=args.strategy,
                threshold=args.threshold,
                max_new_tokens=args.max_new_tokens,
                record_trace=True,
                stop_texts=STOPS,
                on_block=on_block,
            )
            records, ids = token_records(tokenizer, generation, engine.stop_ids)
            decoded = attach_pieces(tokenizer, records, ids)
            completion = cut_at_stops(decoded)
            kept = [record for record in records if record["start"] < len(completion)]
            for record in kept:
                if record["end"] > len(completion):
                    record["piece"] = record["piece"][: len(completion) - record["start"]]
                    record["end"] = len(completion)
                record["runner_up"] = runner_up_text(tokenizer, record)
            scored = grade(problem["answer"], completion)
            span, which = answer_span(completion)
            answer_tokens = overlapping(kept, span)
            block_lines = []
            if answer_tokens:
                block_lines = unmasking(kept, answer_tokens[0]["block"])
            row = {
                "index": problem["index"],
                "question": problem["question"],
                "gold": problem["answer"],
                "completion": completion,
                "prompt_tokens": len(prompt_ids),
                "new_tokens": len(kept),
                "prefill_s": round(generation.prefill_s, 3),
                "decode_s": round(generation.decode_s, 3),
                "passes": generation.passes,
                "stopped": generation.stopped,
                "trace_summary": trace_summary(kept, answer_tokens, args.steps, which),
                "answer_block_unmasking": block_lines,
                "answer_trace": [public_token(token) for token in answer_tokens],
                "tokens": [public_token(token) for token in kept],
            }
            row.update(scored)
            append.write(json.dumps(row, ensure_ascii=False) + "\n")
            append.flush()
            done[row["index"]] = row
            write_summary(summary_path, list(done.values()), protocol)
            print(
                f"index {row['index']}  correct {int(row['correct'])}  "
                f"gold {row['gold_norm']}  got {row['prediction']!r}  "
                f"decode_s {row['decode_s']:.1f}  stop {row['stopped']}",
                flush=True,
            )
            print(row["trace_summary"], flush=True)
            del generation
            torch.cuda.empty_cache()
    finally:
        append.close()
    summary = write_summary(summary_path, list(done.values()), protocol)
    print(
        f"n {summary['n']}  accuracy {summary['accuracy']:.3f}",
        flush=True,
    )
    print(f"wrote {results_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Grade SDAR on GSM8K and record commit traces.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--strategy", choices=("static", "dynamic"), default="static")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default="eval_gsm8k")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-grader", action="store_true")
    args = parser.parse_args()
    if args.check_grader:
        _check_grader()
        print("grader ok", flush=True)
        return
    run(args)


if __name__ == "__main__":
    main()
