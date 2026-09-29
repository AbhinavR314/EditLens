"""
Paraphrase AI-written text with a local open-weights LLM, to test how well a
detector (EditLens or any other) holds up against paraphrase evasion.

Reads a data CSV, picks the AI-generated / AI-edited rows, rewrites each one with
an instruction-tuned model running on your own GPU (no API, no cost), and writes
the rewrite into the `text` column so the result can go straight into
scripts/inference.py.

Labels are kept from the original row on purpose: a paraphrased AI text is still
AI text, so label / text_type / cosine_score still say how much AI was involved
and a robust detector should keep flagging it. The detector scores already in the
CSV describe the *original* text, so they are renamed orig_<col> to avoid mixing
them up with scores computed on the paraphrase.

Rows are paraphrased in chunks of at most --chunk_words words (a few sentences),
then rejoined, so documents of any length fit the model's context. Results are
appended to paraphrases.jsonl after every group of documents, so an interrupted
run resumes where it stopped when re-run with the same arguments.

Usage:
  # plumbing check, no model download
  python scripts/attack/paraphrase.py --dry_run --n_per_type 5

  # real run (downloads the model on first use; ~15 GB for the 7B default)
  python scripts/attack/paraphrase.py --strength heavy --n_per_type 250

  # score the paraphrased set with a detector
  python scripts/inference.py --checkpoint pangram/editlens_roberta-large \
    --base_model FacebookAI/roberta-large --max_length 512 \
    --dataset outputs/paraphrase_heavy_Qwen2.5-7B-Instruct/arrow --text_col text \
    --output outputs/paraphrase_heavy_Qwen2.5-7B-Instruct/scored.jsonl
"""

import argparse
import difflib
import json
import os
import re
import time

import pandas as pd

SYSTEM_PROMPT = "You are a careful text rewriting tool. You output only the rewritten text."

FIDELITY_RULE = (
    "Do not add any name, fact, event, or detail that is not already in the passage, even "
    "if it seems like a natural comparison or elaboration. Every person, place, and claim "
    "in your rewrite must already be present in the original."
)

PROMPTS = {
    # Reword and restructure, keep the register
    "light": (
        "Rewrite the passage below in your own words. Keep every fact and the overall "
        "meaning, keep roughly the same length, and change the wording and sentence "
        f"structure wherever it reads naturally. {FIDELITY_RULE} Output only the rewritten "
        "passage, with no introduction or commentary.\n\nPassage:\n{chunk}"
    ),
    # Evasion-style: strip the register detectors key on
    "heavy": (
        "Rewrite the passage below so it reads like it was written by an ordinary person "
        "rather than an AI assistant: vary the sentence lengths, use plain everyday "
        "wording, drop stock phrases and formal transitions, and reorder or merge ideas "
        "where that reads naturally. Keep every fact and the overall meaning, and keep "
        f"roughly the same length. {FIDELITY_RULE} Output only the rewritten passage, with "
        "no introduction or commentary.\n\nPassage:\n{chunk}"
    ),
}

# Columns that are labels relative to source_text, not detector outputs
LABEL_SCORE_COLS = {"cosine_score", "soft_ngrams_score"}

PREAMBLE = re.compile(r"^\s*(sure|certainly|of course|here(?:'s| is| are))[^\n]*:\s*\n+", re.IGNORECASE)


def split_chunks(text: str, max_words: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most max_words words.
    A single sentence longer than max_words is split by words."""
    chunks, current, n = [], [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        if not words:
            continue
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current, n = [], 0
            chunks.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
            continue
        if current and n + len(words) > max_words:
            chunks.append(" ".join(current))
            current, n = [], 0
        current.append(sentence.strip())
        n += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def clean_output(text: str) -> str:
    """Strip a leading 'Here is the rewrite:' line and wrapping quotes."""
    text = PREAMBLE.sub("", text.strip(), count=1).strip()
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text


_COMMON_CAPITALIZED = {"i", "i'm", "i'll", "i've", "i'd"}


def _normalize_apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def capitalized_words(text: str) -> set[str]:
    """Capitalized word tokens in `text` that occur away from the start of a
    sentence -- a cheap way to exclude ordinary sentence-initial capitals (which
    include, in real text, all sorts of ordinary nouns/verbs a rephrase can land
    at the front of a new sentence, not just "The"/"However"-type words -- a
    fixed word list can't cover that) and keep tokens capitalized because
    they're proper nouns.

    Known gap: a sentence-initial hallucination (e.g. "Smith filed the
    report.") isn't caught. Tried filtering by a common-sentence-starter word
    list instead, but on real text that flagged far more ordinary words than
    it caught real hallucinations, so it isn't worth the noise; this simpler
    away-from-sentence-start rule is what actually caught the one confirmed
    hallucination seen so far, which was mid-sentence.
    """
    text = _normalize_apostrophes(text)
    found = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for i, w in enumerate(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)):
            if not w[0].isupper() or w.lower() in _COMMON_CAPITALIZED:
                continue
            if i > 0:
                found.add(w)
    return found


def _strip_possessive(word: str) -> str:
    lower = word.lower()
    for suffix in ("'s", "s'"):
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return lower


def new_names(orig_text: str, paraphrase: str) -> list[str]:
    """Capitalized words in the paraphrase that never appear (in any form, any
    position, any case) anywhere in the original -- a cheap screen for invented
    names/places. Checks both the word as-is and with a trailing possessive 's
    stripped, so a possessive of a name already in the original (e.g. original
    has "Westbury", paraphrase has "Westbury's") isn't flagged.

    Heuristic, not a hard filter: misses a hallucinated name that only occurs at
    the very start of a sentence in the paraphrase (see capitalized_words), and
    can flag a genuine name if the paraphrase's rewrite lands it right after a
    quote, colon or dash in a way this function's naive '.!?'-only sentence
    splitting treats as "mid-sentence" -- read the new_names column, don't treat
    a nonzero count as confirmed."""
    orig_lower = _normalize_apostrophes(orig_text).lower()
    return sorted(
        w for w in capitalized_words(paraphrase)
        if w.lower() not in orig_lower and _strip_possessive(w) not in orig_lower
    )


def make_generator(args):
    """Returns generate(prompts) -> list of completions."""
    if args.dry_run:
        # No model: echo the passage in upper case so the plumbing can be tested
        return lambda prompts: [p.split("Passage:\n", 1)[1].upper() for p in prompts]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = args.quant
    if quant == "auto":
        quant = "4bit" if torch.cuda.is_available() else "none"
    if quant == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
        )
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    print(f"Loaded {args.model} ({quant}) on {model.device}")

    def generate(prompts):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate


def paraphrase_docs(group: pd.DataFrame, generate, args) -> list[dict]:
    """Paraphrase every document in `group`. Chunks from all documents in the group are
    pooled and sorted by length so each generation batch has similar-length inputs."""
    doc_chunks = [split_chunks(t, args.chunk_words) for t in group["text"]]
    jobs = [(d, c, chunk) for d, chunks in enumerate(doc_chunks) for c, chunk in enumerate(chunks)]
    jobs.sort(key=lambda j: len(j[2]))

    outputs = {}
    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i : i + args.batch_size]
        prompts = [PROMPTS[args.strength].format(chunk=chunk) for _, _, chunk in batch]
        for (d, c, _), completion in zip(batch, generate(prompts)):
            outputs[(d, c)] = clean_output(completion)

    records = []
    for d, (text_id, text) in enumerate(zip(group["text_id"], group["text"])):
        parts = [outputs[(d, c)] for c in range(len(doc_chunks[d]))]
        paraphrase = " ".join(p for p in parts if p)
        orig_words, new_words = text.lower().split(), paraphrase.lower().split()
        ratio = len(new_words) / max(1, len(orig_words))
        records.append(
            {
                "text_id": text_id,
                "paraphrase": paraphrase,
                "para_word_ratio": round(ratio, 3),
                # 1.0 = identical word sequence, lower = more rewritten
                "para_similarity": round(difflib.SequenceMatcher(None, orig_words, new_words, autojunk=False).ratio(), 3),
                # a chunk came back empty or the length is way off (truncated / refused / rambling)
                "paraphrase_ok": bool(all(parts) and 0.5 <= ratio <= 1.6),
            }
        )
    return records


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["text_id"]: r for r in map(json.loads, f)}


def select_rows(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows to paraphrase, unmodified human rows to keep alongside them)."""
    df = df.drop_duplicates("text_id")

    def sample(part):
        if args.n_per_type <= 0:
            return part
        return part.groupby("text_type", group_keys=False).apply(
            lambda g: g.sample(min(args.n_per_type, len(g)), random_state=args.seed)
        )

    targets = sample(df[df["text_type"].isin(args.types)])
    humans = sample(df[df["text_type"] == "human_written"]) if args.keep_human else df.iloc[0:0]
    return targets, humans


def build_output(targets, humans, records: dict, args) -> pd.DataFrame:
    def prepare(part):
        part = part.copy()
        part["orig_text"] = part["text"]
        renames = {c: f"orig_{c}" for c in part.columns if c.endswith("_score") and c not in LABEL_SCORE_COLS}
        return part.rename(columns=renames)

    attacked = prepare(targets)
    rec = pd.DataFrame(records.values()).set_index("text_id")
    attacked = attacked[attacked["text_id"].isin(rec.index)].copy()
    attacked["text"] = attacked["text_id"].map(rec["paraphrase"])
    for col in ("para_word_ratio", "para_similarity", "paraphrase_ok"):
        attacked[col] = attacked["text_id"].map(rec[col])
    attacked["attack_strength"] = args.strength
    # Cheap screen for the paraphraser inventing names/places not in the source
    # (see new_names() docstring for what this heuristic does and doesn't catch)
    invented = attacked.apply(lambda r: new_names(r["orig_text"], r["text"]), axis=1)
    attacked["new_names"] = invented.map(lambda ns: ", ".join(ns))
    attacked["n_new_names"] = invented.map(len)

    kept = prepare(humans)
    kept["para_word_ratio"], kept["para_similarity"], kept["paraphrase_ok"] = 1.0, 1.0, True
    kept["attack_strength"] = "none"
    kept["new_names"], kept["n_new_names"] = "", 0
    return pd.concat([attacked, kept], ignore_index=True)


def parse_args():
    p = argparse.ArgumentParser(description="Paraphrase AI text with a local LLM (detector-robustness test)")
    p.add_argument("--input", default="data/test.csv", help="CSV with text_id, text, text_type columns")
    p.add_argument("--types", nargs="+", default=["ai_generated", "ai_edited"], help="text_type values to paraphrase")
    p.add_argument("--n_per_type", type=int, default=250, help="Rows per text_type to sample (0 = all)")
    p.add_argument("--keep_human", action=argparse.BooleanOptionalAction, default=True,
                   help="Also include unmodified human rows (same count), so one scoring run also gives the false-positive rate")
    p.add_argument("--strength", choices=list(PROMPTS), default="heavy")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Any local-capable HF chat model")
    p.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto",
                   help="auto = 4-bit on CUDA (fits 12 GB for a 7B model), unquantized otherwise")
    p.add_argument("--batch_size", type=int, default=8, help="Chunks per generation batch")
    p.add_argument("--docs_per_step", type=int, default=8, help="Documents between checkpoint writes")
    p.add_argument("--chunk_words", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None, help="Default: outputs/paraphrase_<strength>_<model>")
    p.add_argument("--dry_run", action="store_true", help="No model: echo text in upper case, to test the plumbing")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "dryrun" if args.dry_run else args.model.split("/")[-1]
    out_dir = args.output_dir or f"outputs/paraphrase_{args.strength}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "paraphrases.jsonl")

    df = pd.read_csv(args.input)
    targets, humans = select_rows(df, args)
    records = load_checkpoint(checkpoint)
    todo = targets[~targets["text_id"].isin(records)]
    print(f"{len(targets)} rows to paraphrase ({len(records)} already done), {len(humans)} human rows kept as-is.")

    if len(todo):
        generate = make_generator(args)
        start = time.time()
        for i in range(0, len(todo), args.docs_per_step):
            new = paraphrase_docs(todo.iloc[i : i + args.docs_per_step], generate, args)
            with open(checkpoint, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
            records.update({r["text_id"]: r for r in new})
            done = min(i + args.docs_per_step, len(todo))
            elapsed = time.time() - start
            print(f"  {done}/{len(todo)} documents  ({elapsed:.0f}s elapsed, ~{elapsed / done * (len(todo) - done):.0f}s left)")

    out = build_output(targets, humans, records, args)
    out.to_csv(os.path.join(out_dir, "paraphrased.csv"), index=False)

    from datasets import Dataset

    Dataset.from_pandas(out, preserve_index=False).save_to_disk(os.path.join(out_dir, "arrow"))

    attacked = out[out["attack_strength"] != "none"]
    print(f"\nWrote {len(out)} rows to {out_dir}/ (paraphrased.csv, arrow/)")
    print(f"Paraphrase quality over {len(attacked)} attacked documents:")
    print(f"  paraphrase_ok: {attacked['paraphrase_ok'].mean():.1%}   (length ratio 0.5-1.6, no empty chunks)")
    print(f"  median word-length ratio: {attacked['para_word_ratio'].median():.2f}")
    print(f"  median similarity to original: {attacked['para_similarity'].median():.2f}   (lower = more rewritten)")
    with_new_names = (attacked["n_new_names"] > 0).mean()
    print(f"  possible invented names: {with_new_names:.1%} of documents flag >=1 (heuristic, read new_names column;"
          f" see docstring on new_names() for what it can miss/false-positive on)")


if __name__ == "__main__":
    main()
