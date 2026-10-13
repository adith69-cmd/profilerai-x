"""Synthesize instruction-tuning data with SKILL-ALIGNED sampling.

v2 fix: references and targets are drawn only from rows whose text shares
at least one skill with the profile. This teaches the model skill-conditioning,
not just "output 10 questions."

Output: JSONL, chat-format, one example per line.
Split: 80% train / 10% val / 10% test.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/finetune_v2")
N_EXAMPLES = 2000
SEED = 42
N_REFERENCE = 3
N_TARGETS = 10

PROFILES = {
    "Ai": [
        {"role": "ML Engineer", "years": 3, "skills": "Python, PyTorch, Hugging Face, RAG"},
        {"role": "Data Scientist", "years": 5, "skills": "Python, scikit-learn, XGBoost, SQL"},
        {"role": "ML Engineer", "years": 2, "skills": "Python, TensorFlow, Computer Vision"},
        {"role": "AI Engineer", "years": 4, "skills": "Python, LLMs, LangChain, vector DBs"},
    ],
    "General_Programming": [
        {"role": "Backend Engineer", "years": 3, "skills": "Python, FastAPI, PostgreSQL, Docker"},
        {"role": "Software Engineer", "years": 2, "skills": "Java, Spring Boot, MySQL"},
        {"role": "Backend Engineer", "years": 5, "skills": "Go, gRPC, Redis, Kubernetes"},
        {"role": "Full-Stack Engineer", "years": 4, "skills": "Python, React, Node.js, AWS"},
    ],
    "Web_Development": [
        {"role": "Frontend Engineer", "years": 3, "skills": "React, TypeScript, Vite, Tailwind"},
        {"role": "Frontend Engineer", "years": 2, "skills": "JavaScript, Vue, CSS, REST APIs"},
        {"role": "Full-Stack Engineer", "years": 4, "skills": "React, FastAPI, PostgreSQL"},
        {"role": "Web Developer", "years": 1, "skills": "HTML, CSS, JavaScript, React"},
    ],
}

SYSTEM_PROMPT = (
    "You are an interview coach. Given a candidate profile and reference "
    "Q/A pairs, generate 10 fresh interview questions that probe the "
    "candidate's depth in the listed skills. Do not repeat reference "
    "questions verbatim. Number each question 1. through 10."
)


def fmt_profile(p: dict) -> str:
    return (f"CANDIDATE PROFILE:\n- Role: {p['role']}\n"
            f"- Years of experience: {p['years']}\n- Skills: {p['skills']}")


def fmt_refs(refs: list[dict]) -> str:
    lines = ["REFERENCE Q/A PAIRS:"]
    for i, r in enumerate(refs, 1):
        lines.append(f"{i}. Q: {r['question']}")
        lines.append(f"   A: {r['answer']}")
    return "\n".join(lines)


def fmt_targets(tgts: list[dict]) -> str:
    return "\n".join(f"{i}. {t['question']}" for i, t in enumerate(tgts, 1))


def row_text(r: dict) -> str:
    return (r["question"] + " " + r["answer"]).lower()


def build_example(profile: dict, pool: list[dict], rng: random.Random) -> dict:
    sampled = rng.sample(pool, N_REFERENCE + N_TARGETS)
    refs, targets = sampled[:N_REFERENCE], sampled[N_REFERENCE:]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{fmt_profile(profile)}\n\n{fmt_refs(refs)}"},
            {"role": "assistant", "content": fmt_targets(targets)},
        ],
        "metadata": {"domain": profile["_domain"], "profile": profile["role"]},
    }


def main() -> None:
    rng = random.Random(SEED)
    ds = load_dataset("Shlok307/Interview_questions", split="train")

    by_domain: dict[str, list[dict]] = {}
    for row in ds:
        by_domain.setdefault(row["domain"], []).append(
            {"question": row["question"], "answer": row["answer"]}
        )
    print(f"[load] {len(ds)} rows across {len(by_domain)} domains")
    for d, v in by_domain.items():
        print(f"       {d}: {len(v)}")

    # Precompute per-profile skill-matched pools
    profile_pools: dict[tuple[str, int], list[dict]] = {}
    print("\n[filter] skill-matched pool sizes")
    need = N_REFERENCE + N_TARGETS
    for domain, pool in by_domain.items():
        for pi, profile in enumerate(PROFILES[domain]):
            skills = [s.strip().lower() for s in profile["skills"].split(",")]
            matched = [r for r in pool
                       if any(s in row_text(r) for s in skills)]
            if len(matched) < need:
                print(f"  {domain:22s} / {profile['role']:22s}: "
                      f"{len(matched):4d} -> FALLBACK to domain pool")
                matched = pool
            else:
                print(f"  {domain:22s} / {profile['role']:22s}: {len(matched):4d}")
            profile_pools[(domain, pi)] = matched

    # Build examples
    keys = list(profile_pools.keys())
    examples: list[dict] = []
    for _ in range(N_EXAMPLES):
        domain, pi = rng.choice(keys)
        profile = dict(PROFILES[domain][pi])
        profile["_domain"] = domain
        examples.append(build_example(profile, profile_pools[(domain, pi)], rng))

    rng.shuffle(examples)
    n = len(examples); n_train, n_val = int(0.8 * n), int(0.1 * n)
    splits = {
        "train": examples[:n_train],
        "val":   examples[n_train:n_train + n_val],
        "test":  examples[n_train + n_val:],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {path}  ({len(rows)} examples)")

    print("\n[sample example]")
    for m in splits["train"][0]["messages"]:
        print(f"\n--- {m['role']} ---")
        print(m["content"][:400])


if __name__ == "__main__":
    main()