"""Run the baseline agent (Qwen/GPT-4o) against dev seeds and report scores.

Usage:
    python scripts/run_baseline.py --model gpt-4o --seeds dev
    python scripts/run_baseline.py --model Qwen/Qwen2.5-72B-Instruct --seeds train

Writes results to results/baseline_{model_slug}.json.
Requires API_BASE_URL, MODEL_NAME (or --model), and HF_TOKEN env vars.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_SPLIT = {
    "deductive_liability": {
        "train": list(range(1001, 1016)),
        "dev": list(range(1101, 1106)),
        "test": list(range(1201, 1206)),
    },
    "abductive_conflict": {
        "train": list(range(2001, 2016)),
        "dev": list(range(2101, 2106)),
        "test": list(range(2201, 2206)),
    },
    "adversarial_fabrication": {
        "train": list(range(3001, 3016)),
        "dev": list(range(3101, 3106)),
        "test": list(range(3201, 3206)),
    },
}


def make_ref_db() -> sqlite3.Connection:
    """Build an in-memory reference DB."""
    from server.schema import create_tables
    from data.setup import load_all

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    load_all(db)
    return db


async def run_episode(
    env,
    client,
    model: str,
    task_name: str,
    seed: int,
) -> dict:
    """Run one episode with the model agent and return the result dict."""
    from server.models import ClarusAction

    obs = await env.reset(task_name, seed=seed)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a healthcare billing dispute specialist. "
                "Your job is to investigate the patient's billing complaint, "
                "gather evidence from available records, identify the error, "
                "and file the correct resolution. "
                "Always authenticate first. Cite artifact IDs as evidence. "
                "Complete: authenticate → investigate → diagnose → draft → "
                "submit → communicate → audit → close."
            ),
        }
    ]

    rewards = []
    step_count = 0
    max_steps = {"deductive_liability": 12, "abductive_conflict": 15,
                 "adversarial_fabrication": 22}.get(task_name, 18)

    while not obs.done and step_count < max_steps:
        user_content = (
            f"Step {obs.step_number + 1}/{max_steps}\n"
            f"Case: {obs.patient_complaint}\n"
            f"Patient: {obs.patient_name} (state: {obs.patient_emotional_state})\n"
            f"API calls: {obs.api_calls_used}/{obs.api_call_budget}\n"
            f"Rate-limited tools: {obs.rate_limited_tools}\n"
            f"Last action: {obs.last_action_type} → {obs.last_action_result}\n"
            f"History: {obs.action_log_summary[-5:]}\n\n"
            "Choose the next action. Respond with JSON: "
            '{"action_type": "...", "parameters": {...}}'
        )
        messages.append({"role": "user", "content": user_content})

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": raw})

            # Parse action from model response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                action_dict = json.loads(raw[start:end])
                action = ClarusAction(
                    action_type=action_dict.get("action_type", "close_case"),
                    parameters=action_dict.get("parameters", {}),
                )
            else:
                action = ClarusAction(action_type="close_case")

        except Exception as e:
            print(f"  Model error: {e}", flush=True)
            action = ClarusAction(action_type="close_case")

        result = await env.step(action)
        obs = result.observation
        rewards.append(result.reward)
        step_count += 1

        if result.done:
            episode_score = result.info.get("episode_score", 0.0)
            return {
                "task_name": task_name,
                "seed": seed,
                "episode_score": episode_score,
                "steps": step_count,
                "rewards": rewards,
            }

    # Timeout — close case if not done
    if not obs.done:
        result = await env.step(
            ClarusAction(action_type="close_case", parameters={"outcome_code": "timeout"})
        )
        rewards.append(result.reward)
        episode_score = result.info.get("episode_score", 0.0)
    else:
        episode_score = 0.0

    return {
        "task_name": task_name,
        "seed": seed,
        "episode_score": episode_score,
        "steps": step_count,
        "rewards": rewards,
    }


async def main(model: str, split: str) -> None:
    """Run baseline across all tasks and seeds in the given split."""
    from openai import OpenAI
    from server.env import ClarusEnv

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN", "")
    model = os.getenv("MODEL_NAME", model)

    client = OpenAI(base_url=api_base, api_key=api_key)
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)

    all_results = []
    task_scores: dict[str, list[float]] = {}

    for task_name, split_seeds in SEED_SPLIT.items():
        seeds = split_seeds.get(split, split_seeds["dev"])
        task_scores[task_name] = []
        print(f"\n=== {task_name} ({split} split, {len(seeds)} seeds) ===")

        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            try:
                result = await run_episode(env, client, model, task_name, seed)
                score = result["episode_score"]
                task_scores[task_name].append(score)
                all_results.append(result)
                print(f"score={score:.3f} steps={result['steps']}")
            except Exception as e:
                print(f"ERROR: {e}")
                task_scores[task_name].append(0.0)

    print("\n=== SUMMARY ===")
    overall_scores = []
    for task_name, scores in task_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        overall_scores.extend(scores)
        print(f"  {task_name}: avg={avg:.3f} scores={[f'{s:.2f}' for s in scores]}")

    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"  OVERALL: {overall:.3f}")

    # Save results
    slug = model.replace("/", "_").replace("-", "_").lower()
    out_path = RESULTS_DIR / f"baseline_{slug}.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "model": model,
                "split": split,
                "overall_score": overall,
                "task_scores": {k: sum(v)/len(v) for k, v in task_scores.items() if v},
                "results": all_results,
            },
            fh,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clarus baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--seeds", default="dev", choices=["train", "dev", "test"])
    args = parser.parse_args()
    asyncio.run(main(args.model, args.seeds))
