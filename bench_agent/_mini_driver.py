"""Run mini-swe-agent's headless DefaultAgent once, in a subprocess.

mini-swe-agent's CLI allocates a terminal UI at import time and dies under a pipe
(`OSError: [Errno 22]` from `loop.add_reader`), so the bench drives the library directly.
A subprocess is still used so the bench's wall-clock cap can kill a wedged agent outright.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
import minisweagent.config as cfgmod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--step-limit", type=int, default=25)
    args = ap.parse_args()

    # `default.yaml` drives the model through OpenAI tool calls. qwen2.5-coder:7b on Ollama does
    # not emit them reliably: every one of its first three replies came back "No tool calls found"
    # and the agent exited RepeatedFormatError without touching a file. `mini_textbased.yaml`
    # asks for a fenced bash block instead, which a 7B can actually produce.
    config_path = pathlib.Path(cfgmod.__file__).parent / "mini_textbased.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    agent_cfg = cfg.get("agent", {})
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("environment", {})

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    model_kwargs.update(temperature=0, max_tokens=1024)
    model = LitellmTextbasedModel(model_name=f"openai/{args.model}",
                                  model_kwargs=model_kwargs,
                                  cost_tracking="ignore_errors",
                                  **{k: v for k, v in model_cfg.items()
                                     if k not in {"model_name", "model_kwargs", "cost_tracking"}})
    env = LocalEnvironment(cwd=args.cwd, **{k: v for k, v in env_cfg.items() if k != "cwd"})
    agent = DefaultAgent(
        model, env,
        **{k: v for k, v in agent_cfg.items()
           if k not in {"step_limit", "cost_limit", "output_path", "mode"}},
        step_limit=args.step_limit, cost_limit=0.0, output_path=pathlib.Path(args.traj),
    )
    task = pathlib.Path(args.task_file).read_text()
    try:
        out = agent.run(task)
    except Exception as exc:  # noqa: BLE001
        out = {"exit_status": type(exc).__name__, "error": str(exc)[:1000]}
    print(json.dumps({"exit": out, "n_calls": agent.n_calls}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
