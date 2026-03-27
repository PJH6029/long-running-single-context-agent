from __future__ import annotations

import argparse
from pathlib import Path

from .agents.policies import build_agent_policy
from .agents.runner import RepoAgentRunner, build_memory_backend
from .bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter
from .bench.mixers import PilotEpisodeBuilder
from .bench.types import EpisodeMetrics
from .bench.v0_1 import default_manifest_path, default_output_root, prepare_v0_1_dataset
from .config import load_config
from .utils import ensure_directory, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Interleaved coding benchmark pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-episodes", help="Build mixed pilot episodes")
    build_parser.add_argument("--config", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-v0_1",
        help="Materialize the curated v0.1 sampled benchmark from pinned upstream commits",
    )
    prepare_parser.add_argument("--manifest", default=str(default_manifest_path()))
    prepare_parser.add_argument("--output-root", default=str(default_output_root()))

    single_parser = subparsers.add_parser("run-single", help="Run a single task sanity check")
    single_parser.add_argument("--config", required=True)
    single_parser.add_argument("--source", choices=["swe_bench_verified", "swe_ci"], required=True)
    single_parser.add_argument("--task-id", required=True)
    single_parser.add_argument("--memory-mode", choices=["shared", "filesystem"], default="filesystem")

    mixed_parser = subparsers.add_parser("run-mixed", help="Run one mixed episode")
    mixed_parser.add_argument("--config", required=True)
    mixed_parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    mixed_parser.add_argument("--episode-id", required=True)
    mixed_parser.add_argument("--memory-mode", choices=["shared", "filesystem"], required=True)

    compare_parser = subparsers.add_parser("run-compare", help="Run both memory modes for a split")
    compare_parser.add_argument("--config", required=True)
    compare_parser.add_argument("--split", choices=["dev", "eval"], default="eval")

    args = parser.parse_args()
    if args.command == "build-episodes":
        build_episodes(args.config)
    elif args.command == "prepare-v0_1":
        prepare_v0_1(args.manifest, args.output_root)
    elif args.command == "run-single":
        run_single(args.config, args.source, args.task_id, args.memory_mode)
    elif args.command == "run-mixed":
        run_mixed(args.config, args.split, args.episode_id, args.memory_mode)
    elif args.command == "run-compare":
        run_compare(args.config, args.split)


def build_episodes(config_path: str) -> Path:
    config = load_config(config_path)
    swe_bench_tasks = SWEBenchVerifiedAdapter(config.swe_bench_dataset).load_tasks()
    swe_ci_tasks = SWECIAdapter(config.swe_ci_dataset).load_tasks()
    builder = PilotEpisodeBuilder(
        dev_episodes=config.dev_episodes,
        eval_episodes=config.eval_episodes,
        seed=config.seed,
        max_total_actions=config.max_total_actions,
        max_actions_per_task=config.max_actions_per_task,
        slice_budget=config.slice_budget,
    )
    episodes = builder.build(swe_bench_tasks, swe_ci_tasks)
    output_dir = ensure_directory(Path(config.output_dir))
    return builder.write(output_dir, episodes)


def prepare_v0_1(manifest_path: str, output_root: str) -> Path:
    return prepare_v0_1_dataset(manifest_path=manifest_path, output_root=output_root)


def run_single(config_path: str, source: str, task_id: str, memory_mode: str) -> Path:
    config = load_config(config_path)
    adapter = (
        SWEBenchVerifiedAdapter(config.swe_bench_dataset)
        if source == "swe_bench_verified"
        else SWECIAdapter(config.swe_ci_dataset)
    )
    tasks = adapter.load_tasks()
    task = next(item for item in tasks if item.task_id == task_id)
    runner = RepoAgentRunner(build_agent_policy(config.agent))
    output_dir = _agent_output_dir(config)
    metrics = runner.run_single_task(
        task,
        memory_backend=build_memory_backend(
            memory_mode,
            output_dir / ".task-memory",
            tail_events=config.memory_tail_events,
        ),
        output_root=output_dir,
        max_actions=config.max_actions_per_task,
    )
    result_path = output_dir / f"single-{source}-{task_id}-{memory_mode}.json"
    write_json(result_path, metrics.to_dict())
    return result_path


def run_mixed(config_path: str, split: str, episode_id: str, memory_mode: str) -> Path:
    config = load_config(config_path)
    episode_file = Path(config.output_dir) / "mixed_episodes.json"
    if not episode_file.exists():
        build_episodes(config_path)
    payload = read_json(episode_file)
    episode_payload = next(item for item in payload[split] if item["episode_id"] == episode_id)
    from .bench.types import MixedEpisode

    episode = MixedEpisode.from_dict(episode_payload)
    runner = RepoAgentRunner(build_agent_policy(config.agent))
    output_dir = _agent_output_dir(config)
    metrics = runner.run_mixed_episode(
        episode,
        memory_backend=build_memory_backend(
            memory_mode,
            output_dir / ".task-memory",
            tail_events=config.memory_tail_events,
        ),
        output_root=output_dir,
    )
    result_path = output_dir / f"{episode_id}-{memory_mode}.json"
    write_json(result_path, metrics.to_dict())
    return result_path


def run_compare(config_path: str, split: str) -> Path:
    config = load_config(config_path)
    episode_file = Path(config.output_dir) / "mixed_episodes.json"
    if not episode_file.exists():
        build_episodes(config_path)
    payload = read_json(episode_file)
    runner = RepoAgentRunner(build_agent_policy(config.agent))
    output_dir = _agent_output_dir(config)
    comparisons = []
    shared_metrics: list[EpisodeMetrics] = []
    filesystem_metrics: list[EpisodeMetrics] = []
    from .bench.types import MixedEpisode

    for episode_payload in payload[split]:
        episode = MixedEpisode.from_dict(episode_payload)
        shared = runner.run_mixed_episode(
            episode,
            memory_backend=build_memory_backend(
                "shared", output_dir / ".task-memory", tail_events=config.memory_tail_events
            ),
            output_root=output_dir,
        )
        shared_metrics.append(shared)
        filesystem = runner.run_mixed_episode(
            episode,
            memory_backend=build_memory_backend(
                "filesystem", output_dir / ".task-memory", tail_events=config.memory_tail_events
            ),
            output_root=output_dir,
        )
        filesystem_metrics.append(filesystem)
        shared_success = sum(1 for item in shared.per_task if item.success)
        filesystem_success = sum(1 for item in filesystem.per_task if item.success)
        comparisons.append(
            {
                "agent_name": runner.policy.name,
                "episode_id": episode.episode_id,
                "shared_path": str(output_dir / "shared_transcript" / episode.episode_id / "episode_metrics.json"),
                "filesystem_path": str(output_dir / "filesystem_per_task" / episode.episode_id / "episode_metrics.json"),
                "success_delta": filesystem_success - shared_success,
                "token_delta": filesystem.cumulative_prompt_tokens - shared.cumulative_prompt_tokens,
            }
        )
    result_path = output_dir / f"comparison-{split}.json"
    summary_path = output_dir / f"summary-{split}.json"
    write_json(result_path, comparisons)
    write_json(
        summary_path,
        {
            "agent_name": runner.policy.name,
            "split": split,
            "shared_transcript": _summarize_mode(shared_metrics),
            "filesystem_per_task": _summarize_mode(filesystem_metrics),
        },
    )
    return result_path


def _agent_output_dir(config) -> Path:
    return ensure_directory(Path(config.output_dir) / "runs" / config.agent.slug)


def _summarize_mode(metrics: list[EpisodeMetrics]) -> dict[str, float | int]:
    if not metrics:
        return {
            "episodes": 0,
            "episodes_both_tasks_solved": 0,
            "task_success_total": 0,
            "average_cumulative_prompt_tokens": 0.0,
            "average_stale_memory_errors": 0.0,
            "average_duplicate_work_rate": 0.0,
            "average_wall_clock_seconds": 0.0,
            "policy_error_count": 0,
        }
    episode_count = len(metrics)
    task_success_total = sum(sum(1 for item in metric.per_task if item.success) for metric in metrics)
    return {
        "episodes": episode_count,
        "episodes_both_tasks_solved": sum(metric.both_tasks_solved for metric in metrics),
        "task_success_total": task_success_total,
        "average_cumulative_prompt_tokens": sum(metric.cumulative_prompt_tokens for metric in metrics)
        / episode_count,
        "average_stale_memory_errors": sum(metric.stale_memory_errors for metric in metrics) / episode_count,
        "average_duplicate_work_rate": sum(metric.duplicate_work_rate for metric in metrics) / episode_count,
        "average_wall_clock_seconds": sum(metric.wall_clock_seconds for metric in metrics) / episode_count,
        "policy_error_count": sum(metric.policy_error_count for metric in metrics),
    }


if __name__ == "__main__":
    main()
