from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..bench.memory import FilesystemPerTaskMemory, MemoryBackend, SharedTranscriptMemory
from ..bench.types import EpisodeMetrics, MixedEpisode, ScheduleEvent, TaskEvaluation, TaskSpec, TaskState
from ..utils import ensure_directory, write_json
from .actions import AgentAction
from .policies import AgentPolicy


@dataclass(slots=True)
class ActionExecution:
    action: AgentAction
    observation: str
    prompt_tokens: int
    stale_memory_error: bool


class RepoAgentRunner:
    def __init__(self, policy: AgentPolicy):
        self.policy = policy

    def run_single_task(
        self,
        task_spec: TaskSpec,
        *,
        memory_backend: MemoryBackend,
        output_root: str | Path,
        max_actions: int,
    ) -> EpisodeMetrics:
        episode = MixedEpisode(
            episode_id=f"single-{task_spec.task_id}",
            tasks=[task_spec],
            schedule=[],
            seed=0,
            max_total_actions=max_actions,
            max_actions_per_task=max_actions,
            split=task_spec.split,
        )
        for index in range(max_actions):
            episode.schedule.append(ScheduleEvent(step_idx=index, active_task_id=task_spec.task_id, slice_budget=1))
        return self.run_mixed_episode(episode, memory_backend=memory_backend, output_root=output_root)

    def run_mixed_episode(
        self, episode: MixedEpisode, *, memory_backend: MemoryBackend, output_root: str | Path
    ) -> EpisodeMetrics:
        run_root = ensure_directory(Path(output_root) / memory_backend.name / episode.episode_id)
        workspace_root = ensure_directory(run_root / "workspaces")
        task_states = self._prepare_task_states(episode.tasks, workspace_root, run_root)
        memory_backend.initialize_episode(episode, task_states)

        prompt_tokens_per_slice: list[int] = []
        stale_memory_errors = 0
        action_fingerprints: dict[str, list[str]] = {task.task_id: [] for task in episode.tasks}
        last_active_task_id: str | None = None

        for event in episode.schedule:
            state = task_states[event.active_task_id]
            if state.done or state.action_count >= episode.max_actions_per_task:
                continue
            if last_active_task_id and last_active_task_id != state.task_id and state.action_count > 0:
                state.resume_count += 1
            prompt = memory_backend.build_prompt(state.task_id, task_states)
            action = self.policy.next_action(memory_backend.task_specs[state.task_id], state, prompt)
            execution = self._execute_action(memory_backend.task_specs[state.task_id], state, action, prompt)
            state.action_count += 1
            state.status = "finished" if action.kind == "finish" else "running"
            state.done = action.kind == "finish"
            state.last_result = execution.observation
            state.history.append(
                {
                    "step_idx": event.step_idx,
                    "action_kind": action.kind,
                    "fingerprint": action.fingerprint(),
                    "observation": execution.observation,
                }
            )
            prompt_tokens_per_slice.append(execution.prompt_tokens)
            stale_memory_errors += int(execution.stale_memory_error)
            action_fingerprints[state.task_id].append(action.fingerprint())
            memory_backend.append_event(
                state.task_id,
                "assistant",
                json.dumps({"action": action.to_dict(), "observation": execution.observation}, indent=2),
            )
            memory_backend.sync_task_state(state, task_states)
            last_active_task_id = state.task_id

        evaluations = [self._evaluate_task(task, task_states[task.task_id]) for task in episode.tasks]
        metrics = self._summarize_episode(
            episode,
            memory_mode=memory_backend.name,
            task_states=task_states,
            evaluations=evaluations,
            prompt_tokens_per_slice=prompt_tokens_per_slice,
            stale_memory_errors=stale_memory_errors,
            action_fingerprints=action_fingerprints,
        )
        write_json(run_root / "episode_metrics.json", metrics.to_dict())
        return metrics

    def _prepare_task_states(
        self, tasks: list[TaskSpec], workspace_root: Path, run_root: Path
    ) -> dict[str, TaskState]:
        task_states: dict[str, TaskState] = {}
        for task in tasks:
            if not task.repo_source_path:
                raise ValueError(f"Task {task.task_id} is missing repo_source_path.")
            workspace = workspace_root / task.task_id
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(task.repo_source_path, workspace)
            self._run_setup(task, workspace)
            task_states[task.task_id] = TaskState(
                task_id=task.task_id,
                status="pending",
                workspace_path=str(workspace),
                memory_path=str(run_root / ".task-memory" / task.task_id),
            )
        return task_states

    def _run_setup(self, task_spec: TaskSpec, workspace: Path) -> None:
        for command in task_spec.setup_cmds:
            subprocess.run(command, shell=True, cwd=workspace, check=True, capture_output=True, text=True)

    def _execute_action(
        self, task_spec: TaskSpec, task_state: TaskState, action: AgentAction, prompt
    ) -> ActionExecution:
        observation = ""
        if action.kind == "read":
            target = task_state.workspace / (action.path or "")
            if target.exists():
                observation = target.read_text()
            else:
                observation = f"Missing file: {target}"
        elif action.kind == "shell":
            observation = self._run_command(task_state.workspace, action.command or "")
        elif action.kind == "edit":
            target = task_state.workspace / (action.path or "")
            if not target.exists():
                observation = f"Missing file: {target}"
            else:
                current = target.read_text()
                if action.old_text is not None and action.old_text in current:
                    updated = current.replace(action.old_text, action.new_text or "", 1)
                    target.write_text(updated)
                    observation = f"Edited {target.name}"
                elif action.new_text is not None:
                    target.write_text(action.new_text)
                    observation = f"Overwrote {target.name}"
                else:
                    observation = f"No-op edit for {target.name}"
        elif action.kind == "test":
            observation = self._run_command(task_state.workspace, action.command or "")
        elif action.kind == "finish":
            observation = "finish"
        else:
            raise ValueError(f"Unsupported action kind: {action.kind}")
        return ActionExecution(
            action=action,
            observation=observation,
            prompt_tokens=prompt.estimated_tokens,
            stale_memory_error=prompt.contains_inactive_task_history,
        )

    def _run_command(self, workspace: Path, command: str) -> str:
        result = subprocess.run(command, shell=True, cwd=workspace, capture_output=True, text=True)
        return "\n".join(
            [
                f"command={command}",
                f"returncode={result.returncode}",
                result.stdout.strip(),
                result.stderr.strip(),
            ]
        ).strip()

    def _evaluate_task(self, task_spec: TaskSpec, task_state: TaskState) -> TaskEvaluation:
        harness = task_spec.eval_harness
        if harness.get("type") != "shell":
            raise ValueError(f"Unsupported eval harness type: {harness.get('type')}")
        details = self._run_command(task_state.workspace, harness["command"])
        passed = "returncode=0" in details.splitlines()[:2]
        task_state.eval_passed = passed
        task_state.score = 1.0 if passed and task_state.done else 0.0
        return TaskEvaluation(
            task_id=task_spec.task_id,
            eval_passed=passed,
            finished=task_state.done,
            success=passed and task_state.done,
            score=task_state.score,
            details=details,
        )

    def _summarize_episode(
        self,
        episode: MixedEpisode,
        *,
        memory_mode: str,
        task_states: dict[str, TaskState],
        evaluations: list[TaskEvaluation],
        prompt_tokens_per_slice: list[int],
        stale_memory_errors: int,
        action_fingerprints: dict[str, list[str]],
    ) -> EpisodeMetrics:
        success_count = sum(1 for item in evaluations if item.success)
        duplicate_total = 0
        action_total = 0
        for fingerprints in action_fingerprints.values():
            action_total += len(fingerprints)
            duplicate_total += max(0, len(fingerprints) - len(set(fingerprints)))
        duplicate_work_rate = 0.0 if action_total == 0 else duplicate_total / action_total
        unfinished = sum(1 for state in task_states.values() if not state.done)
        return EpisodeMetrics(
            episode_id=episode.episode_id,
            memory_mode=memory_mode,
            per_task=evaluations,
            total_slices=sum(state.action_count for state in task_states.values()),
            per_task_slices={task_id: state.action_count for task_id, state in task_states.items()},
            prompt_tokens_per_slice=prompt_tokens_per_slice,
            cumulative_prompt_tokens=sum(prompt_tokens_per_slice),
            duplicate_work_rate=duplicate_work_rate,
            stale_memory_errors=stale_memory_errors,
            unfinished_task_abandonment_rate=unfinished / max(1, len(task_states)),
            both_tasks_solved=success_count == len(task_states),
            one_task_solved=success_count == 1,
            zero_tasks_solved=success_count == 0,
        )


def build_memory_backend(mode: str, root_dir: str | Path, *, tail_events: int = 6) -> MemoryBackend:
    if mode == "shared":
        return SharedTranscriptMemory(root_dir)
    if mode == "filesystem":
        return FilesystemPerTaskMemory(root_dir, tail_events=tail_events)
    raise ValueError(f"Unsupported memory mode: {mode}")
