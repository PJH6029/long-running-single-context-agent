from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..bench.types import PromptBundle, TaskSpec, TaskState
from .actions import AgentAction


class AgentPolicy(Protocol):
    def next_action(self, task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> AgentAction:
        ...


@dataclass
class ToyPromptLeakPolicy:
    """Prompt-sensitive toy policy used to make memory-mode differences observable."""

    def next_action(self, task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> AgentAction:
        metadata = _extract_markers(prompt.text)
        target_file = metadata["TARGET_FILE"]
        search_text = metadata["SEARCH_TEXT"]
        replace_text = metadata["REPLACE_TEXT"]
        eval_command = metadata["EVAL_COMMAND"]
        inactive_history = metadata["INACTIVE_TASK_HISTORY_PRESENT"].lower() == "true"

        if task_state.action_count == 0:
            return AgentAction(kind="read", description="Inspect target file", path=target_file)

        if inactive_history and task_state.resume_count > 0 and not _already_repeated(task_state):
            task_state.history.append({"kind": "redundant_read_marker"})
            return AgentAction(kind="read", description="Re-read due to prompt contamination", path=target_file)

        phase = _logical_phase(task_state)
        if phase == "edit":
            return AgentAction(
                kind="edit",
                description="Apply patch",
                path=target_file,
                old_text=search_text,
                new_text=replace_text,
            )
        if phase == "test":
            return AgentAction(kind="test", description="Run evaluation harness", command=eval_command)
        return AgentAction(kind="finish", description="Finish task")


def _extract_markers(prompt_text: str) -> dict[str, str]:
    fields = {
        "TARGET_FILE": "",
        "SEARCH_TEXT": "",
        "REPLACE_TEXT": "",
        "EVAL_COMMAND": "",
        "INACTIVE_TASK_HISTORY_PRESENT": "false",
    }
    for key in fields:
        matches = re.findall(rf"^{key}=(.*)$", prompt_text, flags=re.MULTILINE)
        if matches:
            fields[key] = matches[-1].strip()
    return fields


def _already_repeated(task_state: TaskState) -> bool:
    return any(item.get("kind") == "redundant_read_marker" for item in task_state.history)


def _logical_phase(task_state: TaskState) -> str:
    effective_actions = [
        item
        for item in task_state.history
        if item.get("action_kind") in {"read", "edit", "test", "finish"}
    ]
    performed = [item["action_kind"] for item in effective_actions]
    if "edit" not in performed:
        return "edit"
    if "test" not in performed:
        return "test"
    return "finish"

