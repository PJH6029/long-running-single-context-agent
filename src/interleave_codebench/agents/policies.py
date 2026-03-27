from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..bench.types import PromptBundle, TaskSpec, TaskState
from ..config import AgentConfig
from .actions import AgentAction


ACTION_KINDS = {"read", "shell", "edit", "test", "finish"}
ACTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": sorted(ACTION_KINDS)},
        "description": {"type": "string", "minLength": 1},
        "command": {"type": ["string", "null"]},
        "path": {"type": ["string", "null"]},
        "old_text": {"type": ["string", "null"]},
        "new_text": {"type": ["string", "null"]},
    },
    "required": ["kind", "description", "command", "path", "old_text", "new_text"],
}


@dataclass(slots=True)
class PolicyDecision:
    action: AgentAction | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class AgentPolicy(Protocol):
    name: str

    def next_action(self, task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> PolicyDecision:
        ...


@dataclass(slots=True)
class ToyPromptLeakPolicy:
    """Prompt-sensitive toy policy used to make memory-mode differences observable."""

    name: str = "toy-prompt-leak"

    def next_action(self, task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> PolicyDecision:
        metadata = _extract_markers(prompt.text)
        target_file = metadata["TARGET_FILE"]
        search_text = metadata["SEARCH_TEXT"]
        replace_text = metadata["REPLACE_TEXT"]
        eval_command = metadata["EVAL_COMMAND"]
        inactive_history = metadata["INACTIVE_TASK_HISTORY_PRESENT"].lower() == "true"

        if task_state.action_count == 0:
            return PolicyDecision(
                action=AgentAction(kind="read", description="Inspect target file", path=target_file)
            )

        if inactive_history and task_state.resume_count > 0 and not _already_repeated(task_state):
            task_state.history.append({"kind": "redundant_read_marker"})
            return PolicyDecision(
                action=AgentAction(
                    kind="read",
                    description="Re-read due to prompt contamination",
                    path=target_file,
                )
            )

        phase = _logical_phase(task_state)
        if phase == "edit":
            return PolicyDecision(
                action=AgentAction(
                    kind="edit",
                    description="Apply patch",
                    path=target_file,
                    old_text=search_text,
                    new_text=replace_text,
                )
            )
        if phase == "test":
            return PolicyDecision(
                action=AgentAction(kind="test", description="Run evaluation harness", command=eval_command)
            )
        return PolicyDecision(action=AgentAction(kind="finish", description="Finish task"))


@dataclass(slots=True)
class ExternalCLIJsonPolicy:
    name: str
    command_template: list[str]
    timeout_seconds: int = 180

    def next_action(self, task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> PolicyDecision:
        prompt_text = _render_external_cli_prompt(task_spec, task_state, prompt)
        with tempfile.TemporaryDirectory(prefix="interleave-codebench-") as tmp:
            schema_path = Path(tmp) / "action_schema.json"
            response_path = Path(tmp) / "response.json"
            schema_path.write_text(json.dumps(ACTION_RESPONSE_SCHEMA, indent=2))
            command = [
                item.format(
                    workspace_path=str(task_state.workspace),
                    schema_path=str(schema_path),
                    response_path=str(response_path),
                )
                for item in self.command_template
            ]
            try:
                result = subprocess.run(
                    command,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return PolicyDecision(
                    error=f"Agent CLI timed out after {self.timeout_seconds} seconds.",
                    details={
                        "command": command,
                        "stdout": (exc.stdout or "").strip(),
                        "stderr": (exc.stderr or "").strip(),
                    },
                )
            details = {
                "command": command,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            if result.returncode != 0:
                return PolicyDecision(
                    error=f"Agent CLI exited with returncode={result.returncode}.",
                    details=details,
                )
            response_text = ""
            if response_path.exists():
                response_text = response_path.read_text().strip()
            elif result.stdout.strip():
                response_text = result.stdout.strip()
            if not response_text:
                return PolicyDecision(
                    error="Agent CLI did not produce a JSON action response.",
                    details=details,
                )
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError as exc:
                return PolicyDecision(
                    error=f"Agent CLI returned invalid JSON: {exc.msg}.",
                    details={**details, "response_text": response_text},
                )
            action, validation_error = _parse_action_payload(payload)
            if validation_error:
                return PolicyDecision(
                    error=validation_error,
                    details={**details, "response_text": response_text},
                )
            return PolicyDecision(
                action=action,
                details={**details, "response_text": response_text},
            )


def build_agent_policy(config: AgentConfig) -> AgentPolicy:
    if config.kind == "toy":
        return ToyPromptLeakPolicy(name=config.name)
    if config.kind == "external_cli_json":
        return ExternalCLIJsonPolicy(
            name=config.name,
            command_template=list(config.command),
            timeout_seconds=config.timeout_seconds,
        )
    raise ValueError(f"Unsupported agent kind: {config.kind}")


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


def _render_external_cli_prompt(task_spec: TaskSpec, task_state: TaskState, prompt: PromptBundle) -> str:
    history_lines = []
    for item in task_state.history[-6:]:
        observation = str(item.get("observation", "")).splitlines()
        summary = observation[0].strip() if observation else ""
        history_lines.append(
            "- step={step} action={action} observation={summary}".format(
                step=item.get("step_idx", "?"),
                action=item.get("action_kind", item.get("kind", "unknown")),
                summary=summary or "<empty>",
            )
        )
    history_text = "\n".join(history_lines) if history_lines else "No task-local history yet."
    last_result = task_state.last_result.strip() or "No prior result."
    return "\n\n".join(
        [
            "You are selecting exactly one benchmark action for the active task.",
            "\n".join(
                [
                    "Rules:",
                    "- Return exactly one JSON object that matches the provided schema.",
                    "- Do not modify files or run shell commands yourself.",
                    "- The benchmark runner will execute your chosen action.",
                    "- `read` requires `path`.",
                    "- `edit` requires `path` and uses `old_text`/`new_text` for one exact replacement.",
                    "- `test` and `shell` require `command`.",
                    "- `finish` ends the task so final evaluation can run.",
                ]
            ),
            "\n".join(
                [
                    f"ACTIVE_TASK_ID={task_spec.task_id}",
                    f"TASK_STATUS={task_state.status}",
                    f"TASK_ACTION_COUNT={task_state.action_count}",
                    f"TASK_RESUME_COUNT={task_state.resume_count}",
                    f"TASK_DONE={'true' if task_state.done else 'false'}",
                ]
            ),
            "[RECENT TASK HISTORY]",
            history_text,
            "[LAST RESULT]",
            last_result,
            "[BENCHMARK PROMPT]",
            prompt.text,
        ]
    )


def _parse_action_payload(payload: Any) -> tuple[AgentAction | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Agent CLI response must be a JSON object."
    kind = payload.get("kind")
    description = payload.get("description")
    if not isinstance(kind, str) or kind not in ACTION_KINDS:
        return None, f"Agent CLI response has invalid action kind: {kind!r}."
    if not isinstance(description, str) or not description.strip():
        return None, "Agent CLI response must include a non-empty description."
    path, path_error = _optional_string(payload, "path")
    if path_error:
        return None, path_error
    command, command_error = _optional_string(payload, "command")
    if command_error:
        return None, command_error
    old_text, old_text_error = _optional_string(payload, "old_text")
    if old_text_error:
        return None, old_text_error
    new_text, new_text_error = _optional_string(payload, "new_text")
    if new_text_error:
        return None, new_text_error
    if kind in {"read", "edit"} and not path:
        return None, f"Agent CLI response for {kind} must include `path`."
    if kind in {"shell", "test"} and not command:
        return None, f"Agent CLI response for {kind} must include `command`."
    if kind == "edit" and old_text is None and new_text is None:
        return None, "Agent CLI response for edit must include `old_text`, `new_text`, or both."
    return (
        AgentAction(
            kind=kind,
            description=description.strip(),
            command=command,
            path=path,
            old_text=old_text,
            new_text=new_text,
        ),
        None,
    )


def _optional_string(payload: dict[str, Any], key: str) -> tuple[str | None, str | None]:
    value = payload.get(key)
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"Agent CLI response field `{key}` must be a string or null."
    return value, None
