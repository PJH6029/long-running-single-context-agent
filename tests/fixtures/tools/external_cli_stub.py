from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="External CLI stub for benchmark tests")
    parser.add_argument("--mode", default="toy")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--output-schema")
    parser.add_argument("-o", "--output-last-message", dest="output_last_message")
    parser.add_argument("-C", "--cd")
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args()

    prompt_text = sys.stdin.read() if args.prompt in {None, "-"} else args.prompt

    if args.mode == "sleep":
        time.sleep(args.sleep_seconds)
    if args.mode == "exit-1":
        raise SystemExit(1)
    if args.mode == "invalid-json":
        response_text = "{invalid-json"
    elif args.mode == "missing-field":
        response_text = json.dumps({"description": "missing action kind"})
    else:
        response_text = json.dumps(_choose_action(prompt_text))

    if args.output_last_message:
        Path(args.output_last_message).write_text(response_text)
    else:
        sys.stdout.write(response_text)


def _choose_action(prompt_text: str) -> dict[str, str]:
    markers = _extract_markers(prompt_text)
    target_file = markers["TARGET_FILE"]
    search_text = markers["SEARCH_TEXT"]
    replace_text = markers["REPLACE_TEXT"]
    eval_command = markers["EVAL_COMMAND"]
    action_count = int(markers["TASK_ACTION_COUNT"] or "0")
    resume_count = int(markers["TASK_RESUME_COUNT"] or "0")
    inactive_history = markers["INACTIVE_TASK_HISTORY_PRESENT"].lower() == "true"

    if action_count == 0:
        return {
            "kind": "read",
            "description": "Inspect target file",
            "path": target_file,
        }
    if inactive_history and resume_count > 0 and action_count == 1:
        return {
            "kind": "read",
            "description": "Re-read due to prompt contamination",
            "path": target_file,
        }
    logical_phase = action_count
    if inactive_history and resume_count > 0 and action_count >= 2:
        logical_phase -= 1
    if logical_phase == 1:
        return {
            "kind": "edit",
            "description": "Apply patch",
            "path": target_file,
            "old_text": search_text,
            "new_text": replace_text,
        }
    if logical_phase == 2:
        return {
            "kind": "test",
            "description": "Run evaluation harness",
            "command": eval_command,
        }
    return {
        "kind": "finish",
        "description": "Finish task",
    }


def _extract_markers(prompt_text: str) -> dict[str, str]:
    keys = {
        "TARGET_FILE": "",
        "SEARCH_TEXT": "",
        "REPLACE_TEXT": "",
        "EVAL_COMMAND": "",
        "TASK_ACTION_COUNT": "0",
        "TASK_RESUME_COUNT": "0",
        "INACTIVE_TASK_HISTORY_PRESENT": "false",
    }
    for key in keys:
        matches = re.findall(rf"^{key}=(.*)$", prompt_text, flags=re.MULTILINE)
        if matches:
            keys[key] = matches[-1].strip()
    return keys


if __name__ == "__main__":
    main()
