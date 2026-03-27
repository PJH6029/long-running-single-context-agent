from __future__ import annotations

from collections import defaultdict

from .types import ScheduleEvent


def build_round_robin_schedule(
    task_ids: list[str],
    *,
    max_total_actions: int,
    max_actions_per_task: int,
    slice_budget: int = 1,
) -> list[ScheduleEvent]:
    if not task_ids:
        return []
    per_task = defaultdict(int)
    schedule: list[ScheduleEvent] = []
    step_idx = 0
    task_index = 0
    while step_idx < max_total_actions:
        active_task_id = task_ids[task_index % len(task_ids)]
        if per_task[active_task_id] < max_actions_per_task:
            schedule.append(
                ScheduleEvent(
                    step_idx=step_idx,
                    active_task_id=active_task_id,
                    slice_budget=slice_budget,
                )
            )
            per_task[active_task_id] += 1
            step_idx += 1
        if all(per_task[task_id] >= max_actions_per_task for task_id in task_ids):
            break
        task_index += 1
    return schedule

