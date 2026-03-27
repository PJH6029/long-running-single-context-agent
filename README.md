# Single-Context Interleaved Multitask Agent

This repository explores a long-running agent that can handle multiple interleaved tasks without turning each task into a separate chat thread.

The working hypothesis is simple:

- treat the context window as a bounded working cache
- persist durable task memory in the filesystem
- model multitasking as cache switching over per-task memory, not as replaying one giant interleaved transcript

## What Is Implemented Today

This repo currently contains a `v0` pilot benchmark for interleaved coding tasks.

- Source task families: fixture-shaped `SWE-bench Verified` and `SWE-CI`
- Domain: Python repository tasks only
- Mixing policy: one task from each source benchmark, interleaved with strict round robin scheduling
- Slice granularity: one atomic agent iteration per task switch
- Memory modes:
  - `shared`: one global transcript for the whole mixed episode
  - `filesystem`: one persistent task-memory directory per task
- Runner: a lightweight repo-agent loop with read, edit, test, shell, and finish actions
- Policy: a deterministic `ToyPromptLeakPolicy` that makes cross-task prompt contamination observable

This is intentionally a pilot scaffold, not a leaderboard-ready benchmark release.

## Why This Repo Exists

Most current agent systems handle multitasking by opening separate sessions or separate histories per task. That works, but it sidesteps the question this repo cares about:

Can one long-running agent keep a shared operating identity while switching between multiple active tasks without polluting its working context?

The naive baseline keeps one interleaved transcript in prompt:

```text
task-A-1 | task-B-1 | task-A-2 | task-B-2 | ...
```

This repo studies the alternative:

- keep only compact multitask control information in prompt
- store detailed task state in the filesystem
- load only the active task memory on each slice

## Filesystem Task Memory

The filesystem-backed method writes task-local memory under:

```text
.task-memory/
  <episode>/
    registry.json
    <task-id>/
      spec.md
      interaction.md
      summary.md
      state.json
```

The intended prompt shape for the filesystem mode is:

- global multitask instruction
- compact task registry
- active task spec
- active task summary
- active task interaction tail

Inactive task interaction history should stay out of the prompt.

## Repository Layout

```text
src/interleave_codebench/
  bench/
    adapters.py
    memory.py
    mixers.py
    scheduler.py
    types.py
  agents/
    actions.py
    policies.py
    runner.py
  cli.py
  config.py
scripts/
  run_benchmark.py
  run_tests.py
configs/
  pilot_fixture.toml
tests/
  fixtures/
  test_*.py
related_works/
  literature_review.md
```

## Quickstart

The intended environment for this repo is the `single-context-agent` conda environment.

Run the test suite:

```bash
conda run -n single-context-agent python scripts/run_tests.py
```

Build deterministic mixed episodes from the pilot fixtures:

```bash
conda run -n single-context-agent python scripts/run_benchmark.py build-episodes --config configs/pilot_fixture.toml
```

Compare shared-transcript memory against filesystem task memory on the eval split:

```bash
conda run -n single-context-agent python scripts/run_benchmark.py run-compare --config configs/pilot_fixture.toml --split eval
```

Outputs are written under `.runs/pilot_fixture/`.

If you want the CLI installed as a package entrypoint instead of using the wrapper script:

```bash
conda run -n single-context-agent python -m pip install -e .
conda run -n single-context-agent interleave-codebench run-compare --config configs/pilot_fixture.toml --split eval
```

## Current Scope And Limitations

This `v0` is intentionally narrow.

- The shipped datasets are toy fixtures shaped like `SWE-bench Verified` and `SWE-CI`, not official benchmark checkouts.
- The current agent policy is deterministic and hand-written; it is not yet an LLM-driven repo agent.
- Only Python tasks are supported.
- Only 2-task episodes are supported.
- Interleaving is strict round robin only.
- The benchmark focuses on interruption and resumption under bounded working context, not on benchmark-native multi-turn traces.

## Research Context

The project framing in [AGENTS.md](/Users/jeonghunpark/code/long-running-single-context-agent/AGENTS.md) captures the motivating idea, nearby literature, and the gap this benchmark is trying to study.

Additional research notes live in [related_works/literature_review.md](/Users/jeonghunpark/code/long-running-single-context-agent/related_works/literature_review.md).

## Practical Goal

The practical goal is not just to build another long-context agent.

It is to test whether a long-running multitask agent can be made more scalable and more robust by:

- keeping one global operating identity
- persisting task state outside the prompt
- loading only the active task memory into working context
