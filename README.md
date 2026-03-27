# Single-Context Interleaved Multitask Agent

This repository explores a long-running agent that can handle multiple interleaved tasks without turning each task into a separate chat thread.

The working hypothesis is simple:

- treat the context window as a bounded working cache
- persist durable task memory in the filesystem
- model multitasking as cache switching over per-task memory, not as replaying one giant interleaved transcript

## What Is Implemented Today

This repo currently contains two small benchmark setups for interleaved coding tasks.

- `v0` fixture pilot:
  - source task families: fixture-shaped `SWE-bench Verified` and `SWE-CI`
  - intended for validating memory-mode behavior and runner mechanics
- `v0.1` sampled micro-benchmark:
  - source task families: real sampled task IDs from `SWE-bench Verified` and `SWE-CI`
  - repo snapshots are exported from pinned upstream commits
  - evaluation is normalized to narrow local regression checks, not the official full harness
- Domain: Python repository tasks only
- Mixing policy: one task from each source benchmark, interleaved with strict round robin scheduling
- Slice granularity: one atomic agent iteration per task switch
- Memory modes:
  - `shared`: one global transcript for the whole mixed episode
  - `filesystem`: one persistent task-memory directory per task
- Runner: a lightweight repo-agent loop with read, edit, test, shell, and finish actions
- Agent modes:
  - `toy`: deterministic `ToyPromptLeakPolicy` used for the built-in memory baseline
  - `external_cli_json`: any CLI agent that can return one JSON action per slice, with `codex exec` shipped as the first config example

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
  v0_1_realistic_codex.toml
benchmark_samples/
  v0_1_realistic.json
tests/
  fixtures/
  test_*.py
related_works/
  literature_review.md
```

## Quickstart

The fixture benchmark has no Python package dependencies, so the repo can be run directly with `python3`.

Run the test suite:

```bash
python3 scripts/run_tests.py
```

Build deterministic mixed episodes from the pilot fixtures:

```bash
python3 scripts/run_benchmark.py build-episodes --config configs/pilot_fixture.toml
```

Run the built-in toy baseline and compare shared-transcript memory against filesystem task memory on the eval split:

```bash
python3 scripts/run_benchmark.py run-compare --config configs/pilot_fixture.toml --split eval
```

Run the Codex CLI baseline on the same fixture benchmark:

```bash
python3 scripts/run_benchmark.py run-compare --config configs/pilot_codex.toml --split eval
```

Materialize the sampled `v0.1` benchmark from the curated upstream task manifest:

```bash
python3 scripts/run_benchmark.py prepare-v0_1
```

Build episodes for the sampled `v0.1` benchmark:

```bash
python3 scripts/run_benchmark.py build-episodes --config configs/v0_1_realistic_codex.toml
```

Run the Codex CLI baseline on the sampled `v0.1` benchmark:

```bash
python3 scripts/run_benchmark.py run-compare --config configs/v0_1_realistic_codex.toml --split eval
```

Outputs are written under per-agent directories, for example:

```text
.runs/pilot_fixture/runs/toy-prompt-leak/
.runs/pilot_codex/runs/codex/
.runs/v0_1_realistic/runs/codex/
```

Each agent run writes per-episode metrics plus:

- `comparison-<split>.json`
- `summary-<split>.json`

If you want the CLI installed as a package entrypoint instead of using the wrapper script:

```bash
python3 -m pip install -e .
interleave-codebench run-compare --config configs/pilot_fixture.toml --split eval
```

## Current Scope And Limitations

The scope is still intentionally narrow.

- The shipped datasets are toy fixtures shaped like `SWE-bench Verified` and `SWE-CI`, not official benchmark checkouts.
- The sampled `v0.1` benchmark uses real upstream task IDs and commits, but it still is not the official benchmark harness. It verifies a curated local regression per task rather than replaying the full upstream Docker or CI stack.
- The built-in baseline is deterministic and hand-written; external CLI agents are supported only through a one-action-per-slice JSON contract.
- Only Python tasks are supported.
- Only 2-task episodes are supported.
- Interleaving is strict round robin only.
- The benchmark focuses on interruption and resumption under bounded working context, not on benchmark-native multi-turn traces.
- Prompt-token accounting remains a cheap estimate over the benchmark prompt bundle; it is not provider-native token usage from an external CLI agent.

The current `v0.1` sampled task set includes:

- `SWE-bench Verified`: `pallets__flask-5014`, `psf__requests-1921`, `psf__requests-5414`, `psf__requests-6028`
- `SWE-CI`: `oauthlib__oauthlib__3bcbb2__bf0241`, `pygments__pygments__eeb568__35732f`, `modelcontextprotocol__python-sdk__281fd4__a9cc82`, `pydantic__pydantic-settings__e16290__0497ef`

## Research Context

The project framing in [AGENTS.md](/Users/jeonghunpark/code/long-running-single-context-agent/AGENTS.md) captures the motivating idea, nearby literature, and the gap this benchmark is trying to study.

Additional research notes live in [related_works/literature_review.md](/Users/jeonghunpark/code/long-running-single-context-agent/related_works/literature_review.md).

## Practical Goal

The practical goal is not just to build another long-context agent.

It is to test whether a long-running multitask agent can be made more scalable and more robust by:

- keeping one global operating identity
- persisting task state outside the prompt
- loading only the active task memory into working context
