# Single-Context Interleaved Multitask Agent

This repository explores a long-running agent that can handle multiple interleaved tasks without treating each task as a fully separate chat thread or identity.

The core project hypothesis is:

- the context window should be treated as a bounded working cache
- durable task memory should live outside the prompt, in the filesystem
- multitasking should be implemented as cache switching over persistent per-task memory, not as replaying one giant interleaved transcript

## Current Repo Status

This repository already contains a `v0` pilot benchmark scaffold in [src/interleave_codebench](/Users/jeonghunpark/code/long-running-single-context-agent/src/interleave_codebench).

Today, the implemented scope is:

- coding tasks only
- Python repositories only
- two source benchmark families normalized into one schema:
  - `SWE-bench Verified`
  - `SWE-CI`
- mixed episodes built as `1 SWE-bench Verified task + 1 SWE-CI task`
- strict round-robin task switching at one action slice per switch
- two memory modes:
  - shared transcript baseline
  - filesystem-backed per-task memory

The current runner is intentionally lightweight and uses a deterministic toy policy to make prompt contamination visible during evaluation. It is a benchmark scaffold first, not yet a production LLM agent stack.

## Intro

Most current agent systems effectively solve multitasking by using separate sessions, histories, or threads for separate tasks. That avoids interference, but it also means the agent does not really operate with one long-running shared memory in the human sense.

The naive alternative is to keep all interleaved history in one context:

- `task-A-1 | task-B-1 | task-C-1 | task-B-2 | task-A-2 | ...`

This quickly pollutes the working context, wastes tokens, and makes retrieval of the currently relevant task state harder.

This project studies a different approach:

- keep only compact multitask control information in context
- keep detailed task histories in the filesystem
- load only the active task's memory into working context when needed

## Related Works

The closest related works are adjacent, but none exactly define the target problem.

### Memory and Context Systems

- `MemGPT` proposes virtual context management and memory tiers. It is the clearest inspiration for treating the prompt as fast memory and external storage as slower backing memory. However, its main focus is long single-task or long-session continuity, not explicitly interleaved independent tasks.
- `Everything is Context` is the closest match to the filesystem side of this project. It treats agent-relevant artifacts as a file-based context space and dynamically loads the needed slice under token limits.
- `LLM as OS, Agents as Apps` provides the strongest conceptual analogy: context window as memory, external storage as file system.
- `CoALA` gives the right cognitive framing: working memory in-context, plus episodic, semantic, and procedural memory outside it.

### Task and Workflow Memory

- `Agent Workflow Memory` shows that agents benefit from storing reusable workflows instead of repeatedly replaying raw trajectories.
- `A-MEM`, `SimpleMem`, `AriadneMem`, and `PlugMem` all move toward structured persistent memory instead of raw transcript retrieval.
- `HiAgent` shows that even within one task, keeping the entire trajectory in working memory is inefficient; only the subgoal-relevant slice should stay active.

### Benchmarks Near the Target

- `Beyond Prompts` is the closest benchmark signal for this project. It explicitly studies concurrent multitask conversations with regular context switching.
- `MemoryArena` studies interdependent multi-session agentic tasks, which is close to task carryover and resumability.
- `MEMTRACK` studies chronologically interleaved, noisy, cross-platform agent memory in realistic workflows.
- `Mem2ActBench` and related memory benchmarks study interrupted interactions and delayed memory use, but not a filesystem-first per-task memory architecture.

## Limitations of Existing Work

Existing work still leaves a concrete gap.

### Gap 1: Interleaved Independent Tasks Are Understudied

Most memory papers study one of these:

- a single long-horizon task
- lifelong transfer across many tasks
- long-term conversation
- multi-session continuity

Very few directly study:

- multiple independent tasks interleaved in one active stream
- explicit task switching and task resumption
- competition for limited working context across tasks

### Gap 2: Filesystem-Backed Per-Task Memory Is Not Yet a Standard Design

Many systems use:

- vector stores
- graphs
- note networks
- latent or structured memory modules

But there is still little work that makes the filesystem itself the primary durable memory substrate for per-task state, with loading and unloading driven by task focus.

### Gap 3: There Is No Canonical Benchmark for This Exact Setting

There does not appear to be a standard benchmark that combines:

- interleaved task trajectories such as `A1 | B1 | A2 | B2`
- per-task persistent memory
- resumability under bounded working context
- evaluation of context pollution vs selective loading

## My Method

The proposed method is intentionally simple and concrete.

### Core Idea

Use the prompt as a working cache, not as the source of truth.

Keep only a compact multitask control layer in the active context, such as:

- multitask operating instructions
- task registry
- compact task specs
- the currently active task memory

Persist the rest in the filesystem.

### Filesystem Memory Design

A basic layout may look like:

```text
.task-memory/
  registry.json
  task-A/
    spec.md
    interaction.md
    summary.md
    state.json
  task-B/
    spec.md
    interaction.md
    summary.md
    state.json
  task-C/
    spec.md
    interaction.md
    summary.md
    state.json
```

Where:

- `spec.md` stores stable task requirements
- `interaction.md` stores detailed task-local interaction history
- `summary.md` stores compact resumable summaries
- `state.json` stores machine-readable task state
- `registry.json` stores global task metadata and switching information

### Context Loading Policy

When working on task A, the active context should resemble:

- multitask instruction
- task registry or compact task index
- task A spec
- task A active summary or interaction slice

not:

- the full detailed histories of tasks B and C

Task switching becomes a cache swap:

- unload irrelevant task-local detail
- load the active task's persistent memory
- continue work with minimal interference from unrelated tasks

### Why This Matters

This design aims to:

- reduce context pollution from unrelated tasks
- scale to more simultaneous tasks
- make resumability explicit
- separate durable memory from transient reasoning state
- support benchmarking of switching quality and memory efficiency

## Current Benchmark Shape

The benchmark implementation in this repo currently assumes:

- `TaskSpec`, `TaskState`, `MixedEpisode`, and `ScheduleEvent` are the shared normalized task abstractions
- mixed episodes are built deterministically from fixture-shaped source benchmark tasks
- interleaving is controlled by the benchmark scheduler, not by source-benchmark-native turn traces
- evaluation compares:
  - `shared_transcript`
  - `filesystem_per_task`

The default filesystem memory layout is:

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

Preserve this layout unless there is a strong reason to change it, and update the tests if you do.

## Working Assumptions for This Repo

When contributing to this repository, align work with the following assumptions:

- treat context as scarce working memory
- prefer persistent filesystem state over hidden prompt accumulation
- model multitasking as task switching over shared infrastructure
- distinguish clearly between global memory and task-local memory
- favor simple, inspectable memory formats first

## Repo Workflow Notes

Keep [README.md](/Users/jeonghunpark/code/long-running-single-context-agent/README.md) aligned with the actual implemented benchmark behavior. If the CLI, file layout, or benchmark scope changes, update the README in the same change.

When working in this repository:

- prefer deterministic fixtures and reproducible runs for `v0`
- keep benchmark adapters small and inspectable
- keep prompt construction logic explicit
- preserve the distinction between benchmark generation, memory backends, and agent execution
- update or add tests when changing schemas, scheduling, memory layout, or CLI behavior

Useful commands:

```bash
conda run -n single-context-agent python scripts/run_tests.py
conda run -n single-context-agent python scripts/run_benchmark.py build-episodes --config configs/pilot_fixture.toml
conda run -n single-context-agent python scripts/run_benchmark.py run-compare --config configs/pilot_fixture.toml --split eval
```

Do not commit generated or workspace-local artifacts unless explicitly requested. In particular, avoid committing:

- `.runs/`
- `.task-memory/`
- `.omx/`
- cloned third-party repositories under `related_works/`

## Suggested Benchmark Direction

The benchmark direction for this repository is:

- start from tasks that require multiple turns or resumptions
- interleave them into one chronology, for example:
  - `task-A-1 | task-B-1 | task-A-2 | task-B-2`
- compare at least two settings:
  - naive full interleaved history in prompt
  - selective per-task loading from filesystem memory

Useful evaluation targets include:

- task completion
- recovery after interruption
- context efficiency
- cross-task interference
- correctness of task state after switching

## Practical Goal

The practical goal of this repo is not just to build another long-context agent.

It is to explore whether an agent can appear long-running and multi-task capable by:

- keeping one persistent identity and global operating policy
- storing task memory durably outside the prompt
- loading only the right task memory at the right time

That is the main design lens for future code, experiments, and benchmarks in this repository.
