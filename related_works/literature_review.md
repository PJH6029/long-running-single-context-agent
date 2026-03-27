# Literature Review: Long-Running Multi-Task LLM Agents with a Single Shared History

Date: 2026-03-26

## Question

Is there research on AI agents that can handle multiple tasks over long periods while keeping one shared memory or history, instead of splitting each task into its own isolated thread/session?

## Short Answer

Yes, but not under one clean, dominant name.

The closest research lines are:

- lifelong or continual LLM agents
- long-horizon agent memory
- working, episodic, semantic, and procedural memory for language agents
- workflow or skill memory
- cognitive architectures for language agents

The most relevant papers do not usually say "single-history multitasking." Instead, they solve nearby problems by giving an agent:

- one persistent memory substrate across time
- a smaller working context for the currently active step
- retrieval, compression, reflection, or skill reuse to resume old tasks without replaying the full raw transcript

So the emerging answer is:

- raw single-history agents do exist as an aspiration
- practical systems usually approximate this with persistent shared memory plus compressed active context
- true concurrent multitasking within one live history is still underexplored

## Main Takeaway

There is no mature subfield yet whose main benchmark is "many interleaved tasks in one uninterrupted history."

What does exist is a strong and growing cluster of work on:

1. memory architectures that preserve continuity across long interactions
2. lifelong agents that accumulate reusable knowledge across tasks
3. benchmarks showing that stateless or thread-local agents fail badly once interactions become long, interdependent, or multi-session

This means your framing is valid and timely. It is better viewed as an open problem at the intersection of:

- long-horizon agent memory
- lifelong learning for agents
- task switching and workflow reuse
- cognitive architectures with working memory plus long-term memory

## Recommended Taxonomy

### 1. Architectural Framing Papers

These papers are the closest conceptual match for "one mind, many tasks."

| Paper | Core idea | Why it matters here |
|---|---|---|
| [Cognitive Architectures for Language Agents (CoALA), 2023](https://arxiv.org/pdf/2309.02427v3) | Frames language agents using modular memory systems and structured action spaces inspired by cognitive science. | Gives the clearest vocabulary for shared working memory, episodic memory, semantic memory, and procedural memory. |
| [Empowering Working Memory for Large Language Model Agents, 2023](https://arxiv.org/pdf/2312.17259v2) | Proposes a centralized working-memory hub and episodic buffer across episodes. | Explicitly argues that isolated dialogue episodes are a design limitation. |
| [Generative Agents, 2023](https://arxiv.org/pdf/2304.03442v2) | Agents keep a natural-language record of experience, reflect on it, and retrieve it for planning. | One of the earliest influential examples of a persistent autobiographical memory stream. |

### 2. Memory Systems for Long-Running Agents

These papers are the closest systems papers to your exact question.

| Paper | Core idea | Relevance to "single history" |
|---|---|---|
| [MemGPT, 2023](https://arxiv.org/pdf/2310.08560v2) | OS-style virtual context management with memory tiers. | Treats the context window like scarce RAM and external memory like backing storage; strong match for long-running shared context. |
| [HiAgent, 2024](https://arxiv.org/pdf/2408.09559v1) | Hierarchical working-memory management via subgoals. | Handles long active trajectories by chunking current work instead of replaying everything. |
| [Agent Workflow Memory, 2024](https://arxiv.org/pdf/2409.07429v1) | Learns reusable workflows from prior tasks and injects them into later tasks. | Important for multi-task continuity because memory becomes routines, not just facts. |
| [A-MEM: Agentic Memory for LLM Agents, 2025](https://arxiv.org/pdf/2502.12110v11) | Dynamically linked and evolving notes inspired by Zettelkasten. | Moves from passive retrieval to an actively reorganized memory network. |
| [SimpleMem, 2026](https://arxiv.org/pdf/2601.02553v3) | Semantic structured compression, online synthesis, intent-aware retrieval. | Very close to a practical "single history, compressed working state" architecture. |
| [AriadneMem, 2026](https://arxiv.org/pdf/2603.03290v1) | Handles disconnected evidence and state updates with graph construction plus bridge discovery. | Especially relevant when an agent revisits older tasks and must reconcile changed state. |

### 3. Learning Across Tasks or Episodes

These do not always keep one literal transcript, but they do build one evolving memory across tasks.

| Paper | Core idea | Why it matters |
|---|---|---|
| [Reflexion, 2023](https://arxiv.org/pdf/2303.11366v4) | Stores reflective verbal feedback from prior attempts. | A simple and influential example of inter-episode memory. |
| [ExpeL, 2023](https://arxiv.org/pdf/2308.10144v3) | Extracts reusable lessons from accumulated experience. | Strong fit for multi-task transfer without weight updates. |
| [Voyager, 2023](https://arxiv.org/pdf/2305.16291v2) | Open-ended lifelong agent with an ever-growing skill library. | Important because it turns raw history into reusable executable skills. |
| [Large Language Models Are Semi-Parametric Reinforcement Learning Agents / REMEMBERER, 2023](https://arxiv.org/pdf/2306.07929v2) | Combines LLMs with long-term experience memory for reuse across task goals. | Directly studies experience reuse across different goals. |

### 4. Benchmarks and Surveys

These papers show the gap between current systems and the "single-memory multitasking" goal.

| Paper | Core idea | Why it matters |
|---|---|---|
| [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo), 2024](https://arxiv.org/pdf/2402.17753v1) | Benchmark with long multi-session conversations and temporal/causal reasoning. | Shows that long-term continuity is hard even before true multitasking. |
| [Lifelong Learning of Large Language Model based Agents: A Roadmap, 2025](https://arxiv.org/pdf/2501.07278v2) | First broad survey of lifelong LLM agents. | Best entry point for the overall area. |
| [LifelongAgentBench, 2025](https://arxiv.org/pdf/2505.11942v3) | Interdependent tasks across DB, OS, and KG environments. | Closest benchmark to multi-task accumulation and transfer. |
| [LIFELONG-SOTOPIA, 2025](https://arxiv.org/pdf/2506.12666v1) | Lifelong social interactions across many episodes. | Useful if your notion of one history includes social continuity and relationship tracking. |

## What The Literature Actually Says

### The field is moving away from "just make the context window bigger"

Recent work increasingly treats long-running agency as a memory-management problem, not a pure long-context problem.

The common pattern is:

- store all experience somewhere persistent
- keep only a small, task-relevant slice in working memory
- let the agent decide what to summarize, retrieve, revise, or link

This is visible in MemGPT, HiAgent, A-MEM, SimpleMem, and AriadneMem.

### The field is also moving away from passive vector search

Older or simpler memory systems often treat history as a bag of chunks for similarity retrieval.

Newer systems try to preserve:

- temporal order
- causal structure
- state updates
- reusable workflows or skills
- explicit links between memories

This is exactly the direction needed for one agent that revisits many unfinished tasks over time.

### But most papers still serialize the active task

This is the biggest gap relative to your question.

Most current systems do one of these:

- solve one long task
- solve many tasks sequentially and transfer knowledge between them
- maintain long-term conversational continuity

Very few explicitly study:

- interleaving multiple active tasks in one live stream
- interruption and resumption policies
- conflict between several simultaneously active goals
- working-memory contention between tasks
- priority scheduling over a shared autobiographical memory

So the literature is near your question, but not fully on top of it yet.

## Closest Existing Design Pattern

If we translate the literature into a system design for your exact problem, the best current pattern is:

1. one persistent event log
2. one evolving long-term memory layer
3. one bounded working-memory layer
4. multiple active task frames that all reference the same long-term memory

That differs from "one thread per task" in an important way:

- the agent does not swap minds when it switches tasks
- it swaps active task frames inside one shared memory substrate

The most relevant ingredients from the literature are:

- CoALA for the memory decomposition
- MemGPT for tiered context management
- HiAgent and SimpleMem for bounded working memory
- A-MEM and AriadneMem for structured, evolving long-term memory
- Agent Workflow Memory and Voyager for reusable routines and skills
- LifelongAgentBench and LoCoMo for evaluation

## What Is Still Missing

This appears to be the open research gap.

The literature still lacks a standard benchmark for:

- many interleaved live tasks in one ongoing interaction stream
- interruptions, resumptions, and deadline changes
- state mutation across tasks in shared memory
- task switching cost and cross-task interference
- evaluation of "remembers the right unfinished thing at the right moment"

It also lacks standard agent abstractions for:

- active task stack or task set
- resumption pointers
- shared-memory garbage collection
- cross-task salience scoring
- memory writes that distinguish facts, plans, commitments, and obsolete state

## Practical Interpretation

As of 2026-03-26, the best answer is:

- yes, there are strong approaches in this direction
- no, the community has not yet converged on a standard name or benchmark for "multi-tasking within a single history"
- the state of the art is converging on shared persistent memory plus compressed working context, not on replaying one raw transcript forever

If you want to build such an agent, you would not literally keep one uncompressed chat history. You would build:

- a single autobiographical memory store
- a small active working set
- explicit task objects with status, dependencies, and resumption state
- retrieval and consolidation policies that are aware of task identity, recency, state changes, and commitments

That is the closest research-backed interpretation of "human-like multitasking in one memory."

## Local Artifacts Added

I cloned two relevant repositories into this workspace:

- `refs/awesome-lifelong-llm-agent`
- `refs/A-mem-sys`

Why these two:

- `awesome-lifelong-llm-agent` is a curated survey repo that maps the field into perception, memory, and action modules.
- `A-mem-sys` is a concrete implementation of an evolving agent memory system.

### Implementation Note from `A-mem-sys`

The `A-mem-sys` code is useful because it makes the abstraction concrete:

- memory units are explicit notes with content, keywords, context, tags, links, timestamps, and evolution history
- retrieval uses enriched content, not just raw text
- new memories can trigger updates to older memories

That is much closer to a "single mind with evolving memory" than a plain thread-per-task chat log.

## Source Links

Primary papers:

- [CoALA](https://arxiv.org/pdf/2309.02427v3)
- [Generative Agents](https://arxiv.org/pdf/2304.03442v2)
- [MemGPT](https://arxiv.org/pdf/2310.08560v2)
- [Reflexion](https://arxiv.org/pdf/2303.11366v4)
- [ExpeL](https://arxiv.org/pdf/2308.10144v3)
- [Voyager](https://arxiv.org/pdf/2305.16291v2)
- [REMEMBERER / Semi-Parametric RL Agents](https://arxiv.org/pdf/2306.07929v2)
- [LoCoMo](https://arxiv.org/pdf/2402.17753v1)
- [HiAgent](https://arxiv.org/pdf/2408.09559v1)
- [Agent Workflow Memory](https://arxiv.org/pdf/2409.07429v1)
- [A-MEM](https://arxiv.org/pdf/2502.12110v11)
- [Roadmap survey](https://arxiv.org/pdf/2501.07278v2)
- [LifelongAgentBench](https://arxiv.org/pdf/2505.11942v3)
- [LIFELONG-SOTOPIA](https://arxiv.org/pdf/2506.12666v1)
- [SimpleMem](https://arxiv.org/pdf/2601.02553v3)
- [AriadneMem](https://arxiv.org/pdf/2603.03290v1)

Project and code resources:

- [awesome-lifelong-llm-agent](https://github.com/qianlima-lab/awesome-lifelong-llm-agent)
- [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)
- [SimpleMem code](https://github.com/aiming-lab/SimpleMem)
- [AriadneMem code](https://github.com/LLM-VLM-GSL/AriadneMem)
- [LifelongAgentBench project page](https://caixd-220529.github.io/LifelongAgentBench/)

## Bottom-Line Verdict

The answer is "yes, partially."

The closest serious approaches already exist, but they mostly study:

- memory continuity
- long-horizon reasoning
- transfer across episodes or tasks

rather than:

- genuinely concurrent multi-task management inside one uninterrupted history

That exact formulation still looks like an open and promising research target.
