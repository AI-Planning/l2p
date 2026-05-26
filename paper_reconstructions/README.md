# Paper Reconstructions — L2P v0.3.3

> Reproductions of five influential LLM+PDDL papers, re-implemented using **L2P version 0.3.3**.

Each reconstruction in this directory faithfully follows the methodology of the original paper while replacing ad‑hoc LLM scaffolding with L2P's structured builders and validation framework demonstrating the library's ability to serve as a unified substrate for research in NL-to-PDDL generation.

Most-up-to-date paper feed and reconstructions can be found here: | [PAPER RECONSTRUCTIONS](https://ai-planning.github.io/l2p/docs/paper_recreations.html) | [PAPER FEED](https://ai-planning.github.io/l2p/docs/paper_feed.html) |

---

## Papers

### 1. `llm+dm/` — *Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning*

**Guan et al. (2023)** — [GitHub](https://github.com/GuanSuns/LLMs-World-Models-for-Planning)

An action-by-action construction loop: given NL action descriptions and a type hierarchy, the LLM incrementally generates one PDDL action at a time (parameters, preconditions, effects), growing the predicate list as it goes. This reconstruction covers the *construct action models* phase (Steps 1+2).

**L2P components used:** `DomainBuilder`, PDDL syntax validator.

**Domains:** household, logistics, tyreworld.

---

### 2. `llm+p/` — *LLM+P: Empowering Large Language Models with Optimal Planning Proficiency*

**Liu et al. (2023)** — [GitHub](https://github.com/Cranial-XIX/llm-pddl)

A two-phase approach: (1) given a ground-truth PDDL domain and an NL task description with few-shot examples, the LLM generates the PDDL problem file (objects, initial state, goal); (2) a classical planner solves it. This reconstruction covers Phase 1 exclusively.

**L2P components used:** `ProblemBuilder`, `FastDownward` planner.

**Domains:** barman, blocksworld, floortile, grippers, storage, termes, tyreworld.

---

### 3. `nl2plan/` — *NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions*

**Gestrin et al. (2024)** — [GitHub](https://github.com/mrlab-ai/NL2Plan)

A fully automated five‑step pipeline that constructs a complete PDDL domain and problem from *minimal* natural language — no ground-truth PDDL, no predefined types. Each step includes LLM-based feedback for robustness.

**L2P components used:** `PromptBuilder`, `DomainBuilder`, `ProblemBuilder`, `FeedbackBuilder`, `SyntaxValidator`, `FastDownward` planner.

**Domains:** blocksworld, household, tyreworld.

---

### 4. `p+s/` — *Structured, Flexible, and Robust: Benchmarking and Improving Large Language Models Towards More Human-Like Behavior in Out-of-Distribution Reasoning Tasks*

**Collins et al. (2022)** — [GitHub](https://github.com/collinskatie/structured_flexible_and_robust)

A *Language-of-Thought* (LOT) prompting paradigm where the LLM translates NL initial and goal states into PDDL predicates via few-shot structured reasoning. The generated problem is solved against a fixed domain.

**L2P components used:** `ProblemBuilder`, `FastDownward` planner.

**Domain:** simple-blocks (custom domain with `stack`, `unstack`, `stackfromtable`).

---

### 5. `proc2pddl/` — *PROC2PDDL: Open-Domain Planning Representations from Texts*

**Zhang et al. (2024)** — [GitHub](https://github.com/zharry29/proc2pddl)

Converts procedural texts (wikiHow articles) into PDDL domains using a two-stage pipeline: (1) Zero-shot Plan Description (ZPD) — the LLM summarizes each action with entity state changes; (2) PDDL Translation — structured NL descriptions are converted into formal PDDL action models using pre-annotated types and predicates.

**L2P components used:** `PromptBuilder`, `DomainBuilder`, `FastDownward` planner.

**Domain:** "Survive on a Deserted Island" (15 actions from wikiHow).

---

## Comparison

|                             | LLM+DM      | LLM+P       | NL2Plan     | P+S          | PROC2PDDL   |
|-----------------------------|-------------|-------------|-------------|--------------|-------------|
| **Year**                    | 2023        | 2023        | 2024        | 2022         | 2024        |
| **Input**                   | NL actions  | NL task +   | NL domain + | NL states +  | Procedural  |
|                             | + types     | GT domain   | NL problem  | GT domain    | text + ann. |
| **Output**                  | PDDL domain | PDDL prob.  | Domain +    | PDDL prob.   | PDDL domain |
|                             |             | + plan      | prob. + plan| + plan       | + plan      |
| **Needs GT PDDL?**         | No          | Yes (dom.)  | No          | Yes (dom.)   | No*         |
| **LLM strategy**            | Action-by-  | Few-shot    | 5‑step pipe-| Few-shot LOT | ZPD → PDDL  |
|                             | action gen. | ICL         | line + fb.  | prompting    | translation |
| **Default LLM**             | gpt-4o-mini | gpt-4o-mini | o1-mini     | gpt-4o-mini  | gpt-4o      |

\*PROC2PDDL requires pre-annotated types, predicates, and action names.

---

## Running the Reconstructions

Each directory is self-contained. Consult the local README or `python main.py --help` within each subdirectory for usage.

All reconstructions load their LLM configuration from L2P's config manager (`~/.l2p/config.yaml`), expecting a model configured. The planner path defaults to the FastDownward submodule at `downward/fast-downward.py`.

---

> These reproductions were built on **L2P v0.3.3** and may not be directly compatible with later API changes.
