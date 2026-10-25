# RAGspector

**An autonomous, budget-aware reliability auditor for Retrieval-Augmented Generation (RAG) systems.**

RAGspector investigates how a RAG system behaves under controlled changes to queries, retrieved evidence, and retrieval settings. Rather than repairing answers at runtime, it adaptively selects diagnostic tests, evaluates the results, and builds an evidence-backed profile of the system's reliability.

## Why RAGspector?

A RAG system can produce incorrect or unsupported answers when queries are rephrased, irrelevant or conflicting evidence is introduced, retrieval settings change, or the order of retrieved passages shifts. Running every possible test can be expensive. RAGspector explores whether **adaptive test selection can uncover meaningful weaknesses with fewer tests than fixed, random, or uniform testing**.

## How it works

1. **Generate probes** — create controlled tests for query, retrieval, noise, conflicting-evidence, and context-order robustness.
2. **Select a test** — prioritize candidate probes using estimated information gain, testing cost, coverage, and previous observations.
3. **Execute and evaluate** — run the probe against a target RAG system and assess correctness, retrieval quality, faithfulness, and behavioral changes where the required evidence is available.
4. **Update reliability state** — record outcomes and update per-category risk estimates.
5. **Report findings** — produce a reliability profile, reproducible failure cases, and experiment-cost statistics.

## Research question

> Can an information-guided, cost-aware testing agent discover RAG reliability weaknesses more efficiently than fixed, random, or uniform test selection under the same testing budget?

## Planned components

- **RAG adapter:** a built-in baseline RAG pipeline plus an adapter interface for compatible external RAG APIs.
- **Probe generator:** deterministic and optional LLM-assisted test generation.
- **Test selector:** an adaptive information-gain-per-cost policy, compared with fixed, random, and uniform baselines.
- **Evaluator:** benchmark-based correctness, retrieval metrics, evidence faithfulness, and robustness comparisons.
- **Audit engine:** bounded execution, budget tracking, reproducible experiment logging, and reliability-state updates.
- **Dashboard and reports:** inspect test traces, reliability estimates, supporting evidence, and policy comparisons.

## Initial scope

The first version targets text-based RAG and five probe families: query paraphrasing/noise, retrieval-configuration changes, irrelevant evidence, conflicting evidence, and context-order changes. It will use a small demo corpus by default and support benchmark imports where data are available.

Candidate resources include [RAGTruth](https://github.com/ParticleMedia/RAGTruth) for faithfulness-evaluator validation and [RARE](https://github.com/LeiLiLab/RARE) for robustness-testing resources. Dataset availability, schema, version, and license will be verified before experiments; demo data will be clearly labelled.

## Planned technology stack

- **Backend:** Python, FastAPI, Pydantic
- **RAG and embeddings:** Sentence Transformers, FAISS, Hugging Face Transformers/Datasets
- **Persistence:** SQLite
- **Frontend:** React, Vite, TypeScript
- **Testing:** pytest
- **Model providers:** configurable local or OpenAI-compatible endpoints, with a deterministic mock mode for development and tests

The project does not require LLM fine-tuning. The main research focus is the adaptive test-selection policy and its evaluation against comparable baselines.

## Research evaluation

All selection policies will use comparable candidate pools and testing budgets. The evaluation will track faults and failure categories discovered, test redundancy, coverage, information gain, execution cost, latency, and model calls. Results will be measured rather than assumed; no improvement is claimed in advance.

## Development status

🚧 **Under development.** The initial milestone is an end-to-end demo audit, followed by the adaptive selector, benchmark integrations, policy comparisons, and dashboard.
