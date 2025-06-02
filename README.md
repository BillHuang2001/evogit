# EvoGit

**EvoGit** is a decentralized framework for autonomous software development, built on Git and powered by a population of LLM-based coding agents. It transforms software engineering into an evolutionary process—where agents independently propose changes, evaluate improvements, and evolve a version graph without centralized control.

## 🌐 Live Demos

We provide two demos that demonstrate the capabilities of EvoGit in real-world coding tasks,
we welcome you to explore them and see with your own eyes how EvoGit works:

1. [📃 Task 1: EvoGit Web](https://github.com/BillHuang2001/evogit_web)
   An autonomous multi-agent system builds a one-page website from scratch, including layout, animations, and dark mode support.

2. [🧠 Task 2: EvoGit LLM](https://github.com/BillHuang2001/evogit_llm)
   EvoGit generates a program that itself uses LLMs to evolve code for solving an optimization problem (bin packing).

## 🧬 How It Works

Each project begins with a minimal human-authored setup committed to the `main` branch. From there, EvoGit agents branch out to create independent development lines:

- AI-generated branches follow the pattern:
  `host<i>-individual-<j>`
  where `i` is the node index and `j` is the agent number.

- Every commit, mutation, and merge is tracked using Git, forming a fully traceable version graph.

> [!NOTE]
> GitHub sometimes hides some branches. Click **“View all branches”** on the project page to see the full version graph.

## 📖 More Information

For a deeper understanding of the EvoGit framework—its design principles, methodology, and evaluation—please refer to the accompanying paper.
