# EvoGit

Welcome to EvoGit: Decentralized Code Evolution via Git-Based Multi-Agent Collaboration

## Demo

We provide two demos to showcase the capabilities of EvoGit:

1. [EvoGit Task 1](https://github.com/BillHuang2001/evogit_web)
2. [EvoGit Task 2](https://github.com/BillHuang2001/evogit_llm)

Both demos are **100% written by AI**, except for the initial setup.
The initial setup is performed by a human and placed in the `main` branch, while the AI-generated code is placed in the `host<i>-individual-<j>` branch for each node `i` and agent number `j`.

> [!NOTE]
> The GitHub UI may not display all branches by default. Click the `view all branches` link to see the complete list.

# EvoGit

**EvoGit** is a decentralized framework for autonomous software development, built on Git and powered by a population of LLM-based coding agents. It transforms software engineering into an evolutionary process—where agents independently propose changes, evaluate improvements, and evolve a version graph without centralized control.

## 🌐 Live Demos

We provide two demos that demonstrate the capabilities of EvoGit in real-world coding tasks:

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
