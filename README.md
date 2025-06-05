<h1 align="center">
  <img src="static/evogit_logo_with_text.svg" alt="EvoGit Logo" height="64"/>
</h1>

<h2 align="center"><strong><em>Decentralized Code Evolution via Git-Based Multi-Agent Collaboration</em></strong></h2>

<p align="center">
  <a href="https://www.arxiv.org/abs/2506.02049">
    <img src="https://img.shields.io/badge/arXiv-2506.02049-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/BillHuang2001/evogit">
    <img src="https://img.shields.io/github/stars/BillHuang2001/evogit?style=social" alt="GitHub Stars">
  </a>
</p>

---



## 🚀 Overview

**EvoGit** is a decentralized, Git-native framework for autonomous software development. It leverages a population of LLM-based agents that evolve code collaboratively—without centralized control—mirroring principles of natural selection and mutation.

Each agent independently proposes, mutates, and merges code changes, forming a fully traceable version graph managed by Git.

For detailed methodology and experimental results, refer to our [paper](https://arxiv.org/abs/2506.02049).

## ✨ Key Features

* ⚙️ **Git-Native Evolution**: All code changes are tracked as Git commits and branches, leveraging Git as the core infrastructure.
* 🧠 **Fully Autonomous Multi-Agent Development**: Decentralized agents simulate independent developers, evolving codebases without coordination.
* 🌿 **Transparent Lineage Tracking**: Every commit, branch, and merge is fully version-controlled and inspectable.
* 🔁 **Minimal Human Supervision**: After initialization, the system progresses with only sparse, strategic human feedback.

## 📦 Live Demos

See EvoGit in action on two real-world tasks:

### [📃 EvoGit Web](https://github.com/BillHuang2001/evogit_web)

> A multi-agent system collaboratively builds a complete, interactive one-page website—from layout to animation to dark mode. The human product manager initialized the repo and gave \~10 feedbacks over the course of development.

### [🧠 EvoGit LLM](https://github.com/BillHuang2001/evogit_llm)

> Agents develop an LLM-powered optimizer that evolves code to solve the bin packing problem. The human product manager provided an initial setup and \~5 pieces of feedback.


## 🧬 How to Explore the Results

EvoGit uses Git not only as a version control tool, but also as a transparent window into the code evolution process. Here's how to inspect our demos:

1. 🧑‍💻 The human-initialized seed lives in the `main` branch.
2. 🤖 AI-generated code lives in branches named:
   `host<i>-individual-<j>`,
   where `i` = host node index, `j` = agent index.
3. 🔍 Each agent branch contains an independent development trajectory. You can explore these using GitHub’s commit history or local Git tools.
4. 📈 Git diffs and logs reveal the precise changes made in each commit.
5. 🧭 Use `git log --graph` or GitHub’s branch visualization to see how code diverged and converged over time.

All changes are versioned and traceable. Every commit represents an autonomous decision by an agent—captured, auditable, and reproducible through Git.

> [!NOTE]
> GitHub may hide some branches. Click **“View all branches”** on the repo page to see the complete version graph.


## 📚 Paper

Read the full framework design, evaluation methodology, and results in our paper:
- **[ArXiv:2506.02049](https://arxiv.org/abs/2506.02049)**
