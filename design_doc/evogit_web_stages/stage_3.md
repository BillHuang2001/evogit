The current task is to develop the EvoX homepage, which will serve as the primary entry point for users. The goal is to create a clean, modern, and informative landing page that effectively communicates the purpose and capabilities of EvoX.
At this stage, please focus on enriching and revising each section, including the adding more content, improving the color and design, etc.
Here is the outline for the EvoX homepage development:

# 🧭 EvoX Homepage Development Outline

---

## 1. Objective

* **Primary Goal**: Introduce EvoX to new users (developers, researchers)
* **Secondary Goals**:

  * Drive adoption (install, GitHub stars, contributions)
  * Provide fast access to docs and examples
  * Convey credibility and ease of use

---

## 2. Overall Layout Structure

```
|---------------------------|
|   News / Updates          |
|---------------------------|
|   Navigation Bar          |
|---------------------------|
|   Hero Section            |
|---------------------------|
|   Key Features Section    |
|---------------------------|
|   Code Example Section    |
|---------------------------|
|   Community / Ecosystem   |
|---------------------------|
|   Footer                  |
|---------------------------|
```

---

## 3. Sections Breakdown

---

### 🔹 News / Updates

* **Purpose**: Highlight recent updates, releases, or important announcements
* **Format**: A small banner or ticker at the top of the page

### 🔹 Navigation Bar

* **Logo** (EvoX)
* **Links**:

  * Home
  * Docs
  * GitHub
  * Install
  * Community (optional)

**Sticky on scroll**, responsive for mobile.

---

### 🔹 Hero Section

* **Headline**:

  > *“EvoX: Evolutionary Computation Reimagined”*
* **Subtext**:

  > *“A powerful, flexible distributed and GPU-accelerated framework for evolutionary algorithms in modern AI workflows.”*
* **Call-to-Action (CTA)**:

  * ✅ **Get Started** (link to docs)
  * ✅ **GitHub Repo**
  * ✅ **Install** snippet: `pip install evox`

Optional: background animation or 3D visuals (keep minimal).

---

### 🔹 Key Features Section

Highlighting some of the key features of EvoX, no need to list all of them, prioritize the layout and readability.
Optional: use icons or emojis to represent each feature to make it visually appealing.

### 🔹 Code Example Section

**Show a section of minimal, readable code snippets, optionally show the code output alongside**.

* Syntax-highlighted block with "copy" button
* Include high-level EvoX workflow
* Emphasize simplicity
* Some possible examples are given in the EvoX readme, and their outputs are given as assets to the project.

---

### 🔹 Community / Ecosystem Section

* **Links**:

  * GitHub (stars, issues)
  * Discord community
  * Blog / Tutorials
  * Sister projects

Include:

* Badges (e.g. PyPI version, license)
* Optional: Logo wall for adopters/contributors

---

### 🔹 Footer

Keep it minimal and consistent:

| Column 1 | Column 2   | Column 3        |
| -------- | ---------- | --------------- |
| Docs     | GitHub     | License         |
| Install  | Contribute | Acknowledgments |

---

## 4. **Design Guidelines**

* **Fonts**: Sans-serif (e.g., Inter, Roboto)
* **Color Scheme**: 1–2 primary colors, vibrant but not overwhelming.
* **Layout**: Responsive (mobile-first), grid or flexbox layout
* **Dark Mode**: Optional, but encouraged

---

# Related information: EvoX's Github Readme

## 🔥 News
- [2025-05-13] Released **EvoX 1.2.1** - 🚀 EvoX v1.2.1 release is now available, featuring the new Mujoco Playground and an official tutorial! [[Details](https://evox.group/index.php?m=home&c=View&a=index&aid=157)]
- [2025-02-03] Released **EvoRL**: A GPU-accelerated framework for **Evolutionary Reinforcement Learning**, powered by **JAX** ! [[Paper](https://arxiv.org/abs/2501.15129)] [[Code](https://github.com/EMI-Group/evorl)]
- [2025-01-30] Released **EvoGP**: A GPU-accelerated framework for **Genetic Programming**, powered by **PyTorch** & **CUDA**! [[Paper](http://arxiv.org/abs/2501.17168)] [[Code](https://github.com/EMI-Group/evogp)]
- [2025-01-14] Released **EvoX 1.0.0** - now fully compatible with **PyTorch**, with full `torch.compile` support! Users of the previous **JAX-based version** can access it on the **v0.9.0 branch**.

## Overview

EvoX is a distributed GPU-accelerated evolutionary computation framework compatible with **PyTorch**.  With a user-friendly programming model, it offers a comprehensive suite of **50+ Evolutionary Algorithms (EAs)** and a wide range of **100+ Benchmark Problems/Environments**.

## Key Features

### 💻 High-Performance Computing

#### 🚀 Ultra Performance
- Supports acceleration on heterogeneous hardware, including both **CPUs** and **GPUs**, achieving over **100x speedups**.
- Integrates **distributed workflows** that scale seamlessly across multiple nodes or devices.

#### 🌐 All-in-One Solution
- Includes **50+ algorithms** for a wide range of use cases, fully supporting **single- and multi-objective optimization**.
- Provides a **hierarchical architecture** for complex tasks such as **meta learning**, **hyperparameter optimization**, and **neuroevolution**.

#### 🛠️ Easy-to-Use Design
- Fully compatible with **PyTorch** and its ecosystem, simplifying algorithmic development with a **tailored programming model**.
- Ensures effortless setup with **one-click installation** for Windows users.


### 📊 Versatile Benchmarking

#### 📚 Extensive Benchmark Suites
- Features **100+ benchmark problems** spanning single-objective optimization, multi-objective optimization, and real-world engineering challenges.

#### 🎮 Support for Physics Engines
- Integrates seamlessly with physics engines like **Brax** and other popular frameworks for reinforcement learning.

#### ⚙️ Customizable Problems
- Provides an **encapsulated module** for defining and evaluating custom problems tailored to user needs, with seamless integration into real-world applications and datasets.


### 📈 Flexible Visualization

#### 🔍 Ready-to-Use Tools
- Offers a comprehensive set of **visualization tools** for analyzing evolutionary processes across various tasks.

#### 🛠️ Customizable Modules
- Enables users to integrate their own **visualization code**, allowing for tailored and flexible visualizations.

#### 📂 Real-Time Data Streaming
- Leverages the tailored **.exv format** to simplify and accelerate real-time data streaming.

## Quick Start

Here are some examples to get you started with EvoX:

### Single-objective Optimization

Solve the Ackley problem using the PSO algorithm:

```python
import torch
from evox.algorithms import PSO
from evox.problems.numerical import Ackley
from evox.workflows import StdWorkflow, EvalMonitor

# torch.set_default_device("cuda") # Uncomment this line if you want to use GPU by default

algorithm = PSO(pop_size=100, lb=-32 * torch.ones(10), ub=32 * torch.ones(10))
problem = Ackley()
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()

monitor.plot() # or monitor.plot().show() if you are using headless mode
```

### Multi-objective Optimization

Solve the DTLZ2 problem using the RVEA algorithm:

```python
import torch
from evox.algorithms import RVEA
from evox.metrics import igd
from evox.problems.numerical import DTLZ2
from evox.workflows import StdWorkflow, EvalMonitor

# torch.set_default_device("cuda") # Uncomment this line if you want to use GPU by default

prob = DTLZ2(m=2)
pf = prob.pf()
algo = RVEA(
    pop_size=100,
    n_objs=2,
    lb=-torch.zeros(12),
    ub=torch.ones(12)
)
monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()

monitor.plot() # or monitor.plot().show() if you are using headless mode
```

### Neuroevolution

Evolving a simple MLP model to solve the Brax HalfCheetah environment:

```python
import torch
import torch.nn as nn
from evox.algorithms import PSO
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

# torch.set_default_device("cuda") # Uncomment this line if you want to use GPU by default

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # observation space is 17-dim, action space is 6-dim.
        self.features = nn.Sequential(nn.Linear(17, 8), nn.Tanh(), nn.Linear(8, 6))

    def forward(self, x):
        return torch.tanh(self.features(x))

# Initialize the MLP model
model = SimpleMLP()
adapter = ParamsAndVector(dummy_model=model)
# Set the population size
POP_SIZE = 1024
# Get the bound of the PSO algorithm
model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)
lb = torch.full_like(pop_center, -5)
ub = torch.full_like(pop_center, 5)
# Initialize the PSO, and you can also use any other algorithms
algorithm = PSO(pop_size=POP_SIZE, lb=lb, ub=ub)
# Initialize the Brax problem
problem = BraxProblem(
    policy=model,
    env_name="halfcheetah",
    max_episode_length=1000,
    num_episodes=3,
    pop_size=POP_SIZE,
)
# set an monitor, and it can record the top 3 best fitnesses
monitor = EvalMonitor(topk=3)
# Initiate an workflow
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    opt_direction="max",
    solution_transform=adapter,
)
workflow.init_step()
for i in range(50):
    workflow.step()

monitor.plot() # or monitor.plot().show() if you are using headless mode
```

## Sister Projects
- **EvoRL**:GPU-accelerated framework for Evolutionary Reinforcement Learning. Check out [here](https://github.com/EMI-Group/evorl).
- **EvoGP**:GPU-accelerated framework for Genetic Programming. Check out [here](https://github.com/EMI-Group/evogp).
- **TensorNEAT**: Tensorized NeuroEvolution of Augmenting Topologies (NEAT) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorneat).
- **TensorRVEA**: Tensorized Reference Vector Guided Evolutionary Algorithm (RVEA) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensorrvea).
- **TensorACO**: Tensorized Ant Colony Optimization (ACO) for GPU Acceleration. Check out [here](https://github.com/EMI-Group/tensoraco).
- **EvoXBench**: A real-world benchmark platform for solving various optimization problems, such as Neural Architecture Search (NAS). It operates without the need for GPUs/PyTorch/TensorFlow and supports multiple programming environments. Check out [here](https://github.com/EMI-Group/evoxbench).

## Links

- [GitHub Repo](https://github.com/EMI-Group/evox)
- [Documentation](https://evox.readthedocs.io/en/latest/)
- [Paper](https://arxiv.org/abs/2301.12457)
- [Discord Community](https://discord.gg/Vbtgcpy7G4)
- [QQ Group](https://qm.qq.com/q/vTPvoMUGAw)
