# AI Coding Agent Instructions for CS336 Assignment 2

Welcome to the CS336 Assignment 2 codebase! This document provides essential guidance for AI coding agents to be productive in this project. The focus is on understanding the architecture, workflows, and conventions specific to this repository.

## Project Overview

This repository is structured around two main modules:

1. **`cs336_basics`**: Contains the staff implementation of the language model from Assignment 1. This module includes foundational components like tokenizers, optimizers, and model utilities. Key files:
   - `cs336_basics/model.py`: Core language model implementation.
   - `cs336_basics/optimizer.py`: Optimizer implementations.
   - `cs336_basics/transformer/`: Transformer-related modules.

2. **`cs336_systems`**: The primary focus of Assignment 2. This module is where students implement optimized Transformer models, distributed training, and other advanced features. It starts as a skeleton and is meant to be populated with new code.

## Developer Workflows

### Building and Testing
- **Install Dependencies**: Run `scripts/install_deps.sh` to install required dependencies.
- **Run Tests**: Use `pytest` to run tests located in the `tests/` directory. Example:
  ```bash
  pytest tests/
  ```
- **Submission Script**: Use `test_and_make_submission.sh` to validate your implementation and package it for submission.

### Debugging
- Use the provided `.ipynb` notebooks (e.g., `cs336_basics/5_triton.ipynb`) for interactive debugging and experimentation.
- Profiling tools like `nsys` are used in some scripts (e.g., `cs336_basics/2_nsys.ipynb`).

## Project-Specific Conventions

- **File Organization**: Keep Assignment 1 code in `cs336_basics` and Assignment 2 code in `cs336_systems`. Avoid mixing the two.
- **Testing**: Place all test files in the `tests/` directory. Use descriptive names (e.g., `test_ddp.py` for distributed data parallel tests).
- **Documentation**: Follow the example in `README.md` for documenting new modules or scripts.

## Integration Points

- **Distributed Training**: Implement distributed training logic in `cs336_systems/`. Refer to `cs336_basics/naive_ddp.py` for a starting point.
- **External Dependencies**: Key dependencies include:
  - `torch`: For deep learning.
  - `wandb`: For experiment tracking.
  - `pytest`: For testing.

## Examples

### Adding a New Optimizer
1. Create a new file in `cs336_basics/optimizer/` (e.g., `my_optimizer.py`).
2. Implement the optimizer class, ensuring it adheres to the interface in `cs336_basics/optimizer.py`.
3. Add tests in `tests/test_my_optimizer.py`.

### Implementing a New Transformer Feature
1. Add the feature in `cs336_systems/`.
2. Use `cs336_basics/transformer/` as a reference for structure and style.
3. Test the feature using scripts or notebooks in `cs336_systems/`.

---

For any issues or questions, refer to the `README.md` or open a GitHub issue. Happy coding!