# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## How to run

### 1. naive_ddp.py
```aiignore
uv run cs336-basics.naive_ddp
```

## ToDo
- The first whole part
  1. 1.1 `Profiling and Benchmarking` (need to do)
- 2.2 `Problem (naive_ddp_benchmarking): 3 points`
- 2.3.1 `Problem (minimal_ddp_flat_benchmarking): 2 points`
- 2.3.2 `Problem (ddp_overlap_individual_parameters_benchmarking): 1 point`
- 2.3.3 `Problem (ddp_bucketed_benchmarking): 3 points`
- 2.4 `4D Parallelism`
- 3 `Optimizer State Sharding`





1. 检查 Git 状态
cd /Users/richard/Documents/GitHub/cs336_assignment2
git status
2. 添加文件到暂存区
# 添加所有修改的文件git add .# 或者只添加特定文件git add cs336-basics/cs336_basics/flash_attention7.py
3. 提交更改
git commit -m "Update flash attention implementation"
4. 推送到 GitHub
# 如果已经设置了远程仓库git push origin main# 或者git push origin master

方法 2：如果是新项目或需要创建新仓库
1. 初始化 Git 仓库（如果还没有）
Documents
cd /Users/richard/Documents/GitHub/cs336_assignment2git init
2. 创建 .gitignore（可选但推荐）
# 创建 .gitignore 文件，排除不需要上传的文件
cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/
.venv/
venv/
wandb/
*.prof
*.log
.DS_Store
EOF

3. 添加文件并提交
git add .
git commit -m "Initial commit: FlashAttention implementation"

4. 在 GitHub 上创建新仓库
访问 https://github.com/new
输入仓库名称（例如：cs336_assignment2）
选择 Public 或 Private
不要勾选 "Initialize with README"（因为本地已有代码）
点击 "Create repository"

5. 添加远程仓库并推送
# 替换 YOUR_USERNAME 为你的 GitHub 用户名
git remote add origin https://github.com/YOUR_USERNAME/cs336_assignment2.git
git branch -M main
git push -u origin main

git remote set-url origin https://github.com/Aenchanteda/CS336
