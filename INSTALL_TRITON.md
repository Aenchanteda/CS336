# Triton 安装指南

## macOS 上的 Triton 安装

### ⚠️ 重要限制
- **Triton 在 macOS 上支持有限**，特别是 Apple Silicon (ARM64)
- Triton 主要设计用于 **CUDA GPU**（Linux/Windows）
- macOS 不支持 CUDA，所以 Triton 的功能会受限

### 安装方法

#### 方法 1: 从 GitHub 安装（推荐）
```bash
pip install git+https://github.com/openai/triton.git
```

#### 方法 2: 使用 PyPI（如果可用）
```bash
# 不使用镜像源，直接从 PyPI 安装
pip install triton --no-index --find-links https://download.pytorch.org/whl/torch_stable.html
```

#### 方法 3: 使用 uv（项目推荐）
```bash
# 在项目目录下
cd /Users/richard/Documents/GitHub/cs336_assignment2
uv pip install triton
```

#### 方法 4: 如果以上都不行，尝试安装特定版本
```bash
pip install triton==2.1.0  # 或其他版本
```

### 验证安装
```python
import triton
import triton.language as tl
print(f"Triton version: {triton.__version__}")
```

### 注意事项
1. **macOS 限制**: 在 macOS 上，Triton 可能无法使用 GPU 加速，只能使用 CPU
2. **开发环境**: 如果需要在 GPU 上运行 Triton，建议使用 Linux 系统或 Google Colab
3. **测试**: 即使安装了，某些 Triton 功能在 macOS 上可能无法正常工作

### 替代方案
如果 Triton 在 macOS 上无法正常工作，可以考虑：
- 使用 PyTorch 的原生实现（如 `torch.nn.functional.scaled_dot_product_attention`）
- 在 Linux 服务器或云平台上运行
- 使用 Google Colab（提供免费的 GPU）
