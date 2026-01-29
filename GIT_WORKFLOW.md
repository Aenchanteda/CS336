# Git 工作流程指南

## 📚 作为学习记录仓库的使用方法

这个仓库用于保存和记录你的作业代码，支持不定期更新。

---

## 🚀 首次上传（已完成）

✅ 代码已成功上传到：`https://github.com/Aenchanteda/CS336`

---

## 📝 后续更新代码的标准流程

当你修改了代码，想要更新到 GitHub 时，按以下步骤操作：

### 方法 1：标准三步流程（推荐）

```bash
# 1. 查看更改状态
git status

# 2. 添加所有更改的文件
git add .

# 3. 提交更改（写清楚你做了什么）
git commit -m "描述你的更改内容"

# 4. 推送到 GitHub
git push
```

#### 💡 关于 `git add .` 中的 `.`

**`.` 表示"当前目录"**，那么"当前目录"是怎么确定的？

- **当前目录 = 你执行命令时所在的目录**
- 使用 `pwd` 命令可以查看当前目录
- 使用 `cd` 命令可以切换目录

**示例：**

```bash
# 假设你的项目在：/Users/richard/Documents/GitHub/cs336_assignment2

# 情况 1：在项目根目录执行
cd /Users/richard/Documents/GitHub/cs336_assignment2
git add .
# ✅ 会添加项目根目录下的所有文件

# 情况 2：在子目录执行
cd /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics
git add .
# ⚠️ 只会添加 cs336-basics/ 目录下的文件，不包括其他目录

# 情况 3：在错误的目录执行
cd /Users/richard/Documents
git add .
# ❌ 会添加 Documents/ 目录下的所有文件（错误！）
```

**最佳实践：**

```bash
# 1. 先确认你在项目根目录
pwd
# 应该输出：/Users/richard/Documents/GitHub/cs336_assignment2

# 2. 或者先切换到项目根目录
cd /Users/richard/Documents/GitHub/cs336_assignment2

# 3. 然后再执行 git 命令
git add .
```

**快速检查：**

```bash
# 查看当前目录
pwd

# 查看当前目录的文件（确认你在正确的位置）
ls -la

# 应该能看到 .git 目录（说明这是 Git 仓库的根目录）
```

### 方法 2：快速更新（如果只修改了几个文件）

```bash
# 1. 添加特定文件
git add cs336-basics/cs336_basics/flash_attention7.py

# 2. 提交
git commit -m "Fix FlashAttention backward pass"

# 3. 推送
git push
```

---

## 📋 提交信息的最佳实践

### ✅ 好的提交信息示例：

```bash
git commit -m "Add FlashAttention-2 backward pass implementation"
git commit -m "Fix Triton kernel memory access bug"
git commit -m "Update benchmarking script with new metrics"
git commit -m "Add documentation for attention mechanism"
```

### ❌ 避免的提交信息：

```bash
git commit -m "update"           # 太模糊
git commit -m "fix"             # 不清楚修复了什么
git commit -m "changes"         # 没有信息量
```

### 💡 提交信息模板：

```
<类型>: <简短描述>

<详细说明（可选）>

示例：
- feat: 添加新功能
- fix: 修复 bug
- docs: 更新文档
- refactor: 重构代码
- test: 添加测试
```

---

## 🔍 常用命令

### 查看状态
```bash
# 查看哪些文件被修改了
git status

# 查看详细的更改内容
git diff

# 查看提交历史
git log --oneline -10
```

### 撤销操作
```bash
# 撤销工作区的更改（未 add 的文件）
git restore <文件名>

# 撤销暂存区的更改（已 add 但未 commit）
git restore --staged <文件名>

# 修改最后一次提交（如果还没 push）
git commit --amend -m "新的提交信息"
```

### 查看远程仓库
```bash
# 查看远程仓库地址
git remote -v

# 查看远程分支
git branch -r
```

### 确认推送到正确的仓库（重要！）

推送前务必确认推送到你的仓库，而不是别人的：

```bash
# 1. 检查远程仓库地址
git remote -v
# 输出示例：
# origin  https://github.com/YOUR_USERNAME/YOUR_REPO.git (fetch)
# origin  https://github.com/YOUR_USERNAME/YOUR_REPO.git (push)
# 
# ✅ 确认：URL 中的用户名应该是你的 GitHub 用户名

# 2. 查看详细信息
git remote show origin
# 会显示远程仓库 URL、分支跟踪关系等

# 3. 查看将要推送的内容
git log origin/main..main --oneline
# 显示本地有但远程没有的提交

# 4. 确认无误后推送
git push origin main
```

#### 如果发现推送到错误的仓库

```bash
# 修改远程仓库地址
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 验证修改
git remote -v
```

#### 推送前安全检查清单

```bash
# ✅ 1. 检查远程仓库地址
git remote -v
# 确认：origin 指向你的 GitHub 仓库

# ✅ 2. 检查当前分支
git branch
# 确认：在正确的分支上（通常是 main）

# ✅ 3. 检查要推送的内容
git log origin/main..main --oneline
# 确认：只有你想推送的提交

# ✅ 4. 最后推送
git push origin main
```

**记住：推送前看一眼 `git remote -v`，避免推错仓库！**

---

## 🎯 典型使用场景

### 场景 1：完成了一个作业部分
```bash
git add .
git commit -m "Complete FlashAttention forward pass implementation"
git push
```

### 场景 2：修复了一个 bug
```bash
git add cs336-basics/cs336_basics/flash_attention7.py
git commit -m "Fix backward pass gradient computation"
git push
```

### 场景 3：添加了新的学习笔记
```bash
git add cs336-basics/cs336_basics/TRITON_VS_PYTORCH.md
git commit -m "Add notes on Triton vs PyTorch comparison"
git push
```

### 场景 4：更新了多个文件
```bash
git add .
git commit -m "Update FlashAttention implementation and add tests"
git push
```

---

## ⚠️ 注意事项

### 1. 提交前检查
```bash
# 提交前先查看会提交什么
git status
git diff --cached  # 查看已暂存的更改
```

### 2. 不要提交敏感信息
- 密码、API keys
- 个人数据
- 大型数据文件（使用 `.gitignore` 排除）

### 3. 定期推送
- 完成一个功能就推送一次
- 不要积累太多更改再推送
- 这样即使本地文件丢失，GitHub 上也有备份

### 4. 分支管理（可选）
```bash
# 创建新分支用于实验
git checkout -b experiment-branch

# 切换回主分支
git checkout main

# 合并分支
git merge experiment-branch
```

---

## 🔗 查看你的代码

访问你的 GitHub 仓库：
**https://github.com/Aenchanteda/CS336**

---

## 💡 小贴士

1. **每天结束前推送一次**：确保代码安全
2. **提交信息要清晰**：方便以后回顾
3. **使用 `.gitignore`**：自动排除不需要的文件
4. **定期查看 GitHub**：确认代码已成功上传
5. **推送前检查远程仓库**：执行 `git remote -v` 确认推送到正确的仓库
6. **确认当前目录**：执行 `git add .` 前用 `pwd` 确认你在项目根目录

---

## 🆘 遇到问题？

### 推送失败
```bash
# 先拉取远程更改
git pull origin main

# 解决冲突后再推送
git push
```

### 忘记提交了什么
```bash
# 查看最后一次提交
git show

# 查看提交历史
git log --oneline -5
```

---

**记住：Git 是你的代码时光机，好好利用它！** 🚀
