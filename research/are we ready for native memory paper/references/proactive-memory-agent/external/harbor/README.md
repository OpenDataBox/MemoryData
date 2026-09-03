# Terminal-Bench + Enroot Setup Guide

This document describes how to run Terminal-Bench experiments on an AWS SLURM cluster using Enroot as the container runtime.

**Important:** Enroot must be run inside a SLURM job (via `srun`), not on the login node. The login node does not have the required permissions to run containers.

## Start an interactive SLURM session first
srun --time=4:00:00 --cpus-per-task=8 --mem=32G --pty bash -c "source ~/.bashrc && exec bash"
```

## 1. Install Harbor (with Enroot Support)

```bash
# In your conda env, clone our fork with Enroot support
cd /fsx/$USER
git clone https://github.com/yifannnwu/harbor.git
cd harbor
pip install -e .
```

Verify installation:
```bash
hr --help
```

**Note:** The Enroot environment implementation is in `src/harbor/environments/enroot.py`. This is our modification to the original Harbor codebase.

## 2. Configure Environment Variables

```bash
# OpenRouter API
export OPENROUTER_API_KEY="your-api-key-here"

# Enroot cache directory - USE YOUR OWN PATH
export ENROOT_CACHE_PATH="/fsx/$USER/enroot-cache"
export ENROOT_DATA_PATH="/fsx/$USER/enroot-data"
mkdir -p $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
```

Add these to your `~/.bashrc` for persistence:
```bash
echo 'export ENROOT_CACHE_PATH="/fsx/$USER/enroot-cache"' >> ~/.bashrc
echo 'export ENROOT_DATA_PATH="/fsx/$USER/enroot-data"' >> ~/.bashrc
source ~/.bashrc
```

## 3. Download and Pre-cache Terminal-Bench Images

Terminal-Bench uses Docker images hosted on GitHub Container Registry (ghcr.io). Since we use Enroot instead of Docker, we need to:
1. Download the Docker images
2. Convert them to Enroot squashfs format
3. Cache them locally

### Option A: Copy Pre-cached Images

If someone on your cluster has already built the images, copy them into your cache:

```bash
# Copy all cached .sqsh files
cp /path/to/shared/enroot-cache/*.sqsh $ENROOT_CACHE_PATH/

# Verify
ls -la $ENROOT_CACHE_PATH/*.sqsh
```

### Option B: Download Fresh Images

If you don't have access to the pre-cached images, download them yourself:

#### B.1 Download Dataset and List Required Images

First, download the dataset to get task definitions:
```bash
hr datasets download terminal-bench-sample@2.0
```

Then extract the Docker image names from the task configs:
```bash
# List all required images
for dir in ~/.cache/harbor/tasks/*/; do 
    cat "$dir"/*/task.toml 2>/dev/null | grep docker_image
done | sort -u
```

#### B.2 Configure GitHub Credentials

The terminal-bench images are hosted on GitHub Container Registry (ghcr.io). You need to authenticate:

```bash
# Create a GitHub Personal Access Token (PAT) with `read:packages` scope
# Go to: https://github.com/settings/tokens/new
# Select scope: read:packages

# Set credentials for enroot
export ENROOT_LOGIN="<your-github-username>"
export ENROOT_PASSWORD="<your-github-pat>"
```

#### B.3 Download and Import Images

**Important:** Run `enroot import` inside a SLURM job, not on the login node.

**For terminal-bench-sample@2.0 (10 tasks, requires GitHub auth):**
```bash
for task in build-cython-ext chess-best-move configure-git-webserver \
            fix-code-vulnerability log-summary-date-ranges polyglot-c-py \
            qemu-alpine-ssh qemu-startup regex-log sqlite-with-gcov; do
    echo "=== Importing $task ==="
    enroot import docker://ghcr.io/laude-institute/terminal-bench/$task:2.0
done
```

**For terminal-bench@2.0 (89 tasks, Docker Hub - no auth needed):**
```bash
# Extract image names and import each one
for dir in ~/.cache/harbor/tasks/*/; do 
    cat "$dir"/*/task.toml 2>/dev/null | grep docker_image
done | sort -u | grep "alexgshaw" | sed 's/docker_image = "//g' | sed 's/"//g' | while read image; do
    echo "=== Importing $image ==="
    enroot import docker://$image
done
```

**What happens:**
1. `enroot import` pulls the Docker image
2. Converts the image layers to a single squashfs file
3. Saves it to `$ENROOT_CACHE_PATH`


### 3.1 Verify Image Cache

After import (or copy), you should see these files:

```bash
ls -la $ENROOT_CACHE_PATH/*.sqsh
```

Expected output:
```
ghcr.io+laude-institute+terminal-bench+build-cython-ext+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+chess-best-move+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+configure-git-webserver+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+fix-code-vulnerability+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+log-summary-date-ranges+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+polyglot-c-py+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+qemu-alpine-ssh+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+qemu-startup+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+regex-log+2.0.sqsh
ghcr.io+laude-institute+terminal-bench+sqlite-with-gcov+2.0.sqsh
```

## 4. Run Experiments

**Important:** All `hr run` commands must be executed inside a SLURM job (via `srun`), not on the login node.

### 4.1 Basic Command Format

```bash
hr run \
    --dataset terminal-bench-sample@2.0 \
    --agent terminus-2 \
    --model <provider>/<model-name> \
    --env enroot
```

### 4.2 Examples

**Claude Sonnet 4.5 (via OpenRouter):**
```bash
hr run \
    --dataset terminal-bench-sample@2.0 \
    --agent terminus-2 \
    --model openrouter/anthropic/claude-sonnet-4.5 \
    --env enroot 
```

**Qwen3-32B (via OpenRouter):**
```bash
hr run \
    --dataset terminal-bench-sample@2.0 \
    --agent terminus-2 \
    --model openrouter/qwen/qwen3-32b \
    --env enroot 
```

## 5. View Results

Results are saved under the `jobs/` directory.

