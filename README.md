# SkillFlow

SkillFlow is a benchmark for evaluating autonomous agents on executable office and data workflows, with support for both baseline runs and iterative shared-skill evolution.

## What is SkillFlow?

SkillFlow focuses on two settings:

- **Baseline**: run each workflow family without cross-task skill evolution
- **Iterative**: evolve shared skills across tasks within a workflow family

This repository contains the code, runners, analysis scripts, and Docker setup for the benchmark.
Task data is distributed separately via Hugging Face.

## Quick Start

```bash
# Install Harbor
uv tool install 'harbor @ git+https://github.com/laude-institute/harbor.git'

# Install project dependencies
uv sync

# Download task data from Hugging Face
hf download zhang-ziao/SkillFlow-Task --repo-type dataset --local-dir test_tasks

# Build the base image
./docker/harbor-cli-base/build.sh

# Optionally prebuild task images
python utils/prebuild_task_images.py --tasks-root test_tasks --image-prefix skillflow-prebuilt
```

After downloading, the local layout is expected to look like:

```text
test_tasks/
  <workflow-family>/
    ALL_TASK_DIFFICULTY_RANKING.json
    <task-name>/
      instruction.md
      task.toml
      environment/
      tests/
      solution/
```

## Run the Benchmark

### Baseline

Edit `configs/baseline.yaml`, then run:

```bash
python family_job_runner.py
```

### Iterative Shared Skills

Edit `configs/iter.yaml`, then run:

```bash
python iterative_shared_skills_runner.py
```

The iterative setting uses `shared_skills_template/skills` as the default initial shared-skill directory.

## Repository Layout

- `configs/`: example configs for baseline and iterative runs
- `docker/harbor-cli-base/`: base image with preinstalled agent CLIs
- `analysis/`: result summarization and plotting scripts
- `utils/prebuild_task_images.py`: prebuild task images and write `docker_image` into `task.toml`
- `shared_skills_template/`: initial shared-skill template

## Notes

- This release does **not** include OpenHands in the base image.
- Domestic package mirrors are intentionally removed from the Docker setup.
- Replace API keys, model names, and endpoints in the example configs before running.
