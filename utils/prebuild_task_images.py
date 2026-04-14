#!/usr/bin/env python3
"""Prebuild Harbor task Docker images and write docker_image into task.toml.

Features:
- Recursively scans tasks for directories containing task.toml + environment/Dockerfile
- Builds one image per task (no cross-task grouping)
- Reuses local image if <image-prefix>:task-<sha16> already exists (skip rebuild)
- Hash input is Dockerfile file bytes + instruction.md bytes
- Writes target hash image into each task.toml
- Deletes mismatched old images recorded in task.toml by default
- Streams docker build output in real time
- Summary statistics for discovered tasks, built/reused images, and task.toml updates
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SECTION_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")
ENVIRONMENT_SECTION_RE = re.compile(r"^\s*\[\s*environment\s*\]\s*$", re.IGNORECASE)
DOCKER_IMAGE_RE = re.compile(r"^\s*docker_image\s*=\s*['\"]([^'\"]+)['\"]\s*$")


@dataclass(frozen=True)
class TaskEntry:
    task_dir: Path
    task_toml: Path
    environment_dir: Path
    dockerfile: Path
    instruction_md: Path | None
    relative_task: str
    content_sha: str


@dataclass
class Stats:
    discovered_task_toml: int = 0
    tasks_with_dockerfile: int = 0
    tasks_missing_dockerfile: int = 0
    unique_tasks: int = 0
    images_built: int = 0
    images_reused_local: int = 0
    images_failed: int = 0
    task_toml_updated: int = 0
    task_toml_unchanged: int = 0
    deleted_images: int = 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_tasks_root = repo_root / "tasks"

    parser = argparse.ArgumentParser(
        description=(
            "Prebuild Docker images for Harbor tasks and write docker_image "
            "into task.toml."
        )
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=default_tasks_root,
        help=f"Tasks root directory to scan recursively (default: {default_tasks_root})",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="harbor-prebuilt",
        help="Image repository prefix. Final tag format: <prefix>:task-<sha16>",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional docker build platform (example: linux/amd64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions only. Do not build images or modify files.",
    )
    parser.add_argument(
        "--force",
        "--no-cache",
        action="store_true",
        dest="force",
        help="Force rebuild all images even if a matching local image already exists.",
    )
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def discover_tasks(tasks_root: Path, stats: Stats) -> list[TaskEntry]:
    entries: list[TaskEntry] = []

    for task_toml in sorted(tasks_root.rglob("task.toml"), key=lambda p: p.as_posix()):
        stats.discovered_task_toml += 1

        task_dir = task_toml.parent
        environment_dir = task_dir / "environment"
        dockerfile = environment_dir / "Dockerfile"

        if not dockerfile.exists():
            stats.tasks_missing_dockerfile += 1
            continue

        # Compute content hash from Dockerfile + instruction.md (if exists)
        instruction_md = task_dir / "instruction.md"
        hash_inputs: list[bytes] = [dockerfile.read_bytes()]
        if instruction_md.exists():
            hash_inputs.append(instruction_md.read_bytes())
        else:
            instruction_md = None
        content_sha = sha256_bytes(b"".join(hash_inputs))

        relative_task = str(task_dir.relative_to(tasks_root))

        entries.append(
            TaskEntry(
                task_dir=task_dir,
                task_toml=task_toml,
                environment_dir=environment_dir,
                dockerfile=dockerfile,
                instruction_md=instruction_md,
                relative_task=relative_task,
                content_sha=content_sha,
            )
        )
        stats.tasks_with_dockerfile += 1

    return entries


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def run_cmd_stream(cmd: list[str]) -> int:
    try:
        process = subprocess.Popen(cmd)
    except OSError as exc:
        print(f"ERROR: failed to start command {' '.join(cmd)}: {exc}", file=sys.stderr)
        return 127
    return process.wait()


def list_local_images() -> set[str]:
    result = run_cmd(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to list local images")

    images = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.strip().endswith(":<none>")
    }
    return images


def extract_task_toml_image(task_toml: Path) -> str | None:
    text = task_toml.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = DOCKER_IMAGE_RE.match(line)
        if match:
            return match.group(1).strip()
    return None


def delete_images(images: list[str], dry_run: bool) -> int:
    deleted = 0
    for image in sorted(set(images)):
        if dry_run:
            print(f"[DRY-RUN] DELETE IMAGE: {image}")
            deleted += 1
            continue

        result = run_cmd(["docker", "image", "rm", "-f", image], check=False)
        if result.returncode == 0:
            print(f"DELETE IMAGE: {image}")
            deleted += 1
        else:
            stderr = result.stderr.strip() or "unknown error"
            print(f"WARN: failed to delete image {image}: {stderr}")
    return deleted


def build_image(image_name: str, dockerfile: Path, context_dir: Path, platform: str | None, dry_run: bool) -> bool:
    cmd = [
        "docker",
        "build",
        "--progress=plain",
        "-f",
        str(dockerfile),
        "-t",
        image_name,
    ]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.append(str(context_dir))

    if dry_run:
        print("[DRY-RUN] BUILD:", " ".join(cmd))
        return True

    print("BUILD:", " ".join(cmd), flush=True)
    return_code = run_cmd_stream(cmd)
    if return_code == 0:
        print(f"BUILD OK: {image_name}")
        return True

    print(f"BUILD FAILED: {image_name} (exit_code={return_code})", file=sys.stderr)
    return False


def upsert_docker_image_in_task_toml(task_toml: Path, image_name: str, dry_run: bool) -> bool:
    text = task_toml.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    has_trailing_newline = text.endswith("\n")

    lines = text.splitlines()

    env_start = None
    for i, line in enumerate(lines):
        if ENVIRONMENT_SECTION_RE.match(line):
            env_start = i
            break

    if env_start is None:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.extend(["[environment]", f'docker_image = "{image_name}"'])
        changed = True

        new_text = newline.join(lines)
        if has_trailing_newline:
            new_text += newline

        if dry_run:
            print(f"[DRY-RUN] ADD [environment] + docker_image in task.toml: {task_toml}")
            return True

        task_toml.write_text(new_text, encoding="utf-8")
        print(f"ADD [environment] + docker_image in task.toml: {task_toml}")
        return True

    env_end = len(lines)
    for i in range(env_start + 1, len(lines)):
        if SECTION_RE.match(lines[i]):
            env_end = i
            break

    changed = False
    for i in range(env_start + 1, env_end):
        if DOCKER_IMAGE_RE.match(lines[i]):
            new_line = f'docker_image = "{image_name}"'
            if lines[i] != new_line:
                lines[i] = new_line
                changed = True
            break
    else:
        insert_at = env_end
        while insert_at > env_start + 1 and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        lines.insert(insert_at, f'docker_image = "{image_name}"')
        changed = True

    if not changed:
        return False

    new_text = newline.join(lines)
    if has_trailing_newline:
        new_text += newline

    if dry_run:
        print(f"[DRY-RUN] UPDATE task.toml: {task_toml} -> docker_image={image_name}")
        return True

    task_toml.write_text(new_text, encoding="utf-8")
    print(f"UPDATE task.toml: {task_toml} -> docker_image={image_name}")
    return True


def main() -> int:
    args = parse_args()
    stats = Stats()

    tasks_root = args.tasks_root.expanduser().resolve()
    if not tasks_root.exists() or not tasks_root.is_dir():
        print(f"ERROR: tasks root not found: {tasks_root}", file=sys.stderr)
        return 1

    entries = discover_tasks(tasks_root, stats)
    stats.unique_tasks = len(entries)

    print(f"tasks_root={tasks_root}")
    print(
        "discovered_task_toml=",
        stats.discovered_task_toml,
        "tasks_with_dockerfile=",
        stats.tasks_with_dockerfile,
        "tasks_missing_dockerfile=",
        stats.tasks_missing_dockerfile,
        "unique_tasks=",
        stats.unique_tasks,
    )

    if not entries:
        print("No buildable tasks found. Exiting.")
        return 0

    current_images_by_task: dict[Path, str | None] = {
        entry.task_toml: extract_task_toml_image(entry.task_toml) for entry in entries
    }

    try:
        local_images = list_local_images()
    except RuntimeError as exc:
        print(f"ERROR: failed to list local images: {exc}", file=sys.stderr)
        return 1

    mismatched_old_images: set[str] = set()
    target_images_in_scope = {f"{args.image_prefix}:task-{entry.content_sha[:16]}" for entry in entries}
    failed_docker_paths: list[str] = []

    for entry in entries:
        image_name = f"{args.image_prefix}:task-{entry.content_sha[:16]}"

        print()
        print(f"TASK {entry.relative_task} sha={entry.content_sha[:16]} image={image_name}")

        if image_name in local_images and not args.force:
            stats.images_reused_local += 1
            print("SKIP BUILD: found matching local hash image.")
        else:
            build_ok = build_image(
                image_name=image_name,
                dockerfile=entry.dockerfile,
                context_dir=entry.environment_dir,
                platform=args.platform,
                dry_run=args.dry_run,
            )

            if build_ok:
                stats.images_built += 1
                local_images.add(image_name)
            else:
                stats.images_failed += 1
                failed_docker_paths.append(str(entry.dockerfile))
                print("SKIP updating task.toml because image build failed.")
                continue

        previous_image = current_images_by_task.get(entry.task_toml)
        if previous_image and previous_image != image_name:
            mismatched_old_images.add(previous_image)

        changed = upsert_docker_image_in_task_toml(entry.task_toml, image_name, args.dry_run)
        if changed:
            stats.task_toml_updated += 1
        else:
            stats.task_toml_unchanged += 1

        current_images_by_task[entry.task_toml] = image_name

    stale_images_to_delete = sorted(
        image for image in mismatched_old_images if image in local_images and image not in target_images_in_scope
    )
    if stale_images_to_delete:
        print(f"\nDeleting {len(stale_images_to_delete)} mismatched old images from task.toml...")
        stats.deleted_images += delete_images(stale_images_to_delete, args.dry_run)
    elif mismatched_old_images:
        print("\nNo deletable mismatched old images found locally.")

    print("\n=== SUMMARY ===")
    print(f"tasks_root: {tasks_root}")
    print(f"task.toml discovered: {stats.discovered_task_toml}")
    print(f"tasks with Dockerfile: {stats.tasks_with_dockerfile}")
    print(f"tasks missing Dockerfile: {stats.tasks_missing_dockerfile}")
    print(f"unique tasks: {stats.unique_tasks}")
    print(f"images built/rebuilt (or planned in dry-run): {stats.images_built}")
    print(f"images reused from local cache: {stats.images_reused_local}")
    print(f"images failed: {stats.images_failed}")
    if failed_docker_paths:
        unique_failed_paths = sorted(set(failed_docker_paths))
        print(f"failed docker paths: {len(unique_failed_paths)}")
        for docker_path in unique_failed_paths:
            print(f"  - {docker_path}")
    print(f"task.toml updated: {stats.task_toml_updated}")
    print(f"task.toml unchanged: {stats.task_toml_unchanged}")
    print(f"deleted mismatched old images: {stats.deleted_images}")

    return 1 if stats.images_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
