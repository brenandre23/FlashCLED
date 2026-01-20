#!/usr/bin/env python3
"""
Helper script to build a structured Git commit message with category notes.

Categories are derived from the top-level directory of each changed file.
The script can optionally stage all changes and run `git commit` using the
generated message.

Usage examples
--------------
- Interactive notes, stage all changes, and commit:
    python scripts/github_commit.py --message "Add landcover features" --stage-all --commit

- Generate the message only (no commit), save to a custom file:
    python scripts/github_commit.py --message "Refactor pipeline" --output tmp_commit_message.txt

Flags
-----
--message/-m       Commit subject line (if omitted, you will be prompted)
--stage-all        Run `git add -A` before committing
--commit           Run `git commit -F <output>` after building the message
--non-interactive  Skip note prompts; fill each category with a files summary
--output/-o        Path to write the composed commit message (default: commit_message.txt)
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run(cmd):
    """Run a shell command and return stdout as text."""
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()


def get_changed_files():
    """
    Returns a mapping of category -> list of (status, path) tuples.
    Category is the first path segment (or 'root' for files in repo root).
    """
    try:
        raw = run(["git", "status", "--porcelain"])
    except subprocess.CalledProcessError as exc:
        print(f"Error running git status: {exc.output.decode()}", file=sys.stderr)
        sys.exit(1)

    if not raw:
        return {}

    categories = defaultdict(list)
    for line in raw.splitlines():
        status = line[:2].strip()
        path = line[3:].strip()
        # Handle rename syntax "old -> new"
        if "->" in path:
            path = path.split("->", 1)[1].strip()
        category = path.split("/", 1)[0] if "/" in path else "root"
        categories[category].append((status, path))
    return categories


def prompt_notes(categories, non_interactive=False):
    """
    Collects a single note per category. In non-interactive mode,
    the note is auto-filled with a list of changed files.
    """
    notes = {}
    for category, entries in sorted(categories.items()):
        file_list = ", ".join(p for _, p in entries)
        if non_interactive:
            notes[category] = f"Changes: {file_list}"
            continue

        print(f"\n[{category}]")
        for status, path in entries:
            print(f"  {status or '?'} {path}")
        note = input("Note for this category (leave blank to auto-fill): ").strip()
        if not note:
            note = f"Changes: {file_list}"
        notes[category] = note
    return notes


def build_message(subject, notes):
    """Constructs a commit message string."""
    lines = [subject, ""]
    for category, note in sorted(notes.items()):
        lines.append(f"{category}:")
        lines.append(f"- {note}")
        lines.append("")  # blank line between categories
    return "\n".join(lines).rstrip() + "\n"


def stage_all():
    """Stage all tracked/untracked changes."""
    subprocess.check_call(["git", "add", "-A"])


def commit_with_message(path):
    """Run git commit using the provided message file."""
    subprocess.check_call(["git", "commit", "-F", str(path)])


def main():
    parser = argparse.ArgumentParser(description="Structured GitHub commit helper")
    parser.add_argument("-m", "--message", help="Commit subject line")
    parser.add_argument("--stage-all", action="store_true", help="Run git add -A before committing")
    parser.add_argument("--commit", action="store_true", help="Run git commit after generating the message")
    parser.add_argument("--non-interactive", action="store_true", help="Skip prompts and auto-fill notes")
    parser.add_argument("-o", "--output", default="commit_message.txt", help="Path to write the commit message")
    args = parser.parse_args()

    categories = get_changed_files()
    if not categories:
        print("No changes detected. Nothing to commit.")
        sys.exit(0)

    subject = args.message or input("Commit subject: ").strip()
    if not subject:
        print("Commit subject is required.", file=sys.stderr)
        sys.exit(1)

    notes = prompt_notes(categories, non_interactive=args.non_interactive)
    message = build_message(subject, notes)

    output_path = Path(args.output)
    output_path.write_text(message, encoding="utf-8")
    print(f"\nWrote structured commit message to {output_path}")
    print(message)

    if args.stage_all:
        print("Staging all changes (git add -A)...")
        stage_all()

    if args.commit:
        print("Committing with generated message...")
        commit_with_message(output_path)
        print("Commit created.")
    else:
        print("Skipping git commit (use --commit to commit automatically).")


if __name__ == "__main__":
    main()
