#!/usr/bin/env python3
"""
Dump/Merge Codebase Script

Combines all source code files of the project into a single text/markdown file.
Features:
- Skips virtual environments, git tracking, caches, binary files, and datasets.
- Automatically detects and skips binary files.
- Generates a Table of Contents / File Index with line counts.
- Wraps file contents with clear delimiters and syntax-highlighted code fences.
- Zero external dependencies (uses standard library only).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Default directory patterns to ignore
DEFAULT_IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    ".venv*",
    "venv",
    "env",
    "data",
    "datasets",
    "results",
    "outputs",
    "checkpoints",
    "runs",
    "wandb",
    "build",
    "dist",
    "*.egg-info",
}

# Known binary and non-text extensions to skip
IGNORE_EXTENSIONS = {
    # Images & Media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".ico", ".svg",
    ".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav",
    # Model weights & Checkpoints
    ".pt", ".pth", ".ckpt", ".bin", ".onnx", ".pkl", ".pickle", ".h5", ".npy", ".npz",
    # Archives & Executables
    ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib",
    # Databases & Fonts
    ".db", ".sqlite", ".sqlite3", ".ttf", ".otf", ".woff", ".woff2",
    # PDFs & Docs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
}

# Mapping extensions to markdown code fence languages
LANG_MAP = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".ini": "ini",
    ".cfg": "ini",
    ".env": "bash",
}


def is_binary_file(filepath: Path) -> bool:
    """Checks if a file is binary by checking for null bytes in the first 1024 bytes."""
    if filepath.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except Exception:
        return True
    return False


def should_skip_dir(dir_name: str, custom_ignore_dirs: set) -> bool:
    """Checks if directory name matches any ignore pattern."""
    import fnmatch
    for pattern in custom_ignore_dirs:
        if fnmatch.fnmatch(dir_name, pattern) or fnmatch.fnmatch(dir_name.lower(), pattern.lower()):
            return True
    return False


def collect_files(
    root_dir: Path,
    output_path: Path,
    ignore_dirs: set,
    include_ext: set = None,
    max_size_bytes: int = 500 * 1024,
) -> list:
    """Recursively collects all valid text source files from root_dir."""
    collected = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter directories in-place to prevent os.walk from descending into them
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d, ignore_dirs)]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename

            # Skip the output file itself
            try:
                if filepath.resolve() == output_path.resolve():
                    continue
            except Exception:
                pass

            # Filter by specific extensions if specified
            if include_ext and filepath.suffix.lower() not in include_ext:
                continue

            # Skip known binary extensions
            if filepath.suffix.lower() in IGNORE_EXTENSIONS:
                continue

            # Skip files larger than max_size
            try:
                if filepath.stat().st_size > max_size_bytes:
                    continue
            except OSError:
                continue

            # Check if file is binary
            if is_binary_file(filepath):
                continue

            collected.append(filepath)

    return sorted(collected, key=lambda p: str(p.relative_to(root_dir)).lower())


def build_codebase_dump(
    root_dir: Path,
    output_path: Path,
    ignore_dirs: set,
    include_ext: set = None,
    max_size_bytes: int = 500 * 1024,
):
    """Gathers all code files and writes them formatted into output_path."""
    root_dir = root_dir.resolve()
    output_path = output_path.resolve()

    print(f"Scanning directory: {root_dir}")
    files = collect_files(root_dir, output_path, ignore_dirs, include_ext, max_size_bytes)
    print(f"Found {len(files)} files to combine.")

    # Create parent folder if not existing
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", errors="replace") as out:
        # Header banner
        out.write("# COMBINED PROJECT CODEBASE\n\n")
        out.write(f"- **Project Root:** `{root_dir}`\n")
        out.write(f"- **Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"- **Total Files Combined:** {len(files)}\n\n")

        # Table of Contents / File Index
        out.write("## 📁 File Index\n\n")
        out.write("| # | File Path | Lines | Size |\n")
        out.write("|---|-----------|-------|------|\n")

        file_stats = []
        for idx, file in enumerate(files, 1):
            rel_path = file.relative_to(root_dir).as_posix()
            try:
                size_kb = file.stat().st_size / 1024
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                size_kb = 0
                line_count = 0
            file_stats.append((idx, rel_path, line_count, size_kb))
            out.write(f"| {idx} | `{rel_path}` | {line_count} | {size_kb:.1f} KB |\n")

        out.write("\n---\n\n")

        # Individual File Contents
        for idx, file in enumerate(files, 1):
            rel_path = file.relative_to(root_dir).as_posix()
            ext = file.suffix.lower()
            lang = LANG_MAP.get(ext, "")

            separator = "=" * 80
            out.write(f"{separator}\n")
            out.write(f"### File [{idx}/{len(files)}]: {rel_path}\n")
            out.write(f"{separator}\n\n")

            out.write(f"```{lang}\n")
            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")
            except Exception as e:
                out.write(f"// Error reading file: {e}\n")
            out.write("```\n\n")

    print(f" Successfully written combined codebase to:\n  {output_path}")


def main():
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Consolidate all project code files into a single structured file."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root),
        help=f"Project root directory to scan (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_root / "combined_codebase.md"),
        help="Path of the combined output file (default: combined_codebase.md in root)",
    )
    parser.add_argument(
        "--include-ext",
        nargs="*",
        default=None,
        help="Whitelist specific extensions (e.g., --include-ext .py .yaml .md)",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=[],
        help="Additional directory names/patterns to exclude",
    )
    parser.add_argument(
        "--max-size-kb",
        type=int,
        default=500,
        help="Maximum file size in KB to include (default: 500 KB)",
    )

    args = parser.parse_args()

    root_dir = Path(args.root)
    output_path = Path(args.output)

    ignore_dirs = set(DEFAULT_IGNORE_DIRS)
    if args.exclude_dirs:
        ignore_dirs.update(args.exclude_dirs)

    include_ext = None
    if args.include_ext:
        include_ext = {ext if ext.startswith(".") else f".{ext}" for ext in args.include_ext}

    build_codebase_dump(
        root_dir=root_dir,
        output_path=output_path,
        ignore_dirs=ignore_dirs,
        include_ext=include_ext,
        max_size_bytes=args.max_size_kb * 1024,
    )


if __name__ == "__main__":
    main()
