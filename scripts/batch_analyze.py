#!/usr/bin/env python
"""
Batch analyzer for CodeVoyant - analyzes multiple repositories in codevoyant_output.
Processes folders matching pattern: <name>-<branch> where branch is main, master, dev, or develop.
Outputs to folders named <name> and removes original folders.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from src.graph import analyze_repository


def parse_folder_name(folder_name: str) -> tuple[str, str] | None:
    """
    Parse folder name to extract base name and branch.

    Args:
        folder_name: Name of the folder to parse

    Returns:
        Tuple of (base_name, branch) if matches pattern, None otherwise
    """
    # Match folders ending with -main, -master, -dev, or -develop
    pattern = r"^(.+)-(main|master|dev|develop)$"
    match = re.match(pattern, folder_name)

    if match:
        return match.group(1), match.group(2)
    return None


def analyze_batch(
    base_dir: Path,
    formats: list[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, dict]:
    """
    Analyze all matching repositories in the base directory.

    Args:
        base_dir: Base directory containing repository folders
        formats: Output formats for graphs
        dry_run: If True, only print what would be done without executing
        verbose: If True, print detailed progress information

    Returns:
        Dictionary mapping folder names to analysis results
    """
    if not base_dir.exists():
        raise ValueError(f"Directory does not exist: {base_dir}")

    results: dict[str, dict] = {}

    # Find all matching folders
    matching_folders = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        parsed = parse_folder_name(folder.name)
        if parsed:
            matching_folders.append((folder, parsed[0], parsed[1]))

    if not matching_folders:
        print(f"No matching folders found in {base_dir}")
        return results

    print(f"Found {len(matching_folders)} repositories to analyze:")
    for folder, base_name, branch in matching_folders:
        print(f"  - {folder.name} => {base_name} (branch: {branch})")

    if dry_run:
        print("\nDry run mode - no changes will be made")
        return results

    print("\n" + "=" * 70)

    # Process each folder
    for idx, (folder, base_name, branch) in enumerate(matching_folders, 1):
        print(f"\n[{idx}/{len(matching_folders)}] Processing: {folder.name}")
        print("-" * 70)

        try:
            # Define output directory
            output_dir = base_dir / base_name

            if verbose:
                print(f"  Repository: {folder}")
                print(f"  Output: {output_dir}")
                print(f"  Branch: {branch}")
                print(f"  Formats: {', '.join(formats)}")

            # Run analysis
            output_files = analyze_repository(
                repo_path=str(folder),
                output_dir=str(output_dir),
                formats=formats,
            )

            print("[OK]Analysis complete")
            if verbose:
                for name, path in output_files.items():
                    print(f"    - {name}: {path}")

            # Remove original folder
            if verbose:
                print(f"  Removing original folder: {folder}")
            shutil.rmtree(folder)
            print(f"[OK]Cleaned up {folder.name}")

            results[folder.name] = {
                "status": "success",
                "output_dir": str(output_dir),
                "files": output_files,
            }

        except Exception as e:
            print(f"[ERROR] processing {folder.name}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

            results[folder.name] = {
                "status": "error",
                "error": str(e),
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze repositories in codevoyant_output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all matching folders with default settings
  python batch_analyze.py
  
  # Dry run to see what would be processed
  python batch_analyze.py --dry-run
  
  # Specify custom base directory and formats
  python batch_analyze.py --base-dir ./repos --formats json graphml gexf
  
  # Verbose output
  python batch_analyze.py --verbose
        """,
    )

    parser.add_argument(
        "--base-dir",
        default="./codevoyant_output",
        help="Base directory containing repository folders (default: ./codevoyant_output)",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["json", "graphml"],
        choices=["json", "cytoscape", "d3", "graphml", "gexf", "pickle"],
        help="Output formats (default: json graphml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    try:
        base_dir = Path(args.base_dir).resolve()

        print("CodeVoyant - Batch Analyzer")
        print("=" * 70)
        print(f"Base directory: {base_dir}")
        print(f"Output formats: {', '.join(args.formats)}")
        if args.dry_run:
            print("Mode: DRY RUN")
        print("=" * 70)

        results = analyze_batch(
            base_dir=base_dir,
            formats=args.formats,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if not results:
            print("No repositories were processed.")
            return 0

        success_count = sum(1 for r in results.values() if r["status"] == "success")
        error_count = len(results) - success_count

        print(f"Total: {len(results)}")
        print(f"Success: {success_count}")
        print(f"Errors: {error_count}")

        if error_count > 0:
            print("\nFailed repositories:")
            for name, result in results.items():
                if result["status"] == "error":
                    print(f"  • {name}: {result['error']}")

        return 0 if error_count == 0 else 1

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
