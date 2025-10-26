#!/usr/bin/env python
"""
Command-line interface for CodeVoyant graph analysis.
"""

import argparse
import sys
from pathlib import Path

from src.graph import CodeAnalyzer, analyze_repository


def main():
    parser = argparse.ArgumentParser(
        description="CodeVoyant - Extract and visualize code graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current directory
  python analyze.py
  
  # Analyze specific repository
  python analyze.py --repo /path/to/repo
  
  # Specify output directory and formats
  python analyze.py --repo ./myproject --output ./graphs --formats json cytoscape d3
  
  # Extract only dependency graph
  python analyze.py --dependencies-only
  
  # Extract only call graph
  python analyze.py --calls-only --entry-points main.py app.py
        """,
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the repository to analyze (default: current directory)",
    )

    parser.add_argument(
        "--output",
        help="Output directory for graphs (default: repo_path/codevoyant_output)",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["json", "graphml"],
        choices=["json", "cytoscape", "d3", "graphml", "gexf", "pickle"],
        help="Output formats (default: json graphml)",
    )

    parser.add_argument(
        "--dependencies-only", action="store_true", help="Extract only dependency graph"
    )

    parser.add_argument(
        "--calls-only", action="store_true", help="Extract only call graph"
    )

    parser.add_argument(
        "--entry-points", nargs="+", help="Entry point files for call graph analysis"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate arguments
    if args.dependencies_only and args.calls_only:
        print("Error: Cannot specify both --dependencies-only and --calls-only")
        return 1

    # Determine what to extract
    extract_dependencies = not args.calls_only
    extract_calls = not args.dependencies_only

    # Run analysis
    try:
        print("CodeVoyant - Code Graph Analyzer")
        print("=" * 70)
        print(f"Repository: {args.repo}")
        print(f"Output: {args.output or Path(args.repo) / 'codevoyant_output'}")
        print(f"Formats: {', '.join(args.formats)}")
        print("=" * 70)

        # Use simple API if possible
        if not args.verbose:
            output_files = analyze_repository(
                repo_path=args.repo,
                output_dir=args.output,
                formats=args.formats,
                entry_points=args.entry_points,
            )

            print("\n" + "=" * 70)
            print("Analysis Complete!")
            print("=" * 70)
            print("\nGenerated files:")
            for name, path in output_files.items():
                print(f"  • {name}: {path}")
        else:
            # Use detailed API for verbose output
            analyzer = CodeAnalyzer(args.repo, args.output)

            print("\nExtracting graphs...")
            graphs = analyzer.analyze(
                extract_dependencies=extract_dependencies,
                extract_calls=extract_calls,
                entry_points=args.entry_points,
            )

            print("\nSaving graphs...")
            output_files = analyzer.save_graphs(formats=args.formats)

            print("\n" + "=" * 70)
            print("Analysis Complete!")
            print("=" * 70)
            print("\nGenerated files:")
            for name, path in output_files.items():
                print(f"  • {name}: {path}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
