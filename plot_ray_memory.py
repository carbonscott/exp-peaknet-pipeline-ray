#!/usr/bin/env python3
"""
Plot number of objects over time from Ray memory monitoring CSV files.
"""
import argparse
import pandas as pd
import plotext as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize number of objects over time from CSV"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to CSV file (e.g., 6p_compiled_yes.csv)"
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Export plot to PDF (filename inferred from input CSV)"
    )

    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv_file)

    # Calculate relative time from start
    time_relative = df['timestamp'] - df['timestamp'].iloc[0]

    # Terminal plot (always shown)
    plt.plot(time_relative.tolist(), df['num_objects'].tolist())
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Objects')
    plt.title(f'Ray Object Store: {Path(args.csv_file).stem}')
    plt.show()

    # Export to PDF if requested
    if args.pdf:
        import matplotlib.pyplot as mplt

        mplt.figure(figsize=(10, 6))
        mplt.plot(time_relative, df['num_objects'], linewidth=1.5)
        mplt.xlabel('Time (seconds)', fontsize=12)
        mplt.ylabel('Number of Objects', fontsize=12)
        mplt.title(f'Ray Object Store: {Path(args.csv_file).stem}', fontsize=14)
        mplt.grid(True, alpha=0.3)
        mplt.tight_layout()

        pdf_path = Path(args.csv_file).with_suffix('.pdf')
        mplt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
