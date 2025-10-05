#!/usr/bin/env python
"""
Monitor Ray memory usage over time with optional plotting and CSV export.

Usage:
    python monitor_ray_memory.py --interval 1.0 --plot --output memory_log.csv
"""

import argparse
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any


class MemoryMonitor:
    def __init__(self, interval: float = 1.0, output_file: str = None, write_interval: int = 10):
        self.interval = interval
        self.output_file = output_file
        self.write_interval = write_interval
        self.data: List[Dict[str, Any]] = []
        self.running = True
        self.csv_file = None
        self.csv_writer = None
        self.write_buffer: List[Dict[str, Any]] = []

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)

        # Initialize CSV file if output is specified
        if self.output_file:
            self._init_csv_file()

    def _init_csv_file(self):
        """Initialize CSV file and write header"""
        import csv
        try:
            self.csv_file = open(self.output_file, 'w', newline='')
            # Define expected fieldnames
            fieldnames = ['timestamp', 'datetime', 'num_objects', 'memory_mib',
                         'spilled_objects', 'spilled_mib', 'restored_objects', 'restored_mib', 'error']
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames, extrasaction='ignore')
            self.csv_writer.writeheader()
            self.csv_file.flush()
            print(f"CSV output initialized: {self.output_file}")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            self.csv_file = None
            self.csv_writer = None

    def _flush_csv_buffer(self):
        """Write buffered samples to CSV and flush to disk"""
        if self.csv_writer and self.write_buffer:
            try:
                self.csv_writer.writerows(self.write_buffer)
                self.csv_file.flush()
                self.write_buffer.clear()
            except Exception as e:
                print(f"\nError writing to CSV: {e}")

    def _close_csv_file(self):
        """Flush any remaining data and close CSV file"""
        if self.csv_file:
            self._flush_csv_buffer()
            self.csv_file.close()
            print(f"\nCSV file closed: {self.output_file}")

    def _signal_handler(self, signum, frame):
        """Handle signals gracefully and flush data"""
        signal_names = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM', signal.SIGHUP: 'SIGHUP'}
        signal_name = signal_names.get(signum, f'Signal {signum}')
        print(f"\n\n[Received {signal_name}, stopping monitoring...]")
        self.running = False

    def collect_sample(self) -> Dict[str, Any]:
        """Collect a single memory sample using ray memory --stats-only"""
        timestamp = time.time()

        sample = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
        }

        try:
            # Run ray memory --stats-only command
            result = subprocess.run(
                ['ray', 'memory', '--stats-only'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                sample['error'] = f"Command failed: {result.stderr}"
                return sample

            output = result.stdout

            # Parse output for Plasma memory stats
            # Example: "Plasma memory usage 0 MiB, 0 objects, 0.0% full, 0.0% needed"
            plasma_match = re.search(r'Plasma memory usage (\d+) MiB, (\d+) objects', output)
            if plasma_match:
                sample['memory_mib'] = int(plasma_match.group(1))
                sample['num_objects'] = int(plasma_match.group(2))

            # Parse spilled stats
            # Example: "Spilled 469674 MiB, 33398 objects, avg write throughput 8008 MiB/s"
            spilled_match = re.search(r'Spilled (\d+) MiB, (\d+) objects', output)
            if spilled_match:
                sample['spilled_mib'] = int(spilled_match.group(1))
                sample['spilled_objects'] = int(spilled_match.group(2))

            # Parse restored stats
            # Example: "Restored 14892 MiB, 1059 objects, avg read throughput 6714 MiB/s"
            restored_match = re.search(r'Restored (\d+) MiB, (\d+) objects', output)
            if restored_match:
                sample['restored_mib'] = int(restored_match.group(1))
                sample['restored_objects'] = int(restored_match.group(2))

        except subprocess.TimeoutExpired:
            sample['error'] = "Command timeout"
        except Exception as e:
            sample['error'] = str(e)

        return sample

    def run(self):
        """Main monitoring loop"""
        print(f"Starting Ray memory monitoring (sampling every {self.interval}s)")
        print("Press Control-C to stop and display results\n")

        sample_count = 0
        while self.running:
            sample = self.collect_sample()
            self.data.append(sample)
            sample_count += 1

            # Add to CSV buffer if incremental writing is enabled
            if self.csv_writer:
                self.write_buffer.append(sample)
                # Flush buffer every write_interval samples
                if len(self.write_buffer) >= self.write_interval:
                    self._flush_csv_buffer()

            # Display live update
            if 'error' not in sample:
                print(f"[{sample_count:04d}] {sample['datetime']} | "
                      f"Objects: {sample.get('num_objects', 0):6d} | "
                      f"Memory: {sample.get('memory_mib', 0):6d} MiB",
                      end='\r')
            else:
                print(f"[{sample_count:04d}] Error: {sample['error'][:60]}", end='\r')

            time.sleep(self.interval)

        print()  # New line after loop

        # Close CSV file if it was opened
        self._close_csv_file()

    def print_summary(self):
        """Print summary statistics"""
        if not self.data:
            print("No data collected.")
            return

        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)

        # Filter out samples with errors
        valid_samples = [s for s in self.data if 'error' not in s]

        if not valid_samples:
            print("No valid samples collected.")
            return

        print(f"Total samples: {len(self.data)}")
        print(f"Valid samples: {len(valid_samples)}")
        print(f"Duration: {self.data[-1]['timestamp'] - self.data[0]['timestamp']:.2f} seconds")

        # Calculate statistics
        memory_mib = [s.get('memory_mib', 0) for s in valid_samples]
        objects = [s.get('num_objects', 0) for s in valid_samples]

        if memory_mib:
            print(f"\nMemory Usage (MiB):")
            print(f"  Min:  {min(memory_mib):10d}")
            print(f"  Max:  {max(memory_mib):10d}")
            print(f"  Avg:  {sum(memory_mib)//len(memory_mib):10d}")

        if objects:
            print(f"\nObject Count:")
            print(f"  Min:  {min(objects):10d}")
            print(f"  Max:  {max(objects):10d}")
            print(f"  Avg:  {sum(objects)//len(objects):10d}")

    def plot(self):
        """Generate terminal-based plot"""
        try:
            import plotext as plt
        except ImportError:
            print("\nWarning: plotext not installed. Install with 'pip install plotext' to enable plotting.")
            return

        if not self.data:
            print("No data to plot.")
            return

        valid_samples = [s for s in self.data if 'error' not in s and 'memory_mib' in s]

        if not valid_samples:
            print("No valid samples to plot.")
            return

        # Extract data
        times = [(s['timestamp'] - self.data[0]['timestamp']) for s in valid_samples]
        memory_mib = [s['memory_mib'] for s in valid_samples]

        # Plot
        print("\n" + "="*70)
        print("OBJECT STORE MEMORY USAGE OVER TIME")
        print("="*70)

        plt.clear_figure()
        plt.plot(times, memory_mib, marker="braille")
        plt.title("Object Store Memory Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory (MiB)")
        plt.theme("clear")
        plt.show()

    def save_csv(self, filename: str):
        """Save data to CSV file (only if not already written incrementally)"""
        # If CSV was already written incrementally, skip
        if self.output_file == filename:
            print(f"\nData already saved incrementally to: {filename}")
            return

        if not self.data:
            print(f"No data to save to {filename}")
            return

        import csv

        # Get all possible keys
        keys = set()
        for sample in self.data:
            keys.update(sample.keys())
        keys = sorted(keys)

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.data)

            print(f"\nData saved to: {filename}")
        except Exception as e:
            print(f"\nError saving CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Ray object store memory usage over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with 1 second interval
  python monitor_ray_memory.py --interval 1.0

  # Monitor with plotting enabled
  python monitor_ray_memory.py --interval 0.5 --plot

  # Monitor and save to CSV
  python monitor_ray_memory.py --interval 2.0 --output memory_log.csv --plot
        """
    )

    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Display terminal-based plot after collection'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Save data to CSV file (e.g., memory_log.csv)'
    )

    parser.add_argument(
        '--write-interval',
        type=int,
        default=10,
        help='Write to CSV every N samples (default: 10). Lower values = more I/O but safer against data loss.'
    )

    args = parser.parse_args()

    # Create and run monitor
    monitor = MemoryMonitor(
        interval=args.interval,
        output_file=args.output,
        write_interval=args.write_interval
    )
    monitor.run()

    # Process results after Control-C
    monitor.print_summary()

    if args.plot:
        monitor.plot()

    if args.output:
        monitor.save_csv(args.output)


if __name__ == '__main__':
    main()
