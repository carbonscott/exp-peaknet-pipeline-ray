#!/usr/bin/env python3
"""
Concurrent Pipeline Visualizer
Simulates a GPU pipeline with concurrent buffers and generates a Gantt chart
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


def simulate_pipeline(n_buffers, n_iterations, compute_time, h2d_time, d2h_time, rayput_time, cpu_prep_time=0):
    """
    Simulate a concurrent pipeline and return event timelines

    Pipeline stages per buffer:
    1. [CPU prep] -> 2. H2D -> 3. Compute -> 4. D2H -> 5. ray.put -> repeat

    Constraints:
    - Only one Compute at a time (single GPU)
    - H2D and Compute can overlap (different resources)
    - D2H and Compute can overlap (different resources)
    - ray.put blocks CPU (but not GPU)
    """

    # Color scheme matching the PDFs
    buffer_colors = ['#3333FF', '#FF1493', '#00CC00', '#FF9900', '#9933FF']  # Blue, Pink, Green, Orange, Purple
    rayput_color = '#CCCCCC'  # Gray

    events = {
        'H2D': [],
        'Compute': [],
        'D2H': [],
        'CPU': []
    }

    # Track when each resource becomes available
    compute_available = 0
    h2d_available = 0
    d2h_available = 0
    cpu_available = 0

    # Track when each buffer completes each stage
    buffer_states = defaultdict(lambda: {'stage': 'init', 'ready_time': 0})

    current_time = 0

    # Phase 1: Initial H2D for all buffers (can happen in parallel or series)
    for i in range(n_buffers):
        color_idx = i % len(buffer_colors)
        start_time = max(current_time, h2d_available)
        events['H2D'].append((start_time, h2d_time, color_idx))
        h2d_available = start_time + h2d_time
        buffer_states[i]['ready_time'] = h2d_available
        buffer_states[i]['stage'] = 'ready_for_compute'

    # Phase 2: Pipeline execution
    total_batches = n_buffers * n_iterations
    batches_completed = 0

    while batches_completed < total_batches:
        # Find next buffer ready for compute
        buffer_idx = batches_completed % n_buffers
        color_idx = buffer_idx % len(buffer_colors)

        # Compute (GPU resource)
        compute_start = max(compute_available, buffer_states[buffer_idx]['ready_time'])
        events['Compute'].append((compute_start, compute_time, color_idx))
        compute_available = compute_start + compute_time

        # D2H (can overlap with next compute)
        d2h_start = compute_available
        events['D2H'].append((d2h_start, d2h_time, color_idx))
        d2h_available = d2h_start + d2h_time

        # CPU ray.put (blocks CPU)
        cpu_start = max(d2h_available, cpu_available)
        events['CPU'].append((cpu_start, rayput_time, color_idx))
        cpu_available = cpu_start + rayput_time

        batches_completed += 1

        # Schedule next H2D for this buffer (if more iterations remain)
        if batches_completed < total_batches:
            # Buffer can do H2D after ray.put completes
            h2d_start = max(cpu_available, h2d_available)
            events['H2D'].append((h2d_start, h2d_time, color_idx))
            h2d_available = h2d_start + h2d_time
            buffer_states[buffer_idx]['ready_time'] = h2d_available

    return events, buffer_colors, rayput_color


def plot_gantt(events, buffer_colors, rayput_color, n_buffers, output_file):
    """
    Create a Gantt chart from the events
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    row_names = ['CPU', 'D2H', 'Compute', 'H2D']
    row_positions = {name: i for i, name in enumerate(row_names)}

    # Plot each event
    for row_name, event_list in events.items():
        row_y = row_positions[row_name]
        for start_time, duration, color_idx in event_list:
            if row_name == 'CPU':
                color = rayput_color
            else:
                color = buffer_colors[color_idx]

            ax.barh(row_y, duration, left=start_time, height=0.8, 
                   color=color, edgecolor='white', linewidth=1)

    # Formatting
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_xlim(left=0)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_elements = []
    buffer_labels = ['A', 'B', 'C', 'D', 'E']
    for i in range(min(n_buffers, len(buffer_colors))):
        legend_elements.append(mpatches.Patch(color=buffer_colors[i], label=f'Buffer {buffer_labels[i]}'))
    legend_elements.append(mpatches.Patch(color=rayput_color, label='ray.put'))
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), 
             ncol=min(n_buffers + 1, 6), frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gantt chart saved to {output_file}")

    # Calculate and print statistics
    total_time = max(start + dur for events_list in events.values() 
                    for start, dur, _ in events_list)
    compute_time_total = sum(dur for start, dur, _ in events['Compute'])
    gpu_utilization = compute_time_total / total_time * 100

    print(f"\nPipeline Statistics:")
    print(f"  Total time: {total_time:.1f}")
    print(f"  Compute time: {compute_time_total:.1f}")
    print(f"  GPU utilization: {gpu_utilization:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate concurrent pipeline Gantt chart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--n-buffers', type=int, default=3, 
                       help='Number of concurrent buffers (concurrency level)')
    parser.add_argument('--n-iterations', type=int, default=2, 
                       help='Number of iterations through all buffers')
    parser.add_argument('--compute-time', type=float, default=180, 
                       help='GPU compute time per batch')
    parser.add_argument('--h2d-time', type=float, default=30, 
                       help='Host to Device (H2D) transfer time')
    parser.add_argument('--d2h-time', type=float, default=30, 
                       help='Device to Host (D2H) transfer time')
    parser.add_argument('--rayput-time', type=float, default=20, 
                       help='ray.put time (CPU blocking operation)')
    parser.add_argument('--output', type=str, default='pipeline_gantt.png', 
                       help='Output PNG file name')

    args = parser.parse_args()

    print(f"Simulating pipeline with:")
    print(f"  Buffers: {args.n_buffers}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Compute time: {args.compute_time}")
    print(f"  H2D time: {args.h2d_time}")
    print(f"  D2H time: {args.d2h_time}")
    print(f"  ray.put time: {args.rayput_time}")

    events, buffer_colors, rayput_color = simulate_pipeline(
        args.n_buffers,
        args.n_iterations,
        args.compute_time,
        args.h2d_time,
        args.d2h_time,
        args.rayput_time
    )

    plot_gantt(events, buffer_colors, rayput_color, args.n_buffers, args.output)


if __name__ == '__main__':
    main()
