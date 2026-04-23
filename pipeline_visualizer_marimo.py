# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    #!/usr/bin/env python3
    """
    Concurrent Pipeline Visualizer
    Simulates a GPU pipeline with concurrent buffers and generates a Gantt chart
    """

    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict


    def simulate_pipeline(n_buffers, n_iterations, compute_time, h2d_time, d2h_time, rayput_time, cpu_prep_time=10):
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
            'CPU': [],
            'CPU_prep': []  # CPU scheduling/prep blocks (colored)
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

            # CPU prep before H2D
            cpu_prep_start = max(current_time, cpu_available)
            events['CPU_prep'].append((cpu_prep_start, cpu_prep_time, color_idx))
            cpu_available = cpu_prep_start + cpu_prep_time

            start_time = max(cpu_available, h2d_available)
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
                # CPU prep before H2D
                cpu_prep_start = max(cpu_available, cpu_available)
                events['CPU_prep'].append((cpu_prep_start, cpu_prep_time, color_idx))
                cpu_available = cpu_prep_start + cpu_prep_time

                # Buffer can do H2D after CPU prep completes
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
            if row_name == 'CPU_prep':
                continue  # Handle separately

            row_y = row_positions.get(row_name)
            if row_y is None:
                continue

            for start_time, duration, color_idx in event_list:
                if row_name == 'CPU':
                    color = rayput_color
                else:
                    color = buffer_colors[color_idx]

                ax.barh(row_y, duration, left=start_time, height=0.8, 
                       color=color, edgecolor='white', linewidth=1)

        # Plot CPU prep blocks (colored) on CPU row
        cpu_row_y = row_positions['CPU']
        for start_time, duration, color_idx in events['CPU_prep']:
            ax.barh(cpu_row_y, duration, left=start_time, height=0.8,
                   color=buffer_colors[color_idx], edgecolor='white', linewidth=1)

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
        if output_file is not None:
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
        return ax
    return plot_gantt, simulate_pipeline


@app.cell(disabled=True)
def _(plot_gantt, simulate_pipeline):
    def test():
        n_buffers = 2
        n_iterations= 4
        compute_time= 180
        h2d_time= 50
        d2h_time= 70
        rayput_time= 120
        cpu_prep_time = 10
        output=None

        events, buffer_colors, rayput_color = simulate_pipeline(
            n_buffers,
            n_iterations,
            compute_time,
            h2d_time,
            d2h_time,
            rayput_time,
            cpu_prep_time,
        )

        return plot_gantt(events, buffer_colors, rayput_color, n_buffers, output)

    test()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    n_buffers = mo.ui.slider(
        start=0,           # minimum value
        stop=10,          # maximum value
        step=1,            # increment
        value=2,          # default value
        label="n_buffers"
    )
    n_iterations= mo.ui.slider(
        start=1,           # minimum value
        stop=20,          # maximum value
        step=1,            # increment
        value=2,          # default value
        label="n_iterations"
    )
    compute_time= mo.ui.slider(
        start=10,           # minimum value
        stop=500,          # maximum value
        step=10,            # increment
        value=180,          # default value
        label="compute_time"
    )
    h2d_time= mo.ui.slider(
        start=10,           # minimum value
        stop=200,          # maximum value
        step=10,            # increment
        value=50,          # default value
        label="h2d_time"
    )
    d2h_time= mo.ui.slider(
        start=10,           # minimum value
        stop=200,          # maximum value
        step=10,            # increment
        value=70,          # default value
        label="d2h_time"
    )
    rayput_time= mo.ui.slider(
        start=0,           # minimum value
        stop=200,          # maximum value
        step=10,            # increment
        value=120,          # default value
        label="rayput_time"
    )
    cpu_prep_time = mo.ui.slider(
        start=1,           # minimum value
        stop=50,          # maximum value
        step=1,            # increment
        value=10,          # default value
        label="cpu_prep_time"
    )
    output=None

    n_buffers,n_iterations, compute_time, h2d_time, d2h_time, rayput_time, cpu_prep_time
    return (
        compute_time,
        cpu_prep_time,
        d2h_time,
        h2d_time,
        n_buffers,
        n_iterations,
        output,
        rayput_time,
    )


@app.cell(hide_code=True)
def _(
    compute_time,
    cpu_prep_time,
    d2h_time,
    h2d_time,
    n_buffers,
    n_iterations,
    output,
    plot_gantt,
    rayput_time,
    simulate_pipeline,
):
    def test2():

        events, buffer_colors, rayput_color = simulate_pipeline(
            n_buffers.value,
            n_iterations.value,
            compute_time.value,
            h2d_time.value,
            d2h_time.value,
            rayput_time.value,
            cpu_prep_time.value,
        )

        return plot_gantt(events, buffer_colors, rayput_color, n_buffers.value, output)

    test2()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
