import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    #!/usr/bin/env python3
    """
    Visualize CXI output with debug datasets (detector, segmap, logit maps)

    This script creates a 1x4 visualization showing:
    - Detector image with peaks
    - Segmentation map with peaks
    - Logit map class 0 (background) with peaks
    - Logit map class 1 (peaks) with peaks

    Usage:
        python viz__cxi_writer_from_dump.py --cxi test_cxi_output_debug/test_cxi_*.cxi --idx 0
        python viz__cxi_writer_from_dump.py --cxi test_cxi_output_debug/test_cxi_*.cxi --idx 16 --output viz_image16.png
    """

    import argparse
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from pathlib import Path


    def visualize_cxi_event(cxi_path, idx, show_peaks=True, box_size=20, output_path=None, show=True):
        """
        Visualize a single event from CXI file with all debug datasets.

        Args:
            cxi_path: Path to CXI file
            idx: Event index to visualize
            box_size: Size of peak boxes in pixels
            output_path: Optional path to save figure
            show: Whether to display figure interactively
        """
        print(f"\n=== Visualizing CXI Event {idx} ===")
        print(f"File: {cxi_path}")

        with h5py.File(cxi_path, 'r') as f:
            # Check total number of events
            total_events = len(f['entry_1/data_1/data'])
            print(f"Total events in file: {total_events}")

            if idx >= total_events:
                raise ValueError(f"Index {idx} out of range (0-{total_events-1})")

            # Load datasets
            detector_image = f['entry_1/data_1/data'][idx]

            # Check if debug datasets exist
            has_segmap = '/entry_1/result_1/segmentation_map' in f
            has_logit0 = '/entry_1/result_1/logit_map_class0' in f
            has_logit1 = '/entry_1/result_1/logit_map_class1' in f

            if has_segmap:
                segmentation_map = f['entry_1/result_1/segmentation_map'][idx]
            else:
                print("WARNING: No segmentation_map found in CXI file")
                segmentation_map = None

            if has_logit0:
                logit_class0 = f['entry_1/result_1/logit_map_class0'][idx]
            else:
                print("WARNING: No logit_map_class0 found in CXI file")
                logit_class0 = None

            if has_logit1:
                logit_class1 = f['entry_1/result_1/logit_map_class1'][idx]
            else:
                print("WARNING: No logit_map_class1 found in CXI file")
                logit_class1 = None

            # Load peak positions
            peak_x = f['entry_1/result_1/peakXPosRaw'][idx]
            peak_y = f['entry_1/result_1/peakYPosRaw'][idx]
            n_peaks_raw = f['entry_1/result_1/nPeaks'][idx]

            # Filter valid peaks
            valid = (peak_x >= 0) & (peak_y >= 0)
            peak_x = peak_x[valid]
            peak_y = peak_y[valid]

            # print(f"\nDataset shapes:")
            # print(f"  Detector image: {detector_image.shape}")
            # if segmentation_map is not None:
            #     print(f"  Segmentation map: {segmentation_map.shape}")
            # if logit_class0 is not None:
            #     print(f"  Logit class 0: {logit_class0.shape}")
            # if logit_class1 is not None:
            #     print(f"  Logit class 1: {logit_class1.shape}")

            print(f"\nPeak statistics:")
            print(f"  Number of peaks (from nPeaks): {n_peaks_raw}")
            print(f"  Number of valid peaks: {len(peak_x)}")

            print(f"\nDetector image statistics:")
            print(f"  Mean: {detector_image.mean():.2f}")
            print(f"  Std: {detector_image.std():.2f}")
            print(f"  Min: {detector_image.min():.2f}")
            print(f"  Max: {detector_image.max():.2f}")

            # if segmentation_map is not None:
            #     print(f"\nSegmentation map statistics:")
            #     print(f"  Unique values: {np.unique(segmentation_map)}")
            #     print(f"  Peak pixels (class 1): {(segmentation_map == 1).sum()}")
            #     print(f"  Background pixels (class 0): {(segmentation_map == 0).sum()}")

            # if logit_class0 is not None:
            #     print(f"\nLogit class 0 (background) statistics:")
            #     print(f"  Mean: {logit_class0.mean():.4f}")
            #     print(f"  Std: {logit_class0.std():.4f}")
            #     print(f"  Min: {logit_class0.min():.4f}")
            #     print(f"  Max: {logit_class0.max():.4f}")

            # if logit_class1 is not None:
            #     print(f"\nLogit class 1 (peaks) statistics:")
            #     print(f"  Mean: {logit_class1.mean():.4f}")
            #     print(f"  Std: {logit_class1.std():.4f}")
            #     print(f"  Min: {logit_class1.min():.4f}")
            #     print(f"  Max: {logit_class1.max():.4f}")

        # Create figure with 1x4 layout (optimized for tall/narrow images)
        n_cols = 1 + sum([segmentation_map is not None, logit_class0 is not None, logit_class1 is not None])
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 80))
        # fig, axes = plt.subplots(1, n_cols, figsize=(40, 60))

        if n_cols == 1:
            axes = [axes]

        col_idx = 0

        # Panel 1: Detector image
        ax = axes[col_idx]
        vmin = detector_image.mean()
        vmax = detector_image.mean() + 4 * detector_image.std()
        im = ax.imshow(detector_image, vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
        ax.set_title(f'Detector Image\n({len(peak_x)} peaks)', fontsize=12)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        # plt.colorbar(im, ax=ax, label='Intensity', fraction=0.046, pad=0.04)

        if show_peaks:
            # Draw peak boxes
            for x, y in zip(peak_x, peak_y):
                rect = Rectangle((x - box_size/2, y - box_size/2), box_size, box_size,
                                linewidth=1.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            col_idx += 1

        # Panel 2: Segmentation map (if available)
        if segmentation_map is not None:
            ax = axes[col_idx]
            im = ax.imshow(segmentation_map, vmin=0, vmax=0.5, cmap='viridis', origin='lower')
            ax.set_title('Segmentation Map\n(0=background, 1=peak)', fontsize=12)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            # plt.colorbar(im, ax=ax, label='Class', fraction=0.046, pad=0.04, ticks=[0, 1])

            if show_peaks:
                # Draw peak boxes
                for x, y in zip(peak_x, peak_y):
                    rect = Rectangle((x - box_size/2, y - box_size/2), box_size, box_size,
                                    linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                col_idx += 1

        # Panel 3: Logit class 0 (background) (if available)
        if logit_class0 is not None:
            ax = axes[col_idx]
            # Use symmetric colormap centered at 0
            vmin = logit_class0.mean()
            vmax = logit_class0.mean() + 4 * logit_class0.std()
            im = ax.imshow(logit_class0, vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
            ax.set_title('Logit Map Class 0\n(Background logits)', fontsize=12)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            # plt.colorbar(im, ax=ax, label='Logit value', fraction=0.046, pad=0.04)

            if show_peaks:
                # Draw peak boxes
                for x, y in zip(peak_x, peak_y):
                    rect = Rectangle((x - box_size/2, y - box_size/2), box_size, box_size,
                                    linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                col_idx += 1

        # Panel 4: Logit class 1 (peaks) (if available)
        if logit_class1 is not None:
            ax = axes[col_idx]
            # Use symmetric colormap centered at 0
            vmin = logit_class1.mean()
            vmax = logit_class1.mean() + 4 * logit_class1.std()
            im = ax.imshow(logit_class1, vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
            ax.set_title('Logit Map Class 1\n(Peak logits)', fontsize=12)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            # plt.colorbar(im, ax=ax, label='Logit value', fraction=0.046, pad=0.04)

            if show_peaks:
                # Draw peak boxes
                for x, y in zip(peak_x, peak_y):
                    rect = Rectangle((x - box_size/2, y - box_size/2), box_size, box_size,
                                    linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

        # plt.suptitle(f'CXI Event {idx} - All Debug Datasets', fontsize=16, y=0.995)
        # plt.tight_layout()
        return ax
    return (visualize_cxi_event,)


@app.cell
def _(mo):
    cxi_path = "peaknet_673m_results/peaknet_cxi_20251030_160700_528651_chunk0004.cxi"

    idx = mo.ui.slider(
        start=0,           # minimum value
        stop=30,          # maximum value
        step=1,            # increment
        value=0,          # default value
        label="idx"
    )
    box_size=20
    show_peaks=True
    output_path=None
    show=True
    idx
    return box_size, cxi_path, idx, output_path, show, show_peaks


@app.cell
def _(
    box_size,
    cxi_path,
    idx,
    output_path,
    show,
    show_peaks,
    visualize_cxi_event,
):
    visualize_cxi_event(cxi_path, idx.value, show_peaks, box_size, output_path, show)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
