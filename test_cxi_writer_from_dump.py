#!/usr/bin/env python3
"""
Test CXI Writer Using Dumped Q2 Data

This script processes dumped Q2 data from HDF5 through the CXI pipeline
logic, allowing rapid development and debugging without running the full
ML pipeline.

Usage:
    python test_cxi_writer_from_dump.py --input q2_dump_10batches.h5 --output-dir test_cxi_output
"""

import argparse
import logging
import sys
import h5py
import numpy as np
from pathlib import Path
from scipy import ndimage

# Add cxi-pipeline-ray to path
sys.path.insert(0, '/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray')
from cxi_pipeline_ray.core.file_writer import CXIFileWriterActor
from cxi_pipeline_ray.core.coordinator import group_panels_into_events


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def find_peaks_numpy(logits_2d: np.ndarray, min_prob: float = 0.5) -> np.ndarray:
    """
    Find peaks from 2-class logits using scipy.ndimage.label.

    Args:
        logits_2d: (2, H, W) logits from model
        min_prob: Minimum probability threshold

    Returns:
        peaks: (N, 3) array of [panel_idx=0, y, x]
    """
    # Convert logits to probability: softmax then argmax
    # logits shape: (2, H, W)
    probs = np.exp(logits_2d) / np.exp(logits_2d).sum(axis=0, keepdims=True)
    seg_map = np.argmax(probs, axis=0)  # (H, W) with values 0 or 1

    # Get peak probability map (class 1)
    peak_prob = probs[1]  # (H, W)

    # Threshold
    peak_mask = (seg_map == 1) & (peak_prob >= min_prob)

    # Find connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=np.float32)
    labeled, num_features = ndimage.label(peak_mask, structure=structure)

    if num_features == 0:
        return np.array([]).reshape(0, 3)

    # Find center of mass for each component
    peaks = []
    for label_id in range(1, num_features + 1):
        component_mask = labeled == label_id
        y_coords, x_coords = np.where(component_mask)

        # Center of mass
        y_center = y_coords.mean()
        x_center = x_coords.mean()

        peaks.append([0, y_center, x_center])  # panel_idx=0 for single panel

    return np.array(peaks, dtype=np.float32)


def reconstruct_detector_image(original_image: np.ndarray, original_shape: tuple, preprocessed_shape: tuple) -> np.ndarray:
    """
    Reconstruct detector image from preprocessed format.

    Args:
        original_image: (B*C, 1, H, W) preprocessed image
        original_shape: (B, C, H_orig, W_orig)
        preprocessed_shape: (B*C, 1, H, W)

    Returns:
        detector_image: (B, C, H_orig, W_orig) in original coordinates
    """
    B, C, H_orig, W_orig = original_shape
    BC, _, H, W = preprocessed_shape

    # Validate shapes
    if original_image.shape != (BC, 1, H, W):
        raise ValueError(f"Image shape {original_image.shape} doesn't match preprocessed_shape {preprocessed_shape}")

    # Reshape: (B*C, 1, H, W) → (B, C, H, W)
    image_reshaped = original_image.reshape(B, C, H, W)

    # Unpad: (B, C, H, W) → (B, C, H_orig, W_orig)
    # Bottom-right padding means original data is at [0:H_orig, 0:W_orig]
    image_original = image_reshaped[:, :, :H_orig, :W_orig]

    return image_original


def wavelength_to_energy(wavelength: float) -> float:
    """Convert wavelength (Å) to energy (eV)."""
    HC_EV_ANGSTROM = 12398.4193
    return HC_EV_ANGSTROM / wavelength if wavelength != 0 else 0.0


def process_batch(batch_group, batch_idx: int, file_writer, verbose: bool = False):
    """
    Process a single batch from HDF5 dump through CXI pipeline.

    Args:
        batch_group: HDF5 group containing batch data
        batch_idx: Batch index for logging
        file_writer: CXIFileWriterActor instance (local mode)
        verbose: Enable verbose logging
    """
    logging.info(f"\n=== Processing Batch {batch_idx} ===")

    # Load data from HDF5
    logits = batch_group['logits'][:]  # (B*C, num_classes, H, W)
    original_image = batch_group['original_image'][:]  # (B*C, 1, H, W)

    # Load preprocessing metadata
    original_shape = tuple(batch_group['preprocessing_metadata/original_shape'][:])
    preprocessed_shape = tuple(batch_group['preprocessing_metadata/preprocessed_shape'][:])
    B, C, H_orig, W_orig = original_shape

    logging.info(f"Data shapes:")
    logging.info(f"  Logits: {logits.shape}")
    logging.info(f"  Original image: {original_image.shape}")
    logging.info(f"  Original shape (B, C, H, W): {original_shape}")
    logging.info(f"  Preprocessed shape: {preprocessed_shape}")

    # Load metadata (event-level, length B)
    metadata_group = batch_group['metadata']
    photon_wavelength = metadata_group['photon_wavelength'][:]  # (B,)
    timestamp = metadata_group['timestamp'][:]  # (B,)

    logging.info(f"Metadata:")
    logging.info(f"  Photon wavelength: {photon_wavelength} (length {len(photon_wavelength)})")
    logging.info(f"  Timestamp: {timestamp} (length {len(timestamp)})")

    # Step 1: Reconstruct detector images
    logging.info("\nStep 1: Reconstructing detector images...")
    detector_images_4d = reconstruct_detector_image(original_image, original_shape, preprocessed_shape)
    logging.info(f"  Reconstructed shape: {detector_images_4d.shape} (B, C, H_orig, W_orig)")

    # Step 2: Run peak finding on logits and compute seg_maps
    logging.info("\nStep 2: Running peak finding and computing seg_maps...")
    all_peaks = []
    all_seg_maps = []  # Store seg maps for each panel
    all_logit_maps = []  # Store logit maps for each panel
    _, _, H_padded, W_padded = preprocessed_shape  # 512, 512

    for panel_idx in range(logits.shape[0]):  # Iterate over B*C panels
        panel_logits = logits[panel_idx]  # (2, H, W)
        peaks = find_peaks_numpy(panel_logits)  # (N, 3) in preprocessed coordinates

        # Compute seg_map from logits
        probs = np.exp(panel_logits) / np.exp(panel_logits).sum(axis=0, keepdims=True)
        seg_map = np.argmax(probs, axis=0).astype(np.uint8)  # (H, W) with values 0 or 1

        # Clip seg_map to original bounds (bottom-right padding)
        seg_map_clipped = seg_map[:H_orig, :W_orig]  # (H_orig, W_orig)

        # Clip logits to original bounds
        logits_clipped = panel_logits[:, :H_orig, :W_orig]  # (2, H_orig, W_orig)

        all_seg_maps.append(seg_map_clipped)
        all_logit_maps.append(logits_clipped)

        # Transform peak coordinates from preprocessed space to original space
        # Preprocessed: (512, 512), Original: (352, 384)
        # Bottom-right padding means: y' = y, x' = x (just clip if needed)
        # Actually peaks in 512x512 space should be valid as-is if they're within 352x384
        # Just need to make sure they don't exceed original bounds
        peaks_transformed = []
        for peak in peaks:
            _, y, x = peak
            # Clip to original bounds (bottom-right padding)
            if y < H_orig and x < W_orig:
                peaks_transformed.append([0, y, x])
            # else: skip peaks outside original detector area

        all_peaks.append(np.array(peaks_transformed) if peaks_transformed else np.array([]).reshape(0, 3))

        if verbose and len(peaks_transformed) > 0:
            logging.debug(f"  Panel {panel_idx}: {len(peaks_transformed)} peaks (clipped to original bounds)")

    logging.info(f"  Total panels processed: {len(all_peaks)}")
    logging.info(f"  Total peaks found: {sum(len(p) for p in all_peaks)}")

    # Reshape seg_maps and logit_maps to group by events
    # all_seg_maps: list of (H_orig, W_orig) arrays for B*C panels
    # all_logit_maps: list of (2, H_orig, W_orig) arrays for B*C panels
    # Reshape to (B, C, H_orig, W_orig) and (B, C, 2, H_orig, W_orig) respectively
    seg_maps_4d = np.array(all_seg_maps).reshape(B, C, H_orig, W_orig)  # (B, C, H, W)
    logit_maps_5d = np.array(all_logit_maps).reshape(B, C, 2, H_orig, W_orig)  # (B, C, 2, H, W)
    # Transpose to (B, 2, C, H, W) so num_classes is the second dimension
    logit_maps_5d = np.transpose(logit_maps_5d, (0, 2, 1, 3, 4))  # (B, 2, C, H, W)

    logging.info(f"  Seg maps shape: {seg_maps_4d.shape} (B, C, H_orig, W_orig)")
    logging.info(f"  Logit maps shape: {logit_maps_5d.shape} (B, num_classes, C, H_orig, W_orig)")

    # Step 3: Group panels into events and prepare for CXI writing
    logging.info("\nStep 3: Grouping panels into events...")

    # Prepare batch_info structure for coordinator's group_panels_into_events()
    # Convert all_peaks from [(N,3) array per panel] to [(panel_idx, peaks, None) tuples]
    completed_panels = []
    for panel_idx in range(len(all_peaks)):
        completed_panels.append((panel_idx, all_peaks[panel_idx], None))

    # Convert wavelength array to energy array
    if isinstance(photon_wavelength, np.ndarray):
        photon_energy_array = np.array([wavelength_to_energy(w) for w in photon_wavelength])
    else:
        photon_energy_array = wavelength_to_energy(photon_wavelength)

    batch_info = {
        'completed_panels': completed_panels,
        'B': B,
        'C': C,
        'H_orig': H_orig,
        'W_orig': W_orig,
        'detector_images_4d': detector_images_4d,
        'photon_energy': photon_energy_array,
        'photon_wavelength': photon_wavelength,
        'timestamp': timestamp,
        'metadata': {},
    }

    # Use coordinator's grouping function (testing the actual code!)
    # Note: group_panels_into_events returns 4 values (images, peaks, metadata, seg_maps)
    batch_images, batch_peaks, batch_metadata, _ = group_panels_into_events(batch_info)

    # Group seg_maps and logit_maps by events (same logic as images)
    # Each event has C panels, so we split along the C dimension
    batch_seg_maps = []
    batch_logit_maps = []
    for event_idx in range(B):
        # For each event, extract its C panels
        event_seg_map = seg_maps_4d[event_idx]  # (C, H, W)
        event_logit_map = logit_maps_5d[event_idx]  # (2, C, H, W)
        batch_seg_maps.append(event_seg_map)
        batch_logit_maps.append(event_logit_map)

    if verbose:
        for event_idx, (img, peaks, meta) in enumerate(zip(batch_images, batch_peaks, batch_metadata)):
            if peaks:
                panel_indices = [p[0] for p in peaks]
                min_panel = min(panel_indices)
                max_panel = max(panel_indices)
                logging.debug(f"  Event {event_idx}: shape {img.shape}, {len(peaks)} total peaks, panel range [{min_panel}, {max_panel}]")
            else:
                logging.debug(f"  Event {event_idx}: shape {img.shape}, 0 total peaks")

    # Submit batch to file writer (Ray actor requires .remote())
    file_writer.submit_processed_batch.remote(batch_images, batch_peaks, batch_metadata, batch_seg_maps, batch_logit_maps)

    logging.info(f"Submitted {B} events to file writer")


def main():
    parser = argparse.ArgumentParser(
        description="Test CXI writer using dumped Q2 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input HDF5 dump file (from dump_q2_data.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_cxi_output',
        help='Output directory for CXI files'
    )
    parser.add_argument(
        '--geom-file',
        type=str,
        default=None,
        help='Geometry file for CheetahConverter (optional)'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10,
        help='Buffer size for CXI file writer'
    )
    parser.add_argument(
        '--min-num-peak',
        type=int,
        default=10,
        help='Minimum number of peaks to save event'
    )
    parser.add_argument(
        '--max-num-peak',
        type=int,
        default=2048,
        help='Maximum number of peaks per event'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=None,
        help='Number of batches to process (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Test CXI Writer from Dump ===")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Geometry file: {args.geom_file}")
    logging.info(f"Buffer size: {args.buffer_size}")
    logging.info(f"Min peaks: {args.min_num_peak}")

    # Create file writer (local instance, not Ray actor)
    logging.info("\nInitializing CXI file writer...")

    # We need to create a local version of the file writer, not a Ray actor
    # For simplicity, let's directly instantiate the class without @ray.remote
    # We'll need to modify how we call it

    # Actually, let's just import and use it directly without Ray
    # This requires understanding the file_writer implementation

    # For now, let's create a simple inline version
    from datetime import datetime
    import ray

    # Initialize Ray for object store (needed for file_writer)
    ray.init(ignore_reinit_error=True)

    # Create actor with debug outputs enabled
    file_writer = CXIFileWriterActor.remote(
        output_dir=str(output_dir),
        geom_file=args.geom_file,
        buffer_size=args.buffer_size,
        min_num_peak=args.min_num_peak,
        max_num_peak=args.max_num_peak,
        file_prefix="test_cxi",
        save_segmentation_maps=True,  # Enable seg map output for debugging
        save_logit_maps=True  # Enable logit map output for debugging
    )

    # Process batches
    try:
        with h5py.File(input_path, 'r') as f:
            # Get number of batches
            batch_keys = [k for k in f.keys() if k.startswith('batch_')]
            num_batches = len(batch_keys)

            if args.num_batches is not None:
                num_batches = min(num_batches, args.num_batches)

            logging.info(f"\nProcessing {num_batches} batches from {input_path.name}...")

            for batch_idx in range(num_batches):
                batch_key = f'batch_{batch_idx}'
                if batch_key not in f:
                    logging.warning(f"Batch {batch_idx} not found, stopping")
                    break

                batch_group = f[batch_key]
                process_batch(batch_group, batch_idx, file_writer, verbose=args.verbose)

        # Flush final buffer
        logging.info("\nFlushing final CXI file...")
        stats = ray.get(file_writer.flush_final.remote())

        logging.info("\n=== Processing Complete ===")
        logging.info(f"Total events written: {stats['total_events_written']}")
        logging.info(f"Total events filtered: {stats['total_events_filtered']}")
        logging.info(f"CXI chunks: {stats['chunks_written']}")
        logging.info(f"Output directory: {output_dir.absolute()}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        ray.shutdown()
        logging.info("Ray shutdown complete")


if __name__ == '__main__':
    main()
