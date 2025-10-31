#!/usr/bin/env python3
"""
Simple Q2-to-CXI Writer (Synchronous)

Pulls batches from Q2 queue and writes CXI files every N batches.
Repurposed from test_cxi_writer_from_dump.py with proven working logic.

Usage:
    python simple_q2_to_cxi.py --config cxi_writer.yaml
"""

import argparse
import logging
import sys
import yaml
import numpy as np
from pathlib import Path
from scipy import ndimage
import ray

# Add cxi-pipeline-ray to path
sys.path.insert(0, '/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray')
from cxi_pipeline_ray.core.file_writer import CXIFileWriterActor
from cxi_pipeline_ray.core.coordinator import group_panels_into_events


def setup_logging(level: str = 'INFO'):
    """Configure logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def find_peaks_numpy(logits_2d: np.ndarray, min_prob: float = 0.5, return_seg_map: bool = False):
    """
    Find peaks from 2-class logits using scipy.ndimage.label.

    Args:
        logits_2d: (2, H, W) logits from model
        min_prob: Minimum probability threshold
        return_seg_map: If True, return (peaks, seg_map) tuple

    Returns:
        peaks: (N, 3) array of [panel_idx=0, y, x]
        OR
        (peaks, seg_map): If return_seg_map=True, where seg_map is (H, W) uint8 with values 0 or 1
    """
    # Convert logits to probability: softmax then argmax
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
        peaks = np.array([]).reshape(0, 3)
    else:
        # Find center of mass for each component
        peaks = []
        for label_id in range(1, num_features + 1):
            component_mask = labeled == label_id
            y_coords, x_coords = np.where(component_mask)

            # Center of mass
            y_center = y_coords.mean()
            x_center = x_coords.mean()

            peaks.append([0, y_center, x_center])  # panel_idx=0 for single panel

        peaks = np.array(peaks, dtype=np.float32)

    if return_seg_map:
        return peaks, seg_map.astype(np.uint8)
    else:
        return peaks


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


def process_batch_from_q2(pipeline_output, file_writer, save_segmentation_maps: bool = False, verbose: bool = False):
    """
    Process a single batch from Q2 through CXI pipeline.

    Args:
        pipeline_output: PipelineOutput object from Q2
        file_writer: CXIFileWriterActor instance
        save_segmentation_maps: Save segmentation maps to CXI (debug mode)
        verbose: Enable verbose logging
    """
    # Extract logits
    logits = pipeline_output.get_torch_tensor(device='cpu').numpy()  # (B*C, num_classes, H, W)

    # Extract B, C, H_orig, W_orig from preprocessing metadata FIRST
    # This ensures peak grouping works correctly even if image reconstruction fails
    B, C, H_orig, W_orig = None, None, None, None

    if hasattr(pipeline_output, 'preprocessing_metadata') and pipeline_output.preprocessing_metadata is not None:
        original_shape = pipeline_output.preprocessing_metadata.original_shape
        B, C, H_orig, W_orig = original_shape
        logging.debug(f"Extracted shape from metadata: B={B}, C={C}, H={H_orig}, W={W_orig}")
    else:
        logging.warning("NO preprocessing_metadata found! This may cause detector image/seg map mismatch!")

    # Now try to extract detector images (can fail independently)
    detector_images_4d = None

    if hasattr(pipeline_output, 'original_image_ref') and pipeline_output.original_image_ref is not None:
        try:
            if hasattr(pipeline_output, 'preprocessing_metadata') and pipeline_output.preprocessing_metadata is not None:
                # Reconstruct to original size
                original_image = ray.get(pipeline_output.original_image_ref)
                preprocessed_shape = pipeline_output.preprocessing_metadata.preprocessed_shape

                logging.debug(f"original_image shape: {original_image.shape}, preprocessed_shape: {preprocessed_shape}, original_shape: {original_shape}")
                detector_images_4d = reconstruct_detector_image(original_image, original_shape, preprocessed_shape)
                logging.debug(f"Reconstructed detector images: {detector_images_4d.shape}")
            else:
                # No preprocessing metadata - use as-is
                original_image_raw = ray.get(pipeline_output.original_image_ref)
                logging.warning(f"NO preprocessing metadata - using images as-is: {original_image_raw.shape}")
                logging.warning(f"This will likely cause mismatch! Expected (B,C,H,W) but got {original_image_raw.shape}")
                detector_images_4d = original_image_raw
        except Exception as e:
            logging.warning(f"Failed to extract detector images: {e}")
            detector_images_4d = None
    else:
        logging.warning("NO original_image_ref found! Detector images will be None")

    # Extract physics metadata
    metadata = pipeline_output.metadata if hasattr(pipeline_output, 'metadata') else {}

    # Handle photon wavelength → energy conversion
    photon_wavelength = metadata.get('photon_wavelength', None)
    photon_energy = metadata.get('photon_energy', None)

    if photon_wavelength is not None:
        if isinstance(photon_wavelength, (list, np.ndarray)):
            photon_energy = np.array([wavelength_to_energy(w) for w in photon_wavelength])
        else:
            photon_energy = wavelength_to_energy(float(photon_wavelength))

    # Extract timestamp
    timestamp = metadata.get('timestamp', None)

    # Run peak finding on logits
    logging.debug(f"Running peak finding on {logits.shape[0]} panels (logits shape: {logits.shape})...")
    logging.debug(f"Expected B*C = {B}*{C} = {B*C if B and C else 'unknown'}")
    if B and C and logits.shape[0] != B*C:
        logging.error(f"MISMATCH: logits.shape[0]={logits.shape[0]} but B*C={B*C}!")

    all_peaks = []
    all_seg_maps = [] if save_segmentation_maps else None

    for panel_idx in range(logits.shape[0]):  # Iterate over B*C panels
        panel_logits = logits[panel_idx]  # (2, H, W)

        # Get peaks and optionally seg_map
        if save_segmentation_maps:
            peaks, seg_map = find_peaks_numpy(panel_logits, return_seg_map=True)  # (N, 3), (H, W)
        else:
            peaks = find_peaks_numpy(panel_logits)  # (N, 3)

        # Transform peak coordinates if needed (clip to original bounds)
        if H_orig and W_orig:
            peaks_transformed = []
            for peak in peaks:
                _, y, x = peak
                if y < H_orig and x < W_orig:
                    peaks_transformed.append([0, y, x])
            all_peaks.append(np.array(peaks_transformed) if peaks_transformed else np.array([]).reshape(0, 3))

            # Clip seg_map to original bounds
            if save_segmentation_maps:
                all_seg_maps.append(seg_map[:H_orig, :W_orig])
        else:
            all_peaks.append(peaks)
            if save_segmentation_maps:
                all_seg_maps.append(seg_map)

    # Group panels into events
    completed_panels = []
    for panel_idx in range(len(all_peaks)):
        completed_panels.append((panel_idx, all_peaks[panel_idx], None))

    batch_info = {
        'completed_panels': completed_panels,
        'B': B if B is not None else 1,
        'C': C if C is not None else len(all_peaks),
        'H_orig': H_orig,
        'W_orig': W_orig,
        'detector_images_4d': detector_images_4d,
        'photon_energy': photon_energy,
        'photon_wavelength': photon_wavelength,
        'timestamp': timestamp,
        'metadata': metadata,
        'num_panels': len(all_peaks),
        'segmentation_maps': all_seg_maps,  # None if save_segmentation_maps=False
    }

    # Use coordinator's grouping function
    event_images, event_peaks, event_metadata, event_seg_maps = group_panels_into_events(batch_info)

    # Submit to file writer
    file_writer.submit_processed_batch.remote(event_images, event_peaks, event_metadata, event_seg_maps)

    total_peaks = sum(len(p) for p in all_peaks)
    logging.debug(f"Submitted {len(event_images)} events, {total_peaks} total peaks")

    return len(event_images)


def main():
    parser = argparse.ArgumentParser(
        description="Simple Q2-to-CXI writer (synchronous)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to writer configuration YAML file'
    )
    parser.add_argument(
        '--batches-per-file',
        type=int,
        default=10,
        help='Write CXI file every N batches'
    )
    parser.add_argument(
        '--save-segmentation-maps',
        action='store_true',
        help='Save segmentation maps to /entry_1/result_1/segmentation_map (debug mode)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=== Simple Q2-to-CXI Writer (Synchronous) ===")
    logger.info(f"Ray namespace: {config['ray']['namespace']}")
    logger.info(f"Queue: {config['queue']['name']} ({config['queue']['num_shards']} shards)")
    logger.info(f"Output dir: {config['output']['output_dir']}")
    logger.info(f"Batches per file: {args.batches_per_file}")
    logger.info(f"Save segmentation maps: {args.save_segmentation_maps}")

    # Connect to Ray
    logger.info("Connecting to Ray cluster...")
    ray.init(
        namespace=config['ray']['namespace'],
        ignore_reinit_error=True
    )
    logger.info(f"Connected to Ray: {ray.cluster_resources()}")

    # Connect to Q2 queue
    logger.info(f"Connecting to Q2 queue: {config['queue']['name']}")
    try:
        from peaknet_pipeline_ray.utils.queue import ShardedQueueManager

        q2_manager = ShardedQueueManager(
            base_name=config['queue']['name'],
            num_shards=config['queue']['num_shards'],
            maxsize_per_shard=config['queue'].get('maxsize_per_shard', 1600)
        )
        logger.info("Successfully connected to Q2 queue")
    except ImportError as e:
        logger.error("Failed to import ShardedQueueManager")
        logger.error("Make sure peaknet-pipeline-ray is installed")
        sys.exit(1)

    # Create file writer actor
    logger.info("Creating CXI file writer actor...")
    file_writer = CXIFileWriterActor.remote(
        output_dir=config['output']['output_dir'],
        geom_file=config.get('geometry', {}).get('geom_file'),
        buffer_size=config['output'].get('buffer_size', 100),
        min_num_peak=config['peak_finding']['min_num_peak'],
        max_num_peak=config['peak_finding']['max_num_peak'],
        file_prefix=config['output'].get('file_prefix', 'peaknet_cxi'),
        crystfel_mode=config.get('geometry', {}).get('geom_file') is not None,
        save_segmentation_maps=args.save_segmentation_maps
    )

    # Main processing loop
    logger.info("Starting main processing loop...")
    batch_count = 0
    total_events = 0
    batches_since_flush = 0

    try:
        while True:
            # Pull batch from Q2 (blocking)
            pipeline_output = q2_manager.get(timeout=0.1)

            if pipeline_output is None:
                continue

            # Process batch
            num_events = process_batch_from_q2(
                pipeline_output,
                file_writer,
                save_segmentation_maps=args.save_segmentation_maps,
                verbose=(args.log_level == 'DEBUG')
            )
            batch_count += 1
            total_events += num_events
            batches_since_flush += 1

            logger.info(f"Processed batch {batch_count}: {num_events} events (total: {total_events})")

            # Periodic flush: write CXI file every N batches
            if batches_since_flush >= args.batches_per_file:
                logger.info(f"=== Writing CXI file after {batches_since_flush} batches ===")
                stats = ray.get(file_writer.flush_final.remote())
                logger.info(f"Wrote CXI: {stats['chunks_written']} files, "
                           f"{stats['total_events_written']} events written, "
                           f"{stats['total_events_filtered']} events filtered")
                batches_since_flush = 0

            # Progress logging
            if batch_count % 50 == 0:
                logger.info(f"Progress: {batch_count} batches, {total_events} events")

    except KeyboardInterrupt:
        logger.info("\n=== Interrupted by user ===")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Final flush
        logger.info("Flushing final CXI file...")
        stats = ray.get(file_writer.flush_final.remote())

        logger.info("\n=== Processing Complete ===")
        logger.info(f"Total batches: {batch_count}")
        logger.info(f"Total events: {total_events}")
        logger.info(f"Events written: {stats['total_events_written']}")
        logger.info(f"Events filtered: {stats['total_events_filtered']}")
        logger.info(f"CXI files: {stats['chunks_written']}")

        ray.shutdown()


if __name__ == '__main__':
    main()
