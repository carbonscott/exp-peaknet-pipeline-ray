#!/usr/bin/env python3
"""
Multi-Actor Q2-to-CXI Writer (Round-Robin)

Creates N postprocessing actors that pull batches from Q2 in round-robin fashion.
Each actor processes synchronously and writes independent CXI files.

Usage:
    python multi_actor_q2_to_cxi.py --config cxi_writer.yaml --num-actors 4
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


@ray.remote
class PostProcessorActor:
    """
    Postprocessing actor that handles batches independently.

    Each actor:
    - Has its own file writer
    - Processes batches synchronously
    - Tracks batch count and flushes periodically
    - Writes CXI files with actor ID in filename
    """

    def __init__(
        self,
        actor_id: int,
        output_dir: str,
        geom_file: str,
        buffer_size: int,
        min_num_peak: int,
        max_num_peak: int,
        file_prefix: str,
        batches_per_cxi_file: int,
        crystfel_mode: bool = False
    ):
        """
        Initialize postprocessing actor.

        Args:
            actor_id: Unique actor identifier
            output_dir: Output directory for CXI files
            geom_file: Geometry file for CheetahConverter
            buffer_size: Events to buffer before writing
            min_num_peak: Minimum peaks to save event
            max_num_peak: Maximum peaks per event
            file_prefix: Filename prefix
            batches_per_cxi_file: Batches per CXI file
            crystfel_mode: Enable CrystFEL mode
        """
        self.actor_id = actor_id
        self.batches_per_cxi_file = batches_per_cxi_file
        self.batch_count = 0
        self.total_events = 0

        # Create file writer with actor ID in filename
        actor_prefix = f"{file_prefix}_actor{actor_id}"
        self.file_writer = CXIFileWriterActor.remote(
            output_dir=output_dir,
            geom_file=geom_file,
            buffer_size=buffer_size,
            min_num_peak=min_num_peak,
            max_num_peak=max_num_peak,
            file_prefix=actor_prefix,
            crystfel_mode=crystfel_mode
        )

        logging.info(f"PostProcessorActor {actor_id} initialized")

    def process_batch(self, pipeline_output):
        """
        Process a single batch from Q2.

        Args:
            pipeline_output: PipelineOutput object from Q2

        Returns:
            dict: Statistics about processing
        """
        # Extract logits
        logits = pipeline_output.get_torch_tensor(device='cpu').numpy()  # (B*C, num_classes, H, W)

        # Extract B, C, H_orig, W_orig from preprocessing metadata FIRST
        # This ensures peak grouping works correctly even if image reconstruction fails
        B, C, H_orig, W_orig = None, None, None, None

        if hasattr(pipeline_output, 'preprocessing_metadata') and pipeline_output.preprocessing_metadata is not None:
            original_shape = pipeline_output.preprocessing_metadata.original_shape
            B, C, H_orig, W_orig = original_shape
            logging.debug(f"Actor {self.actor_id}: Extracted shape from metadata: B={B}, C={C}, H={H_orig}, W={W_orig}")

        # Now try to extract detector images (can fail independently)
        detector_images_4d = None

        if hasattr(pipeline_output, 'original_image_ref') and pipeline_output.original_image_ref is not None:
            try:
                if hasattr(pipeline_output, 'preprocessing_metadata') and pipeline_output.preprocessing_metadata is not None:
                    # Reconstruct to original size
                    original_image = ray.get(pipeline_output.original_image_ref)
                    preprocessed_shape = pipeline_output.preprocessing_metadata.preprocessed_shape

                    detector_images_4d = reconstruct_detector_image(original_image, original_shape, preprocessed_shape)
                else:
                    # No preprocessing metadata - use as-is
                    detector_images_4d = ray.get(pipeline_output.original_image_ref)
            except Exception as e:
                logging.warning(f"Actor {self.actor_id}: Failed to extract detector images: {e}")
                detector_images_4d = None

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
        all_peaks = []

        for panel_idx in range(logits.shape[0]):  # Iterate over B*C panels
            panel_logits = logits[panel_idx]  # (2, H, W)
            peaks = find_peaks_numpy(panel_logits)  # (N, 3)

            # Transform peak coordinates if needed (clip to original bounds)
            if H_orig and W_orig:
                peaks_transformed = []
                for peak in peaks:
                    _, y, x = peak
                    if y < H_orig and x < W_orig:
                        peaks_transformed.append([0, y, x])
                all_peaks.append(np.array(peaks_transformed) if peaks_transformed else np.array([]).reshape(0, 3))
            else:
                all_peaks.append(peaks)

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
        }

        # Use coordinator's grouping function
        event_images, event_peaks, event_metadata = group_panels_into_events(batch_info)

        # Submit to file writer
        self.file_writer.submit_processed_batch.remote(event_images, event_peaks, event_metadata)

        num_events = len(event_images)
        self.total_events += num_events
        self.batch_count += 1

        # Periodic flush: write CXI file every N batches
        if self.batch_count % self.batches_per_cxi_file == 0:
            stats = ray.get(self.file_writer.flush_final.remote())
            logging.info(f"Actor {self.actor_id}: Wrote CXI after {self.batch_count} batches "
                        f"({stats['chunks_written']} files, {stats['total_events_written']} events)")

        return {
            'actor_id': self.actor_id,
            'batch_count': self.batch_count,
            'num_events': num_events,
            'total_events': self.total_events
        }

    def get_statistics(self):
        """Get actor statistics."""
        file_writer_stats = ray.get(self.file_writer.get_statistics.remote())
        return {
            'actor_id': self.actor_id,
            'batch_count': self.batch_count,
            'total_events': self.total_events,
            'events_written': file_writer_stats['total_events_written'],
            'events_filtered': file_writer_stats['total_events_filtered'],
            'chunks_written': file_writer_stats['chunks_written']
        }

    def flush_final(self):
        """Flush any remaining data."""
        stats = ray.get(self.file_writer.flush_final.remote())
        logging.info(f"Actor {self.actor_id}: Final flush - {stats['chunks_written']} files written")
        return self.get_statistics()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-actor Q2-to-CXI writer (round-robin)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to writer configuration YAML file'
    )
    parser.add_argument(
        '--num-actors',
        type=int,
        default=4,
        help='Number of postprocessing actors'
    )
    parser.add_argument(
        '--batches-per-file',
        type=int,
        default=10,
        help='Write CXI file every N batches (per actor)'
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

    logger.info("=== Multi-Actor Q2-to-CXI Writer (Round-Robin) ===")
    logger.info(f"Number of actors: {args.num_actors}")
    logger.info(f"Batches per file (per actor): {args.batches_per_file}")
    logger.info(f"Ray namespace: {config['ray']['namespace']}")
    logger.info(f"Queue: {config['queue']['name']} ({config['queue']['num_shards']} shards)")
    logger.info(f"Output dir: {config['output']['output_dir']}")

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

    # Create postprocessing actors
    logger.info(f"Creating {args.num_actors} postprocessing actors...")
    actors = []
    for actor_id in range(args.num_actors):
        actor = PostProcessorActor.remote(
            actor_id=actor_id,
            output_dir=config['output']['output_dir'],
            geom_file=config.get('geometry', {}).get('geom_file'),
            buffer_size=config['output'].get('buffer_size', 100),
            min_num_peak=config['peak_finding']['min_num_peak'],
            max_num_peak=config['peak_finding']['max_num_peak'],
            file_prefix=config['output'].get('file_prefix', 'peaknet_cxi'),
            batches_per_cxi_file=args.batches_per_file,
            crystfel_mode=config.get('geometry', {}).get('geom_file') is not None
        )
        actors.append(actor)

    logger.info(f"Created {len(actors)} actors")

    # Main round-robin loop
    logger.info("Starting round-robin processing...")
    batch_count = 0
    total_events = 0

    try:
        while True:
            # Pull batch from Q2 (blocking)
            pipeline_output = q2_manager.get(timeout=0.1)

            if pipeline_output is None:
                continue

            # Round-robin: assign to actor[batch_count % num_actors]
            actor_idx = batch_count % args.num_actors

            # Submit to actor (non-blocking)
            result_ref = actors[actor_idx].process_batch.remote(pipeline_output)

            # Get result to track progress
            result = ray.get(result_ref)

            batch_count += 1
            total_events += result['num_events']

            if batch_count % 10 == 0:
                logger.info(f"Progress: {batch_count} batches distributed, {total_events} total events")

    except KeyboardInterrupt:
        logger.info("\n=== Interrupted by user ===")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Flush all actors
        logger.info("Flushing all actors...")
        flush_refs = [actor.flush_final.remote() for actor in actors]
        all_stats = ray.get(flush_refs)

        # Aggregate statistics
        total_batches = sum(s['batch_count'] for s in all_stats)
        total_events_written = sum(s['events_written'] for s in all_stats)
        total_events_filtered = sum(s['events_filtered'] for s in all_stats)
        total_chunks = sum(s['chunks_written'] for s in all_stats)

        logger.info("\n=== Processing Complete ===")
        logger.info(f"Total batches distributed: {batch_count}")
        logger.info(f"Total batches processed: {total_batches}")
        logger.info(f"Total events written: {total_events_written}")
        logger.info(f"Total events filtered: {total_events_filtered}")
        logger.info(f"Total CXI files: {total_chunks}")

        logger.info("\nPer-actor statistics:")
        for stats in all_stats:
            logger.info(f"  Actor {stats['actor_id']}: {stats['batch_count']} batches, "
                       f"{stats['events_written']} events, {stats['chunks_written']} files")

        ray.shutdown()


if __name__ == '__main__':
    main()
