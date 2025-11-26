#!/usr/bin/env python3
"""
Batch Structural Graph Construction

This script constructs structural brain graphs for all subjects in the
preprocessed dataset using T1w MRI data and brain masks.

Since fMRI data is not available, we focus on structural connectivity based on:
1. Morphometric features (GM/WM volume, brain regions)
2. Atlas-based parcellation (Schaefer 200 ROIs)
3. Region-based similarity for connectivity

Author: YOPD Pipeline
Date: November 2025
"""

import sys
import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import nibabel as nib
from datetime import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Neuroimaging libraries
from nilearn import datasets, image, plotting
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_construction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class StructuralGraphConstructor:
    """
    Construct structural brain graphs from preprocessed T1w data
    """
    
    def __init__(self, n_rois: int = 200, correlation_threshold: float = 0.25):
        self.n_rois = n_rois
        self.correlation_threshold = correlation_threshold
        self.atlas = None
        self.atlas_labels = None
        self._load_atlas()
    
    def _load_atlas(self):
        """Load Schaefer atlas for parcellation"""
        logger.info(f"Loading Schaefer {self.n_rois} ROI atlas...")
        try:
            # Load Schaefer atlas
            self.atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=self.n_rois,
                yeo_networks=7,
                resolution_mm=2
            )
            self.atlas_labels = self.atlas['labels']
            logger.info(f"✓ Atlas loaded: {len(self.atlas_labels)} regions")
        except Exception as e:
            logger.error(f"Failed to load atlas: {e}")
            raise
    
    def extract_roi_features(self, t1w_file: Path, gm_mask: Path, wm_mask: Path, 
                            brain_mask: Path) -> Dict:
        """
        Extract morphometric features for each ROI
        
        Returns:
            Dictionary with ROI features and connectivity
        """
        logger.info(f"Processing: {t1w_file.name}")
        
        try:
            # Load images
            t1w_img = nib.load(t1w_file)
            gm_img = nib.load(gm_mask)
            wm_img = nib.load(wm_mask)
            brain_img = nib.load(brain_mask)
            
            # Resample atlas to match T1w space
            logger.info("  Resampling atlas to T1w space...")
            atlas_img = nib.load(self.atlas['maps'])
            atlas_resampled = image.resample_to_img(
                atlas_img, 
                t1w_img, 
                interpolation='nearest'
            )
            
            # Create masker for extracting ROI signals
            masker = NiftiLabelsMasker(
                labels_img=atlas_resampled,
                standardize=False,
                memory='nilearn_cache',
                verbose=0
            )
            
            # Extract mean intensity for each ROI from different modalities
            logger.info("  Extracting ROI features...")
            t1w_signals = masker.fit_transform(t1w_img)
            gm_signals = masker.transform(gm_img)
            wm_signals = masker.transform(wm_img)
            brain_signals = masker.transform(brain_img)
            
            # Compute additional morphometric features per ROI
            atlas_data = atlas_resampled.get_fdata()
            t1w_data = t1w_img.get_fdata()
            gm_data = gm_img.get_fdata()
            wm_data = wm_img.get_fdata()
            
            roi_features = []
            for roi_idx in range(1, self.n_rois + 1):
                roi_mask = (atlas_data == roi_idx)
                
                if roi_mask.sum() == 0:
                    # Empty ROI
                    roi_features.append(np.zeros(8))
                    continue
                
                # Extract features for this ROI
                features = [
                    np.mean(t1w_data[roi_mask]),           # Mean T1w intensity
                    np.std(t1w_data[roi_mask]),            # T1w std
                    np.mean(gm_data[roi_mask]),            # Mean GM probability
                    np.std(gm_data[roi_mask]),             # GM std
                    np.mean(wm_data[roi_mask]),            # Mean WM probability
                    np.sum(gm_data[roi_mask]),             # Total GM volume
                    np.sum(wm_data[roi_mask]),             # Total WM volume
                    roi_mask.sum()                          # ROI size (voxels)
                ]
                roi_features.append(features)
            
            roi_features = np.array(roi_features)
            
            # Normalize features
            logger.info("  Normalizing features...")
            scaler = StandardScaler()
            roi_features_normalized = scaler.fit_transform(roi_features)
            
            # Compute structural connectivity based on morphometric similarity
            logger.info("  Computing morphometric similarity matrix...")
            # Use correlation between feature vectors as similarity
            similarity_matrix = np.corrcoef(roi_features_normalized)
            
            # Apply threshold to create sparse connectivity
            adjacency_matrix = (np.abs(similarity_matrix) > self.correlation_threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections
            
            # Compute graph metrics
            logger.info("  Computing graph metrics...")
            G = nx.from_numpy_array(adjacency_matrix)
            
            # Node-level metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            # Graph-level metrics
            try:
                avg_clustering = nx.average_clustering(G)
            except:
                avg_clustering = 0.0
            
            num_edges = G.number_of_edges()
            density = nx.density(G)
            
            # Create comprehensive graph dictionary
            graph_data = {
                'subject_id': t1w_file.stem.split('_')[0],
                'n_rois': self.n_rois,
                'roi_labels': list(self.atlas_labels),
                'node_features': roi_features_normalized,
                'raw_node_features': roi_features,
                'feature_names': [
                    't1w_mean', 't1w_std', 'gm_mean', 'gm_std', 
                    'wm_mean', 'gm_volume', 'wm_volume', 'roi_size'
                ],
                'similarity_matrix': similarity_matrix,
                'adjacency_matrix': adjacency_matrix,
                'edge_weights': similarity_matrix * adjacency_matrix,
                'graph_metrics': {
                    'num_nodes': self.n_rois,
                    'num_edges': num_edges,
                    'density': density,
                    'avg_clustering': avg_clustering,
                },
                'node_metrics': {
                    'degree_centrality': np.array([degree_centrality[i] for i in range(self.n_rois)]),
                    'betweenness_centrality': np.array([betweenness_centrality[i] for i in range(self.n_rois)]),
                    'closeness_centrality': np.array([closeness_centrality[i] for i in range(self.n_rois)]),
                },
                'atlas': f'Schaefer2018_{self.n_rois}Parcels_7Networks',
                'construction_date': datetime.now().isoformat(),
                'modality': 'structural_t1w'
            }
            
            logger.info(f"✓ Graph constructed: {num_edges} edges, density={density:.4f}")
            return graph_data
            
        except Exception as e:
            logger.error(f"Failed to process {t1w_file.name}: {e}")
            traceback.print_exc()
            raise


def process_single_subject(args):
    """Wrapper function for parallel processing"""
    subject_dir, output_dir, n_rois, threshold = args
    
    try:
        subject_id = subject_dir.name
        logger.info(f"Processing {subject_id}...")
        
        # Find required files
        t1w_files = list(subject_dir.glob("*T1w_brain.nii*"))
        gm_masks = list(subject_dir.glob("*GM_mask.nii*"))
        wm_masks = list(subject_dir.glob("*WM_mask.nii*"))
        brain_masks = list(subject_dir.glob("*brain_mask.nii*"))
        
        if not (t1w_files and gm_masks and wm_masks and brain_masks):
            logger.warning(f"Skipping {subject_id}: Missing required files")
            return None
        
        # Construct graph
        constructor = StructuralGraphConstructor(n_rois=n_rois, correlation_threshold=threshold)
        graph_data = constructor.extract_roi_features(
            t1w_files[0], gm_masks[0], wm_masks[0], brain_masks[0]
        )
        
        # Save graph
        output_file = output_dir / f"{subject_id}_structural_graph.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(graph_data, f)
        
        # Save summary as JSON
        summary = {
            'subject_id': graph_data['subject_id'],
            'n_rois': graph_data['n_rois'],
            'num_edges': graph_data['graph_metrics']['num_edges'],
            'density': graph_data['graph_metrics']['density'],
            'avg_clustering': graph_data['graph_metrics']['avg_clustering'],
            'modality': graph_data['modality']
        }
        
        summary_file = output_dir / f"{subject_id}_structural_graph_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Saved: {output_file.name}")
        return (subject_id, True, output_file)
        
    except Exception as e:
        logger.error(f"Failed to process {subject_dir.name}: {e}")
        return (subject_dir.name, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Batch construct structural brain graphs from preprocessed T1w data'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing preprocessed subject data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/sukiperumal/Documents/yopd/outputs',
        help='Output directory for graph files'
    )
    
    parser.add_argument(
        '--n-rois',
        type=int,
        default=200,
        help='Number of ROIs in Schaefer atlas (100, 200, 400, 600)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.25,
        help='Correlation threshold for creating edges'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (-1 for all CPUs)'
    )
    
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='*',
        help='Specific subjects to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("BATCH STRUCTURAL GRAPH CONSTRUCTION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Atlas: Schaefer {args.n_rois} ROIs")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("")
    
    # Find subjects
    if args.subjects:
        subject_dirs = [data_dir / f"sub-{s}" if not s.startswith('sub-') else data_dir / s 
                       for s in args.subjects]
        subject_dirs = [d for d in subject_dirs if d.exists()]
    else:
        subject_dirs = sorted(list(data_dir.glob("sub-*")))
    
    if not subject_dirs:
        logger.error("No subjects found!")
        return 1
    
    logger.info(f"Found {len(subject_dirs)} subjects to process")
    logger.info("")
    
    # Prepare arguments for parallel processing
    process_args = [
        (subject_dir, output_dir, args.n_rois, args.threshold)
        for subject_dir in subject_dirs
    ]
    
    # Process subjects
    results = []
    successful = []
    failed = []
    
    if args.n_jobs == 1:
        # Sequential processing
        for pargs in tqdm(process_args, desc="Processing subjects"):
            result = process_single_subject(pargs)
            if result:
                results.append(result)
                if result[1]:
                    successful.append(result[0])
                else:
                    failed.append(result[0])
    else:
        # Parallel processing
        n_jobs = args.n_jobs if args.n_jobs > 0 else None
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_single_subject, pargs) for pargs in process_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
                result = future.result()
                if result:
                    results.append(result)
                    if result[1]:
                        successful.append(result[0])
                    else:
                        failed.append(result[0])
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total subjects: {len(subject_dirs)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.warning(f"\nFailed subjects: {', '.join(failed)}")
    
    # Save overall summary
    summary = {
        'processing_date': datetime.now().isoformat(),
        'data_directory': str(data_dir),
        'output_directory': str(output_dir),
        'n_rois': args.n_rois,
        'threshold': args.threshold,
        'total_subjects': len(subject_dirs),
        'successful': len(successful),
        'failed': len(failed),
        'successful_subjects': successful,
        'failed_subjects': failed
    }
    
    summary_file = output_dir / 'batch_processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Summary saved to: {summary_file}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
