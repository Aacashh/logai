"""
ML-Based Rock Type Classification for Well Log Data

Implements unsupervised clustering algorithms for rock type detection:
- K-means clustering with automatic optimal K selection
- Gaussian Mixture Model (GMM) for soft clustering
- PCA preprocessing for dimensionality reduction

Key Features Used:
- GR (Gamma Ray) - Primary shale indicator
- RHOB (Bulk Density) - Porosity and lithology indicator
- NPHI (Neutron Porosity) - Gas/fluid detection
- DT (Sonic) - Porosity and lithology (optional)

References:
- SPE papers on electrofacies classification
- K-means++ initialization for robust clustering
- BIC criterion for GMM component selection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples


@dataclass
class RockClassificationResult:
    """Container for rock classification results."""
    cluster_labels: np.ndarray  # Cluster assignment for each sample
    cluster_probabilities: Optional[np.ndarray]  # Soft clustering probabilities (GMM only)
    num_clusters: int
    cluster_centers: np.ndarray  # Cluster centroids in feature space
    feature_columns: List[str]
    method: str  # 'kmeans' or 'gmm'
    
    # Quality metrics
    silhouette_score: float
    inertia: Optional[float]  # Within-cluster sum of squares (KMeans)
    bic: Optional[float]  # Bayesian Information Criterion (GMM)
    
    # Petrophysical interpretation
    cluster_interpretations: Dict[int, str]  # e.g., {0: 'Sand', 1: 'Shale'}
    cluster_stats: Dict[int, Dict[str, float]]  # Statistics per cluster
    
    # PCA info (if used)
    pca_explained_variance: Optional[np.ndarray]
    pca_components: Optional[np.ndarray]
    used_pca: bool
    
    # Optimal K analysis
    k_analysis: Optional[Dict[str, Any]] = None


def find_optimal_clusters(
    data: np.ndarray,
    max_k: int = 8,
    min_k: int = 2
) -> Dict[str, Any]:
    """
    Find the optimal number of clusters using Elbow and Silhouette methods.
    
    Args:
        data: Scaled feature matrix (n_samples, n_features)
        max_k: Maximum number of clusters to test
        min_k: Minimum number of clusters to test
        
    Returns:
        Dictionary containing:
        - k_range: List of K values tested
        - inertias: WCSS for each K
        - silhouette_scores: Silhouette score for each K
        - optimal_k_elbow: Suggested K from elbow method
        - optimal_k_silhouette: Suggested K from silhouette method
    """
    k_range = list(range(min_k, max_k + 1))
    inertias = []
    silhouette_scores_list = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        
        inertias.append(kmeans.inertia_)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(data, labels)
            silhouette_scores_list.append(sil_score)
        else:
            silhouette_scores_list.append(0)
    
    # Elbow method: find point of maximum curvature
    inertias_arr = np.array(inertias)
    # Use second derivative to find elbow
    if len(inertias) >= 3:
        second_diff = np.diff(np.diff(inertias_arr))
        elbow_idx = np.argmax(second_diff) + 1  # +1 because of double diff
        optimal_k_elbow = k_range[min(elbow_idx, len(k_range) - 1)]
    else:
        optimal_k_elbow = min_k
    
    # Silhouette method: find maximum silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores_list)]
    
    return {
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_silhouette': optimal_k_silhouette,
        'recommended_k': optimal_k_silhouette  # Silhouette is more reliable
    }


def apply_pca_preprocessing(
    data: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction and noise filtering.
    
    Args:
        data: Scaled feature matrix
        n_components: Number of components (if None, uses variance_threshold)
        variance_threshold: Keep components explaining this much variance
        
    Returns:
        Transformed data and fitted PCA object
    """
    if n_components is None:
        # First fit to determine number of components needed
        pca_full = PCA()
        pca_full.fit(data)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_components = max(2, min(n_components, data.shape[1]))  # At least 2, at most all features
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    return transformed_data, pca


def interpret_clusters_petrophysically(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_columns: List[str],
    gr_column: Optional[str] = None
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]]]:
    """
    Interpret clusters based on petrophysical properties.
    
    Uses Gamma Ray as primary indicator:
    - Low GR (< 75 API) = Sand (Reservoir)
    - High GR (> 75 API) = Shale (Non-Reservoir)
    
    Args:
        df: Original DataFrame with log data
        cluster_labels: Cluster assignments
        feature_columns: Columns used for clustering
        gr_column: Name of GR column (auto-detected if None)
        
    Returns:
        Tuple of (cluster_interpretations, cluster_stats)
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_stats = {}
    cluster_interpretations = {}
    
    # Auto-detect GR column
    if gr_column is None:
        gr_candidates = ['GR', 'SGR', 'CGR', 'GAMMA', 'GRGC']
        for col in gr_candidates:
            if col in df.columns:
                gr_column = col
                break
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_data = df.loc[mask]
        
        # Calculate statistics for each feature
        stats = {
            'count': int(mask.sum()),
            'percentage': float(mask.sum() / len(cluster_labels) * 100)
        }
        
        for col in feature_columns:
            if col in df.columns:
                col_data = cluster_data[col].dropna()
                if len(col_data) > 0:
                    stats[f'{col}_mean'] = float(col_data.mean())
                    stats[f'{col}_std'] = float(col_data.std())
                    stats[f'{col}_min'] = float(col_data.min())
                    stats[f'{col}_max'] = float(col_data.max())
        
        cluster_stats[cluster_id] = stats
    
    # Interpret based on GR values
    if gr_column and gr_column in df.columns:
        # Calculate mean GR for each cluster
        gr_means = {}
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            gr_data = df.loc[mask, gr_column].dropna()
            if len(gr_data) > 0:
                gr_means[cluster_id] = gr_data.mean()
            else:
                gr_means[cluster_id] = np.nan
        
        # Sort clusters by GR value
        sorted_clusters = sorted(gr_means.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
        
        # Assign interpretations based on GR ranking
        n_clusters = len(unique_clusters)
        for i, (cluster_id, gr_mean) in enumerate(sorted_clusters):
            if np.isnan(gr_mean):
                cluster_interpretations[cluster_id] = "Unknown"
            elif n_clusters == 2:
                # Binary classification
                cluster_interpretations[cluster_id] = "Sand (Reservoir)" if i == 0 else "Shale (Non-Reservoir)"
            elif n_clusters == 3:
                # Ternary classification
                if i == 0:
                    cluster_interpretations[cluster_id] = "Clean Sand"
                elif i == 1:
                    cluster_interpretations[cluster_id] = "Shaly Sand"
                else:
                    cluster_interpretations[cluster_id] = "Shale"
            else:
                # Multi-class: use GR thresholds
                if gr_mean < 45:
                    cluster_interpretations[cluster_id] = "Clean Sand"
                elif gr_mean < 75:
                    cluster_interpretations[cluster_id] = "Shaly Sand"
                elif gr_mean < 100:
                    cluster_interpretations[cluster_id] = "Sandy Shale"
                else:
                    cluster_interpretations[cluster_id] = "Shale"
    else:
        # No GR available - use generic names
        for cluster_id in unique_clusters:
            cluster_interpretations[cluster_id] = f"Facies {cluster_id + 1}"
    
    return cluster_interpretations, cluster_stats


def classify_rocks_kmeans(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_clusters: Optional[int] = None,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    auto_interpret: bool = True
) -> RockClassificationResult:
    """
    Classify rock types using K-means clustering.
    
    K-means++ initialization ensures robust, reproducible results.
    Optimal K is automatically determined if not specified.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use for clustering
        n_clusters: Number of clusters (auto-detected if None)
        use_pca: Whether to apply PCA preprocessing
        pca_variance: Variance ratio to retain in PCA
        auto_interpret: Whether to interpret clusters petrophysically
        
    Returns:
        RockClassificationResult with all clustering information
    """
    # Prepare data
    valid_columns = [col for col in feature_columns if col in df.columns]
    if len(valid_columns) < 2:
        raise ValueError(f"Need at least 2 valid feature columns. Found: {valid_columns}")
    
    # Handle missing values
    data_df = df[valid_columns].copy()
    valid_mask = ~data_df.isnull().any(axis=1)
    clean_data = data_df[valid_mask].values
    
    if len(clean_data) < 10:
        raise ValueError("Insufficient valid data points for clustering (need at least 10)")
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    # Apply PCA if requested
    pca_info = None
    pca_components = None
    if use_pca and scaled_data.shape[1] > 2:
        transformed_data, pca = apply_pca_preprocessing(scaled_data, variance_threshold=pca_variance)
        pca_info = pca.explained_variance_ratio_
        pca_components = pca.components_
        clustering_data = transformed_data
    else:
        clustering_data = scaled_data
        use_pca = False
    
    # Find optimal K if not specified
    k_analysis = None
    if n_clusters is None:
        k_analysis = find_optimal_clusters(clustering_data, max_k=min(8, len(clean_data) // 10))
        n_clusters = k_analysis['recommended_k']
    
    # Run K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    labels_clean = kmeans.fit_predict(clustering_data)
    
    # Map back to original indices
    full_labels = np.full(len(df), -1)
    full_labels[valid_mask] = labels_clean
    
    # Calculate silhouette score
    sil_score = silhouette_score(clustering_data, labels_clean) if len(np.unique(labels_clean)) > 1 else 0
    
    # Interpret clusters
    interpretations = {}
    cluster_stats = {}
    if auto_interpret:
        interpretations, cluster_stats = interpret_clusters_petrophysically(
            df[valid_mask].reset_index(drop=True),
            labels_clean,
            valid_columns
        )
    
    return RockClassificationResult(
        cluster_labels=full_labels,
        cluster_probabilities=None,
        num_clusters=n_clusters,
        cluster_centers=kmeans.cluster_centers_,
        feature_columns=valid_columns,
        method='kmeans',
        silhouette_score=sil_score,
        inertia=kmeans.inertia_,
        bic=None,
        cluster_interpretations=interpretations,
        cluster_stats=cluster_stats,
        pca_explained_variance=pca_info,
        pca_components=pca_components,
        used_pca=use_pca,
        k_analysis=k_analysis
    )


def classify_rocks_gmm(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_components: Optional[int] = None,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    auto_interpret: bool = True,
    covariance_type: str = 'full'
) -> RockClassificationResult:
    """
    Classify rock types using Gaussian Mixture Model.
    
    GMM provides soft clustering with probability assignments.
    Uses BIC for automatic component selection.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use for clustering
        n_components: Number of GMM components (auto-detected if None)
        use_pca: Whether to apply PCA preprocessing
        pca_variance: Variance ratio to retain in PCA
        auto_interpret: Whether to interpret clusters petrophysically
        covariance_type: 'full', 'tied', 'diag', or 'spherical'
        
    Returns:
        RockClassificationResult with all clustering information
    """
    # Prepare data
    valid_columns = [col for col in feature_columns if col in df.columns]
    if len(valid_columns) < 2:
        raise ValueError(f"Need at least 2 valid feature columns. Found: {valid_columns}")
    
    # Handle missing values
    data_df = df[valid_columns].copy()
    valid_mask = ~data_df.isnull().any(axis=1)
    clean_data = data_df[valid_mask].values
    
    if len(clean_data) < 10:
        raise ValueError("Insufficient valid data points for clustering (need at least 10)")
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    # Apply PCA if requested
    pca_info = None
    pca_components = None
    if use_pca and scaled_data.shape[1] > 2:
        transformed_data, pca = apply_pca_preprocessing(scaled_data, variance_threshold=pca_variance)
        pca_info = pca.explained_variance_ratio_
        pca_components = pca.components_
        clustering_data = transformed_data
    else:
        clustering_data = scaled_data
        use_pca = False
    
    # Find optimal number of components using BIC
    best_bic = None
    best_n = 2
    if n_components is None:
        max_components = min(8, len(clean_data) // 20)
        bics = []
        for n in range(2, max_components + 1):
            gmm_test = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=42,
                n_init=3
            )
            gmm_test.fit(clustering_data)
            bics.append(gmm_test.bic(clustering_data))
        
        best_idx = np.argmin(bics)
        best_n = best_idx + 2
        n_components = best_n
        best_bic = bics[best_idx]
    
    # Fit final GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        n_init=5,
        max_iter=200
    )
    labels_clean = gmm.fit_predict(clustering_data)
    probabilities_clean = gmm.predict_proba(clustering_data)
    
    # Map back to original indices
    full_labels = np.full(len(df), -1)
    full_labels[valid_mask] = labels_clean
    
    full_probabilities = np.zeros((len(df), n_components))
    full_probabilities[valid_mask] = probabilities_clean
    
    # Calculate silhouette score
    sil_score = silhouette_score(clustering_data, labels_clean) if len(np.unique(labels_clean)) > 1 else 0
    
    # Interpret clusters
    interpretations = {}
    cluster_stats = {}
    if auto_interpret:
        interpretations, cluster_stats = interpret_clusters_petrophysically(
            df[valid_mask].reset_index(drop=True),
            labels_clean,
            valid_columns
        )
    
    return RockClassificationResult(
        cluster_labels=full_labels,
        cluster_probabilities=full_probabilities,
        num_clusters=n_components,
        cluster_centers=gmm.means_,
        feature_columns=valid_columns,
        method='gmm',
        silhouette_score=sil_score,
        inertia=None,
        bic=gmm.bic(clustering_data) if best_bic is None else best_bic,
        cluster_interpretations=interpretations,
        cluster_stats=cluster_stats,
        pca_explained_variance=pca_info,
        pca_components=pca_components,
        used_pca=use_pca,
        k_analysis=None
    )


def get_facies_colors(n_clusters: int) -> List[str]:
    """
    Get visually distinct colors for facies visualization.
    
    Colors are chosen for maximum contrast and petrophysical meaning:
    - Yellow/Gold for sand (reservoir)
    - Gray/Brown for shale (non-reservoir)
    - Intermediate colors for transitional facies
    """
    if n_clusters == 2:
        return ['#FFD700', '#708090']  # Gold (Sand), SlateGray (Shale)
    elif n_clusters == 3:
        return ['#FFD700', '#DAA520', '#708090']  # Gold, Goldenrod, SlateGray
    elif n_clusters == 4:
        return ['#FFD700', '#FFA500', '#8B4513', '#708090']  # Gold, Orange, SaddleBrown, SlateGray
    else:
        # Use a colormap for more clusters
        import matplotlib.cm as cm
        cmap = cm.get_cmap('Spectral')
        return [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                for c in cmap(np.linspace(0.1, 0.9, n_clusters))]


def get_facies_color_by_interpretation(interpretation: str) -> str:
    """Get color based on rock type interpretation."""
    interpretation_lower = interpretation.lower()
    if 'clean sand' in interpretation_lower:
        return '#FFD700'  # Bright Gold
    elif 'shaly sand' in interpretation_lower:
        return '#DAA520'  # Goldenrod
    elif 'sandy shale' in interpretation_lower:
        return '#8B7355'  # Burlywood
    elif 'shale' in interpretation_lower:
        return '#708090'  # Slate Gray
    elif 'sand' in interpretation_lower or 'reservoir' in interpretation_lower:
        return '#FFA500'  # Orange
    else:
        return '#A0A0A0'  # Generic gray
