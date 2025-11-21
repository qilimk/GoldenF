import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import hdbscan

# If you use Louvain:
#   pip install python-louvain
# and then:
import community as community_louvain  # a.k.a. python-louvain
from data_processing.embedding_operation import EmbeddingMerger


# ======================
# 0) Argument parsing
# ======================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict clusters from merged GO/PAGER embeddings")
    parser.add_argument('--num', type=int, default=2, help='Number of clusters (for k-means / agglomerative / spectral)')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha weight for weighted merge (0..1)')
    parser.add_argument('--pager_embedding_file', type=str, default="GO2025/node2vec_model_biological_process_256",
                        help="Path to PAGER (graph) embeddings NPZ file")
    parser.add_argument('--go_embedding_file', type=str, default="GO2025/openai_go_2025_embeddings.npz",
                        help="Path to GO (description) embeddings NPZ file")
    parser.add_argument('--gpu', action="store_true", help='Use GPU (if your merger supports it)')
    parser.add_argument('--gs_file', type=str, default=None,
                        help='Path to a text file with GS IDs (one per line) [REQUIRED]')
    parser.add_argument('--save_csv', action="store_true", help='Save merged matrix and labels as CSV')
    parser.add_argument('--merge_output', type=str, default="demo_test.csv",
                        help='CSV file to save merged embeddings (used with --save_csv)')
    # New, commonly needed knobs:
    parser.add_argument('--merge_method', type=str, default='concatenate',
                        choices=['weighted', 'concatenate'],
                        help='Merge method for embeddings')
    parser.add_argument('--clustering_algo', type=str, default='agglomerative_clustering',
                        choices=['kmeans', 'agglomerative_clustering', 'spectral_clustering',
                                 'louvain', 'girvan_newman', 'hdbscan'],
                        help='Clustering algorithm')
    parser.add_argument('--cut_dim', type=int, default=512,
                        help='Target dimension after merge/projection (if applicable)')
    parser.add_argument('--labels_output', type=str, default='predicted_labels.csv',
                        help='CSV to save predicted labels (used with --save_csv)')
    return parser.parse_args()


# ======================
# 1) Config container
# ======================

@dataclass
class ClusterConfig:
    merge_method: str = "weighted"      # {"weighted", "concatenate"}
    alpha: float = 0.6
    clustering_algo: str = "agglomerative_clustering"
    num_clusters: int = 2
    cut_dim: int = 256
    use_gpu: bool = False


# =========================
# 2) Embedding preparation
# =========================

def load_gs_ids(gs_file: str) -> List[str]:
    if gs_file is None:
        raise ValueError("A GS ID file is required. Provide --gs_file path (one GS ID per line).")
    with open(gs_file, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No GS IDs found in {gs_file}.")
    return ids


def merge_embeddings(
    go_embeddings: str,
    pager_embeddings: str,
    gs_id_list: List[str],
    merge_method: str,
    alpha: float,
    cut_dim: int,
    use_gpu: bool = False
) -> Tuple[List[str], np.ndarray]:
    """
    Align + merge GO and PAGER embeddings for the requested GS IDs.
    Delegates to your existing EmbeddingMerger.merge(...) implementation.
    Returns (merged_ids, merged_emb) in the aligned order.
    """
    merged_ids, merged_emb = EmbeddingMerger.merge(
        npz_file1=go_embeddings,
        npz_file2=pager_embeddings,
        gs_id_list=gs_id_list,
        merge_method=merge_method,
        alpha=alpha,
        output_file="merged_embeddings",
        cut_dim=cut_dim
    )
    return merged_ids, merged_emb


def clean_matrix(mat: np.ndarray, make_symmetric_for_graph: bool = True) -> np.ndarray:
    """
    Replace NaNs and (optionally) enforce symmetry for graph-based methods.
    """
    mat = np.nan_to_num(mat, nan=0.0)
    if make_symmetric_for_graph:
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
    return mat


# ==========================
# 3) Clustering back-ends
# ==========================

def louvain_clustering(affinity: np.ndarray, ids: List[str]) -> List[int]:
    G = nx.from_numpy_array(affinity)
    G = nx.relabel_nodes(G, {i: ids[i] for i in range(len(ids))})
    partition = community_louvain.best_partition(G)
    return [partition[g] for g in ids]


def girvan_newman_clustering(affinity: np.ndarray, ids: List[str]) -> List[int]:
    G = nx.from_numpy_array(affinity)
    G = nx.relabel_nodes(G, {i: ids[i] for i in range(len(ids))})
    communities = next(nx.algorithms.community.girvan_newman(G))
    label_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            label_map[node] = idx
    return [label_map[g] for g in ids]


def spectral_clustering_affinity(affinity: np.ndarray, k: int) -> List[int]:
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    return sc.fit_predict(affinity).tolist()


def agglomerative_clustering_features(X: np.ndarray, k: int) -> List[int]:
    ac = AgglomerativeClustering(n_clusters=k)
    return ac.fit_predict(X).tolist()


def kmeans_clustering_features(X: np.ndarray, k: int) -> List[int]:
    km = KMeans(n_clusters=k, random_state=42)
    return km.fit_predict(X).tolist()


def hdbscan_clustering_features(X: np.ndarray) -> List[int]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    return clusterer.fit_predict(X).tolist()  # noise labeled as -1


def run_clustering(
    merged_emb: np.ndarray,
    merged_ids: List[str],
    algo: str,
    num_clusters: int
) -> List[int]:
    """
    Dispatch to the requested clustering back-end.
    NOTE:
      - Louvain / Girvan–Newman / Spectral expect an AFFINITY (precomputed similarity).
      - Agglomerative / KMeans / HDBSCAN treat rows as FEATURES.
    """
    algo = algo.lower()

    if algo == "louvain":
        affinity = clean_matrix(merged_emb, make_symmetric_for_graph=True)
        return louvain_clustering(affinity, merged_ids)

    elif algo == "girvan_newman":
        affinity = clean_matrix(merged_emb, make_symmetric_for_graph=True)
        return girvan_newman_clustering(affinity, merged_ids)

    elif algo == "spectral_clustering":
        affinity = clean_matrix(merged_emb, make_symmetric_for_graph=True)
        return spectral_clustering_affinity(affinity, num_clusters)

    elif algo == "agglomerative_clustering":
        X = clean_matrix(merged_emb, make_symmetric_for_graph=False)
        return agglomerative_clustering_features(X, num_clusters)

    elif algo == "kmeans":
        X = clean_matrix(merged_emb, make_symmetric_for_graph=False)
        return kmeans_clustering_features(X, num_clusters)

    elif algo == "hdbscan":
        X = clean_matrix(merged_emb, make_symmetric_for_graph=False)
        return hdbscan_clustering_features(X)

    else:
        raise ValueError(f"Unknown clustering_algo: {algo}")


# ==========================
# 4) Saving utilities
# ==========================

def save_merged_matrix_csv(path: str, ids: List[str], mat: np.ndarray) -> None:
    """
    Save merged embeddings as CSV: first column is GS ID, remaining columns are vector values.
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["gs_id"] + [f"dim_{i}" for i in range(mat.shape[1])]
        writer.writerow(header)
        for gid, row in zip(ids, mat):
            writer.writerow([gid] + list(map(float, row)))


def save_labels_csv(path: str, ids: List[str], labels: List[int]) -> None:
    """
    Save predicted labels as CSV: columns = gs_id, pred_label.
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["gs_id", "pred_label"])
        for gid, lab in zip(ids, labels):
            writer.writerow([gid, int(lab)])


# ==========================
# 5) Public entry point
# ==========================

def predict_clusters(
    go_embeddings: str,
    pager_embeddings: str,
    gs_id_list: List[str],
    merge_method: str = "weighted",
    alpha: float = 0.6,
    clustering_algo: str = "agglomerative_clustering",
    num_clusters: int = 2,
    cut_dim: int = 256,
    use_gpu: bool = False
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    High-level wrapper: merge embeddings, run clustering, return labels + aligned IDs + merged matrix.
    """
    merged_ids, merged_emb = merge_embeddings(
        go_embeddings=go_embeddings,
        pager_embeddings=pager_embeddings,
        gs_id_list=gs_id_list,
        merge_method=merge_method,
        alpha=alpha,
        cut_dim=cut_dim,
        use_gpu=use_gpu
    )
    pred_labels = run_clustering(
        merged_emb=merged_emb,
        merged_ids=merged_ids,
        algo=clustering_algo,
        num_clusters=num_clusters
    )
    return pred_labels, merged_ids, merged_emb


# ==========================
# 6) Main
# ==========================

def main():
    args = parse_arguments()

    # Load GS IDs
    gs_ids = load_gs_ids(args.gs_file)

    # Config (handy if you later pass this around)
    cfg = ClusterConfig(
        merge_method=args.merge_method,
        alpha=args.alpha,
        clustering_algo=args.clustering_algo,
        num_clusters=args.num,
        cut_dim=args.cut_dim,
        use_gpu=args.gpu
    )

    # Predict clusters
    pred_labels, merged_ids, merged_emb = predict_clusters(
        go_embeddings=args.go_embedding_file,
        pager_embeddings=args.pager_embedding_file,
        gs_id_list=gs_ids,
        merge_method=cfg.merge_method,
        alpha=cfg.alpha,
        clustering_algo=cfg.clustering_algo,
        num_clusters=cfg.num_clusters,
        cut_dim=cfg.cut_dim,
        use_gpu=cfg.use_gpu
    )

    # Report summary
    print(f"[OK] Clustering done with algo='{cfg.clustering_algo}', k={cfg.num_clusters}")
    print(f"     Merged {len(merged_ids)} GS IDs; example: {merged_ids[:3]}")
    print(f"     Labels example: {pred_labels[:10]}")

    # Optional saving
    if args.save_csv:
        save_merged_matrix_csv(args.merge_output, merged_ids, merged_emb)
        save_labels_csv(args.labels_output, merged_ids, pred_labels)
        print(f"[Saved] Merged embeddings -> {args.merge_output}")
        print(f"[Saved] Predicted labels  -> {args.labels_output}")


if __name__ == "__main__":
    main()

