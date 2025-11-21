# clustering/clustering_evaluator.py
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from networkx.algorithms.community import girvan_newman
from data_processing.embedding_operation import EmbeddingMerger
import hdbscan

class ClusteringEvaluator:
    """
    Class to evaluate clustering performance on merged matrices computed from
    embedding similarity and GS relationship adjacency matrices.
    """
    
    @staticmethod
    def evaluate(true_labels: list, pred_labels: list) -> tuple:
        """
        Compute the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
        between true and predicted cluster labels.
        """
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        return ari, nmi

    @staticmethod
    def run_clustering_on_merged_matrix(merged_matrix: np.ndarray, gs_ids: list, true_labels: list, 
                                        algorithm: str = 'louvain', num_clusters: int = 2, plot: bool = False) -> tuple:
        """
        Run a clustering algorithm on a merged matrix and evaluate performance.
        
        Parameters:
            merged_matrix (np.ndarray): The merged matrix (either used as an affinity matrix or feature matrix).
            gs_ids (list): The list of GS IDs in the same order as merged_matrix rows.
            true_labels (list): Ground truth labels corresponding to the GS IDs.
            algorithm (str): Which clustering algorithm to use. Options include:
                - 'louvain'
                - 'girvan_newman'
                - 'spectral_clustering'
                - 'agglomerative_clustering'
                - 'kmeans'
            num_clusters (int): Number of clusters (if applicable).
            plot (bool): Whether to plot the clustering result (if implemented).
        
        Returns:
            tuple: (ARI, NMI)
        """
        merged_matrix = np.nan_to_num(merged_matrix, nan=0.0)

        if algorithm == 'louvain':
            # Convert the merged matrix (affinity) to a graph.
            G = nx.from_numpy_array(merged_matrix)
            # Relabel nodes using gs_ids
            mapping = {i: gs_ids[i] for i in range(len(gs_ids))}
            G = nx.relabel_nodes(G, mapping)
            partition = community_louvain.best_partition(G)
            pred_labels = [partition[gs] for gs in gs_ids]
            title = "Louvain"
        elif algorithm == 'girvan_newman':
            G = nx.from_numpy_array(merged_matrix)
            mapping = {i: gs_ids[i] for i in range(len(gs_ids))}
            G = nx.relabel_nodes(G, mapping)
            communities = next(girvan_newman(G))
            partition = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    partition[node] = idx
            pred_labels = [partition[gs] for gs in gs_ids]
            title = "Girvan-Newman"
        elif algorithm == 'spectral_clustering':
            sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
            pred_labels = sc.fit_predict(merged_matrix)
            title = "Spectral Clustering"
        elif algorithm == 'agglomerative_clustering':
            # Here we treat the merged matrix as features.
            ac = AgglomerativeClustering(n_clusters=num_clusters)
            pred_labels = ac.fit_predict(merged_matrix)
            title = "Agglomerative Clustering"
        elif algorithm == 'kmeans':
            km = KMeans(n_clusters=num_clusters, random_state=42)
            pred_labels = km.fit_predict(merged_matrix)
            title = "KMeans"
        elif algorithm == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
            pred_labels = clusterer.fit_predict(merged_matrix)
            title = "HDBSCAN"
        else:
            raise ValueError("Unknown algorithm option.")

        ari, nmi = ClusteringEvaluator.evaluate(true_labels, pred_labels)
        print(f"{title}: ARI = {ari:.3f}, NMI = {nmi:.3f}")
        # (Optionally, add code here to generate plots.)
        return ari, nmi

    @staticmethod
    def merge_and_evaluate(npz_file: str, pager_file: str, gs_id_list: list, alpha: float, 
                             true_labels: list, clustering_algo: str = 'louvain', use_gpu: bool = False) -> tuple:
        """
        Compute a partial similarity matrix (from embeddings) and an adjacency matrix (from relationship data)
        for a given list of GS IDs, merge them using the specified alpha value, and evaluate clustering performance.
        
        Parameters:
            npz_file (str): Path to NPZ file containing embeddings.
            pager_file (str): Path to GS relationship file.
            gs_id_list (list): List of GS IDs to process.
            alpha (float): Weight for the similarity matrix in merging.
            true_labels (list): Ground truth labels corresponding to the GS IDs.
            clustering_algo (str): Clustering algorithm to use for evaluation.
            use_gpu (bool): Use GPU (if available) for computing similarity.
        
        Returns:
            tuple: (merged_matrix, ARI, NMI)
        """
        # Compute the partial similarity matrix for the provided GS IDs.
        # Here we use MatrixOperations.get_similarity_from_npz() which computes the full similarity
        # for the given subset.
        partial_similarity, found_ids = MatrixOperations.get_similarity_from_npz(npz_file, gs_id_list, use_gpu=use_gpu)
        # Generate the adjacency matrix (ensuring the same order as found_ids)
        adj_matrix = MatrixOperations.generate_adj_matrix(found_ids, pager_file, value_option='SIMILARITY')
        # Merge the two matrices with the given alpha weight.
        merged_matrix = MatrixOperations.merge_matrices(partial_similarity, adj_matrix, alpha)
        print("Merged matrix computed.")
        # Evaluate clustering on the merged matrix.
        ari, nmi = ClusteringEvaluator.run_clustering_on_merged_matrix(merged_matrix, found_ids, true_labels, algorithm=clustering_algo)
        return merged_matrix, ari, nmi
    
    def merge_embeddings_and_evaluate(go_embeddings: str, pager_embeddings: str, gs_id_list: list,
                                      true_labels: list, merge_method: str = "weighted",
                                      alpha: float = 0.5, clustering_algo: str = "louvain",
                                      num_clusters: int = 2, use_gpu: bool = True, cut_dim=256) -> tuple:
        """
        Merge embeddings from two NPZ files (GO embeddings and Pager embeddings) for a set of GS IDs
        (loaded from a text file), then evaluate clustering performance on the merged embeddings.
        
        The merge is done by calling EmbeddingMerger.merge(), which aligns the embeddings for the GS IDs 
        in the text file and merges them using the specified method (weighted or concatenation). The merged
        matrix is then evaluated using run_clustering_on_merged_matrix().
        
        Parameters:
            go_embeddings (str): Path to the NPZ file with GO embeddings.
            pager_embeddings (str): Path to the NPZ file with Pager embeddings.
            gs_id_list (list): The list containing GS IDs.
            true_labels (list): Ground truth labels for the GS IDs.
            merge_method (str): Merge method; either "weighted" or "concatenate".
            alpha (float): Weight for the weighted merge (ignored for concatenation).
            clustering_algo (str): Clustering algorithm to use for evaluation.
            num_clusters (int): Number of clusters (if applicable).
            use_gpu (bool): Whether to use GPU for similarity computation (if needed).
        
        Returns:
            tuple: (merged_matrix, ARI, NMI)
        """
        
        # Use the EmbeddingMerger.merge() function to merge embeddings.
        # It is assumed that EmbeddingMerger.merge() takes parameters:
        #   npz_file1 (GO embeddings), npz_file2 (Pager embeddings), gs_file (the text file),
        #   merge_method, alpha, output_file.
        merged_ids, merged_emb = EmbeddingMerger.merge(
            npz_file1=go_embeddings,
            npz_file2=pager_embeddings,
            gs_id_list=gs_id_list,
            merge_method=merge_method,
            alpha=alpha,
            output_file="merged_embeddings",
            cut_dim=cut_dim
        )
        
        
        indices = [merged_ids.index(node) for node in merged_ids]
        # sub_merged_matrix = merged_emb[np.ix_(indices, indices)]
        
        # Evaluate clustering on the submatrix.
        # ari, nmi = ClusteringEvaluator.run_clustering_on_merged_matrix(
        #     sub_merged_matrix, merged_ids, true_labels,
        #     algorithm=clustering_algo, num_clusters=num_clusters, plot=False
        # )

        ari, nmi = ClusteringEvaluator.run_clustering_on_merged_matrix(
            merged_emb, merged_ids, true_labels,
            algorithm=clustering_algo, num_clusters=num_clusters, plot=False
        )
        print("Merged embeddings evaluated.")
        return merged_emb, ari, nmi