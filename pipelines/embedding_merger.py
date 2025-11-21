# pipelines/embedding_merger.py
import numpy as np
from data_processing.data_loader import DataLoader

class EmbeddingMerger:
    @staticmethod
    def merge(npz_file1: str, npz_file2: str, gs_id_list: list, merge_method: str = "weighted", 
              alpha: float = 0.5, output_file: str = "merged_embeddings.npz", if_save: bool = False):
        ids1, emb1 = DataLoader.load_embeddings(npz_file1)
        ids2, emb2 = DataLoader.load_embeddings(npz_file2)
        ids1 = list(ids1)
        ids2 = list(ids2)
        common_ids = sorted(set(ids1) & set(ids2) & set(gs_id_list))
        if not common_ids:
            raise ValueError("No common GS IDs found.")
        missing_in_file1 = set(gs_id_list) - set(ids1)
        missing_in_file2 = set(gs_id_list) - set(ids2)
        if missing_in_file1:
            print(f"Warning: IDs missing in file1: {missing_in_file1}")
        if missing_in_file2:
            print(f"Warning: IDs missing in file2: {missing_in_file2}")
        idx1 = [ids1.index(gs) for gs in common_ids]
        idx2 = [ids2.index(gs) for gs in common_ids]
        emb1_aligned = emb1[idx1]
        emb2_aligned = emb2[idx2]
        if merge_method == "weighted":
            if emb1_aligned.shape != emb2_aligned.shape:
                raise ValueError("For weighted merging, shapes must be equal.")
            merged_embeddings = alpha * emb1_aligned + (1 - alpha) * emb2_aligned
        elif merge_method == "concatenate":
            merged_embeddings = np.concatenate((alpha * emb1_aligned, (1 - alpha) * emb2_aligned), axis=1)
        else:
            raise ValueError("merge_method must be 'weighted' or 'concatenate'.")
        if if_save:
            np.savez(output_file, ID=common_ids, embeddings=merged_embeddings)
        print(f"Merged embeddings saved to {output_file}.npz")
        return common_ids, merged_embeddings

