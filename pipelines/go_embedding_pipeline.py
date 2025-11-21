# pipelines/go_embedding_pipeline.py
import numpy as np
from data_processing.data_loader import DataLoader
from data_processing.encoder import Encoder

class GOEmbeddingPipeline:
    """
    Pipeline to compute and save embeddings for GO nodes.
    """
    def __init__(self, file_path, output_file, model_choice='default', use_gpu=True):
        self.file_path = file_path
        self.output_file = output_file  # Base name; NPZ file will be output_file+".npz"
        self.encoder = Encoder(model_name=model_choice, use_gpu=use_gpu)

    def compute_and_save_embeddings(self):
        df = DataLoader.load_data(self.file_path)
        goids = df['GOID'].tolist()
        descriptions = df['DESCRIPTION'].tolist()
        embeddings = self.encoder.encode(descriptions)
        embeddings_dict = {goid: emb for goid, emb in zip(goids, embeddings)}
        node_list = sorted(goids)
        self.save_embeddings(embeddings_dict, node_list)

    def save_embeddings(self, embeddings_dict, node_list):
        emb_matrix = np.array([embeddings_dict[node] for node in node_list])
        np.savez(self.output_file, ID=node_list, embeddings=emb_matrix)
        print(f"Embeddings saved to {self.output_file}.npz")

    def extract_embeddings(self, gs_file):
        from data_processing.data_loader import DataLoader
        gs_id_list = DataLoader.load_gs_ids(gs_file)
        npz_file = self.output_file if self.output_file.endswith(".npz") else self.output_file + ".npz"
        ids, embeddings = DataLoader.load_embeddings(npz_file)
        id_to_embedding = {id_: emb for id_, emb in zip(ids, embeddings)}
        result = {}
        for gs in gs_id_list:
            if gs in id_to_embedding:
                result[gs] = id_to_embedding[gs]
            else:
                print(f"Warning: GS ID {gs} not found in the embedding file.")
        return result
    
    def run(self):
        self.compute_and_save_embeddings()
        print("GO embedding pipeline finished.")

