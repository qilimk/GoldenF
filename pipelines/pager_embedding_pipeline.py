# pipelines/pager_embedding_pipeline.py
import numpy as np
import pandas as pd
import networkx as nx
from data_processing.data_loader import DataLoader

class PagerEmbeddingPipeline:
    """
    Pipeline to compute Pager embeddings using Node2Vec.
    """
    def __init__(self, input_file, output_file, weight_column="NLOGPMF",
                 embedding_length=384, walk_length=30, num_walks=200, window=10,
                 workers=4, use_gpu=True):
        self.input_file = input_file
        self.output_file = output_file
        self.weight_column = weight_column
        self.embedding_length = embedding_length
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.workers = workers
        self.use_gpu = use_gpu

    def build_graph(self):
        df = pd.read_csv(self.input_file, delimiter="\t")
        print(f"Loaded {len(df)} rows from {self.input_file}")
        G = nx.Graph()
        for _, row in df.iterrows():
            gs_a = row["GS_A_ID"]
            gs_b = row["GS_B_ID"]
            weight = row[self.weight_column]
            G.add_edge(gs_a, gs_b, weight=weight)
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def compute_embeddings(self):
        G = self.build_graph()
        if self.use_gpu:
            try:
                import torch
                from torch_geometric.nn import Node2Vec
            except ImportError:
                raise ImportError("PyTorch Geometric is required for GPU acceleration.")
            
            node_list = sorted(G.nodes())
            node_to_index = {node: i for i, node in enumerate(node_list)}
            edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges() if u in node_to_index and v in node_to_index]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Node2Vec(
                edge_index,
                embedding_dim=self.embedding_length,
                walk_length=self.walk_length,
                context_size=self.window,
                walks_per_node=self.num_walks,
                p=1, q=1,
                sparse=True
            ).to(device)
            loader = model.loader(batch_size=24, shuffle=True, num_workers=self.workers)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            model.train()
            for epoch in range(1, 51):
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")
            model.eval()
            with torch.no_grad():
                z = model.embedding.weight.cpu().numpy()
            embeddings_dict = {node: z[node_to_index[node]] for node in node_list}
            print(f"Computed GPU embeddings for {len(node_list)} nodes.")
            return embeddings_dict, node_list
        else:
            from node2vec import Node2Vec
            node2vec = Node2Vec(G, dimensions=self.embedding_length,
                                walk_length=self.walk_length, num_walks=self.num_walks,
                                workers=self.workers, weight_key="weight")
            model = node2vec.fit(window=self.window, min_count=1, batch_words=4)
            node_list = sorted(G.nodes())
            embeddings_dict = {node: model.wv[node] for node in G.nodes()}
            print(f"Computed CPU embeddings for {len(node_list)} nodes.")
            return embeddings_dict, node_list

    def save_embeddings(self, embeddings_dict, node_list):
        import numpy as np
        emb_matrix = np.array([embeddings_dict[node] for node in node_list])
        np.savez(self.output_file, ID=node_list, embeddings=emb_matrix)
        print(f"Embeddings saved to {self.output_file}.npz")

    def extract_embeddings(self, gs_file):
        """
        Extract embeddings for GS IDs read from a text file from the saved NPZ file.
        
        Parameters:
            gs_file (str): Path to a text file containing one GS ID per line.
            
        Returns:
            dict: A dictionary mapping each found GS ID to its embedding vector.
        """
        # Read the GS IDs from the text file.
        gs_id_list = DataLoader.load_gs_ids(gs_file)
        
        # Ensure the NPZ file name ends with .npz.
        npz_file = self.output_file if self.output_file.endswith(".npz") else self.output_file + ".npz"
        # ids, embeddings = DataLoader.load_embeddings(npz_file)
        data = np.load(npz_file, allow_pickle=True)
        # Use key 'ID' if available; otherwise assume 'PAGER_ID'
        ids = data['ID'] if 'ID' in data else data['PAGER_ID']
        embeddings = data['embeddings']
        
        # Build a mapping from ID to embedding vector.
        id_to_embedding = {id_: emb for id_, emb in zip(ids, embeddings)}
        
        # Extract embeddings for each GS ID from the file.
        result = {}
        for gs in gs_id_list:
            if gs in id_to_embedding:
                result[gs] = id_to_embedding[gs]
            else:
                print(f"Warning: GS ID {gs} not found in the embedding file.")
        print(len(result), result['GO:0000715'], len(result['GO:0000715']))
        return result

    def run(self):
        embeddings_dict, node_list = self.compute_embeddings()
        self.save_embeddings(embeddings_dict, node_list)

