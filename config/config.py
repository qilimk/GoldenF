# config/config.py

CONFIG = {
     "paths": {
     "WHOLE_GRAPH": "GO2025/2025_GO_wholegraph.graphml",
     # "BP_GRAPH": "GO2025/2025_biological_process_graph_w_verification.graphml",
     "BP_GRAPH": "GO2025/2025_cross_filtered_graph_w_verification.graphml", 
     "META_DATA": "GO2025/GO_info_2025.txt",
     "GO_EMBEDDINGS": "GO2024/go_embeddings.npz",
     "PAGER_EMBEDDINGS": "GO2025/node2vec_model_biological_process",
     "PAGER_FILE": "GO2025/m_type_biological_process_2025.txt",
     "SIMILARITY_FILE": "GO2025/2024_cosine_similarity_matrix.npy",
     "MERGE_OUTPUT": "GO2025/2024_merged_embeddings.csv"
     }

#     "parameters": {
#          "model": "default",
#          "use_gpu": True,
#          "alpha": 0.6,
#          "num_clusters": 2,
#          "num_groups": 2
#     },
#     "visualization": {
#          "label_fontsize": 24,
#          "font_size": 20,
#          "annot_font_size": 16,
#          "figsize": (30, 10)
#     }
}

