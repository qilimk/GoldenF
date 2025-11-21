# graph/graph_operations.py
import networkx as nx
import itertools
from collections import deque

# WHOLE_GRAPH_PATH = 'GO2024/2024_GO_wholegraph.graphml'
WHOLE_GRAPH_PATH = 'GO2025/2025_GO_wholegraph.graphml'

class GraphOperations:
    """Static methods for building graphs from files and performing graph filtering."""
    
    @staticmethod
    def load_graph_from_table(file_path: str, sep: str = '\t') -> nx.DiGraph:
        """Load a directed graph from a text file with edge pairs."""
        G = nx.DiGraph()
        with open(file_path, 'r') as f:
            for ln, line in enumerate(f, start=1):
                try:
                    parent, child = line.strip().split(sep)
                    G.add_edge(parent, child)
                except Exception as e:
                    print(f"Line {ln}: Skipping due to error: {e}")
        return G

    @staticmethod
    def save_graph(G: nx.DiGraph, file_path: str):
        """Save the given graph to a GraphML file."""
        nx.write_graphml(G, file_path)

    @staticmethod
    def load_graph(file_path: str) -> nx.DiGraph:
        """Load a GraphML file into a NetworkX graph."""
        return nx.read_graphml(file_path)
    @staticmethod
    def table_2_graph(file_path='GO2024/topdown_2024.txt'):

        G = nx.DiGraph()

        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    parent, child = line.strip().split('\t')
                    G.add_edge(parent, child)

                except ValueError:
                    # Handle the error (e.g., wrong number of values to unpack)
                    print(f"Skipping malformed line {line_number}: {line.strip()}")
                except Exception as e:
                    # Handle other possible errors
                    print(f"Error on line {line_number}: {e}. Skipping line: {line.strip()}")

        nx.write_graphml(G, "2024_GO_wholegraph.graphml")

    @staticmethod
    def table_to_verified_graph(file_path: str, valid_nodes: set, part_name: str, sep: str = '\t'):
        """Create a directed graph from a table, adding only edges whose nodes are in valid_nodes."""
        G = nx.DiGraph()
        with open(file_path, 'r') as f:
            for ln, line in enumerate(f, start=1):
                try:
                    parent, child = line.strip().split(sep)
                    if parent in valid_nodes and child in valid_nodes:
                        if not G.has_edge(parent, child):
                            print(f"Adding edge: {parent} -> {child}")
                            G.add_edge(parent, child)
                    else:
                        print(f"Line {ln}: Skipping, invalid nodes.")
                except Exception as e:
                    print(f"Line {ln}: Skipping due to error: {e}")
        GraphOperations.save_graph(G, f"2025_{part_name}_graph_w_verification.graphml")

    @staticmethod
    def filtered_roots(graph_path: str, lower_bound: int = 10, upper_bound: int = 100) -> list:
        """Return root nodes (in-degree = 0) whose out-degree is between lower_bound and upper_bound."""
        G = GraphOperations.load_graph(graph_path)
        roots = [node for node, d in G.in_degree() if d == 0]
        filtered = []
        for node in roots:
            out_deg = G.out_degree(node)
            if lower_bound < out_deg < upper_bound:
                filtered.append(node)
                print(f"{node}: out-degree {out_deg}")
        return filtered

    @staticmethod
    def extract_subgraph_given_nodes(G: nx.DiGraph, nodes: list):
        """Extract and save a subgraph induced by the given nodes."""
        subG = nx.DiGraph()
        for node in nodes:
            for child in G.successors(node):
                subG.add_edge(node, child)
        GraphOperations.save_graph(subG, f"subgraph_{'_'.join(nodes)}.graphml")

    @staticmethod
    def get_node_levels(G: nx.Graph, root: str, nodes: list) -> dict:
        """Return a dictionary mapping each node in nodes to its level from root."""
        levels = {root: 0}
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            for neighbor in G.neighbors(curr):
                if neighbor not in levels:
                    levels[neighbor] = levels[curr] + 1
                    queue.append(neighbor)
        return {node: levels.get(node) for node in nodes}

    @staticmethod
    def parents_info(G: nx.DiGraph, nodes: list) -> dict:
        """Return a dictionary mapping each node to its parent (assuming one parent per node)."""
        parents = {}
        for parent, child in G.edges():
            if child not in parents:
                parents[child] = parent
        return {node: parents.get(node) for node in nodes}

    @staticmethod
    def nodes_with_unique_parents(G: nx.DiGraph, nodes: list) -> bool:
        """Return True if all nodes have different parents."""
        parents = GraphOperations.parents_info(G, nodes)
        return len({p for p in parents.values() if p is not None}) == len(nodes)

    @staticmethod
    def filter_experiment_candidates(target_graph_path: str, root: str, group_size: int = 2) -> list:
        """
        Filter candidate groups from a graph based on outdegree and parent uniqueness.
        Returns a list of tuples where each tuple is a candidate group.
        """
        G_whole = GraphOperations.load_graph(WHOLE_GRAPH_PATH)
        filtered = GraphOperations.filtered_roots(target_graph_path)
        levels = GraphOperations.get_node_levels(G_whole, root, filtered)
        groups = []
        level_groups = {}
        for node, lvl in levels.items():
            level_groups.setdefault(lvl, []).append(node)
        for nodes in level_groups.values():
            for group in itertools.combinations(nodes, group_size):
                if GraphOperations.nodes_with_unique_parents(G_whole, group):
                    groups.append(group)
                else:
                    print(f"Group {group} share the same parent.")
        print(f"Total candidate groups: {len(groups)}")
        return groups
    
    @staticmethod
    def get_all_gs_ids_list(filepath):
        """
        Reads a tab-delimited file and returns a list of all unique GS_A_ID and GS_B_ID values.

        Args:
            filepath (str): Path to the input text file.

        Returns:
            List[str]: List of unique GS_A_ID and GS_B_ID values.
        """
        gs_ids = set()

        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()  # skip header
            for line in f:
                if line.strip():  # skip empty lines
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        gs_ids.add(parts[0])
                        gs_ids.add(parts[1])

        return list(gs_ids)
    

