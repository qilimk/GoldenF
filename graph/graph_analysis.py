# graph/graph_analysis.py
import networkx as nx
from collections import deque

class GraphAnalysis:
    """
    Class for traversing a directed graph from a given root, computing node levels,
    finding common ancestors, and extracting connected subgraphs.
    """
    def __init__(self, G: nx.DiGraph, root: str):
        self.G = G
        self.root = root
        self.parent = {root: None}  # Parent of each node
        self.level = {root: 0}      # Level of each node
        self._bfs()                 # Populate parent and level data

    def _bfs(self):
        """Perform BFS from the root to record parent and level information."""
        queue = deque([self.root])
        while queue:
            current = queue.popleft()
            for neighbor in self.G.successors(current):
                if neighbor not in self.parent:
                    self.parent[neighbor] = current
                    self.level[neighbor] = self.level[current] + 1
                    queue.append(neighbor)

    def check_same_parent(self, node1: str, node2: str):
        """Return a tuple (bool, level-info) if both nodes share the same parent."""
        if node1 not in self.parent or node2 not in self.parent:
            return None, None
        same = self.parent[node1] == self.parent[node2]
        return (same, self.level[node1] if same else (self.level[node1], self.level[node2]))

    def find_lowest_nodes_and_common_parent(self, nodes: list):
        """Find nodes at the lowest level and determine their common ancestor (if any)."""
        valid_levels = [self.level[node] for node in nodes if node in self.level]
        lowest_level = min(valid_levels)
        highest_level = max(valid_levels)
        lowest_nodes = [node for node in nodes if self.level[node] == lowest_level]
        if len(lowest_nodes) == 1:
            common_parent = lowest_nodes[0]
        else:
            common_parent = nx.lowest_common_ancestor(self.G, lowest_nodes[0], lowest_nodes[1])
            for node in lowest_nodes[2:]:
                common_parent = nx.lowest_common_ancestor(self.G, common_parent, node)
        return lowest_nodes, common_parent, highest_level

    def create_subgraph(self, nodes: set) -> nx.DiGraph:
        """Return a subgraph induced by the given set of nodes."""
        return self.G.subgraph(nodes)

    def add_intermediate_nodes(self, nodes_list: list) -> set:
        """
        Given a list of nodes, add intermediate nodes on the shortest paths between them
        so that the resulting set is connected.
        """
        lowest_nodes, _, _ = self.find_lowest_nodes_and_common_parent(nodes_list)
        all_nodes = set(nodes_list)
        for i, start in enumerate(nodes_list):
            for end in nodes_list[i+1:]:
                if nx.has_path(self.G, start, end):
                    all_nodes.update(nx.shortest_path(self.G, source=start, target=end))
        return all_nodes

    def descendants_to_level(self, start_node: str, max_depth: int) -> set:
        """Return all descendants of start_node up to a given depth (max_depth)."""
        depths = nx.single_source_shortest_path_length(self.G, start_node)
        return {node for node, depth in depths.items() if depth <= max_depth}

    def find_groups_with_same_m_level_ancestor(self, subtree_root: str, m: int, save_path: str = None):
        """
        Find node groups under a given subtree that have different parents but share
        the same ancestor exactly m levels above them. Optionally save the result to a file.

        Args:
            subtree_root (str): Node used as the root of the subtree.
            m (int): Number of levels above to consider as common ancestor.
            save_path (str, optional): Path to save the results as a text or CSV file.

        Returns:
            List of tuples: [((parent1, children1), (parent2, children2), ancestor), ...]
        """
        from collections import defaultdict

        if subtree_root not in self.G:
            return []

        # Step 1: Get all nodes in the subtree
        subtree_nodes = nx.descendants(self.G, subtree_root) | {subtree_root}

        # Step 2: For each node, trace its m-level ancestor
        ancestor_map = defaultdict(lambda: defaultdict(list))  # {ancestor: {parent: [children]}}

        for node in subtree_nodes:
            parent = self.parent.get(node)
            if parent is None:
                continue

            # Walk up m levels to find ancestor
            current = node
            steps = 0
            while steps < m and current in self.parent:
                current = self.parent[current]
                if current is None:
                    break
                steps += 1

            if steps == m and current in subtree_nodes:
                ancestor = current
                ancestor_map[ancestor][parent].append(node)

        # Step 3: Find groups with same ancestor but different parents
        result = []
        for ancestor, parent_dict in ancestor_map.items():
            parents = list(parent_dict.keys())
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1, p2 = parents[i], parents[j]
                    group1 = (p1, parent_dict[p1])
                    group2 = (p2, parent_dict[p2])
                    result.append((group1, group2, ancestor))

        # Step 4: Save to file if path is provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Ancestor,Parent1,Children1,Parent2,Children2\n")
                for (p1, c1), (p2, c2), anc in result:
                    line = f"{anc},{p1},{'|'.join(c1)},{p2},{'|'.join(c2)}\n"
                    f.write(line)

        return result

