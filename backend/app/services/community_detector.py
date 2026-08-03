"""
Community Detector
Uses the Leiden algorithm (via leidenalg + igraph) to detect communities
in the knowledge graph. Supports hierarchical multi-level communities.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Detect communities in the knowledge graph using the Leiden algorithm.
    Produces hierarchical community assignments at multiple resolution levels.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
        n_iterations: int = 10,
    ):
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.n_iterations = n_iterations

    def detect(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        resolutions: Optional[List[float]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Run Leiden community detection at one or more resolution levels.

        Args:
            nodes: List of {"node_id": ..., "label": ..., "type": ...}
            edges: List of {"source": ..., "target": ..., "weight": ...}
            resolutions: List of resolution parameters for hierarchical detection.
                         Lower = fewer larger communities, higher = more smaller ones.
                         Default: [0.5, 1.0, 2.0] for 3 hierarchy levels.

        Returns:
            Dict with key "levels" → list of level results, each containing community assignments.
        """
        import igraph as ig
        import leidenalg

        if len(nodes) < self.min_community_size:
            logger.warning(f"Too few nodes ({len(nodes)}) for community detection")
            return {"levels": []}

        if resolutions is None:
            resolutions = [0.5, 1.0, 2.0]

        # Build igraph graph
        node_id_to_idx = {n["node_id"]: i for i, n in enumerate(nodes)}
        g = ig.Graph(directed=True)
        g.add_vertices(len(nodes))

        for i, node in enumerate(nodes):
            g.vs[i]["node_id"] = node["node_id"]
            g.vs[i]["label"] = node.get("label", "")
            g.vs[i]["type"] = node.get("type", "CONCEPT")

        valid_edges = []
        weights = []
        for edge in edges:
            src_idx = node_id_to_idx.get(edge["source"])
            tgt_idx = node_id_to_idx.get(edge["target"])
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                valid_edges.append((src_idx, tgt_idx))
                weights.append(edge.get("weight", 1.0))

        if valid_edges:
            g.add_edges(valid_edges)
            g.es["weight"] = weights

        # Convert to undirected for community detection
        g_undirected = g.as_undirected(mode="collapse", combine_edges={"weight": "max"})

        levels = []
        for level_idx, res in enumerate(resolutions):
            try:
                partition = leidenalg.find_partition(
                    g_undirected,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=res,
                    weights="weight" if g_undirected.ecount() > 0 else None,
                    n_iterations=self.n_iterations,
                    seed=42,
                )

                communities = self._extract_communities(
                    partition, g_undirected, level_idx
                )
                levels.append({
                    "level": level_idx,
                    "resolution": res,
                    "num_communities": len(communities),
                    "communities": communities,
                    "modularity": partition.modularity,
                })

                logger.info(
                    f"Level {level_idx} (res={res}): {len(communities)} communities, "
                    f"modularity={partition.modularity:.4f}"
                )
            except Exception as e:
                logger.error(f"Community detection failed at level {level_idx}: {e}")
                continue

        return {"levels": levels}

    def _extract_communities(
        self, partition, graph, level: int
    ) -> List[Dict]:
        """Extract community information from a Leiden partition."""
        communities = []

        for community_id, members in enumerate(partition):
            if len(members) < self.min_community_size:
                continue

            member_data = []
            for vertex_idx in members:
                member_data.append({
                    "node_id": graph.vs[vertex_idx]["node_id"],
                    "label": graph.vs[vertex_idx]["label"],
                    "type": graph.vs[vertex_idx]["type"],
                })

            # Calculate community density
            subgraph = graph.subgraph(members)
            possible_edges = len(members) * (len(members) - 1) / 2
            density = subgraph.ecount() / possible_edges if possible_edges > 0 else 0

            # Find key entities (highest degree within community)
            degrees = subgraph.degree()
            sorted_members = sorted(
                zip(range(len(members)), degrees),
                key=lambda x: x[1],
                reverse=True,
            )
            key_entities = [
                member_data[idx]["label"]
                for idx, _ in sorted_members[:5]
            ]

            communities.append({
                "community_id": community_id,
                "level": level,
                "member_count": len(members),
                "density": round(density, 4),
                "members": member_data,
                "key_entities": key_entities,
            })

        return communities
