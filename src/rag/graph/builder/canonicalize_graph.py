"""
Graph Canonicalization Pipeline: Entity deduplication and merging for knowledge graphs.

This module provides sophisticated entity canonicalization capabilities for knowledge graphs
built from academic papers. It uses a combination of similarity clustering and LLM-powered
decision making to identify and merge duplicate or similar entities while preserving
relationships and maintaining graph integrity.

The canonicalization process works through these stages:
1. Similarity-based clustering of entities within each type
2. LLM-powered merge decisions for candidate clusters
3. Relationship preservation and updating during merges
4. Iterative refinement until convergence

Key features:
- Constraint-based clustering (prevents merging entities from same paper)
- Dynamic threshold adjustment to control cluster sizes
- LLM-guided intelligent merging decisions
- Comprehensive relationship preservation
- Configurable stopping conditions

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import pickle
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import rag.settings as settings
from rag.graph.builder.prompts import get_entity_canonicalization_prompt
from rag.graph.core.systematics_graph import SystematicsGraph, Entity, Relationship


class GraphCanonicalizer:
    """
    Manages the canonicalization process for knowledge graph entities with async parallelization.
    """
    
    _llm = None
    _semaphore = None
    _embedding_model = None

    @classmethod
    def _get_llm(cls):
        """
        Get the pointer to the OpenAI llm
        """
        if cls._llm is None:
            model = settings.CANONICALIZE_CONFIG["model"]

            from llama_index.llms.openai import OpenAI
            cls._llm = OpenAI(temperature=0, model=model)
        
        return cls._llm

    @classmethod
    def _get_semaphore(cls):
        """
        Get the pointer to the semaphore for preventing too many simultaneous API calls
        """
        if cls._semaphore is None:
            max_threads = settings.CANONICALIZE_CONFIG["max_threads"]

            from asyncio import Semaphore
            cls._semaphore = Semaphore(max_threads)
        
        return cls._semaphore
    
    @classmethod
    def _get_embedding_model(cls):
        """
        Get the shared embedding model, loading it if necessary.
        """
        if cls._embedding_model is None:
            model = settings.EMBEDDING_CONFIG["model"]
            device = settings.EMBEDDING_CONFIG["device"]

            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model} on {device}")
            cls._embedding_model = SentenceTransformer(model, device=device)
        
        return cls._embedding_model


    def __init__(self):
        """Initialize the canonicalizer with configured LLM and concurrency limits."""

        self.llm = self._get_llm()
        self.semaphore = self._get_semaphore()
        self.embedding_model = self._get_embedding_model()
        
        # Semaphore to limit concurrent LLM calls to prevent rate limiting
        

    async def canonicalize_type_async(
        self,
        graph: SystematicsGraph,
        entity_type: str,
        stop_ratio: float = 0.1,
        max_cluster_size: int = 100,
    ) -> SystematicsGraph:
        """
        Canonicalize entities of a specific type using iterative clustering and parallel LLM merging.
        """

        if entity_type not in graph.entity_types:
            print(f"[canonicalize] No entities of type '{entity_type}' found.")
            return graph

        iteration = 1
        total_merges = 0

        while True:
            entities = list(graph.entity_types[entity_type])

            print(f"\n=== ITERATION {iteration} | {len(entities)} entities of type '{entity_type}' ===")

            # Cluster entities based on similarity with size constraints
            clusters = self._cluster_entities(
                entities=entities,
                max_cluster_size=max_cluster_size,
            )

            # Only consider clusters with multiple entities
            clusters = [cluster for cluster in clusters if len(cluster) > 1]
            
            if not clusters:
                print("[canonicalize] No clusters. Stopping.")
                break

            # Process all clusters concurrently
            iteration_merges, removed_entities = await self._merge_clusters(clusters, graph)

            total_merges += iteration_merges

            # Check stopping condition: too few merges relative to entity count
            if (len(entities) == 0 or 
                len(removed_entities) < stop_ratio * len(entities)):
                print("[canonicalize] Too few merges this round. Stopping.")
                break

            iteration += 1

    async def _merge_clusters(self, clusters: List[List[Entity]], graph: SystematicsGraph):
        """
        Process all clusters
        """
        # Create tasks for each cluster
        cluster_tasks = []
        for cluster_idx, cluster in enumerate(clusters, start=1):
            task = self._process_single_cluster(cluster)
            cluster_tasks.append(task)
        
        print(f"Processing {len(cluster_tasks)} clusters concurrently...")

        # Wait for all clusters to be processed
        cluster_results = await asyncio.gather(*cluster_tasks, return_exceptions=True)

        # Collect results and apply merges
        iteration_merges = 0
        removed_entities = set()

        for cluster_idx, result in enumerate(cluster_results, start=1):
            if isinstance(result, Exception):
                print(f"[ERROR] Cluster {cluster_idx} failed: {result}")
                continue

            cluster_entities, merge_decisions = result
            
            if merge_decisions:
                print(f"[parallel] Cluster {cluster_idx}: {len(merge_decisions)} merge decisions")

            # Apply each merge decision
            for decision in merge_decisions:
                try:
                    merged_entity = decision["merged_entity"]
                    source_indices = decision["source_indices"]
                    
                    if len(source_indices) < 2:
                        continue  # Not a real merge
                    
                    # Map local cluster indices to actual entities
                    source_entities = [cluster_entities[i] for i in source_indices]

                    print(f"  Merging {len(source_entities)} -> {merged_entity.name}")

                    # Remove old entities
                    for entity in source_entities:
                        graph.remove_entity(entity)
                        removed_entities.add(entity)

                    # Add the new merged entity to the graph
                    graph.add_entity(merged_entity)

                    # Update all relationships involving the merged entities
                    self._update_relationships_for_merge(graph, source_entities, merged_entity)
                    
                    iteration_merges += 1
                
                except Exception as e:
                    print(f"[WARN] Error with llm merge decision: {e}\{decision}")
                    continue

        # Make sure all the relationships are pointing to the right place
        self._fix_relationship_pointers(graph)
        
        return iteration_merges, removed_entities

    async def _process_single_cluster(self, cluster_entities: List[Entity]):
        """
        Process a single cluster with LLM merge decisions (with semaphore rate limiting).
        """
        # Use semaphore to limit concurrent LLM calls
        async with self.semaphore:
            # Get LLM decisions on what should be merged
            merge_decisions = await self._get_llm_merge_decisions(cluster_entities)
            
            return cluster_entities, merge_decisions

    async def _get_llm_merge_decisions(self, cluster_entities: List[Entity]):
        """
        Use LLM to make intelligent merge decisions for a cluster of similar entities.
        Rate-limited by the semaphore in the calling function.
        """
        entity_type = cluster_entities[0].type
        instructions = settings.CANONICALIZE_CONFIG["type_instructions"].get(entity_type, "")
        prompt, default = get_entity_canonicalization_prompt(cluster_entities, instructions)

        try:
            response = (await self.llm.acomplete(prompt)).text
            return self._parse_llm_response(response, cluster_entities)
            
        except Exception as e:
            print(f"[WARN] LLM merge decision failed: {e}")
            return default

    def canonicalize_type(
        self,
        graph: SystematicsGraph,
        entity_type: str,
        stop_ratio: float = 0.1,
        max_cluster_size: int = 100,
    ) -> SystematicsGraph:
        """Synchronous wrapper for the async canonicalization process."""
        return asyncio.run(
            self.canonicalize_type_async(
                graph, entity_type, stop_ratio, max_cluster_size
            )
        )

    def _cluster_entities(self, entities: List[Entity], max_cluster_size: int) -> List[List[Entity]]:
        """Cluster entities using semantic similarity with constraint enforcement."""
        # Build semantic representation combining name and description
        texts = [f"{entity.name} {entity.description}" for entity in entities]

        embeddings = self.embedding_model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)

        # Apply constraints: entities from same paper should never be merged
        self._apply_same_paper_constraints(similarity_matrix, entities)

        # Find optimal clustering threshold to respect max_cluster_size
        cluster_idxs = self._find_optimal_clusters(similarity_matrix, max_cluster_size)
        clusters = [[entities[idx] for idx in cluster] for cluster in cluster_idxs]
        
        return clusters

    def _apply_same_paper_constraints(self, similarity_matrix: np.ndarray, entities: List[Entity]):
        """Apply constraints to prevent merging entities from the same source paper."""
        n_entities = len(entities)
        for i in range(n_entities):
            for j in range(n_entities):
                if i == j:
                    continue
                
                # Check if entities are from the same paper
                arxiv_id_i = entities[i].attributes.get("arxiv_id")
                arxiv_id_j = entities[j].attributes.get("arxiv_id")
                
                if arxiv_id_i and arxiv_id_j and arxiv_id_i == arxiv_id_j:
                    similarity_matrix[i, j] = 0.0

    def _find_optimal_clusters(self, similarity_matrix: np.ndarray, max_cluster_size: int) -> List[List[int]]:
        """Find clustering with optimal threshold to respect size constraints."""
        n_entities = similarity_matrix.shape[0]
        
        # Scan thresholds to find clustering that respects max_cluster_size
        for threshold in np.arange(0.0, 1.0, 0.005):
            distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - threshold,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

            # Group entities by cluster label
            clusters_dict: Dict[int, List[int]] = {}
            for entity_idx, cluster_label in enumerate(labels):
                clusters_dict.setdefault(cluster_label, []).append(entity_idx)
            
            cluster_groups = list(clusters_dict.values())
            max_size = max(len(group) for group in cluster_groups)
            
            if max_size <= max_cluster_size:
                return cluster_groups

        # Fallback: each entity in its own cluster
        return [[i] for i in range(n_entities)]

    def _parse_llm_response(self, response: str, cluster_entities: List[Entity]) -> List[Dict[str, Any]]:
        """Parse the LLM response to extract merge decisions."""
        try:
            # Extract the Python list from the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end <= start:
                return []

            raw_list = response[start:end]
            merge_decisions = eval(raw_list.replace("```python", "").replace("```", ""))

            results = []
            for decision in merge_decisions:
                merged_data = decision.get("merged_entity", {})
                source_indices_1based = decision.get("source_indices", [])

                # Convert to 0-based indices
                source_indices_0based = [
                    max(0, int(i) - 1) for i in source_indices_1based 
                    if isinstance(i, int) and i > 0
                ]
                
                if len(source_indices_0based) < 2:
                    continue  # Not a real merge

                # Create merged entity
                merged_entity = Entity(
                    name=merged_data.get("name", cluster_entities[source_indices_0based[0]].name),
                    type=cluster_entities[0].type,
                    description=merged_data.get("description", ""),
                    attributes=merged_data.get("attributes", {}),
                )

                results.append({
                    "merged_entity": merged_entity,
                    "source_indices": source_indices_0based,
                })

            return results

        except Exception as e:
            print(f"[WARN] Failed to parse LLM response: {e}")
            return []

    def _update_relationships_for_merge(self, graph: SystematicsGraph, source_entities: List[Entity], merged_entity: Entity):
        """Update all relationships to reference the merged entity instead of source entities."""
        
        for relationship in list(graph.relationships):

            # Points entities within a relationship away from the old source entities towards the newly merged entity
            if relationship.source in source_entities and relationship.target in source_entities:
                # These two entities were merged into each other, just remove the relationship fully
                graph.remove_relationship(relationship)
            elif relationship.source in source_entities and relationship.target not in source_entities:
                # The source entity was merged, keep the target
                graph.remove_relationship(relationship)
                graph.add_relationship(Relationship(source=merged_entity, type=relationship.type, target=relationship.target, attributes=relationship.attributes))
            elif relationship.source not in source_entities and relationship.target in source_entities:
                # The target entity was merged, keep the source
                graph.remove_relationship(relationship)
                graph.add_relationship(Relationship(source=relationship.source, type=relationship.type, target=merged_entity, attributes=relationship.attributes))
            else:
                # Neither was merged, do nothing
                pass

    def _fix_relationship_pointers(self, graph: SystematicsGraph):
        """ 
        Handles the rare occurence of entities within relationships getting disconnected from the graph, because they were overwritten by an entity of the same name
        Points the relationship back to the correct entity within the graph
        """

        for relationship in list(graph.relationships):
            if relationship.source.name in graph.entities.keys() and relationship.source not in graph.entities.values():
                relationship.source = graph.entities[relationship.source.name]
            if relationship.target.name in graph.entities.keys() and relationship.target not in graph.entities.values():
                relationship.target = graph.entities[relationship.target.name]

    

def load_graph_from_graph_dicts() -> SystematicsGraph:
    """
    Load all cached graph dict structures from the cache directory and adds them to a single SystematicsGraph
    """
    cache_dir = Path(settings.GRAPH_RAG_CACHE) / "papers"
    graph_dicts = {}
    
    if not cache_dir.exists():
        raise(Exception("The graph dict cache directory does not exist. Did you run extract_graph_dicts.py?"))

    for pkl_path in cache_dir.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                graph_dicts[pkl_path.stem] = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_path.name}: {e}")
    
    if not graph_dicts:
        raise(Exception("No graph dicts found in the cache. Did you run extract_graph_dicts.py?"))
    
    graph = SystematicsGraph()
    
    for arxiv_id, graph_dict in graph_dicts.items():
        try:
            graph.load_graph_dict(arxiv_id, graph_dict)
        except Exception as e:
            print(f"[WARN] Failed to load graph {arxiv_id}: {e}")
    
    return graph

def get_output_path():
    output_dir = Path(settings.GRAPH_RAG_CACHE)
    output_path = output_dir / "systematics_graph.pkl"
    return output_path

def save_canonicalized_graph(graph: SystematicsGraph, output_path: Path) -> None:
    """
    Save the canonicalized graph to the standard output location.
    
    Respects the overwrite configuration setting unless force_overwrite is True.
    """
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved canonicalized SystematicsGraph to {output_path}")

def main() -> None:
    """
    Main canonicalization pipeline entry point.
    
    Orchestrates the complete process: loading graphs, building consolidated graph,
    running canonicalization for each entity type, and saving results.
    """

    output_path = get_output_path()

    if output_path.exists() and not settings.CANONICALIZE_CONFIG.get("overwrite_existing", False):
        print(f"Canonicalized graph already exists at {output_path}. "
              f"Set CANONICALIZE_CONFIG['overwrite_existing']=True to overwrite.")
        return

    # Load all cached graph dictionaries
    graph  = load_graph_from_graph_dicts()
    
    total_entities = sum(len(entities) for entities in graph.entity_types.values())
    print(f"Consolidated graph: {total_entities} entities across "
          f"{len(graph.entity_types)} types; "
          f"{len(graph.relationships)} relationships")

    # Initialize canonicalizer
    canonicalizer = GraphCanonicalizer()

    # Canonicalize each configured entity type
    entity_types_to_canonicalize = list(settings.CANONICALIZE_CONFIG["type_instructions"].keys())
    
    for entity_type in entity_types_to_canonicalize:
        if entity_type in graph.entity_types:
            print(f"\n=== Canonicalizing type: {entity_type} ===")
            canonicalizer.canonicalize_type(
                graph,
                entity_type=entity_type,
                stop_ratio=settings.CANONICALIZE_CONFIG["stop_ratio"],
                max_cluster_size=settings.CANONICALIZE_CONFIG["max_cluster_size"],
            )
        else:
            print(f"No entities of type '{entity_type}' found in graph.")

    # Print final statistics
    final_entities = sum(len(entities) for entities in graph.entity_types.values())
    print("\n=== Canonicalization complete ===")
    print(f"Final graph: {final_entities} entities across "
          f"{len(graph.entity_types)} types; "
          f"{len(graph.relationships)} relationships")

    # Save the canonicalized graph
    save_canonicalized_graph(graph, output_path)


if __name__ == "__main__":
    main()