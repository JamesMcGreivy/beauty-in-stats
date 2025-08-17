"""
SystematicsGraph: A knowledge graph implementation for managing entities and relationships.

This module provides classes for building and managing knowledge graphs with support
for Neo4j integration. It's designed to work with academic papers and research data,
particularly for extracting and organizing structured information.

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import numpy as np
import time
import asyncio
import regex as re
import copy
from typing import List, Dict, Set, Any, Optional, Union, Iterable
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import rag.settings as settings

class Entity:
    """
    Represents a named entity in the knowledge graph.
    
    An entity is a fundamental unit in the graph that represents a concept, person,
    organization, or any identifiable object with a name, type, and description.
    """

    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            model_name = settings.EMBEDDING_CONFIG["model"]
            device = settings.EMBEDDING_CONFIG["device"]

            cls._embedding_model = SentenceTransformer(model_name, device=device)
        
        return cls._embedding_model
    
    @staticmethod
    def generate_embeddings(entities: Iterable, batch_size=1028):
        """
        Generate embeddings for a collection of entities using a sentence transformer model.
        
        Combines entity name and description to create a text representation, then generates
        embeddings using the specified sentence transformer model.
        """

        embedding_model = Entity._get_embedding_model()
        
        # Collect entities that need embeddings
        entities_to_embed = []
        for entity in entities:
            entities_to_embed.append(entity)
        
        print(f"Generating embeddings for {len(entities_to_embed)} entities...")
        
        # Process entities in batches
        for i in range(0, len(entities_to_embed), batch_size):
            batch_entities = entities_to_embed[i:i + batch_size]
            
            # Prepare text representations for the batch
            batch_texts = []
            for entity in batch_entities:
                name = entity.name or ""
                description = entity.description or ""
                text = f"{name} {description}".strip()
                batch_texts.append(text)
            
            # Generate embeddings for the entire batch
            batch_embeddings = embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Assign embeddings back to entities
            for entity, embedding in zip(batch_entities, batch_embeddings):
                entity.embedding = [float(e) for e in embedding]
        
        print("Embedding generation complete.")
    
    def __init__(
        self, 
        name: str, 
        type: str, 
        description: str, 
        attributes: Dict[str, Any] = None
    ):
        """Initialize an Entity with basic properties and optional attributes."""
        self.name = name
        self.type = type
        self.description = description
        self.attributes = attributes or {}
        self.embedding = None
        self._neo4j: Optional[Any] = None  # Will hold Neo4j node reference

    def __repr__(self) -> str:
        """Return a human-readable string representation of the entity."""
        return f"{self.type}: {self.name} \n\t {self.description}"
    
    def add_to_graph(self, graph: Any):
        """
        Add this entity as a node to a Neo4j graph database.
        
        Creates a Neo4j Node with the entity's properties and stores a reference
        to the created node for future relationship creation.
        """
        from py2neo import Node
        
        # Convert non-string attribute values to string representations
        neo4j_attributes = {
            key: value if isinstance(value, str) else repr(value) 
            for key, value in self.attributes.items() 
            if value
        }

        neo4j_embedding = self.embedding
        
        neo4j_node = Node(
            self.type, 
            name=self.name, 
            description=self.description, 
            embedding=neo4j_embedding,
            **neo4j_attributes
        )
        graph.create(neo4j_node)
        self._neo4j = neo4j_node


class Relationship:
    """
    Represents a directed relationship between two entities in the knowledge graph.
    
    Relationships define how entities are connected, with a source entity, target entity,
    relationship type, and optional attributes that describe the relationship.
    """
    
    def __init__(
        self, 
        source: Entity, 
        type: str, 
        target: Entity, 
        attributes: Dict[str, Any] = None
    ):
        """Initialize a Relationship between two entities."""
        self.source = source
        self.type = type
        self.target = target
        self.attributes = attributes or {}
        self._neo4j: Optional[Any] = None  # Will hold Neo4j relationship reference

    def __repr__(self) -> str:
        """Return a human-readable string representation of the relationship."""
        return f"({self.source.name}) --[{self.type}]--> ({self.target.name})"
    
    def add_to_graph(self, graph: Any):
        """
        Add this relationship to a Neo4j graph database.
        
        Ensures both source and target entities exist in the graph before
        creating the relationship between them.
        """
        from py2neo import Relationship as Neo4jRelationship
        
        matcher = graph.nodes
        
        # Find or create source node
        source_node = matcher.match(
            self.source.type, 
            name=self.source.name, 
            description=self.source.description
        ).first()
        
        if source_node is None:
            self.source.add_to_graph(graph)
        else:
            self.source._neo4j = source_node
            
        # Find or create target node
        target_node = matcher.match(
            self.target.type, 
            name=self.target.name, 
            description=self.target.description
        ).first()
        
        if target_node is None:
            self.target.add_to_graph(graph)
        else:
            self.target._neo4j = target_node
            
        # Create the relationship
        neo4j_relationship = Neo4jRelationship(
            self.source._neo4j, 
            self.type, 
            self.target._neo4j, 
            **self.attributes
        )
        graph.create(neo4j_relationship)
        self._neo4j = neo4j_relationship


class SystematicsGraph:
    """
    Main knowledge graph container that manages entities and relationships.
    
    This class provides functionality to build, manage, and export knowledge graphs.
    It supports loading data from structured formats and exporting to Neo4j databases.
    Designed particularly for academic and research applications.
    """
    
    def __init__(self):
        """Initialize an empty SystematicsGraph."""
        # Dictionary mapping entity names to Entity objects
        self.entities: Dict[str, Entity] = {}
        
        # Dictionary mapping entity types to sets of entities of that type
        self.entity_types: Dict[str, Set[Entity]] = {}
        
        # Set of all relationships in the graph
        self.relationships: Set[Relationship] = set()
    
    def get_entity(self, entity_name: str) -> Entity:
        """Retrieve an entity by its name."""
        return self.entities[entity_name]

    def add_entity(self, entity: Entity, overwrite: bool = False):
        """
        Add an entity to the graph.
        
        Updates both the main entities dictionary and the entity_types grouping
        to maintain efficient lookups by both name and type.
        """
        if not overwrite and entity.name in self.entities:
            # Do not overwrite if an entity name already exists in the graph
            return

        self.entities[entity.name] = entity
        
        if entity.type not in self.entity_types:
            self.entity_types[entity.type] = set()
        self.entity_types[entity.type].add(entity)

    def remove_entity(self, entity: Entity):
        """Remove an entity from the graph."""
        self.entities.pop(entity.name, None)
        
        # Remove from entity_types as well
        if entity.type in self.entity_types:
            self.entity_types[entity.type].discard(entity)

    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the graph."""
        self.relationships.add(relationship)

    def remove_relationship(self, relationship: Relationship):
        """Remove a relationship from the graph."""
        self.relationships.remove(relationship)

    def load_graph_dict(self, arxiv_id: str, graph_dict: Dict):
        """
        Load entities and relationships from a structured graph dictionary.
        
        Expected format:
        {
            "entities": {
                "EntityType": [{"name": "...", "description": "...", ...}, ...],
                ...
            },
            "relationships": {
                "RelationshipType": [{"source": "...", "target": "...", ...}, ...],
                ...
            }
        }
        
        The arxiv_id is added as an attribute to all loaded entities for provenance tracking.
        """
        # Load all entities first
        for entity_type in graph_dict["entities"]:
            for entity_dict in graph_dict["entities"][entity_type]:
                entity_data = copy.deepcopy(entity_dict)
                
                # Add provenance information
                entity_data["arxiv_id"] = arxiv_id
                
                # Extract required fields
                name = entity_data.pop("name")
                description = entity_data.pop("description", "")

                entity = Entity(
                    name=name, 
                    type=entity_type, 
                    description=description, 
                    attributes=entity_data
                )

                self.add_entity(entity)
        
        # Load relationships after all entities are created
        for relationship_type in graph_dict["relationships"]:
            for relationship_dict in graph_dict["relationships"][relationship_type]:
                relationship_data = copy.deepcopy(relationship_dict)
                
                # Extract source entity
                source_name = relationship_data.pop("source")
                try:  
                    source = self.get_entity(source_name)
                except KeyError:
                    print(f"Error loading relationship from {arxiv_id}: \n {relationship_dict} \n source {source_name} does not exist")
                    continue
                
                # Extract target entity
                target_name = relationship_data.pop("target")
                try:
                    target = self.get_entity(target_name)
                except KeyError:
                    print(f"Error loading relationship from {arxiv_id}: \n {relationship_dict} \n target {target_name} does not exist")
                    continue
            
                relationship = Relationship(
                    source, 
                    relationship_type, 
                    target, 
                    relationship_data
                )
                self.add_relationship(relationship)

    def generate_embeddings(self):
        """Generates embeddings for all entities in the graph"""
        Entity.generate_embeddings(self.entities.values())

    def push_to_neo4j(self, uri: str, username: str, password: str):
        """
        Export the entire graph to a Neo4j database.
        
        This method clears the target database and recreates all entities and
        relationships. Use with caution as it will delete existing data.
        """
        from py2neo import Graph
        
        graph = Graph(uri, auth=(username, password))
        
        # WARNING: This clears all existing data in the database
        print("Deleting current Neo4j database...")
        graph.delete_all()

        print("Uploading new Neo4j database...")
        # Create all entities first
        for name, entity in self.entities.items():
            entity.add_to_graph(graph)

        # Create all relationships
        for relationship in self.relationships:
            relationship.add_to_graph(graph)