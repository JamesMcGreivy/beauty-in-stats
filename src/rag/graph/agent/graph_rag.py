"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import regex as re

from rag.graph.agent.prompts import get_cypher_query_prompt, generate_synthesis_prompt
import rag.settings as settings

class GraphRAG:
    """
    Agent that converts natural language queries about particle physics systematic uncertainties
    into Cypher queries, executes them against a Neo4j knowledge graph, and synthesizes the results.
    """

    # Class variables for shared model (lazy loading)
    _embedding_model = None
    _neo4j_graph = None
    _llm = None
    
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
    
    @classmethod
    def _get_neo4j_graph(cls):
        """
        Get the neo4j graph pointer, loading it if necessary
        """
        if cls._neo4j_graph is None:
            uri = settings.NEO4J_CONFIG["uri"]
            username = settings.NEO4J_CONFIG["username"]
            password = settings.NEO4J_CONFIG["password"]

            from py2neo import Graph
            print(f"Loading Neo4J Graph at: {uri}")
            cls._neo4j_graph = Graph(uri, auth=(username, password))

        return cls._neo4j_graph

    @classmethod
    def _get_llm(cls):
        """
        Get the pointer to the OpenAI llm
        """
        if cls._llm is None:
            model = settings.GRAPH_CONFIG["model"]

            from llama_index.llms.openai import OpenAI
            cls._llm = OpenAI(temperature=0, model=model)
        
        return cls._llm
    
    def __init__(self):
        self.embedding_model = self._get_embedding_model()
        self.neo4j_graph = self._get_neo4j_graph()
        self.llm = self._get_llm()

    def process_cypher_query(self, cypher_query):
        pattern = r'\$\(["\'](.*?)["\']\)'
        
        def replacement_function(match):
            text = match.group(1)
            embedding = list(self.embedding_model.encode(text))
            return str(embedding)
        
        processed_cypher = re.sub(pattern, replacement_function, cypher_query)
        return processed_cypher
    
    def cursor_to_formatted_results(self, cursor):
        if not cursor:
            return {"columns": [], "results": []}
        
        # Extract column names and data from cursor
        columns = cursor.keys()
        results = []
        
        # Convert to a list of dictionaries for easier processing
        for record in cursor:
            row = {}
            for col in columns:
                row[col] = record[col]
            results.append(row)
            
        return {
            "columns": columns,
            "results": results
        }

    def query(self, query: str, max_tokens: int = 5000):
        try:
            # Step 1: Generate Cypher query from natural language
            prompt, default = get_cypher_query_prompt(query)
            llm_response = self.llm.complete(prompt).text

            response_dict = eval(llm_response)
            cypher = response_dict["cypher"]
            explanation = response_dict["explanation"]
        
            # Step 2: Process and execute the Cypher query
            processed_cypher = self.process_cypher_query(cypher)
            cursor = self.neo4j_graph.query(processed_cypher)
            formatted_results = self.cursor_to_formatted_results(cursor)
        
            # Step 3: Synthesize results with LLM
            prompt, default = generate_synthesis_prompt(query, cypher, formatted_results)
            synthesized_answer = self.llm.complete(prompt).text

            return synthesized_answer
        
        except Exception as e:
            error_msg = f"Error while processing query: {e}"
            return error_msg