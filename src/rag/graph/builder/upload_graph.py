import pickle
import os

from rag.graph.core.systematics_graph import SystematicsGraph
from rag.graph.builder.canonicalize_graph import get_output_path
import rag.settings as settings

def main():
    output_path = get_output_path()
    with open(output_path, "rb") as f:
        graph = pickle.load(f)

    graph.generate_embeddings()
    graph.push_to_neo4j(**settings.NEO4J_CONFIG)

if __name__ == "__main__":
    main()