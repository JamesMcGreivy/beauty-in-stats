To construct the systematics knowledge graph, you must have the LHCb corpus downloaded already. Run the following:

```bash
python build_systematics_graph.py --build-cache
```

This will construct the knowledge graph from individual papers in the corpus, merge and canonicalize it, and then push it to neo4j.

testing.ipynb has a demonstration of querying the Graph RAG.