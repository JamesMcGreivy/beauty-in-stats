You will first need to construct the paper tree cache:

```bash
python build_paper_trees.py
```

Then, run the recursive summarization algorithm to populate the paper tree with summarizations:

```bash
python recursively_summarize.py
```

testing.ipynb has a demonstration of querying the Tree RAG.