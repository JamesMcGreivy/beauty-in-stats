"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import numpy as np
import os
import time
from PaperTree import PaperTree
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device="cuda")

### Tree RAG Method 0: Base RAG
class BaseRAG:

    def __init__(self, papers):
        self.unique_id = str(int(10000 * time.time()) % 2**16)

        self.papers = papers
        self.chunks = []
        self.ids = []
        self.paper_ids = []  # Track which paper each chunk belongs to
        
        for paper_id in papers:
            paper = papers[paper_id]
            self.add_paper(paper, paper_id)

        self.client = chromadb.EphemeralClient()
        self.build_collection()

    def __del__(self):
        collections = self.client.list_collections()
        for collection in collections:
            if self.unique_id in collection:
                self.client.delete_collection(collection)

    def add_paper(self, paper, paper_id):
        if len(paper.sections) == 0:
            self.chunks.append(f"{paper._id_str()} \n {paper.abstract}")
            self.ids.append(paper._id_str())
            self.paper_ids.append(paper_id)  # Store the parent paper ID
        for section in paper.sections:
            self.add_paper(section, paper_id)  # Pass down the paper_id to all sections

    def build_collection(self):
        collection = self.client.get_or_create_collection(name=f"base-rag-{self.unique_id}", metadata={"hnsw:space": "cosine"})

        batch_size = 250
        all_embeddings = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            all_embeddings.append(batch_embeddings)
        embeddings = np.vstack(all_embeddings)

        # Add metadata about which paper each chunk belongs to
        metadatas = [{"paper_id": paper_id} for paper_id in self.paper_ids]
        
        collection.add(
            embeddings=embeddings,
            documents=self.chunks,
            ids=self.ids,
            metadatas=metadatas,
        )
        self.collection = collection

    def query(self, query, n_results, filter_ids=None):
        query_embedding = model.encode(query)
        
        # Apply filtering if filter_ids is provided and not empty
        where_clause = None
        if filter_ids and len(filter_ids) > 0:
            where_clause = {"paper_id": {"$in": filter_ids}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )

        result_ids = results["ids"][0]
        result_docs = results["documents"][0]

        final_results = []
        for i in range(len(result_docs)):
            final_results.append((result_ids[i], result_docs[i]))

        return final_results
    
### Tree RAG Method 0.1: Base RAG + Reranker
class BaseRerankRAG:

    def __init__(self, papers):
        self.unique_id = str(int(10000 * time.time()) % 2**16)

        self.papers = papers
        self.chunks = []
        self.ids = []
        self.paper_ids = []  # Track which paper each chunk belongs to
        
        for paper_id in papers:
            paper = papers[paper_id]
            self.add_paper(paper, paper_id)

        self.client = chromadb.EphemeralClient()
        self.build_collection()

    def __del__(self):
        collections = self.client.list_collections()
        for collection in collections:
            if self.unique_id in collection:
                self.client.delete_collection(collection)

    def add_paper(self, paper, paper_id):
        if len(paper.sections) == 0:
            self.chunks.append(f"{paper._id_str()} \n {paper.abstract}")
            self.ids.append(paper._id_str())
            self.paper_ids.append(paper_id)  # Store the parent paper ID
        for section in paper.sections:
            self.add_paper(section, paper_id)  # Pass down the paper_id to all sections

    def build_collection(self):
        collection = self.client.get_or_create_collection(name=f"base-rag-{self.unique_id}", metadata={"hnsw:space": "cosine"})

        batch_size = 250
        all_embeddings = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            all_embeddings.append(batch_embeddings)
        embeddings = np.vstack(all_embeddings)

        # Add metadata about which paper each chunk belongs to
        metadatas = [{"paper_id": paper_id} for paper_id in self.paper_ids]
        
        collection.add(
            embeddings=embeddings,
            documents=self.chunks,
            ids=self.ids,
            metadatas=metadatas,
        )
        self.collection = collection

    def query(self, query, n_results, filter_ids=None):
        query_embedding = model.encode(query)
        
        # Apply filtering if filter_ids is provided and not empty
        where_clause = None
        if filter_ids and len(filter_ids) > 0:
            where_clause = {"paper_id": {"$in": filter_ids}}
        
        # Get more results than needed for reranking
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2 * n_results,
            where=where_clause,
        )

        result_ids = results["ids"][0]
        result_docs = results["documents"][0]

        rankings = self.rerank(query, result_docs)
        result_ids = np.array(result_ids)[rankings][0:n_results]
        result_docs = np.array(result_docs)[rankings][0:n_results]

        final_results = []
        for i in range(len(result_docs)):
            final_results.append((result_ids[i], result_docs[i]))

        return final_results
    
    def rerank(self, query, paragraphs):
        def rank_indices(lst):
            return [i for i, _ in sorted(enumerate(lst), key=lambda x: -x[1])]
        
        rankings = reranker.compute_score([[query, paragraph] for paragraph in paragraphs], normalize=True)
        return rank_indices(rankings)

### Tree RAG Method 1: Level Search
class LevelSearchRAG:

    def __init__(self, papers):
        self.unique_id = str(int(10000 * time.time()) % 2**16)

        self.papers = papers
        self.id_to_paper = {}
        self.id_to_paper_id = {}  # Map section IDs to parent paper IDs
        self.level_zero = []
        
        for paper_id in papers:
            paper = papers[paper_id]
            self.add_paper(paper, paper_id)

        self.client = chromadb.EphemeralClient()
        self.build_collection()

    def __del__(self):
        collections = self.client.list_collections()
        for collection in collections:
            if self.unique_id in collection:
                self.client.delete_collection(collection)

    def add_paper(self, paper, paper_id):
        if paper.abstract is not None:
            id = paper._id_str()
            self.id_to_paper[id] = paper
            self.id_to_paper_id[id] = paper_id  # Store mapping to parent paper ID
            if paper.get_depth() == 0:
                self.level_zero.append(id)
        for section in paper.sections:
            self.add_paper(section, paper_id)  # Pass down the paper_id to all sections

    def build_collection(self):
        self.collection = self.client.get_or_create_collection(name=f"level-rag-{self.unique_id}", metadata={"hnsw:space": "cosine"})

        ids = list(self.id_to_paper.keys())
        chunks = []
        for id in ids:
            paper = self.id_to_paper[id]
            chunks.append(f"{paper._id_str()} \n {paper.abstract}")

        batch_size = 250
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            all_embeddings.append(batch_embeddings)
        embeddings = np.vstack(all_embeddings)

        # Add both section ID and paper ID to metadata
        metadatas = [{"id": id, "paper_id": self.id_to_paper_id[id]} for id in ids]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query, n_results, filter_ids=None):
        query_embedding = model.encode(query)
        
        # Start with level zero entries, filtered by paper_id if necessary
        if filter_ids and len(filter_ids) > 0:
            # Only consider level-zero papers that match the filter
            focus_ids = [id for id in self.level_zero if self.id_to_paper_id[id] in filter_ids]
        else:
            focus_ids = list(self.level_zero)
            
        final_results = []
        while len(focus_ids) > 0 and len(final_results) < n_results:
            # Filter query to only consider the focus IDs
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"id": {"$in": focus_ids}}
            )
            
            if len(result["ids"][0]) == 0:
                break

            best_id = result['ids'][0][0]
            best_document = result['documents'][0][0]
            best_paper = self.id_to_paper[best_id]

            focus_ids.remove(best_id)
            
            if len(best_paper.sections) > 0:
                # Only add child sections if they match our filter criteria
                for section in best_paper.sections:
                    section_id = section._id_str()
                    if filter_ids and len(filter_ids) > 0:
                        if self.id_to_paper_id[section_id] in filter_ids:
                            focus_ids.append(section_id)
                    else:
                        focus_ids.append(section_id)
            else:
                final_results.append((best_id, best_document))

        return final_results


### Tree RAG Method 2: Level Search with Reranker
class LevelSearchRerankRAG:

    def __init__(self, papers):
        self.unique_id = str(int(10000 * time.time()) % 2**16)

        self.papers = papers
        self.id_to_paper = {}
        self.level_zero = []
        
        for paper in papers:
            self.add_paper(paper)

        self.client = chromadb.EphemeralClient()
        self.build_collection()

        

    def __del__(self):
        collections = self.client.list_collections()
        for collection in collections:
            if self.unique_id in collection:
                self.client.delete_collection(collection)

    def get_paper_id(self, paper):
        if not paper.parent:
            return paper.title
        else:
            return f"{self.get_paper_id(paper.parent)} --> {paper.title}"

    def add_paper(self, paper):
        id = self.get_paper_id(paper)
        self.id_to_paper[id] = paper
        
        if not paper.parent:
            self.level_zero.append(id)
        
        for section in paper.sections:
            self.add_paper(section)

    def build_collection(self):
            self.collection = self.client.get_or_create_collection(
                name=f"level-rag-{self.unique_id}",
                metadata={"hnsw:space": "cosine"}
            )

            ids = list(self.id_to_paper.keys())
            chunks = [self.id_to_paper[i].abstract for i in ids]

            all_embeddings = []
            batch_size = 5000
            print("Encoding abstracts and building vector collection...")
            for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Batches"):
                batch = chunks[i:i + batch_size]
                batch_embeddings = model.encode(batch)
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings)
            metadatas = [{"id": pid, "paper": self.id_to_paper[pid].title} for pid in ids]

            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )

    def query(self, query, n_results, filter=None):
        query_embedding = model.encode(query)
        
        focus_ids = [id for id in self.level_zero if self.id_to_paper[id] in filter] if filter else list(self.level_zero)
        
        final_results = []

        pbar = tqdm(total=n_results, desc="Retrieving Results")

        id_to_relevance = {}
        
        while focus_ids and len(final_results) < n_results:
            where_clause = {"id": {"$in": focus_ids}}

            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=2 * n_results,
                where=where_clause
            )

            if not result['ids'][0]:
                break

            ids = result['ids'][0]
            documents = result['documents'][0]
            needs_relevance_ids = []
            needs_relevance_chunks = []

            for pid, doc in zip(ids, documents):
                if pid not in id_to_relevance:
                    needs_relevance_ids.append(pid)
                    paper = self.id_to_paper[pid]
                    needs_relevance_chunks.append(paper.abstract)

            scores = self.llm_reranker(query, needs_relevance_chunks)
            for pid, score in zip(needs_relevance_ids, scores):
                id_to_relevance[pid] = score

            best_id = max(id_to_relevance, key=id_to_relevance.get)
            best_paper = self.id_to_paper[best_id]
            best_document = best_paper.abstract

            if best_id in focus_ids:
                focus_ids.remove(best_id)
            id_to_relevance.pop(best_id)

            if best_paper.sections:
                for section in best_paper.sections:
                    focus_ids.append(self.get_paper_id(section))
            else:
                final_results.append((best_id, best_document))
                pbar.update(1)

        pbar.close()
        return final_results

    def llm_reranker(self, query, paragraphs):
        if len(paragraphs) == 0:
            return []
        relevances = reranker.compute_score([[query, paragraph] for paragraph in paragraphs], normalize=True)
        return relevances