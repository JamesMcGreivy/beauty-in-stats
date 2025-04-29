import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple, Optional, Any, Union

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4o-mini")

stopwords = [
    # LaTeX formatting
    'begin', 'end', 'align', 'equation', 'itemize', 'ref', 'cite', 'label', 
    'includegraphics', 'width', 'nonumber', 'textrm', 'mathrm', 'textbf',
    'mathcal', 'overline', 'cdots', 'bf', 'frac', 'quad', 'pm',
    
    # Common physics/math notation
    'sigma', 'delta', 'lambda', 'alpha', 'beta', 'theta', 'mu', 'nu', 'pi',
    'partial', 'vec', 'hat', 'tilde', 'bar',
    
    # Common physics terms that are too generic
    'particle', 'physics', 'energy', 'mass', 'time', 'event', 'events',
    'function', 'value', 'values', 'approach', 'case', 'given', 'number', 'proton', 'data', 'eq', 'probability', 'statistics', 
    
    # English stopwords
    'fig', 'figure', 'equation', 'example', 'left', 'right', 'sincere', 'inc', 'seems', 'her', 'already', 'amount', 'etc', 'part', 'ourselves', 'find', 'nor', 'anyhow', 'last', 'always', 'sixty', 'toward', 'two', 'describe', 'please', 'see', 'about', 'herein', 'has', 'was', 'three', 'until', 'which', 'to', 'your', 'why', 'on', 'thick', 'perhaps', 'same', 'often', 'move', 'whose', 'whether', 'side', 'thereupon', 'much', 'fire', 'twelve', 'else', 'show', 'elsewhere', 'whole', 'because', 'serious', 'call', 'fifteen', 'via', 'give', 'interest', 'go', 'ever', 'everywhere', 'all', 'seem', 'among', 'indeed', 'are', 'i', 'whereby', 'moreover', 'eight', 'back', 'us', 'will', 'themselves', 'tchen', 'thru', 'namely', 'after', 'formerly', 'very', 'they', 'at', 'up', 'whereupon', 'herself', 'beside', 'hereby', 'rather', 'third', 'now', 'bottom', 'fifty', 'nothing', 'anyone', 'hers', 'there', 'sometimes', 'behind', 'by', 'though', 'hereupon', 'is', 'eg', 'who', 'former', 're', 'cannot', 'along', 'hasnt', 'whereas', 'again', 'thereafter', 'front', 'nevertheless', 'off', 'both', 'our', 'therefore', 'however', 'therein', 'whoever', 'con', 'everything', 'he', 'me', 'besides', 'five', 'my', 'either', 'have', 'himself', 'too', 'several', 'itself', 'could', 'it', 'mill', 'not', 'had', 'whom', 'bill', 'per', 'whenever', 'over', 'amoungst', 'these', 'co', 'than', 'yourselves', 'four', 'beyond', 'nine', 'out', 'the', 'whereafter', 'thin', 'mostly', 'couldnt', 'a', 'every', 'twenty', 'wherever', 'anything', 'were', 'also', 'one', 'for', 'within', 'whence', 'whither', 'keep', 'another', 'how', 'sometime', 'un', 'throughout', 'few', 'nowhere', 'take', 'an', 'when', 'him', 'none', 'except', 'towards', 'during', 'thence', 'thereby', 'further', 'latter', 'what', 'other', 'can', 'made', 'be', 'into', 'hundred', 'ten', 'meanwhile', 'thus', 'hereafter', 'ours', 'first', 'noone', 'its', 'yours', 'each', 'system', 'between', 'hence', 'full', 'would', 'others', 'once', 'eleven', 'seeming', 'afterwards', 'from', 'although', 'around', 'somehow', 'no', 'but', 'many', 'been', 'get', 'down', 'whatever', 'such', 'here', 'together', 'without', 'even', 'below', 'own', 'or', 'forty', 'fill', 'someone', 'only', 'any', 'this', 'name', 'through', 'found', 'put', 'yet', 'mine', 'should', 'amongst', 'cry', 'must', 'being', 'more', 'seemed', 'next', 'am', 'still', 'as', 'latterly', 'before', 'anywhere', 'everyone', 'somewhere', 'them', 'she', 'cant', 'well', 'of', 'otherwise', 'may', 'in', 'across', 'with', 'nobody', 'might', 'becomes', 'empty', 'anyway', 'beforehand', 'do', 'something', 'most', 'his', 'done', 'their', 'least', 'became', 'less', 'myself', 'enough', 'detail', 'ltd', 'wherein', 'almost', 'you', 'become', 'yourself', 'onto', 'if', 'becoming', 'some', 'we', 'so', 'due', 'top', 'six', 'upon', 'while', 'above', 'ie', 'and', 'that', 'those', 'neither', 'de', 'never', 'since', 'alone', 'where', 'against', 'under'
]

class SubjectCluster:
    """
    A class for hierarchical topic clustering on text data, using UMAP for dimensionality
    reduction and HDBSCAN for clustering.
    
    Each SubjectCluster instance can contain subclusters, creating a hierarchical
    structure of topics and subtopics.
    """
    
    def __init__(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        context: str = "Root Topic",
        reduced_dim: int = 16,
        optimal_clusters: int = 4,
        min_cluster_size: int = 50,
        keyword_cutoff: float = 0.05,
        parent = None
    ):
        """
        Initialize a SubjectCluster object.
        
        Args:
            chunks: List of text chunks to cluster
            embeddings: Embeddings of the text chunks
            context: Description of the context for these clusters
            reduced_dim: Dimensionality to reduce embeddings to for clustering
            optimal_clusters: Optimal number of clusters to find with HDBSCAN
            min_cluster_size: Minimum size of clusters for HDBSCAN
            keyword_cutoff: Cutoff for keyword importance (relative to max)
            parent: Parent SubjectCluster object (if this is a subcluster)
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.context = context
        self.reduced_dim = reduced_dim
        self.optimal_clusters = optimal_clusters
        self.min_cluster_size = min_cluster_size
        self.keyword_cutoff = keyword_cutoff
        self.parent = parent
        
        # These will be populated during clustering
        self.clusters = None  # Cluster assignments for each chunk
        self.cluster_labels = None  # Names for each cluster
        self.cluster_keywords = None  # Keywords for each cluster
        self.subclusters = {} # Dict of SubjectCluster objects for subclusters
        
        # Cache for dimensionality reduction
        self._reduced_embeddings = None  # UMAP reduced embeddings
        self._scatter_embeddings = None  # 2D UMAP for visualization
        self._umap_model = None  # Cached UMAP model
        self._scatter_umap_model = None  # Cached 2D UMAP model
        
        # Perform initial clustering
        self.perform_clustering()
    
    def get_reduced_embeddings(self) -> np.ndarray:
        """
        Get dimensionality-reduced embeddings, computing them if necessary.
        
        Returns:
            Reduced embeddings array
        """
        if self._reduced_embeddings is None:
            self._umap_model = UMAP(
                n_components=self.reduced_dim,
                min_dist=0.0,
                metric='cosine'
            )
            self._reduced_embeddings = self._umap_model.fit_transform(self.embeddings)
        return self._reduced_embeddings
    
    def get_scatter_embeddings(self) -> np.ndarray:
        """
        Get 2D embeddings for visualization, computing them if necessary.
        
        Returns:
            2D embeddings array for visualization
        """
        if self._scatter_embeddings is None:
            self._scatter_umap_model = UMAP(
                n_components=2,
                min_dist=0.0,
                metric='cosine'
            )
            self._scatter_embeddings = self._scatter_umap_model.fit_transform(self.embeddings)
        return self._scatter_embeddings
    
    def perform_clustering(self):
        """
        Perform clustering on the embeddings and extract keywords and labels.
        """
        # Get reduced embeddings
        reduced_embeddings = self.get_reduced_embeddings()

        cluster_sizes, num_clusters, proportion_unclustered = self._test_cluster_sizes()
        best_cluster_size = cluster_sizes[np.argmin(np.abs(num_clusters - self.optimal_clusters))]
        
        # Perform clustering with HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=best_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(reduced_embeddings)
        
        self.clusters = hdbscan_model.labels_
        # If no chunks were assigned to a cluster then return
        if np.max(self.clusters) < 0:
            return
        
        # Extract keywords for each cluster
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            max_df=0.9,
            min_df=2,
            ngram_range=(1, 2)
        )

        # Fit the vectorizer on the entire corpus
        parent = self
        while parent.parent:
            parent = parent.parent
        self.vectorizer.fit(parent.chunks)
        
        self.top_keywords = []
        for cluster in range(0, np.max(self.clusters + 1)):
            cluster_chunks = self.chunks[self.clusters == cluster]
            self.top_keywords.append(self._get_top_keywords(cluster_chunks))
        
        self.unclustered_keywords = self._get_top_keywords(self.chunks[self.clusters == -1])
        
        # Generate labels for clusters
        self.cluster_labels = self._generate_cluster_labels()
        self.subclusters = {i : None for i in range(len(self.cluster_labels))}
    
    def _get_top_keywords(self, cluster_chunks: List[str], n_keywords: int = 50) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a set of chunks using TF-IDF.
        
        Args:
            cluster_chunks: List of text chunks
            n_keywords: Number of top keywords to extract
            
        Returns:
            List of (keyword, importance) tuples
        """
        if len(cluster_chunks) == 0:
            return []
        
        tfidf_matrix = self.vectorizer.transform(cluster_chunks)
        feature_names = self.vectorizer.get_feature_names_out()
        
        tfidf_sums = np.sum(tfidf_matrix.toarray(), axis=0)
        top_indices = tfidf_sums.argsort()[-n_keywords:][::-1]
        top_tfidf = tfidf_sums.max()
        
        top_keywords = [
            (feature_names[i], tfidf_sums[i]) 
            for i in top_indices 
            if tfidf_sums[i] > self.keyword_cutoff * top_tfidf
        ]
        
        return top_keywords
    
    def _generate_cluster_labels(self) -> Dict[int, str]:
        """
        Generate human-readable labels for clusters using an LLM.
        
        Returns:
            Dictionary mapping cluster indices to label strings
        """
        
        keyword_str = " \n".join([
            f"Cluster {cluster} keywords: {{{', '.join([f'{keyword}: {importance:.2f}' for keyword, importance in self.top_keywords[cluster]])}}}" 
            for cluster in range(len(self.top_keywords))
        ])
        
        unclustered_str = ", ".join([
            f"{keyword} : {importance:.2f}" 
            for keyword, importance in self.unclustered_keywords
        ])
        
        prompt = f"""
Using a corpus of documents, I have chunked the text and used a sentence embedding model to generate paragraph embeddings for each chunk. Using UMAP + HDBSCAN as a clustering algorithm, the chunks have been clustered. Comparing chunks within a cluster to chunks outside of it, the TF-IDF algorithm has identified the primary keywords within each cluster.

Below I have provided the primary keywords along with their importance scores for each cluster. I have also provided the top keywords and importances across all the chunks which were not clustered at all. By comparing keywords within each cluster and contrasting them across different clusters, you should construct a few words label that summarizes the unique content specific to each cluster. 

This cluster exists within the context of "{self.context}". Thus, your labels should follow from this context but with more specificity.

Give your answer as a python dictionary. Do not respond with anything else, including with ```python ... ```

{{0: "Cluster 0 Name", 1: "Cluster 1 Name", ...}}

Unclustered Keywords:

{unclustered_str}

Clustered Keywords:

{keyword_str}
"""
        
        answer = llm.complete(prompt, temperature=0.0)
        try:
            return eval(answer.text)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {i: f"Cluster {i}" for i in range(len(self.top_keywords))}
    
    def create_subcluster(self, cluster_idx: int, min_cluster_size: Optional[int] = None) -> 'SubjectCluster':
        """
        Create a subcluster from a specific cluster.
        
        Args:
            cluster_idx: Index of the cluster to create a subcluster from
            min_cluster_size: Override min_cluster_size for the subcluster
            
        Returns:
            New SubjectCluster object
        """
        if cluster_idx > np.max(self.clusters):
            raise ValueError(f"Cluster index {cluster_idx} exceeds maximum cluster index {np.max(self.clusters)}")
        
        # Select chunks and embeddings for this cluster
        cluster_mask = self.clusters == cluster_idx
        cluster_chunks = self.chunks[cluster_mask]
        cluster_embeddings = self.embeddings[cluster_mask]
        
        # Create subcluster context
        subcluster_context = f"{self.context} --> {self.cluster_labels[cluster_idx]}"
        
        subcluster = SubjectCluster(
            chunks=cluster_chunks,
            embeddings=cluster_embeddings,
            context=subcluster_context,
            reduced_dim=self.reduced_dim,
            optimal_clusters=self.optimal_clusters,
            min_cluster_size=self.min_cluster_size,
            keyword_cutoff=self.keyword_cutoff,
            parent=self
        )
        
        # Add to subclusters list
        self.subclusters[cluster_idx] = subcluster
        return subcluster
        
    def create_all_subclusters(self):
        """
        Create all subclusters
        """
        # First check if there are any valid clusters, and only continue subclustering if there are enough chunks to create optimal_clusters possible subclusters
        if np.max(self.clusters) < 0 or len(self.chunks) < self.optimal_clusters*self.min_cluster_size:
            return
            
        for idx in self.subclusters:
            cluster_size = np.sum(self.clusters == idx)
            # Dont build a subcluster if there are too few data points
            if cluster_size < self.optimal_clusters*self.reduced_dim:
                continue
            subcluster = self.create_subcluster(idx)
            subcluster.create_all_subclusters()
    
    def _test_cluster_sizes(self):
        """
        Compute how min_cluster_size affects the number of clusters formed, and the proportion of unclustered points.
        """

        reduced_embeddings = self.get_reduced_embeddings()
        
        num_clusters = []
        proportion_unclustered = []
        
        min_clusters = self.min_cluster_size
        max_clusters = 20 * self.min_cluster_size
        step_size = max(1, (max_clusters - min_clusters) // 20)
        cluster_sizes = list(range(min_clusters, max_clusters, step_size))
        
        for cluster_size in cluster_sizes:
            hdbscan_model = HDBSCAN(
                min_cluster_size=cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(reduced_embeddings)
            
            clusters = hdbscan_model.labels_
            
            num_clusters.append(np.max(clusters) + 1)
            proportion_unclustered.append(np.sum(clusters == -1) / len(clusters))

        return np.array(cluster_sizes), np.array(num_clusters), np.array(proportion_unclustered)


    def plot_cluster_sizes(self, ax=None, title=None):
        """
        Plot analysis of how min_cluster_size affects clustering.
        
        Args:
            ax: Matplotlib axis to plot on (creates a new one if None)
            title: Optional title for the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_sizes, num_clusters, proportion_unclustered = self._test_cluster_sizes()
        
        ax.plot(cluster_sizes, num_clusters, label="Number of clusters")
        ax.set_xlabel("Minimum cluster size")
        ax.set_ylabel("Number of clusters found")
        ax.set_ylim(1, 10 * max(num_clusters) if max(num_clusters) > 0 else 10)
        ax.set_yscale("log")
        
        ax2 = ax.twinx()
        ax2.plot(cluster_sizes, proportion_unclustered, 'r-', label="Proportion unclustered")
        ax2.set_ylabel("Proportion of Unclustered Chunks")
        ax2.set_ylim(0, 1)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Cluster Analysis: {self.context}")
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        return ax
    
    def plot_clusters(self, ax=None, title=None, alpha=0.7, show_unclustered=True):
        """
        Plot 2D visualization of clusters.
        
        Args:
            ax: Matplotlib axis to plot on (creates a new one if None)
            title: Optional title for the plot
            alpha: Transparency of points
            show_unclustered: Whether to show unclustered points
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter_embeddings = self.get_scatter_embeddings()
        
        # Create a colormap for clusters
        cmap = matplotlib.colormaps["cool"]
        norm = plt.Normalize(0, np.max(self.clusters))
        
        # Plot unclustered points
        if show_unclustered:
            unclustered_mask = (self.clusters == -1)
            if np.any(unclustered_mask):
                unclustered_embeddings = scatter_embeddings[unclustered_mask]
                ax.scatter(
                    unclustered_embeddings[:, 0],
                    unclustered_embeddings[:, 1],
                    color="grey",
                    alpha=alpha,
                    label="Unclustered"
                )
        
        # Plot clustered points and labels
        for cluster in range(0, max(self.clusters) + 1):
            cluster_mask = (self.clusters == cluster)
            if not np.any(cluster_mask):
                continue
                
            cluster_embeddings = scatter_embeddings[cluster_mask]
            cluster_color = cmap(norm(cluster))
            
            # Plot points
            ax.scatter(
                cluster_embeddings[:, 0],
                cluster_embeddings[:, 1],
                color=cluster_color,
                alpha=alpha,
                label=f"Cluster {cluster}"
            )
            
            # Add cluster label at mean position
            mean_x, mean_y = cluster_embeddings[:, 0].mean(), cluster_embeddings[:, 1].mean()
            ax.scatter(mean_x, mean_y, color=cluster_color, alpha=0.5, s=4000)
            
            label = self.cluster_labels.get(cluster, f"Cluster {cluster}")
            ax.text(
                mean_x, mean_y, label,
                color=cluster_color, alpha=1.0, fontsize=14,
                horizontalalignment='center', verticalalignment='center',
                path_effects=[path_effects.withStroke(linewidth=3, foreground='white')]
            )
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"UMAP Projection: {self.context}", fontsize=14)
            
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
    
    def plot_full_analysis(self, figsize=(16, 24)):
        """
        Plot a comprehensive analysis with cluster size analysis and visualization.
        
        Args:
            figsize: Size of the figure
            
        Returns:
            Figure and axes objects
        """
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        
        # Plot cluster size analysis
        self.plot_cluster_sizes(ax=axs[0], title=f"Cluster Size Analysis: {self.context}")
        
        # Plot cluster proportions
        if hasattr(self, 'clusters') and self.clusters is not None:
            clusters = self.clusters
            cluster_counts = {i: np.sum(clusters == i) for i in range(-1, np.max(clusters) + 1)}
            cluster_labels = {**{-1: "Unclustered"}, **self.cluster_labels}
            
            labels = [f"{cluster_labels[i]} ({count})" for i, count in cluster_counts.items() if i >= -1]
            counts = [count for i, count in cluster_counts.items() if i >= -1]
            
            axs[1].bar(range(len(counts)), counts)
            axs[1].set_xticks(range(len(counts)))
            axs[1].set_xticklabels(labels, rotation=45, ha='right')
            axs[1].set_title(f"Cluster Sizes: {self.context}")
            axs[1].set_ylabel("Number of Chunks")
        
        # Plot clusters visualization
        self.plot_clusters(ax=axs[2], title=f"Cluster Visualization: {self.context}")
        
        plt.tight_layout()
        return fig, axs

    def visualize_hierarchy(self, level=0, include_keywords=False):
        """
        Print a hierarchical visualization of clusters and subclusters.
        
        Args:
            level: Current level in hierarchy (for indentation)
            include_keywords: Whether to print top keywords for each cluster
        """
            
        indent = "  " * level
        if level == 0:
            print(f"{indent}● {self.context}")

        if np.max(self.clusters) <= 0:
            return
            
        for cluster_idx in sorted(self.cluster_labels.keys()):
            label = self.cluster_labels[cluster_idx]
            cluster_mask = (self.clusters == cluster_idx)
            count = np.sum(cluster_mask)
            
            has_subcluster = (cluster_idx in self.subclusters and 
                            self.subclusters[cluster_idx] is not None)
            
            # Use different connector based on whether this is the last entry
            connector = "└─" if cluster_idx == max(self.cluster_labels.keys()) and not include_keywords else "├─"
            
            print(f"{indent}  {connector} {label} ({count} chunks)")
            
            if include_keywords and self.top_keywords and cluster_idx < len(self.top_keywords):
                keywords = self.top_keywords[cluster_idx]
                if keywords:
                    top_kw = ", ".join([k for k, _ in keywords[:5]])
                    keyword_indent = "  " if has_subcluster else "     "
                    keyword_connector = "├─" if has_subcluster else "└─"
                    print(f"{indent}  {keyword_indent} {keyword_connector} Keywords: {top_kw}")
            
            # Print subcluster for this specific cluster index
            if has_subcluster:
                self.subclusters[cluster_idx].visualize_hierarchy(
                    level=level+1,
                    include_keywords=include_keywords
                )

    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics about clustering.
        
        Returns:
            Dictionary of cluster statistics
        """
        if self.clusters is None:
            return {"status": "not_clustered"}
            
        total_chunks = len(self.chunks)
        unclustered = np.sum(self.clusters == -1)
        num_clusters = np.max(self.clusters)
        
        return {
            "total_chunks": total_chunks,
            "clustered_chunks": total_chunks - unclustered,
            "unclustered_chunks": unclustered,
            "unclustered_percentage": (unclustered / total_chunks) * 100 if total_chunks > 0 else 0,
            "num_clusters": num_clusters,
            "cluster_sizes": {i: sum(self.clusters == i) for i in range(-1, max(self.clusters) + 1) if i >= -1}
        }
    
    def __str__(self):
        """String representation of the cluster."""
        stats = self.get_cluster_stats()
        if "status" in stats and stats["status"] == "not_clustered":
            return f"SubjectCluster('{self.context}', not clustered)"
            
        return (
            f"SubjectCluster('{self.context}', "
            f"{stats['num_clusters']} clusters, "
            f"{stats['clustered_chunks']}/{stats['total_chunks']} chunks clustered, "
            f"{len(self.subclusters)} subclusters)"
        )
    
    def __repr__(self):
        """Detailed representation of the cluster."""
        return self.__str__()