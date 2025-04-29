import numpy as np
import time
import asyncio
from typing import List, Dict, Tuple, Any, Set
import json
from sklearn.cluster import DBSCAN
import copy
import regex as re

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4o-mini")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("BAAI/bge-m3")


# ~~ LLM agent prompts ~~ #


def get_entity_extraction_prompt(text):
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in knowledge graph construction and statistical techniques in particle physics. Your objective is to extract entities from the provided text for a particle physics knowledge graph.

    Extract entities that belong only to these entity types:

    Type: analysis_technique 
    - Statistical or computational methods used in particle physics analyses. Examples include Maximum Likelihood Estimation, Profile Likelihood Ratio, Bootstrap Method, and Unfolding Techniques. 
    
    Type: statistics_concept
    - Fundamental statistical principles, definitions, or frameworks that form the theoretical basis for particle physics analyses. These concepts typically have precise mathematical definitions and are broadly applicable across many types of analyses. Examples include likelihood functions, p-values, and confidence intervals.

    Only extract entities that represent well-established techniques, concepts, or tasks in particle physics analysis. DO NOT extract:
    - Document-specific variables or notation (e.g., "η_s(α)", "p^{{\\mathrm{{lo}}}}")
    - Arbitrary mathematical expressions (e.g. "-2ln L")
    - Generic concepts that are too broad (e.g. "Data Analysis", "Statistics")
    - Text-specific parameter names without wider significance
    - One-off examples used purely for illustration (e.g. "Top Quark Mass Measurement", "Search for the Higgs Boson", )

    Input: Raw text describing statistical methods in particle physics.

    Output: A list of Python dictionaries with the following structure:
    [
        {{"entity_name": "name of the entity", "entity_type": "analysis_technique/statistics_concept", "description": "a comprehensive and detailed description of the entity"}},
        ...
    ]

    If the provided text doesn't include any relevant entities, return an empty list: []

    Do not include anything else in your output, including ```python ... ``` or ```json ... ```.

    ## EXAMPLE ##

    Input:
    The look-elsewhere effect refers to the statistical phenomenon where the apparent significance of an observed signal appears inflated when searching across multiple regions or parameters, increasing the probability of finding a seemingly significant result purely by random chance. To account for this effect, physicists employ the trial factor method, which quantifies how many independent statistical tests were effectively performed during the search. The trial factor represents the ratio between global and local significance, measuring how much the significance of an observed signal should be adjusted to account for multiple testing.
    This correction is especially important in bump hunt analyses, where physicists search for new particles appearing as resonant peaks across wide invariant mass distributions. Since examining numerous mass points constitutes multiple independent statistical tests on the same dataset, the probability of observing a random fluctuation that appears inconsistent with background expectations increases substantially. By applying the trial factor to convert local p-values to global p-values, physicists maintain appropriate statistical rigor and avoid false discovery claims that would otherwise occur from inadequately accounting for the breadth of the search space.

    Output:
    [
    {{ "entity_name": "look-elsewhere effect", "entity_type": "statistics_concept", "description": "A statistical phenomenon in particle physics where the significance of an observed signal is inflated due to testing across multiple regions or parameters, increasing the chance of finding a seemingly significant result by random fluctuation." }}, 
    {{ "entity_name": "trial factor method", "entity_type": "analysis_technique", "description": "A technique used to account for the look-elsewhere effect in particle physics by quantifying the effective number of independent statistical tests conducted, thereby adjusting local significance to obtain global significance." }}, 
    {{ "entity_name": "p-value", "entity_type": "statistics_concept", "description": "A statistical measure representing the probability of obtaining an observed result, or one more extreme, assuming the null hypothesis is true. It is used to assess the significance of results in particle physics analyses." }} 
    ]

    ## INPUT ##

    Input:
    {text}

    Output:
    """


def get_relationship_extraction_prompt(text, entities):
    entities_str = ""
    for entity in entities:
        entities_str += repr(entity) + "\n"
    
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in knowledge graph construction and statistical techniques in particle physics. Your objective is to read the provided text in order to find relationships between the provided entities.

    Input 1: A list of all acceptable entity_names and a description of each entity.

    Input 2: Raw text describing statistical methods in particle physics.

    Output: A list of Python dictionaries with the following structure:
    [
    {{"entity_name_a": "from Input 1", "entity_type_a": "from Input 1", "entity_name_b": "from Input 1", "entity_type_b": "from Input 1", "relationship_name": "from the allowed relationships based on entity types", "description": "a more detailed description of the relationship", "conditions": "are there any conditions or requirements for this relationship to hold true?", "confidence": 0.0}},
    ...
    ]
    The "confidence" field should be a floating-point number between 0.0 and 1.0 that represents your assessment of how confident you are that this relationship is accurately represented in the text:
    - 0.8-1.0: Relationship is explicitly stated with clear, direct language
    - 0.4-0.7: Relationship is heavily implied in the text
    - 0.1-0.3: Relationship can be reasonably inferred given the text plus your own domain knowledge

    Do not include anything else in your output, including ```python ... ``` or ```json ... ```. Do not make anything up.

    ## EXAMPLE ##

    Input 1:
    {{ "entity_name": "look-elsewhere effect", "entity_type": "statistics_concept", "description": "A statistical phenomenon in particle physics where the significance of an observed signal is inflated due to testing across multiple regions or parameters, increasing the chance of finding a seemingly significant result by random fluctuation." }}, 
    {{ "entity_name": "trial factor method", "entity_type": "analysis_technique", "description": "A technique used to account for the look-elsewhere effect in particle physics by quantifying the effective number of independent statistical tests conducted, thereby adjusting local significance to obtain global significance." }}, 
    {{ "entity_name": "p-value", "entity_type": "statistics_concept", "description": "A statistical measure representing the probability of obtaining an observed result, or one more extreme, assuming the null hypothesis is true. It is used to assess the significance of results in particle physics analyses." }} 

    Input 2:
    The look-elsewhere effect refers to the statistical phenomenon where the apparent significance of an observed signal appears inflated when searching across multiple regions or parameters, increasing the probability of finding a seemingly significant result purely by random chance. To account for this effect, physicists employ the trial factor method, which quantifies how many independent statistical tests were effectively performed during the search. The trial factor represents the ratio between global and local significance, measuring how much the significance of an observed signal should be adjusted to account for multiple testing.
    This correction is especially important in bump hunt analyses, where physicists search for new particles appearing as resonant peaks across wide invariant mass distributions. Since examining numerous mass points constitutes multiple independent statistical tests on the same dataset, the probability of observing a random fluctuation that appears inconsistent with background expectations increases substantially. By applying the trial factor to convert local p-values to global p-values, physicists maintain appropriate statistical rigor and avoid false discovery claims that would otherwise occur from inadequately accounting for the breadth of the search space.

    Output:
    [ 
    {{ "entity_name_a": "trial factor method", "entity_type_a": "analysis_technique", "entity_name_b": "look-elsewhere effect", "entity_type_b": "statistics_concept", "relationship_name": "addresses", "description": "The trial factor method is used specifically to address the look-elsewhere effect by adjusting significance levels to account for multiple testing.", "conditions": "The relationship holds in scenarios where multiple statistical tests are conducted during a search, as in bump hunt analyses.", "confidence": 1.0 }}, 
    {{ "entity_name_a": "look-elsewhere effect", "entity_type_a": "statistics_concept", "entity_name_b": "trial factor method", "entity_type_b": "analysis_technique", "relationship_name": "is addressed by", "description": "The look-elsewhere effect is mitigated through the use of the trial factor method, which adjusts for the inflated significance due to multiple testing.", "conditions": "Occurs in contexts where results are drawn from searching many possible signal regions, requiring correction for multiple testing.", "confidence": 1.0 }}, 
    {{ "entity_name_a": "trial factor method", "entity_type_a": "analysis_technique", "entity_name_b": "p-value", "entity_type_b": "statistics_concept", "relationship_name": "modifies", "description": "The trial factor method modifies p-values by converting local p-values to global p-values, accounting for the number of statistical tests performed.", "conditions": "This relationship applies when adjusting for multiple comparisons in signal significance testing.", "confidence": 1.0 }}, 
    {{ "entity_name_a": "look-elsewhere effect", "entity_type_a": "statistics_concept", "entity_name_b": "p-value", "entity_type_b": "statistics_concept", "relationship_name": "affects interpretation of", "description": "The look-elsewhere effect impacts the interpretation of p-values by making seemingly significant results less reliable due to multiple comparisons.", "conditions": "This effect is observed in analyses involving multiple hypotheses or search regions.", "confidence": 0.9 }} 
    ]

    ## INPUT ##

    Input 1:
    {entities_str}
    
    Input 2:
    {text}

    Output:
    """


def get_entity_merger_prompt(cluster_str):
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in knowledge graph construction and statistical techniques in particle physics. Your objective is to merge entities in a knowledge graph that have been deemed similar.
    
    Merging guidelines:
    - All provided entities should be merged into a single entity.
    - Choose a name which encompasses what all of the merged entities have in common. You may either take the name from among the merged entities, or use your domain knowledge to construct a new name.
    Example: "Toy Monte Carlo" + "Monte Carlo Sampling" + "Markov Chain Monte Carlo" -> "Monte Carlo Methods"
    Example: "Type I Statistical Error" + "Type II Statistical Error" -> "Type I and II Statistical Errors"
    Example: "Decision Trees" + "Boosted Decision Trees" + "Decision Tree Classifiers" -> "Decision Trees"
    - The most accurate entity type should be chosen from among the entities, with a preference for analysis_technique
    - The description should combine information from the descriptions of all of the merged entities.
    
    Input: a list of entities which have been deemed similar

    Output: A single merged entity in the following format:
    {{"entity_name" : "comprehensive and generally applicable name", "entity_type" : "analysis_technique/statistics_concept", "description" : "comprehensive description combining all information"}}
    
    Important output rules:
    1. You MUST merge ALL entities in the input list into a single entity.
    2. Do not include anything else in your output, including ```python ... ``` or ```json ... ```.

    ## EXAMPLE ##

    Input:
    [
    {{"entity_name": "Likelihood Ratio Test", "entity_type": "analysis_technique", "description": "A statistical test that compares the goodness of fit of two models, one of which is nested within the other. It's based on the ratio of the likelihood functions of the two models."}},
    {{"entity_name": "Likelihood Ratio", "entity_type": "statistics_concept", "description": "The ratio of the likelihood function under a specific value of the parameter to the likelihood function under the maximum likelihood estimate. Often used in hypothesis testing."}},
    {{"entity_name": "Likelihood Ratio Statistic", "entity_type": "statistics_concept", "description": "A test statistic based on the ratio of likelihood functions that follows a chi-squared distribution under certain conditions."}}
    ]

    Output:
    {{ "entity_name": "Likelihood Ratio Test Statistic", "entity_type": "analysis_technique", "description": "A statistical measure used in hypothesis testing that compares the fit of two nested models by computing the ratio of their likelihoods. This ratio, when expressed as a test statistic, often follows a chi-squared distribution under regular conditions." }}

    ## INPUT ##

    Input:
    {cluster_str}

    Output:
    """


def get_relationship_merger_prompt(relationship_cluster):
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in knowledge graph construction and statistical techniques in particle physics, tasked with merging similar relationships between the same entities.

    Merging guidelines:
    - All relationships in the input list must be merged into a single, comprehensive, and more general relationship.
    - You must maintain the same entity_name_a and entity_name_b fields as provided.
    - Choose the most accurate and descriptive relationship_name from among the provided relationships, or synthesize a new one that better captures the overall relationship if needed.
    - Create a unified description that combines information from all relationships, removing redundancy while preserving important details.
    - Select or synthesize conditions that best represent when this relationship applies.
    - Calculate a new confidence score based on the input relationships.

    Input: A list of relationships that needs merging

    Output: A single merged relationship in the following format:
    {{"entity_name_a": "from Input", "entity_type_a": "from Input", "entity_name_b": "from Input", "entity_type_b": "from Input", "relationship_name": "more general relationship name", "description": "more general relationship description", "conditions": "more general discussion on the conditions", "confidence": 0.0}}

    Important output rules:
    1. You MUST merge ALL relationships in the input list into a single relationship.
    2. The entity_name_a and entity_name_b must match exactly what was in the input relationships.
    3. Do not include anything else in your output, including ```python ... ``` or ```json ... ```.
    4. Return a well-formed Python dictionary that can be directly evaluated with `eval()`.

    ## EXAMPLE ##

    Input:
    [
    {{"entity_name_a": "Parameters of Interest", "entity_type_a": "statistics_concept", "entity_name_b": "Nuisance Parameters", "entity_type_b": "statistics_concept", "relationship_name": "are distinguished from", "description": "Parameters of interest are the specific key parameters in a statistical model that researchers aim to estimate or test, while nuisance parameters are additional parameters that must be accounted for but are not of primary interest.", "conditions": "In the context of statistical modeling and analysis in particle physics", "confidence": 0.9}},
    {{"entity_name_a": "Parameters of Interest", "entity_type_a": "statistics_concept", "entity_name_b": "Nuisance Parameters", "entity_type_b": "statistics_concept", "relationship_name": "are split into", "description": "In statistical analysis, parameters can be categorized into parameters of interest and nuisance parameters, where the former are the key parameters being estimated and the latter are additional parameters that are not the main focus.", "conditions": "When analyzing data with multiple parameters in particle physics", "confidence": 0.9}}
    ]

    Output:
    {{"entity_name_a": "Parameters of Interest", "entity_type_a": "statistics_concept", "entity_name_b": "Nuisance Parameters", "entity_type_b": "statistics_concept", "relationship_name": "are categorized with", "description": "In particle physics statistical modeling, parameters are often categorized into parameters of interest—those that are the focus of estimation or hypothesis testing—and nuisance parameters, which are necessary for the model but not of direct interest. This distinction is essential for accurately modeling uncertainty and for constructing profile likelihoods or marginalizations.", "conditions": "Applies in statistical analyses involving multiple parameters where distinguishing between key and auxiliary quantities improves inference quality and clarity.", "confidence": 0.95}}

    ## INPUT ##

    Input:
    {relationship_cluster}

    Output:
    """


class ChunkProcessingResult:
    def __init__(self, 
                 text: str, 
                 entities: List[Dict] = None):
        self.text = text
        self.entities = entities or []


# ~~ Entity extraction helper functions ~~ #


async def extract_entities(chunk_text: str):
    """Extract entities from a chunk of text"""
    result = ChunkProcessingResult(chunk_text)
    
    response = None
    try:        
        prompt = get_entity_extraction_prompt(result.text)
        #print(prompt)
        response = (await llm.acomplete(prompt)).text
        entities = eval(response)

        # Standardize names to be lowercase and free of punctuation, attach citation
        for entity in entities:
            entity["entity_name"] = re.sub(r"[^\w\s]", "", entity["entity_name"].lower())
            entity["relevant_passages"] = set([chunk_text])

        result.entities = entities
        return result
        
    except Exception as e:
        print(f"Exception occurred while processing chunk: {e} \n Chunk text: \n {chunk_text} \n Got response: \n {response}")
        return result


# ~~ Entity merger helper functions ~~ #


def cluster_entities(entities: List[Dict], threshold: float = 0.7):
    """Cluster entities based on the embedding similarity of their names and descriptions"""
    entity_names = []
    entity_types = []
    entity_descriptions = []
    for entity in entities:
        entity_names.append(entity["entity_name"])
        entity_types.append(entity["entity_type"])
        entity_descriptions.append(entity["description"])
    entity_names = np.array(entity_names)
    entity_types = np.array(entity_types)
    entity_descriptions = np.array(entity_descriptions)

    
    entity_name_embeddings = embedding_model.encode(entity_names)
    entity_description_embeddings = embedding_model.encode(entity_descriptions)
    entity_same_type = (entity_types[:, None] == entity_types[None, :]).astype(int)
    similarity_matrix = entity_same_type * ( (2.0/3.0) * (entity_name_embeddings @ entity_name_embeddings.T) + (1.0/3.0) * (entity_description_embeddings @ entity_description_embeddings.T) )

    distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
    
    clustering = DBSCAN(metric="precomputed", eps=1-threshold, min_samples=1)
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    unique_labels = set(cluster_labels)
    clusters = [[] for _ in range(len(unique_labels))]
    
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
        
    return clusters


async def merge_entity_cluster(cluster_ids: List[int], entities: List[Dict]):
    """Merge entities which have been deemed similar"""
    id_to_merged_entity = {}

    response = None
    try:
        if len(cluster_ids) == 1:
            id_to_merged_entity[cluster_ids[0]] = entities[cluster_ids[0]]
            return id_to_merged_entity
        
        cluster_entities = copy.deepcopy(entities[cluster_ids])
        relevant_passages = set([])
        cluster_str = ""
        for entity in cluster_entities:
            relevant_passages.update(entity.pop("relevant_passages"))
            cluster_str += repr(entity) + "\n"

        prompt = get_entity_merger_prompt(cluster_str)
        #print(prompt)
        response = (await llm.acomplete(prompt)).text
        merged_entity = eval(response)
        
        # Entity names should be free of capitalization and punctuation
        merged_entity["entity_name"] = re.sub(r"[^\w\s]", "", merged_entity["entity_name"].lower())
        merged_entity["relevant_passages"] = relevant_passages

        for id in cluster_ids:
            id_to_merged_entity[id] = merged_entity

        return id_to_merged_entity
    
    except Exception as e:
        print(f"Error merging cluster w/ ids {cluster_ids}: {e} \n with response: \n {response}")
        return id_to_merged_entity
    

# ~~ Relationship extraction helper functions ~~ #


async def extract_relationships(text: str, entities: List[Dict]):
    """Extract relationships from a chunk using a list of entities"""
    response = None
    try:
        entities = copy.deepcopy(entities)
        for entity in entities:
            entity.pop("relevant_passages")
                    
        prompt = get_relationship_extraction_prompt(text, entities)
        #print(prompt)
        response = (await llm.acomplete(prompt)).text
        relationships = eval(response)

        # Entity names should be free of capitalization and punctuation, attach citation
        for relationship in relationships:
            relationship["entity_name_a"] = re.sub(r"[^\w\s]", "", relationship["entity_name_a"].lower())
            relationship["entity_name_b"] = re.sub(r"[^\w\s]", "", relationship["entity_name_b"].lower())
            relationship["relevant_passages"] = set([text])
            

        # Filter out any relationships pointing to entities that don"t exist in the knowledge graph
        entity_names = [e["entity_name"] for e in entities]
        relationships = list(filter(lambda rel : rel["entity_name_a"] in entity_names and rel["entity_name_b"] in entity_names, relationships))
        
        return relationships
        
    except Exception as e:
        print(f"Exception occurred while extracting relationships from chunk: {e} \n with response: \n {response}")
        return []


def get_populate_relevant_entities(merged_entities: List[Dict], n_populate: int = 8):
    """
    Constructs a RAG system for finding additional relevant entities based on text similarity.
    """
    entity_strings = []
    for entity in merged_entities:
        entity_str = f"Name: {entity['entity_name']} \n Description: {entity['description']}"
        entity_strings.append(entity_str)
    
    entity_embeddings = embedding_model.encode(entity_strings)
    
    def populate_relevant_entities(text : str, relevant_entities : List[Dict]):
        """Populates a list of relevant entities with additional entities based on text similarity."""
        
        text_embedding = embedding_model.encode(text)
        
        similarity_scores = np.dot(text_embedding, entity_embeddings.T)
        
        top_indices = np.argsort(similarity_scores)[::-1]
        
        added_count = 0
        
        for idx in top_indices:
            if added_count >= n_populate:
                break

            candidate_entity = merged_entities[idx]
            if candidate_entity not in relevant_entities:
                relevant_entities.append(candidate_entity)
                added_count += 1
    
    return populate_relevant_entities


# ~~ Relationship merger helper functions ~~ #

def cluster_relationships(relationships):
    """Cluster relationships which share a starting and ending entity"""
    relationship_clusters = {}
    
    for relationship in relationships:
        entity_name_a, entity_name_b = relationship["entity_name_a"], relationship["entity_name_b"]
        entity_pair = (entity_name_a, entity_name_b)
        if entity_pair not in relationship_clusters:
            relationship_clusters[entity_pair] = []
        relationship_clusters[entity_pair].append(relationship)
    
    return list(relationship_clusters.values())

async def merge_relationship_cluster(cluster_relationships):
    """Merge relationships which have been deemed similar"""
    response = None
    try:
        if len(cluster_relationships) == 1:
            return cluster_relationships[0]

        cluster_relationships = copy.deepcopy(cluster_relationships)

        relevant_passages = set([])
        cluster_str = ""
        for relationship in cluster_relationships:
            relevant_passages.update(relationship.pop("relevant_passages"))
            cluster_str += repr(relationship) + "\n"

        prompt = get_relationship_merger_prompt(cluster_str)
        #print(prompt)

        response = (await llm.acomplete(prompt)).text
        merged_relationship = eval(response)

        # Entity names should be free of capitalization and punctuation
        merged_relationship["entity_name_a"] = re.sub(r"[^\w\s]", "", merged_relationship["entity_name_a"].lower())
        merged_relationship["entity_name_b"] = re.sub(r"[^\w\s]", "", merged_relationship["entity_name_b"].lower())
        merged_relationship["relevant_passages"] = relevant_passages

        return merged_relationship
    
    except Exception as e:
        print(f"Error merging relationship cluster {cluster_relationships}: {e} \n with response: \n {response}")
        return cluster_relationships[0]


# ~~ Build full knowledge graph ~~ #


async def build_knowledge_graph(chunks: List[str], chunk_size: int = 40, sleep_time: int = 10):
    """
    Build knowledge graph in two phases:
    1. Extract all entities from all chunks
    2. Extract relationships using the complete entity list
    """

    # ~~ Phase 1: Extract entities from all chunks ~~ #
    print("\nPhase 1: Extracting entities from all chunks")

    all_entities = []
    chunk_results = []

    for i in range(0, len(chunks), chunk_size):
        batch_end = min(i + chunk_size, len(chunks))
        print(f"Processing chunks {i} to {batch_end-1}")
        
        tasks = []
        for j in range(i, batch_end):
            task = extract_entities(chunks[j])
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        chunk_results.extend(batch_results)
        
        batch_entities = []
        for result in batch_results:
            batch_entities.extend(result.entities)
        
        all_entities.extend(batch_entities)
        
        if batch_end < len(chunks):
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

    all_entities = np.array(all_entities)
    
    # ~~ Phase 2: Deduplicate entities ~~ #
    print(f"\nPhase 2: Deduplicating {len(all_entities)} extracted entities")

    merged_entities = []
    
    entity_clusters = cluster_entities(all_entities)

    id_to_merged_entities = {}

    tasks = []
    for i in range(0, len(entity_clusters)):
        task = merge_entity_cluster(entity_clusters[i], all_entities)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for result in results:
        id_to_merged_entities.update(result)

    for entity in id_to_merged_entities.values():
        if entity not in merged_entities:
            merged_entities.append(entity)    
    merged_entities = np.array(merged_entities)
    
    print(f"After deduplication: {len(merged_entities)} unique entities")

    # ~~ Phase 3: Extract relationships with the merged entity list ~~ #
    print("\nPhase 3: Extracting relationships using merged entities")

    all_relationships = []

    populate_relevant_entities = get_populate_relevant_entities(merged_entities)

    for i in range(0, len(chunk_results), chunk_size):
        batch_end = min(i + chunk_size, len(chunk_results))
        print(f"Processing relationships for chunks {i} to {batch_end-1}")
        
        tasks = []
        for j in range(i, batch_end):
            chunk_result = chunk_results[j]

            relevant_entities = []
            for entity in chunk_result.entities:
                entity_id = int(np.where(all_entities == entity)[0][0])
                merged_entity = id_to_merged_entities[entity_id]
                if merged_entity not in relevant_entities:
                    relevant_entities.append(merged_entity)

            populate_relevant_entities(chunk_result.text, relevant_entities)

            task = extract_relationships(chunk_result.text, relevant_entities)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        for result in results:
            all_relationships.extend(result)
        
        if batch_end < len(chunk_results):
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

    all_relationships = np.array(all_relationships)

    # ~~ Phase 4: Deduplicate relationships
    print(f"\nPhase 4: Deduplicating {len(all_relationships)} extracted relationships")

    merged_relationships = []

    relationship_clusters = cluster_relationships(all_relationships)

    tasks = []
    for i in range(0, len(relationship_clusters)):
        task = merge_relationship_cluster(relationship_clusters[i])
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for result in results:
        merged_relationships.append(result)

    merged_relationships = np.array(merged_relationships)

    print(f"After deduplication: {len(merged_relationships)} unique relationships")

    return merged_entities, merged_relationships

def process_all_sections(chunks, chunk_size=40, sleep_time=10):
    entities, relationships = asyncio.run(build_knowledge_graph(chunks, chunk_size, sleep_time))
    return entities, relationships


## ~~ ##

def get_lhcb_kg_prompt(text, paper_name, entities):
    entities_str = ""
    for entity in copy.deepcopy(entities):
        entity.pop("relevant_passages", None)
        entities_str += repr(entity) + "\n"
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in knowledge graph construction and statistical techniques in particle physics. Your objective is to identify how a particle physics paper connects to existing statistical methods and concepts in a knowledge graph.

    Your task is to:
    1. Analyze the provided text from the paper
    2. Identify any mentions or uses of the statistical methods and concepts listed in the existing knowledge graph
    3. Create relationships that connect the paper to those existing entities
    4. Each relationship should describe how the paper uses, implements, or discusses the statistical method or concept

    Input 1: Raw text from a particle physics paper.
    Input 2: A list of existing statistical entities in the knowledge graph.
    Input 3: The name of the paper.

    Output: A list of Python dictionaries representing relationships between the paper and existing entities:
    [
        {{"entity_name_a": "{paper_name}", "entity_type_a": "lhcb_paper", "entity_name_b": "name from Input 2", "entity_type_b": "type from Input 2", "relationship_name": "name for the relationship", "description": "detailed description of how the paper uses or relates to this statistical method/concept", "confidence": 0.0}},
        ...
    ]
    The "confidence" field should be a floating-point number between 0.0 and 1.0 that represents your assessment of how clearly the relationship is demonstrated in the text:
    - 0.8-1.0: Relationship is explicitly stated with clear evidence in the text
    - 0.4-0.7: Relationship is heavily implied in the text
    - 0.1-0.3: Relationship can be reasonably inferred given the text plus domain knowledge

    If the provided text doesn't mention or use any of the listed statistical entities, return an empty list: []

    Do not include anything else in your output, including ```python ... ``` or ```json ... ```.

    ## EXAMPLE ##

    Input 1:
    In this paper, we measure the CP violating phase φs in the decay Bs → J/ψφ using proton-proton collision data recorded by the LHCb detector. We employ a time-dependent angular analysis with the sPlot technique to statistically separate signal and background components. The sPlot method allows us to extract signal distributions without explicitly modeling background contributions in the angular variables. For the maximum likelihood fit, we use a double-sided Crystal Ball function to model the invariant mass distribution of the signal component, which accounts for the radiative tails characteristic of LHCb detector resolution. The background is modeled with an exponential function. Systematic uncertainties are evaluated using a modified bootstrap technique specifically developed for decay time acceptance corrections in LHCb analyses. This method involves generating multiple bootstrap samples of the calibration data, recalculating acceptance factors for each sample, and propagating these variations through the analysis chain.

    Input 2:
    {{"entity_name": "maximum likelihood estimation", "entity_type": "analysis_technique", "description": "A method of estimating the parameters of a statistical model by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable."}}
    {{"entity_name": "bootstrap method", "entity_type": "uncertainty_handling", "description": "A resampling technique used to estimate statistics on a population by sampling a dataset with replacement, providing an empirical distribution that can be used to approximate the sampling distribution of a statistic."}}
    {{"entity_name": "splot technique", "entity_type": "analysis_technique", "description": "A statistical method used to disentangle different sources of events in a dataset based on the distributions of discriminating variables."}}
    {{"entity_name": "background subtraction", "entity_type": "analysis_technique", "description": "A technique used to remove the contribution of background events from data, allowing for more accurate analysis of the signal of interest."}}

    Input 3:
    Measurement of CP violation in Bs → J/ψφ decays

    Output:
    [
        {{"entity_name_a": "Measurement of CP violation in Bs → J/ψφ decays", "entity_type_a": "lhcb_paper", "entity_name_b": "splot technique", "entity_type_b": "analysis_technique", "relationship_name": "uses", "description": "The paper uses the sPlot technique to statistically separate signal and background components in their analysis of Bs → J/ψφ decays, allowing them to extract signal distributions without explicitly modeling background contributions in the angular variables.", "conditions": "Applied in the context of a time-dependent angular analysis for CP violation measurement", "confidence": 0.9}},
        {{"entity_name_a": "Measurement of CP violation in Bs → J/ψφ decays", "entity_type_a": "lhcb_paper", "entity_name_b": "maximum likelihood estimation", "entity_type_b": "analysis_technique", "relationship_name": "uses", "description": "The paper employs maximum likelihood estimation with a double-sided Crystal Ball function to model the invariant mass distribution of the signal component in their analysis.", "conditions": "Used for modeling the invariant mass distribution with specific probability density functions", "confidence": 0.9}},
        {{"entity_name_a": "Measurement of CP violation in Bs → J/ψφ decays", "entity_type_a": "lhcb_paper", "entity_name_b": "bootstrap method", "entity_type_b": "uncertainty_handling", "relationship_name": "modifies", "description": "The paper applies a modified version of the bootstrap technique specifically developed for decay time acceptance corrections in LHCb analyses to evaluate systematic uncertainties.", "conditions": "Used when evaluating systematic uncertainties related to decay time acceptance corrections", "confidence": 0.8}},
        {{"entity_name_a": "Measurement of CP violation in Bs → J/ψφ decays", "entity_type_a": "lhcb_paper", "entity_name_b": "background subtraction", "entity_type_b": "analysis_technique", "relationship_name": "discusses", "description": "While not explicitly using traditional background subtraction, the paper discusses the concept in relation to their approach of using sPlot for statistically separating signal from background.", "conditions": "In the context of analyzing angular distributions where explicit background modeling is avoided", "confidence": 0.6}}
    ]

    ## INPUT ##

    Input 1:
    {text}

    Input 2:
    {entities_str}

    Input 3:
    {paper_name}

    Output:
    """

# Type: measurement_task
# - Analysis objectives focused on determining the value (and uncertainty) of physical parameters or properties. These tasks involve quantifying properties of particles, interactions, or physical processes and typically result in a numerical value with associated uncertainties for a physical quantity. Examples include cross-section measurements, branching ratio determinations, and particle mass measurements.

# Type: discovery_task
# - Analysis objectives aimed at finding new particles, interactions, or phenomena not previously observed. These tasks involve discriminating potential signals from background and quantifying the statistical significance of any excess. Examples include bump hunting in invariant mass distributions, searches for rare decays, and anomaly detection.

# Type: exclusion_task
# - Analysis objectives designed to constrain theoretical parameters or rule out potential new physics models when no significant signal is observed. These tasks establish bounds on what is not allowed by the data. Examples include setting upper limits on production cross-sections, constraining parameter spaces of beyond-Standard Model theories, and limit setting on branching ratios.

# Type: uncertainty_handling
# - Methods specifically developed to quantify, propagate, or mitigate uncertainties in particle physics analyses. These methods focus on ensuring the robustness of results against various sources of uncertainty. These approaches deal with both statistical and systematic uncertainties. Examples include systematic error estimation techniques, constraint term modeling, and nuisance parameter profiling.

async def extract_lhcb_kg(text, paper_name, relevant_entities):
    """Extract entities from a chunk of text"""
    
    response = None
    try:
        prompt = get_lhcb_kg_prompt(text, paper_name, relevant_entities)
        response = (await llm.acomplete(prompt)).text

        relationships = []
        for relationship in eval(response):
            relationships.append(relationship)

        for relationship in relationships:
            relationship["relevant_passages"] = set([text])

        return relationships
        
    except Exception as e:
        print(f"Exception occurred while processing chunk: {e} \n Chunk text: \n {text} \n Got response: \n {response}")
        return []


async def build_lhcb_kg_extension(chunks, chunk_to_paper, existing_entities, chunk_size: int = 40, sleep_time: int = 10):
    all_entities = []
    for paper in set(chunk_to_paper.values()):
        entity = {"entity_name": paper, "entity_type": "lhcb_paper", "description": "One of the LHCb corpus papers"}
        all_entities.append(entity)

    all_relationships = []
    populate_relevant_entities = get_populate_relevant_entities(existing_entities, 10)
    for i in range(0, len(chunks), chunk_size):
        batch_end = min(i + chunk_size, len(chunks))
        print(f"Processing chunks {i} to {batch_end-1}")
        
        tasks = []
        for j in range(i, batch_end):
            relevant_entities = []
            populate_relevant_entities(chunks[j], relevant_entities)
            task = extract_lhcb_kg(chunks[j], chunk_to_paper[chunks[j]], relevant_entities)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        
        batch_relationships = []
        for result in batch_results:
            batch_relationships.extend(result)

        all_relationships.extend(batch_relationships)
        
        if batch_end < len(chunks):
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

    all_entities = np.array(all_entities)
    all_relationships = np.array(all_relationships)

    merged_relationships = []

    relationship_clusters = cluster_relationships(all_relationships)

    tasks = []
    for i in range(0, len(relationship_clusters)):
        task = merge_relationship_cluster(relationship_clusters[i])
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for result in results:
        merged_relationships.append(result)

    merged_relationships = np.array(merged_relationships)

    return all_entities, merged_relationships


def lhcb_kg_extension(chunks, chunk_to_paper, existing_entities, chunk_size=40, sleep_time=10):
    entities, relationships = asyncio.run(build_lhcb_kg_extension(chunks, chunk_to_paper, existing_entities, chunk_size, sleep_time))
    return entities, relationships