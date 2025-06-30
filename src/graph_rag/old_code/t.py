import numpy as np
import time
import asyncio

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4o-mini")

def get_summarizer_prompt(text):
    return f"""
    You are an expert in knowledge graph construction and statistical techniques in particle physics, tasked with creating a concise summary of the given text about statistical methods in particle physics. Preserve all essential technical information while reducing the overall length.

    Follow these instructions carefully:
    
    PRESERVE EXACTLY:
    - Well-established statistical method names and terminology (e.g., "Monte Carlo Methods", "Kolmogorov-Smirnov Test")
    - Named probability distributions and their properties (e.g., "Gaussian distribution", "Poisson distribution")
    - Standard statistical test descriptions and frameworks (e.g., "Neyman-Pearson framework")
    - Common particle physics observables and concepts (e.g., "invariant mass", "time-integrated luminosity")
    - Established numerical parameters and measurements with context
    - Foundational relationships between statistical concepts
    
    CONDENSE OR REMOVE:
    - Lengthy explanations or justifications
    - Text-specific variable names, notation, and symbols (e.g., "η_s(α)", "p^{{\\mathrm{{lo}}}}")
    - Equation-specific terms that aren't standard in the field
    - Mathematical formulas, equations, and notations (provide verbal explanations of important concepts instead)
    - One-off examples used only for illustration
    - Historical background information
    - Non-essential comparisons
    - Peripheral details not directly related to the statistical method
    
    IMPORTANT: When referring to statistical concepts, use their standard, widely-recognized names rather than document-specific notation or variables. Focus on describing relationships between established concepts rather than reproducing specific mathematical notation from the text.
    
    Maintain the technical accuracy of the content. Do not simplify technical concepts in a way that changes their meaning.
    
    The output should be a single continuous text (not bullet points) that retains all the technical essence of the original, focusing on established statistical methods and concepts.
    
    Text to summarize:
    
    {text}
    """

def get_entity_extraction_prompt(text, entity_types):
    return f"""
    You are an expert in knowledge graph construction and statistical techniques in particle physics, tasked with extracting entities from summarized text describing statistical methods in particle physics. Your objective is to identify only well-established, reusable statistical concepts and methodologies.
    
    Input: A summarized text about statistical methods in particle physics.
    
    Output: A list of Python dictionaries with the following structure:
    [
        {{"entity_name": "name of the entity", "entity_type": "what type of entity this is", "description": "a comprehensive and detailed description of the entity"}},
        ...
    ]
    
    Entity types include but are not limited to:
    {str(entity_types)[1:-1]}, etc...
    
    STRICT CRITERIA FOR ENTITY EXTRACTION:
    1. ONLY extract entities that represent well-established, named concepts in statistics or physics
    2. The entity must be a recognized methodology, test, distribution, or framework that appears in textbooks or literature
    3. DO NOT extract:
       - Document-specific variables or notation (e.g., "η_s(α)", "p^{{\\mathrm{{lo}}}}")
       - Arbitrary mathematical expressions (e.g., "-2ln L")
       - Generic concepts that are too broad (e.g., "Data-Intensive Science", "Strategic Planning")
       - Text-specific parameter names without wider significance
       - One-off examples used purely for illustration
    
    Examples of GOOD entities (extract these):
    - Monte Carlo Methods
    - Breit-Wigner function
    - XGBoost
    - Bayes' theorem
    - Kolmogorov-Smirnov Test
    - Type-I error
    - Gaussian distribution
    - Look-Elsewhere Effect
    
    Examples of BAD entities (DO NOT extract these):
    - Statistical Model (too generic)
    - master model F_tot(data | α) (this was some sort of function defined in a specific context, not a general technique)
    - shape function σ_s(x | α) (this is a specific function, not a technique)
    - f(q_mu/μ) (unclear what this even is, some sort of math symbol)
    - P, ν, p, M, F (single letters or symbols)
    - Upper limit (too generic without specific statistical meaning)
    - counting events (too generic description)
    
    Important guidelines:
    - Identify multi-word expressions as single entities (e.g., "Piecewise Linear Interpolation")
    - For each entity, verify it represents a well-established concept before including it
    - If an entity appears multiple times, combine the information into a single comprehensive entry
    - Include implementation details, advantages, or limitations mentioned ONLY for legitimate entities
    
    Do not include anything else in your output, including ```python ... ```
    
    Text to process:
    
    {text}
    """

def get_relationship_extraction_prompt(text, entities):
    return f"""
    You are an expert in knowledge graph construction and statistical techniques in particle physics, tasked with extracting relationships from the given summarized text describing statistical methods in particle physics starting from a list of statistical and particle physics entities.

    Relationships are defined as the connection between two of the given entities, e.g., "Bayesian inference is applied to Higgs boson detection" or "Maximum likelihood estimation quantifies W boson mass uncertainty."
    A typical relationship comprises an entity A, a relationship name, and an entity B. For example:
    - Entity A: "XGBoost"
    - Relationship: "is a special case of"
    - Entity B: "Decision Trees"

    Potential relationship types include:
    - "is applied to" (connecting statistical methods to physics problems)
    - "depends on" (showing dependencies between entities)
    - "is a special case of" (showing hierarchical relationships)
    - "is mathematically equivalent to" (showing equivalence relationships)
    - "outperforms" (comparing statistical approaches)
    - "quantifies" (showing measurement relationships)
    - "is implemented in" (connecting methods to computational tools)
    - "is derived from" (showing derivation relationships)
    but you should use other relational language when neccesary.

    Include a detailed description of this relationship, synthesizing all relevant information in the text which describes this relationship.
    
    Include a confidence score between 0 and 1 for each relationship based on how explicitly it is stated in the text.

    At your discretion, extra information about the relationship can be stored as attributes, such as:
    - conditions of applicability (when does this relationship hold true)
    - mathematical formulation (does the text provide an important formula to describe this relationship)
    - etc...

    The output should be a list of python dictionaries: 
    
    [
    {{"entity_name_a" : "...", "entity_name_b" : "...", "relationship_name" : "...", "description" : "...", "confidence_level" : "...", "conditions" : "..."}},
    ...
    ]
    
    Do not include anything else in your output, including ```python ... ```. 


    "entity_name_a" and "entity_name_b" MUST be equal to the "entity_name" field of an entity from the following list of possible entities:

    {entities}
    
    Text to process:

    {text}

    """

def process_section(text, entity_types):
    try:
        prompt = get_summarizer_prompt(text)
        summarized = llm.complete(prompt).text

        prompt = get_entity_extraction_prompt(summarized, entity_types)
        entities = eval(llm.complete(prompt).text)

        prompt = get_relationship_extraction_prompt(summarized, entities)
        relationships = eval(llm.complete(prompt).text)

        return entities, relationships
    
    except Exception as e:
        print(f"Exception occured while processing a section with text: \n {text} \n Exception: {e}")
        return [], []

async def process_section_async(text, entity_types):
    try:
        prompt = get_summarizer_prompt(text)
        summarized_response = await llm.acomplete(prompt)
        summarized = summarized_response.text
        
        prompt = get_entity_extraction_prompt(summarized, entity_types)
        entities_response = await llm.acomplete(prompt)
        entities = eval(entities_response.text)
        
        prompt = get_relationship_extraction_prompt(summarized, entities)
        relationships_response = await llm.acomplete(prompt)
        relationships = eval(relationships_response.text)
        
        return entities, relationships
    
    except Exception as e:
        print(f"Exception occured while processing a section with text: \n {text} \n Exception: {e}")
        return [], []

async def process_all_sections_async(chunks, entity_types):
    tasks = [process_section_async(chunk, entity_types) for chunk in chunks]
    all_results = await asyncio.gather(*tasks)
    
    all_entities = []
    all_relationships = []
    
    for entities, relationships in all_results:
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    return np.array(all_entities), np.array(all_relationships)

def process_all_sections(chunks, entity_types, chunk_size=50, sleep=10):
    all_entities, all_relationships = [], []
    for i in range(0, len(chunks), chunk_size):
        print(f"Processing chunks {i} - {i+chunk_size}")
        temp_chunks = chunks[i:i+chunk_size]
        temp_entities, temp_relationships = asyncio.run(process_all_sections_async(temp_chunks, entity_types))
        all_entities.append(temp_entities), all_relationships.append(temp_relationships)
        
        print(f"Sleeping for {sleep} seconds...")
        time.sleep(sleep)
        
    return np.concatenate(all_entities), np.concatenate(all_relationships)
