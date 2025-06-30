import numpy as np
import time
import asyncio
import regex as re
import copy
from typing import List
from sklearn.cluster import DBSCAN

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4.1-mini")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("BAAI/bge-m3")

class LHCbPaper():
    """_summary_
    """

    def __init__(self, abstract, text, arxiv_id):
        self.abstract = self.clean_latex(abstract)
        self.text = self.clean_latex(text)
        self.arxiv_id = arxiv_id

        self.sections = self.split_sections(self.text)

    def __repr__(self):
        return f"{self.arxiv_id}:\n\t{self.abstract}"

    def clean_latex(self, text):
        environments = [
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
            r'\\begin\{wrapfigure\*?\}.*?\\end\{wrapfigure\*?\}',
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            r'\\label\{.*?\}',
            r'.*\\begin\{document\}',
        ]
        
        for pattern in environments:
            text = re.sub(pattern, ' ', text, flags=re.DOTALL)

        marker = "\\section{Introduction}"
        position = text.find(marker)
        if position > 0:
            text = text[position:]

        text = re.sub(r'\%.*\n', "", text)
        text = re.sub(r"\n\s*", "\n", text)
        text = re.sub(r'([^\S\n])+', ' ', text)
        return text

    def split_sections(self, text, depth=0, title="", max_tokens=6000):
        pattern = r"(\\" + "sub" * depth + r"section[\*\s]*(?:\[[^\]]*\])?\s*({(?:[^{}]*+|(?2))*}))"
        matches = re.finditer(pattern, text)

        if not matches or depth > 3:
            return [text]

        sections = []
        start = 0
        section_title = ""
        for match in [(match.start(), match.end()) for match in matches] + [(-1, -1)]:
            end = match[0]
            section_text = text[start:end]
            if len(re.sub("\s", "", section_text)) > 0:
                new_title = f"{title}\n{section_title}" if len(title) > 0 else section_title
                sections.append(new_title + section_text)

            start = match[1]
            section_title = text[end:start]

        return sections
    
    async def process_paper(self):
        metadata_task = self.classify_abstract(self.abstract)
        entities_task = self.extract_entities(f"{self.abstract} \n {self.text}")

        self.metadata, self.entities = await asyncio.gather(metadata_task, entities_task)
        self.relationships = await self.extract_relationships(f"{self.abstract} \n {self.text}", self.entities)

    def get_abstract_classification_prompt(self, abstract):
        return f"""You are a research scientist in the LHCb collaboration at CERN and a particle physics expert. You will be given an abstract from an LHCb analysis paper and extract the information described below.

        1. DECAYS

        Identify all particle decay channels. For each decay channel, provide the production method, parent particle(s), and children particle(s). You must select production methods and particles from the following lists:
        - production: p-p, Pb-Pb, p-Pb, Xe-Xe, O-O, Pb-Ar, p-O
        - particles: u, u~, d, d~, s, s~, c, c~, b, b~, t, t~, e-, e+, mu-, mu+, tau-, tau+, nu(e), nu(e)~, nu(mu), nu(mu)~, nu(tau), nu(tau)~, g, gamma, Z0, W+, W-, H0, pi0, pi+, pi-, eta, eta', K0, K~0, K+, K-, K*(892)0, K*(892)~0, K*(892)+, K*(892)-, rho(770)0, rho(770)+, rho(770)-, omega(782), phi(1020), J/psi(1S), psi(2S), psi(3770), Upsilon(1S), Upsilon(2S), Upsilon(3S), Upsilon(4S), D0, D~0, D+, D-, D*(2010)+, D*(2010)-, D*(2007)0, D*(2007)~0, D(s)+, D(s)-, D(s)*+, D(s)*-, B0, B~0, B+, B-, B(s)0, B(s)~0, B(c)+, B(c)-, p, p~, n, n~, Lambda, Lambda~, Lambda(b)0, Lambda(b)~0, Lambda(c)+, Lambda(c)~-, Sigma+, Sigma-, Sigma0, Sigma~+, Sigma~-, Sigma~0,  Xi0, Xi-, Xi~0, Xi~+, Xi(b)0, Xi(b)-, Xi(b)~0, Xi(b)~+, Xi(c)0, Xi(c)+, Xi(c)~0, Xi(c)~-, Omega-, Omega~+, Omega(b)-, Omega(b)~+, Omega(c)0, Omega(c)~0, chi(c0)(1P), chi(c1)(1P), chi(c2)(1P), chi(b0)(1P), chi(b1)(1P), chi(b2)(1P), eta(c)(1S), eta(b)(1S), X(1)(3872), X(2)(3872), Z(4430)+, Z(4430)-
        
        You may need to use your knowledge as a particle physics expert when classifying. For a paper that "measures B_s \\to \\mu^+ \\mu^- and B_0 -> \\mu^+ \\mu^- in pp collisions" the "B_s" would be written as "B(s)0" because B_s (strange B-meson) corresponds to B(s)0 in the provided list of allowed particles.

        2. RUN

        Identify the LHCb data-taking period or dataset characteristics
        - "run1": Data collected during 2011-2012 at center-of-mass energies of 7-8 TeV
        - "run2": Data collected during 2015-2018 at a center-of-mass energy of 13 TeV
        - "run1,run2": Combined dataset from both Run 1 and Run 2 periods (2011-2018)
        - "run3": Data collected during 2022-2025 at a center-of-mass energy of 13.6 TeV
        - "ift": Any paper which does not use p-p as its production method.

        3. STRATEGY
        
        Identify which of following labels best describes the analysis strategy employed in the paper.

        - "angular_analysis": Studies that examine angular distributions or asymmetries in particle decays. These include forward-backward asymmetries, angular coefficients, or observables related to Wilson coefficients. These analyses typically perform multi-dimensional fits to angular variables to extract physics parameters and test Standard Model predictions.

        - "amplitude_analysis": Studies focused on decay amplitude structures, including Dalitz plot analyses, determination of resonance properties, spin-parity assignments, or relative phase measurements. These analyses model interfering amplitudes and extract quantities such as fit fractions, strong phases, or resonance parameters.
                
        - "search": Analyses aimed at discovering previously unobserved states, processes, or symmetry-breaking effects. This includes first observations of rare decay modes, searches for new particles, or forbidden processes. Often results in significance measurements for new signals or limit setting at confidence levels (typically 90% or 95%) when no significant signal is observed.
                
        - "other": Any analysis that does not fall into the above three categories. This includes analyses primarily focused on precision measurements of established quantities, such as branching fraction determinations, lifetime measurements, production cross-sections, mass measurements.

        Format your answer exactly as in the below example output. Do not respond with anything else.

        # EXAMPLE - INPUT

        Branching fractions of the decays $H_b\\to H_c\\pi^-\\pi^+\\pi^-$ relative to $H_b\\to H_c\\pi^-$ are presented, where $H_b$ ($H_c$) represents B^0-bar($D^+$), $B^-$ ($D^0$), B_s^0-bar ($D_s^+$) and $\\Lambda_b^0$ ($\\Lambda_c^+$). The measurements are performed with the LHCb detector using 35${{\\rm pb^{{-1}}}}$ of data collected at $\\sqrt{{s}}=7$ TeV. The ratios of branching fractions are measured to be B(B^0-bar -> D^+\\pi^-\\pi^+\\pi^-)/ B(B^0-bar -> D^+\\pi^-) = 2.38\\pm0.11\\pm0.21 B(B^- -> D^0\\pi^-\\pi^+\\pi^-) / B(B^- -> D^0\\pi^-) = 1.27\\pm0.06\\pm0.11 B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-) = 2.01\\pm0.37\\pm0.20 B(\\Lambda_b^0->\\Lambda_c^+\\pi^-\\pi^+\\pi^-) / B(\\Lambda_b^0 -> \\Lambda_c^+\\pi^-) = 1.43\\pm0.16\\pm0.13. We also report measurements of partial decay rates of these decays to excited charm hadrons. These results are of comparable or higher precision than existing measurements.

        # EXAMPLE - OUTPUT
        {{
            "explanation": "Since the production method is not explicitly mentioned, it is likely p-p. This paper studies H_b to H_c pi- pi+ pi- and H_b to H_c pi- for four different values of H_b (H_c). Using names from the provided list of particles: B^0-bar ($D^+$) becomes B~0 (D+), $B^-$ ($D^0$) becomes B- (D0), B_s^0-bar ($D_s^+$) becomes B(s)~0 (D(s)+), and $\\Lambda_b^0$ ($\\Lambda_c^+$) becomes Lambda(b)0 (Lambda(c)). The energy is 7 TeV, indicating run1. The paper reports precision measurements of branching fraction ratios, not a search, angular, or amplitude analysis, so strategy is 'other'.",
            "decays": [
                {{
                    "production": "p-p",
                    "parent": "B~0",
                    "children": ["D+", "pi-", "pi+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "B~0",
                    "children": ["D+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "B-",
                    "children": ["D0", "pi-", "pi+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "B-",
                    "children": ["D0", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "B(s)~0",
                    "children": ["D(s)+", "pi-", "pi+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "B(s)~0",
                    "children": ["D(s)+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "Lambda(b)0",
                    "children": ["Lambda(c)+", "pi-", "pi+", "pi-"]
                }},
                {{
                    "production": "p-p",
                    "parent": "Lambda(b)0",
                    "children": ["Lambda(c)+", "pi-"]
                }}
            ],
            "run": "run1",
            "strategy": "other"
        }}

        # INPUT

        {abstract}

        # OUTPUT
        """

    async def classify_abstract(self, abstract: str):
        response = None
        try:        
            prompt = self.get_abstract_classification_prompt(abstract)
            response = (await llm.acomplete(prompt)).text
            metadata = eval(response.replace("```python","").replace("```",""))
            return metadata
            
        except Exception as e:
            print(f"Exception occurred: {e} \n Paper Abstract: \n {abstract} \n Got Response: \n {response}")
            return {
                "focus": "<UNK>",
                "run": "<UNK>",
                "strategy": "<UNK>"
            }

    def get_entity_extraction_prompt(self, text):
        return f"""# Systematic Uncertainty Knowledge Graph - Entity Extraction

        You are an expert physicist specialized in extracting entities for a knowledge graph of systematic uncertainties in particle physics publications. Your task is to identify and extract all relevant entities related to systematic uncertainties from the provided text.

        # ENTITY TYPES AND PROPERTIES

        1. Observable

        Identify all physical quantities which are being measured or derived from experimental data in the paper (ex: CKM mixing angles, branching ratios, CP violation parameters, differential cross-sections, particle masses, particle lifetimes, etc.). Although some papers measure the same observable across multiple kinematic bins, only one entity should be extracted regardless of the number of bins (ex: an observable may be measured for several di-muon invariant masses, but you should only create one entity). Do not include broad physics questions motivating the analysis such as lepton universality, matter-antimatter asymmetry, etc.

        Categorize observable according to the following definitions:
        - "branching_fraction": Often explicitly called a branching fraction. Ex: B(B^- -> D^0\\pi^-\\pi^+\\pi^-)
        - "branching_ratio": Often explicitly called a branching fraction ratio. Ex: B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-)
        - "physical_constant": Any fundamental physics parameter that is not a branching fraction. Ex: CP-violating phase (gamma), CKM angles, wilson coefficients, particle lifetimes, etc.
        - "angular_observable": This is any observable coming from an angular analysis. Ex: asymmetries in angular distributions, polarization fractions, or helicity amplitudes.
        - "functional_dependence": This refers to observables that are measured as a function of kinematic variables. Ex: distribution of p_T for a specific B-hadron, differential cross sections as a function of rapidity, form factors as a function of q^2, or resonance structures in invariant mass distributions.

        2. Uncertainty Source

        Identify all sources of uncertainty described in the paper which affect the measured values of any of the observables extracted above.
        
        Categorize the uncertainty source according to the following definitions:
        - "statistical": An uncertainty resulting from variability in the measured data due to random fluctuations or sampling limitations.
        - "internal": An uncertainty that is the result of choices made by the authors while performing the analysis, such as choices in reconstruction, modelling of the efficiencies, treatment of the background, etc. Usually related to the LHCb experiment itself, the reconstruction, or the analysis techniques used.
        - "external": An uncertainty related to external inputs to the analysis, such as values taken from a theoretical calculation or a previous measurement which was not done in this analysis. This uncertainty could not be improved by changing anything about the analysis. Ex: using the mass of the phi resonance from the PDG world average or using the PDG branching ratio for B(B0 -> D- pi+) as input when normalizing a rare B-meson decay.

        3. Method

        Identify all mentioned methods, techniques, general strategies, etc used in the paper to estimate or otherwise evaluate the impact of systematic uncertainties upon a measurement.

        # EXTRACTION PRINCIPLES

        1. Comprehensive Coverage: Extract ALL entities which fit the above descriptions. When in doubt, include it.

        2. Precision: Maintain exact numerical values with proper units and distinguish between absolute/relative uncertainties.

        3. Completeness: Ensure all entity properties are filled with the most specific and relevant information available.

        4. Standardization: Use consistent terminology across entities.

        5. Context Preservation: Note when measurements apply only to specific kinematic regions or under specific conditions.

        6. Concise Naming: The "name" field should be as concise as possible while capturing the full meaning of the entity.

        # OUTPUT FORMAT

        Provide your extraction as a Python dictionary with this exact structure:
        {{
        "observable": [
        {{
        "description" : str,
        "name": str,
        "type": <"branching_fraction", "branching_ratio", "physical_constant", "angular_observable", "functional_dependence">
        }},
        ...
        ],
        "uncertainty_source": [
        {{
        "description" : str,
        "name": str,
        "type": <"statistical", "internal", "external">
        }},
        ...
        ],
        "method": [
        {{
        "description" : str,
        "name": str,
        }},
        ...
        ]
        }}
        Do not provide anything else in your output.

        # INPUT

        {text}

        # OUTPUT
        """

    async def extract_entities(self, text: str):
        response = None
        try:        
            prompt = self.get_entity_extraction_prompt(text)
            response = (await llm.acomplete(prompt)).text
            entities = eval(response.replace("```python","").replace("```",""))
            return entities
            
        except Exception as e:
            print(f"Exception occurred: {e} \n Paper Text: \n {text} \n Got Response: \n {response}")
            return {}

    def get_relationship_extraction_prompt(self, text, entities):
        return f"""# Systematic Uncertainty Knowledge Graph - Relationship Extraction

        You are an expert physicist tasked with extracting relationships for a knowledge graph of systematic uncertainties in particle physics publications. You will be given the text from a particle physics paper, as well as a list of entities (nodes) from the knowledge graph. Your task is to read the paper and identify new relationships connecting nodes in the knowledge graph.

        # RELATIONSHIP TYPES
        1. "affects": when an "uncertainty_source" affects the measured value of an "observable". 
        2. "estimated_with": when a "method" is used to estimate or otherwise evaluate the impact of a "systematic_uncertainty".
        
        # EXTRACTION PRINCIPLES

        Be comprehensive. Extract all relationships between entities that are discussed in the paper. For each relationship, justify its creation by including relevant quotations from the paper.

        Focus particularly on:
        - Less obvious connections that may be embedded in technical descriptions
        - The complete uncertainty propagation chain from 
        - Complex correlation structures between different uncertainty sources
        - Temporal or kinematic dependence of relationships
        - Relationships involving important entities that may have been missed in the initial extraction

        IMPORTANT: Ensure that the "source" and "target" fields exactly match names from the provided entities dictionary.
    
        # OUTPUT FORMAT

        Provide your extraction as a Python dictionary with this exact structure:
        {{
        "affects": [
        {{
        "relevant_quotes": List[str] - a list of less than 4 direct quotations from the text which justify the creation of this relationship (do not provide overly lengthy quotations),
        "source": str - "name" attribute from an uncertainty_source entity,
        "target": str - "name" attribute from an observable entity,
        "magnitude": str - the quantitative impact expressed as a percentage, absolute value, or relative contribution (if specified in the paper),
        "dominant": "true"/"false" - does this source of uncertainty dominate the measurement of the observable
        }},
        ...
        ],
        "estimated_with": [
        {{
        "relevant_quotes": List[str] - a list of less than 4 direct quotations from the text which justify the creation of this relationship (do not provide overly lengthy quotations),
        "source": str - "name" attribute from an "uncertainty_source" entity,
        "target": str - "name" attribute from a "method" entity
        }},
        ...
        ]
        }}
        Do not provide anything else in your output.

        # INPUT

        Text:

        {text}

        Entities:

        {entities}

        # OUTPUT
        """

    async def extract_relationships(self, text: str, entities: dict):
        """Extract relationships from a chunk using a list of entities"""
        response = None
        try:            
            prompt = self.get_relationship_extraction_prompt(text, entities)
            response = (await llm.acomplete(prompt)).text
            relationships = eval(response.replace("```python","").replace("```",""))        
            return relationships
            
        except Exception as e:
            print(f"Exception occurred: {e} \n Paper Text: \n {text} \n Got Response: \n {response}")
            return []


class Entity():
    def __init__(self, name: str, type: str, description: str, attributes: dict = {}):
        self.name = name
        self.type = type
        self.description = description
        self.attributes = attributes

    def __repr__(self):
        return f"{self.type}: {self.name}"
    
    def add_to_graph(self, graph):
        from py2neo import Node
        neo4j_node = Node(self.type, name=self.name, description=self.description, **{key : repr(value) for key, value in self.attributes.items()})
        graph.create(neo4j_node)
        self._neo4j = neo4j_node


class Relationship():
    def __init__(self, source: Entity, type: str, target: Entity, attributes: dict = {}):
        self.source = source
        self.type = type
        self.target = target
        self.attributes = attributes

    def __repr__(self):
        return f"({self.source}) --[{self.type}]--> ({self.target})"
    
    def add_to_graph(self, graph):
        from py2neo import Relationship
        
        matcher = graph.nodes
        source_node = matcher.match(self.source.type, name=self.source.name, description=self.source.description).first()
        if source_node is None:
            self.source.add_to_graph(graph)
        else:
            self.source._neo4j = source_node
            
        target_node = matcher.match(self.target.type, name=self.target.name, description=self.target.description).first()
        if target_node is None:
            self.target.add_to_graph(graph)
        else:
            self.target._neo4j = target_node
            
        neo4j_relationship = Relationship(self.source._neo4j, self.type, self.target._neo4j, **self.attributes)
        graph.create(neo4j_relationship)
        self._neo4j = neo4j_relationship


class SystematicsGraph():

    def __init__(self):
        self.entities = {}
        self.entity_types = {}
        self.relationships = set()
    
    def get_entity(self, entity_name: str):
        return self.entities[entity_name]

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity
        if entity.type not in self.entity_types:
            self.entity_types[entity.type] = set()
        self.entity_types[entity.type].add(entity)

    def remove_entity(self, entity: Entity):
        self.entities.pop(entity.name)
        self.entity_types

    def add_relationship(self, relationship: Relationship):
        self.relationships.add(relationship)

    def load_paper(self, paper: LHCbPaper):
        paper_entity = Entity(name=paper.arxiv_id, type="LHCb Paper", description=paper.abstract, attributes=copy.deepcopy(paper.metadata))
        self.add_entity(paper_entity)

        for type in paper.entities:
            for dict in paper.entities[type]:
                entity_dict = copy.deepcopy(dict)
                name = entity_dict.pop("name")
                description = entity_dict.pop("description", "")
                entity = Entity(name=name, type=type, description=description, attributes=entity_dict)
                
                self.add_entity(entity)

                if type in "observable":
                    relationship = Relationship(paper_entity, "MEASURES", entity)
                    self.add_relationship(relationship)
        
        for type in paper.relationships:
            for dict in paper.relationships[type]:
                relationship_dict = copy.deepcopy(dict)
                
                source_name = relationship_dict.pop("source")
                try:  
                    source = self.get_entity(source_name)
                except:
                    print(f"Error loading relationship: \n {dict} \n source {source_name} does not exist")
                    print(f"Available entities: \n {self.entities} \n\n")
                    continue
                
                target_name = relationship_dict.pop("target")
                try:
                    target = self.get_entity(target_name)
                except:
                    print(f"Error loading relationship: \n {dict} \n target {target_name} does not exist")
                    print(f"Available entities: \n {self.entities} \n\n")
                    continue
            
                relationship = Relationship(source, type, target, relationship_dict)
                self.add_relationship(relationship)
    
    def push_to_neo4j(self, uri, username, password):
        from py2neo import Graph
        
        graph = Graph(uri, auth=(username, password))
        graph.delete_all()

        for name, entity in self.entities.items():
            entity.add_to_graph(graph)

        for relationship in self.relationships:
            relationship.add_to_graph(graph)

    def cluster_entities(self, entities: list[Entity], threshold: float = 0.75):
        names = []
        descriptions = []
        for entity in entities:
            names.append(entity.name)
            descriptions.append(entity.description)
        names = np.array(names)
        descriptions = np.array(descriptions)
        
        name_embeddings = embedding_model.encode(names)
        description_embeddings = embedding_model.encode(descriptions)
        similarity_matrix = ( (2.0/3.0) * (name_embeddings @ name_embeddings.T) + (1.0/3.0) * (description_embeddings @ description_embeddings.T) )

        distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
        
        clustering = DBSCAN(metric="precomputed", eps=1-threshold, min_samples=1)
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        unique_labels = set(cluster_labels)
        clusters = [[] for _ in range(len(unique_labels))]
        
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
            
        return clusters

    async def merge_entities_with_llm(self, cluster_entities: List[Entity]) -> Entity:
        entities_str = ""
        for i, entity in enumerate(cluster_entities):
            entity_attrs = ", ".join([f"{k}: {v}" for k, v in entity.attributes.items()])
            entities_str += f"Entity {i+1}:\n"
            entities_str += f"  Name: {entity.name}\n"
            entities_str += f"  Type: {entity.type}\n"
            entities_str += f"  Description: {entity.description}\n"
            entities_str += f"  Attributes: {entity_attrs}\n\n"
        
        prompt = f"""You are an expert physicist specialized in building knowledge graphs of systematic uncertainties in particle physics publications.
        
        # TASK
        
        Merge the following similar entities from a particle physics knowledge graph into one entity. The scope of the entity may need to be made broader to capture all of the sub-entities being merged into it.
        
        # OUTPUT REQUIREMENTS
        
        Create a single merged entity that:
        1. Has a name that best represents all of the entities
        2. Contains a comprehensive description synthesizing all relevant details
        3. Combines all attributes from the source entities
        
        # OUTPUT FORMAT
        
        Provide your output as a valid Python dictionary:
        {{
            "name": "standardized entity name",
            "description": "comprehensive synthesized description",
            "attributes": {{
                "attribute1": "value1",
                "attribute2": "value2",
                ...
            }}
        }}

        # INPUT ENTITIES
        
        {entities_str}
        """
        
        response = None
        try:
            response = (await llm.acomplete(prompt)).text
            start = response.find("{")
            end = response.rfind("}") + 1
            dict_text = response[start:end]
            
            merged_data = eval(dict_text)
            
            base_entity = cluster_entities[0]
            merged_entity = Entity(
                name=merged_data["name"],
                type=base_entity.type,
                description=merged_data["description"],
                attributes=merged_data.get("attributes", {})
            )
            
            return merged_entity
        except Exception as e:
            print(f"Error in entity merging: {e}")
            print(f"LLM response: {response}")
            return cluster_entities[0]

    async def generalize_entities_with_llm(self, cluster_entities: List[Entity], cluster_id: int) -> Entity:
        parent_type = f"Generalized {cluster_entities[0].type}"

        entities_str = ""
        for i, entity in enumerate(cluster_entities):
            entity_attrs = ", ".join([f"{k}: {v}" for k, v in entity.attributes.items()])
            entities_str += f"Entity {i+1}:\n"
            entities_str += f"  Name: {entity.name}\n"
            entities_str += f"  Type: {entity.type}\n"
            entities_str += f"  Description: {entity.description}\n"
            entities_str += f"  Attributes: {entity_attrs}\n\n"
        
        prompt = f"""You are an expert physicist specialized in building knowledge graphs of systematic uncertainties in particle physics publications.
        
        # TASK
        
        Create a parent/generalized entity that represents the common patterns and shared characteristics of these similar physics entities.
        
        # OUTPUT REQUIREMENTS
        
        Create a generalized parent entity that:
        1. Has a descriptive name capturing the common concept
        2. Contains a description explaining what this category represents
        3. Includes any common attributes shared across most/all of the entities
        4. Uses the parent type "{parent_type}"
        
        # OUTPUT FORMAT
        
        Provide your output as a valid Python dictionary:
        {{
            "name": "generalized entity name",
            "description": "explanation of what this category represents",
            "attributes": {{
                "common_attribute1": "value1",
                "common_attribute2": "value2",
                ...
            }}
        }}
        
        # INPUT ENTITIES
        
        {entities_str}
        """
        
        response = None
        try:
            response = (await llm.acomplete(prompt)).text
            start = response.find("{")
            end = response.rfind("}") + 1
            dict_text = response[start:end]
            
            generalized_data = eval(dict_text)
            
            if "attributes" not in generalized_data:
                generalized_data["attributes"] = {}
            generalized_data["attributes"]["cluster_id"] = cluster_id
            generalized_data["attributes"]["cluster_size"] = len(cluster_entities)
            
            generalized_entity = Entity(
                name=generalized_data["name"],
                type=parent_type,
                description=generalized_data["description"],
                attributes=generalized_data.get("attributes", {})
            )
            
            return generalized_entity
        except Exception as e:
            print(f"Error in entity generalization: {e}")
            print(f"LLM response: {response}")
            return Entity(
                name=f"{parent_type}_{cluster_id}",
                type=parent_type,
                description=f"Common entity cluster of {len(cluster_entities)} similar {cluster_entities[0].type} entities",
                attributes={"cluster_size": len(cluster_entities), "cluster_id": cluster_id}
            )

    async def merge_entity_type_async(self, type: str, threshold: float = 0.8):
        if type not in self.entity_types:
            print(f"ERROR: type {type} must be one of {list(self.entity_types.keys())}")
            return
        
        entities = list(self.entity_types[type])
        clusters = self.cluster_entities(entities, threshold)
        
        merge_tasks = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
                
            cluster_entities = [entities[i] for i in cluster]
            merge_tasks.append((cluster, cluster_entities))
        
        merged_results = await asyncio.gather(
            *[self.merge_entities_with_llm(cluster_entities) for _, cluster_entities in merge_tasks]
        )
        
        for i, (cluster, cluster_entities) in enumerate(merge_tasks):
            merged_entity = merged_results[i]
            self.add_entity(merged_entity)
            
            relationships_to_update = []
            for relationship in self.relationships.copy():
                if relationship.source in cluster_entities and relationship.source != merged_entity:
                    new_relationship = Relationship(
                        source=merged_entity,
                        type=relationship.type,
                        target=relationship.target,
                        attributes=relationship.attributes
                    )
                    relationships_to_update.append((relationship, new_relationship))
                
                elif relationship.target in cluster_entities and relationship.target != merged_entity:
                    new_relationship = Relationship(
                        source=relationship.source,
                        type=relationship.type,
                        target=merged_entity,
                        attributes=relationship.attributes
                    )
                    relationships_to_update.append((relationship, new_relationship))
            
            for old_rel, new_rel in relationships_to_update:
                self.relationships.remove(old_rel)
                self.relationships.add(new_rel)
            
            for entity in cluster_entities:
                if entity != merged_entity:
                    self.entities.pop(entity.name, None)
                    self.entity_types[type].discard(entity)

    async def cluster_entity_type_async(self, type: str, threshold: float = 0.8):
        if type not in self.entity_types:
            print(f"ERROR: type {type} must be one of {list(self.entity_types.keys())}")
            return
        
        entities = list(self.entity_types[type])
        clusters = self.cluster_entities(entities, threshold)

        generalize_tasks = []
        for i, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
                    
            cluster_entities = [entities[i] for i in cluster]
            generalize_tasks.append((i, cluster_entities))
        
        parent_entities = await asyncio.gather(
            *[self.generalize_entities_with_llm(cluster_entities, i) 
            for i, cluster_entities in generalize_tasks]
        )
        
        for i, (cluster_idx, cluster_entities) in enumerate(generalize_tasks):
            parent_entity = parent_entities[i]
            self.add_entity(parent_entity)
            
            for entity in cluster_entities:
                subtype_rel = Relationship(
                    source=entity,
                    type="SUBTYPE_OF",
                    target=parent_entity,
                    attributes={}
                )
                self.add_relationship(subtype_rel)
            
            for relationship in self.relationships.copy():
                if (relationship.target in cluster_entities and 
                    relationship.source not in cluster_entities):
                    
                    new_relationship = Relationship(
                        source=relationship.source,
                        type=relationship.type,
                        target=parent_entity,
                        attributes=relationship.attributes.copy()
                    )
                    
                    self.add_relationship(new_relationship)

    def merge_entity_type(self, type: str, threshold: float = 0.8):
        asyncio.run(self.merge_entity_type_async(type, threshold))

    def cluster_entity_type(self, type: str, threshold: float = 0.8):
        asyncio.run(self.cluster_entity_type_async(type, threshold))
