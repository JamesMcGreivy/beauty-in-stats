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
        return f"""You are a research scientist in the LHCb collaboration at CERN and particle physics expert.
        
        # TASK
        
        Classify LHCb physics analysis paper abstracts (not performance/instrumentation papers). Your classifications will power a RAG system for physics analysis recommendations using three categories:

        1. PHYSICS FOCUS - The context of the analysis (similar to LHCb Physics Working Groups)
        
        Definitions: 
            - 'Beauty decays': $B^+$, $B^0$, $B_s^0$, $B_c^+$, $\Lambda_b^0$, $\Xi_b^0$, $\Xi_b^-$, $\Omega_b^-$, $B^{{*0}}$, $B_s^{{*0}}$, $\Lambda_b^{{*0}}$, $\Xi_b^{{0}}$, $\Xi_b^{{-}}$ (charge conjugation implied)
            - 'Charm decays': $D^+$, $D^0$, $D_s^+$, $\Lambda_c^+$, $\Xi_c^+$, $\Xi_c^0$, $\Omega_c^0$, $D^{{+}}$, $D^{{0}}$, $D_s^{{+}}$, $\Lambda_c^{{+}}$, $\Xi_c^{{*+}}$, $\Xi_c^{{*0}}$, $\Omega_c^{{*0}}$ (charge conjugation implied, excludes charmonia)

        Allowed labels (choose up to THREE most relevant labels):
        - `b->sll`: Beauty hadron decays with two same-generation leptons in the final state. Includes rare loop-suppressed penguin decays often used to probe New Physics, such as $B^+\\to K^{{+}} \mu^+ \mu^-$, $B_s^0 \\to \mu^+ \mu^-$, $B^0 \\to K^{{*0}} \mu^+ \mu^-$, $\Lambda_b^0 \\to \Lambda \mu^+ \mu^-$. These transitions are mediated by $b \to s \ell^+ \ell^-$ or $b \to d \ell^+ \ell^-$ FCNC processes.
        
        - `c->sll`: Charm hadron decays with two same-generation leptons in the final state. Includes rare decays that are highly suppressed in the SM, such as $D^0 \\to \pi^+\pi^-e^+e^-$, $D^0 \\to \mu^+ \mu^-$, $\Lambda_c^+ \\to p \mu^+ \mu^-$, $D^+ \\to \pi^+ \mu^+ \mu^-$. These are mediated by $c \to u \ell^+ \ell^-$ FCNC processes and are excellent probes for New Physics.
        
        - `radiative_decays`: Heavy-flavor decays with at least one photon in the final state, such as $B^0 \\to K^{{*0}} \gamma$, $\Lambda_b^0 \\to \Lambda \gamma$, $D^0 \\to K^{{*0}} \gamma$. These are mediated by FCNC transitions like $b \to s \gamma$ or $c \to u \gamma$ and are sensitive to physics beyond the SM.
        
        - `spectroscopy`: Studies focused on exotic multi-quark (>3) QCD bound states including tetraquarks, pentaquarks, and molecular states. Includes both discovery and characterization of states like $T_{{cc}}^+$, $P_c(4450)^+$, or $X(3872)$. Also includes conventional spectroscopy of excited charm and beauty hadrons.
        
        - `semileptonic`: Heavy-flavor decays with a charged lepton and neutrino in the final state, such as $B \\to D^{{(*)}}\ell\\nu$ or $\Lambda_b \\to \Lambda_c^+ \ell^- \\nu$. These are mediated by tree-level $b \to c \ell^- \\nu$ transitions and are often used for $|V_{{cb}}|$ extraction or lepton flavor universality tests.
        
        - `lifetime`: Measurements of the lifetimes or lifetime ratios of heavy-flavored hadrons, which provide sensitive tests of the heavy quark expansion and inform theoretical understanding of strong interaction effects.
        
        - `electoweak`: Electroweak/Higgs physics including production and decay of $W$, $Z$, or Higgs bosons, tests of electroweak theory, and measurements of electroweak parameters such as the weak mixing angle $\sin^2 \theta_W$.
        
        - `dark_sector`: Searches for dark matter or dark sector particles such as dark photons, long-lived particles (LLPs), axion-like particles (ALPs), heavy neutral leptons (HNLs), or any other beyond-SM particles that could explain astrophysical anomalies.
        
        - `forbidden_decays`: Searches for decays forbidden or extremely suppressed in the Standard Model, including baryon/lepton number violation, lepton flavor violation, lepton universality violation tests, or other symmetry-breaking processes.
        
        - `jet_physics`: Studies of QCD jet production, jet properties, jet substructure, or measurements of strong coupling constant $\\alpha_s$. Includes both beauty and charm jets as well as light-flavor jets.
        
        - `heavy_ions`: Physics analyses using non-proton-proton collision data, such as lead-lead ($PbPb$) or proton-lead ($pPb$) collisions. Includes studies of quark-gluon plasma, collective effects, or nuclear modification factors.
        
        - `beauty`: General beauty hadron physics not covered by more specific categories. Includes studies of beauty hadron production, properties, branching fractions, form factors, hadronic decays, or other aspects of beauty hadron phenomenology.
        
        - `charm`: General charm hadron physics not covered by more specific categories. Includes studies of charm hadron production, properties, branching fractions, form factors, hadronic decays, or other aspects of charm hadron phenomenology.
        
        - `CP_asymmetry`: Measurements of CP violation or CP asymmetries in any heavy-flavor decay system. Includes direct and indirect CP asymmetries, mixing-induced CP violation, time-dependent CP violation studies, and extractions of CKM angles ($\\alpha$, $\\beta$, $\\gamma$) or CKM matrix elements.

        2. RUN PERIOD - The LHCb data-taking period and dataset characteristics

        - `Run1`: Data collected during 2011-2012 at center-of-mass energies of 7-8 TeV, corresponding to an integrated luminosity of approximately 3.0 fb-1.
        - `Run2`: Data collected during 2015-2018 at a center-of-mass energy of 13 TeV, corresponding to an integrated luminosity of approximately 5.4-6 fb-1.
        - `Run1+2`: Combined dataset from both Run 1 and Run 2 periods (2011-2018), with a total integrated luminosity of approximately 9 fb-1.
        - `Run3`: Data collected during 2022-2025 at a center-of-mass energy of 13.6 TeV, with an expected integrated luminosity of approximately 15 fb-1 by the end of the period.

        3. ANALYSIS STRATEGY - The observable and statistical inference framework

        - `angular_analysis`: Studies employing angular distributions or asymmetries such as forward-backward asymmetries, angular coefficients, or Wilson coefficient-related observables like $P_5'$. Typically involves multi-dimensional fits to angular variables to extract physics parameters.
        
        - `amplitude_analysis`: Analyses focused on decay amplitudes, including Dalitz plot analyses, determination of resonance structures, spin-parity assignments, or phase measurements. Usually involves modeling interfering amplitudes and extracting fit fractions or strong phases.
        
        - `search`: Analyses aimed at discovering unobserved states, processes, or symmetry-breaking effects, typically resulting in limit setting at 90% or 95% confidence level if no significant signal is observed. Includes searches for new particles, rare decays, or forbidden processes.
        
        - `direct_measurement`: Straightforward measurements of particle properties, decay observables, symmetries, or Standard Model parameters. Includes branching fraction measurements, production cross-sections, mass measurements, or other direct determinations of physical quantities.

        # OUTPUT FORMAT
        
        Your response must follow this exact format:
        {{
            "focus": "<label from physics focus>, ...",
            "run": "<label from run period>, ...",
            "strategy": "<label from analysis strategy>, ..."
            "explanation" : "Justification of why these labels were extracted according to the prompt directions and your expert knowledge."
        }}

        # IMPORTANT CLASSIFICATION GUIDELINES:
        
        1. Papers can have MULTIPLE FOCUS labels - choose up to THREE most relevant ones
        2. For papers studying rare decays with CP asymmetry measurements, include BOTH the decay type label (e.g., `c->sll`) AND `CP_asymmetry`
        3. For charm baryon decays, always include the `charm` label
        4. For beauty baryon decays, always include the `beauty` label
        5. If a paper doesn't clearly fit any specific focus category, choose the hadron type (`charm` or `beauty`) plus the most relevant measurement type
        6. Be attentive to the actual physics goals of the paper, not just the decay mentioned - a paper studying $B^+ \\to K^+ \mu^+ \mu^-$ could be a `b->sll` study or it could be focused on `CP_asymmetry` or both
        7. For Run period, choose the specific data-taking period(s) explicitly mentioned in the abstract

        Do not mix labels between categories. Use <PERF> for all categories if it's a performance paper. Only use labels with medium/high confidence. Use <UNK> for low-confidence categories.

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

        You are an expert physicist specialized in extracting structured knowledge about systematic uncertainties from particle physics publications. Your task is to identify and extract all relevant entities related to systematic uncertainties from the provided text.

        # ENTITY TYPES AND PROPERTIES

        1. Uncertainty Sources
        Categorize each uncertainty source using these precise definitions:
        - **Experimental**: Uncertainties from measurement apparatus, detector performance, calibration, or data collection. Examples: detector efficiency, resolution effects, instrument calibration.
        - **Methodological**: Uncertainties from analysis techniques, fitting procedures, binning strategies, or data processing choices. Examples: model selection, bin migration, background subtraction.
        - **Theoretical**: Uncertainties from theoretical assumptions, external input parameters, or modeling choices. Examples: branching ratios, hadronization models, polarization assumptions.
        For each uncertainty source, extract:
        - `name`: The specific uncertainty source (standardized terminology)
        - `description`: In one or two sentences, summarize any descriptions or explanations of this method provided in the paper.
        - `type`: Experimental/Methodological/Theoretical

        2. Measurements/Observables
        Extract all physical quantities being measured or derived from experimental data (ex: CKM mixing angles, branching ratios, CP violation parameters, differential cross-sections, particle masses, particle lifetimes, etc.)
        - `name`: Standardized measurement name
        - `value`: Numerical value given in the text with units included
        - `statistical_uncertainty`: Numerical value of complete statistical uncertainty with units included
        - `systematic_uncertainty`: Numerical value of complete systematic uncertainty with units included
        - `additional_uncertainty`: Numerical value of other uncertainties with units included (e.g., polarization)

        3. Estimation Methods
        Extract all analysis methods or mathematical techniques used in the paper to reduce, estimate, or otherwise evaluate uncertainties for a measurement.
        - `name`: Method name (use standardized terminology)
        - `description`: In one or two sentences, summarize any descriptions or explanations of this method provided in the paper.

        # EXTRACTION PRINCIPLES

        1. **Comprehensive Coverage**: Extract ALL entities which fit the above descriptions. When in doubt, include it.

        2. **Precision**: Maintain exact numerical values with proper units and distinguish between absolute/relative uncertainties.

        3. **Completeness**: Ensure all entity properties are filled with the most specific and relevant information available.

        4. **Standardization**: Use consistent terminology across entities.

        5. **Context Preservation**: Note when measurements apply only to specific kinematic regions or conditions.

        # OUTPUT FORMAT

        Provide your extraction as a Python dictionary with this structure:

        {{
        "Uncertainty Sources": [
        {{
        "name": str,
        "description": str,
        "type": str
        }},
        ...
        ],
        "Measurements/Observables": [
        {{
        "name": str,
        "value": str,
        "statistical_uncertainty": str,
        "systematic_uncertainty": str,
        "additional_uncertainty": str
        }},
        ...
        ],
        "Estimation Methods": [
        {{
        "name": str,
        "description": str
        }},
        ...
        ]
        }}

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

        You are an expert physicist specialized in extracting structured knowledge about systematic uncertainties from particle physics publications. Your task is to identify all relationships between the provided entities based on the provided text.

        # RELATIONSHIP TYPES

        Uncertainty Source affects Measurement/Observable:
        - `type`: "AFFECTS"
        - `source`: Uncertainty Sources - The name of the uncertainty source that impacts the measurement
        - `target`: Measurements/Observables - The specific measurement being affected
        - `magnitude`: The quantitative impact expressed as a percentage, absolute value, or relative contribution (if specified in the paper)

        Uncertainty Source dominates Measurement/Observable:
        - `type`: "DOMINATES"
        - `source`: Uncertainty Sources - The primary uncertainty source with largest impact
        - `target`: Measurements/Observables - The measurement for which this uncertainty is dominant
        - `magnitude`: N/A (implied to be the largest contribution)

        Uncertainty Source is correlated with Uncertainty Source:
        - `type`: "CORRELATED_WITH"
        - `source`: Uncertainty Sources - The first uncertainty source in the correlation
        - `target`: Uncertainty Sources - The second uncertainty source in the correlation
        - `magnitude`: Correlation coefficient (numerical value if provided) or qualitative strength description (e.g., "strong", "weak")

        Uncertainty Source is estimated with Estimation Method:
        - `type`: "ESTIMATED_WITH"
        - `source`: Uncertainty Sources - The uncertainty being evaluated
        - `target`: Estimation Methods - The method used to quantify or estimate the magnitude of the uncertainty
        - `magnitude`: The resulting precision or uncertainty on the uncertainty (e.g., "±0.5%")

        Uncertainty Source is reduced by Estimation Method:
        - `type`: "REDUCED_BY"
        - `source`: Uncertainty Sources - The uncertainty being mitigated
        - `target`: Estimation Methods - The method used to reduce the uncertainty
        - `magnitude`: Quantitative reduction factor (e.g., "reduced by factor of 2") or before/after values (e.g., "from 3.2% to 1.1%")

        For all:
        - `relevant_quotes`: A list of at most three direct quotations taken from the text which justify the creation of this relationships according to the prompt and your expert knowledge. Do not provide overly lengthy quotations.
        
        # EXTRACTION PRINCIPLES

        **Verifiable Connections**: Extract relationships that are explicitly stated or strongly implied in the text.
        **Relationship Completeness**: Capture all details about the nature and magnitude of each relationship.
        **Correlation Structure**: Pay particular attention to correlation structures between uncertainty sources.
        Focus particularly on:
        - Less obvious connections that may be embedded in technical descriptions
        - Complete uncertainty propagation chains
        - Complex correlation structures between different uncertainty sources
        - Temporal or kinematic dependence of relationships
        - Relationships involving important entities that may have been missed in the initial extraction

        IMPORTANT: Ensure that the "source" and "target" fields exactly match names from the provided entities dictionary.
    
        # OUTPUT FORMAT

        Provide your extraction as a Python list of relationship dictionaries with this structure:

        [
        {{
        "type": str,
        "source": str,
        "target": str,
        "relevant_quotes": List[str],
        "magnitude": str
        }},
        ...
        ]

        # INPUT

        Entities extracted from text:

        {entities}

        Text:

        {text}

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
        neo4j_node = Node(self.type, name=self.name, description=self.description, **self.attributes)
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
    """_summary_
    """

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

                if type in "Measurements/Observables":
                    relationship = Relationship(paper_entity, "MEASURES", entity)
                    self.add_relationship(relationship)
        
        for dict in paper.relationships:
            relationship_dict = copy.deepcopy(dict)
            
            source_name = relationship_dict.pop("source")
            try:  
                source = self.get_entity(source_name)
            except:
                print(f"Error loading relationship: \n {dict} \n source does not exist")
                continue
            
            type = relationship_dict.pop("type")
            
            target_name = relationship_dict.pop("target")
            try:
                target = self.get_entity(target_name)
            except:
                print(f"Error loading relationship: \n {dict} \n target does not exist")
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

    def cluster_entities(self, entities: list[Entity], threshold: float = 0.8):
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
        
        prompt = f"""You are an expert in particle physics and knowledge representation.
        
        # TASK
        
        Merge the following similar entities from a physics knowledge graph into one cohesive entity that synthesizes their information.
        
        # OUTPUT REQUIREMENTS
        
        Create a single merged entity that:
        1. Has a standardized name that best represents all entities
        2. Contains a comprehensive description synthesizing all relevant details
        3. Combines all unique attributes from source entities (keeping the most precise values when attributes overlap)
        
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
            # Extract the dictionary from the response
            # Find the opening and closing braces
            start = response.find("{")
            end = response.rfind("}") + 1
            dict_text = response[start:end]
            
            merged_data = eval(dict_text)
            
            # Create a new merged entity
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
            # Fallback to using the first entity if LLM fails
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
        
        prompt = f"""You are an expert in particle physics and knowledge graph taxonomy.
        
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
