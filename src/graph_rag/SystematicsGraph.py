"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import numpy as np
import time
import asyncio
import regex as re
import copy
from typing import List
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from py2neo import Graph

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4.1-mini")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("BAAI/bge-m3")

def process_latex_figures(text):
    pattern = r'\\begin{figure}.*?(?:\\caption{(.*?)}).*?\\end{figure}'
    def replacement_function(match):
        caption_content = match.group(1)
        if caption_content:
            return f"\n\nFigure with caption: {caption_content} \n\n"
        else:
            return ""
    processed_text = re.sub(pattern, replacement_function, text, flags=re.DOTALL)
    pattern = r'\\begin{figure}.*?\\end{figure}'
    processed_text = re.sub(pattern, "", processed_text, re.DOTALL)
    return processed_text

class LHCbPaper():
    """_summary_
    """

    def __init__(self, abstract, text, arxiv_id):
        self.abstract = abstract
        self.arxiv_id = arxiv_id
        
        self.text = text
        self.section_titles, self.section_texts = self.split_sections(text)
        
        relevant_text, self.relevant_section_titles = self.get_relevant_text(self.section_titles, self.section_texts)
        self.relevant_text = process_latex_figures(relevant_text)
        
        self.processed = False

    def __repr__(self):
        return f"{self.arxiv_id}:\n\t{self.abstract}"
    
    def get_relevant_text(self, section_titles, section_texts):
        section_titles = copy.deepcopy(section_titles)
        section_texts = copy.deepcopy(section_texts)

        remove_flags = ["introduction", "detector", "tables", "figures", "dataset"]
        remove_idxs = np.where([any([flag in title for flag in remove_flags]) for title in section_titles])[0]
        for idx in np.sort(remove_idxs)[::-1]:
            section_titles.pop(idx)
            section_texts.pop(idx)
        
        start_flags = ["error", "uncertaint", "systematic"]
        start_idx = np.where([any([flag in title for flag in start_flags]) for title in section_titles])[0]
        if len(start_idx) > 0:
            start_idx = min(start_idx)
        else:
            start_idx = None

        end_flags = ["appendi", "supplement", "end matter", "acknowle", "aknowle"]
        end_idx = np.where([any([flag in title for flag in end_flags]) for title in section_titles])[0]
        if len(end_idx) > 0:
            end_idx = min(end_idx)
        else:
            end_idx = None

        relevant_section_titles = section_titles[start_idx:end_idx]
        relevant_text = self.abstract + " \n".join(section_texts[0:end_idx])
        
        return relevant_text, relevant_section_titles

    def split_sections(self, text, depth=0, title="", max_tokens=6000):
        pattern = r"(\\" + "sub" * depth + r"section[\*\s]*(?:\[[^\]]*\])?\s*({(?:[^{}]*+|(?2))*}))"
        matches = re.finditer(pattern, text)

        if not matches or depth > 3:
            return [text]

        section_titles = []
        section_texts = []
        start = 0
        section_title = ""
        for match in [(match.start(), match.end()) for match in matches] + [(-1, -1)]:
            end = match[0]
            section_text = text[start:end]
            if len(re.sub("\s", "", section_text)) > 0:
                new_title = re.sub(r"[^\w\s]", " ", f"{title} \t {section_title}" if len(title) > 0 else section_title).lower()
                section_titles.append(new_title)
                section_texts.append(section_text)

            start = match[1]
            section_title = text[end:start]

        return section_titles, section_texts
    
    async def process_paper(self):
        metadata = await self.extract_abstract(self.abstract)
    
        decays = metadata.pop("decay")
        observables = metadata.pop("observable")
        
        self.metadata = metadata
        
        entities = await self.extract_graph(self.relevant_text, observables)
        entities["decay"] = decays
        entities["observable"] = observables

        relationships = {}
        for type in entities:
            for entity in entities[type]:
                relationship_keys = [key for key in entity.keys() if key.startswith("relationship_")]
                for key in relationship_keys:
                    if key.removeprefix("relationship_") not in relationships:
                        relationships[key.removeprefix("relationship_")] = []
                    entity_relationships = entity.pop(key)
                    for relationship in entity_relationships:
                        relationship["source"] = entity["name"]
                        relationships[key.removeprefix("relationship_")].append(relationship)

        self.entities = entities
        self.relationships = relationships
        self.processed = True

    def get_abstract_extraction_prompt(self, abstract):
        return f"""
You are a research scientist in the LHCb collaboration at CERN and a particle physics expert. You will be given an abstract from an LHCb analysis paper and extract the information described below.

1. "run"
Identify the LHCb data-taking period or dataset characteristics
- "run1": Data collected during 2011-2012 at center-of-mass energies of 7-8 TeV
- "run2": Data collected during 2015-2018 at a center-of-mass energy of 13 TeV
- "run1,run2": Combined dataset from both Run 1 and Run 2 periods (2011-2018)
- "run3": Data collected during 2022-2025 at a center-of-mass energy of 13.6 TeV
- "ift": Any paper which does not use p-p as its production method.

2. "strategy"
Identify which of following labels best describes the analysis strategy employed in the paper.

- "angular_analysis": Studies that examine angular distributions or asymmetries in particle decays. These include forward-backward asymmetries, angular coefficients, or observables related to Wilson coefficients. These analyses typically perform multi-dimensional fits to angular variables to extract physics parameters and test Standard Model predictions.

- "amplitude_analysis": Studies focused on decay amplitude structures, including Dalitz plot analyses, determination of resonance properties, spin-parity assignments, or relative phase measurements. These analyses model interfering amplitudes and extract quantities such as fit fractions, strong phases, or resonance parameters.
        
- "search": Analyses aimed at discovering previously unobserved states, processes, or symmetry-breaking effects. This includes first observations of rare decay modes, searches for new particles, or forbidden processes. Often results in significance measurements for new signals or limit setting at confidence levels (typically 90% or 95%) when no significant signal is observed.
        
- "other": Any analysis that does not fall into the above three categories. This includes analyses primarily focused on precision measurements of established quantities, such as branching fraction determinations, lifetime measurements, production cross-sections, mass measurements.

3. "decay"
Identify all particle decay channels. For each decay channel, provide the production method, parent particle(s), and children particle(s). If the production, parent, or children of a decay is unknown then assign it the empty string "". You must select production methods and particles from the following lists:
- production: p-p, Pb-Pb, p-Pb, Xe-Xe, O-O, Pb-Ar, p-O
- particles: d, d~, u, u~, s, s~, c, c~, b, b~, t, t~, e-, e+, nu(e), nu(e)~, mu-, mu+, nu(mu), nu(mu)~, tau-, tau+, nu(tau), nu(tau)~, g, gamma, Z0, W+, W-, H0, R0, R~0, X(u)0, X(u)+, X(u)-, pi0, rho0, a0, K(L)0, B(L)0, pi+, pi-, rho+, rho-, a+, a-, eta, omega, f, K(S)0, K0, K~0, K*0, K*~0, K+, K-, K*+, K*-, eta', phi, f', B(sL)0, D+, D-, D*+, D*-, D0, D~0, D*0, D*~0, D(s)+, D(s)-, D(s)*+, D(s)*-, D(s2)*+, D(s2)*-, eta(c)(1S), J/psi(1S), chi(c2)(1P), B(H)0, B0, B~0, B*0, B*~0, B+, B-, B*+, B*-, B(sH)0, B(s)0, B(s)~0, B(s)*0, B(s)*~0, B(s2)*0, B(s2)*~0, B(c)+, B(c)-, B(c)*+, B(c)*-, B(c2)*+, B(c2)*-, eta(b)(1S), Upsilon(1S), chi(b2)(1P), Upsilon(1D), (dd), (dd)~, Delta-, Delta~+, Delta0, Delta~0, N0, N~0, (ud), (ud)~, n, n~, Delta+, Delta~-, N+, N~-, (uu), (uu)~, p, p~, Delta++, Delta~--, (sd), (sd)~, Sigma-, Sigma~+, Lambda, Lambda~, (su), (su)~, Sigma0, Sigma~0, Sigma+, Sigma~-, (ss), (ss)~, Xi-, Xi~+, Xi0, Xi~0, Omega-, Omega~+, (cd), (cd)~, Sigma(c)0, Sigma(c)~0, Lambda(c)+, Lambda(c)~-, Xi(c)0, Xi(c)~0, (cu), (cu)~, Sigma(c)+, Sigma(c)~-, Sigma(c)++, Sigma(c)~--, Xi(c)+, Xi(c)~-, (cs), (cs)~, Xi(c)'0, Xi(c)'~0, Xi(c)'+, Xi(c)'~-, Omega(c)0, Omega(c)~0, (cc), (cc)~, Xi(cc)+, Xi(cc)~-, Xi(cc)*+, Xi(cc)*~-, Xi(cc)++, Xi(cc)~--, Xi(cc)*++, Xi(cc)*~--, Omega(cc)+, Omega(cc)~-, Omega(cc)*+, Omega(cc)*~-, Omega(ccc)*++, Omega(ccc)*~--, (bd), (bd)~, Sigma(b)-, Sigma(b)~+, Sigma(b)*-, Sigma(b)*~+, Lambda(b)0, Lambda(b)~0, Xi(b)-, Xi(b)~+, Xi(bc)0, Xi(bc)~0, (bu), (bu)~, Sigma(b)0, Sigma(b)~0, Sigma(b)*0, Sigma(b)*~0, Sigma(b)+, Sigma(b)~-, Sigma(b)*+, Sigma(b)*~-, Xi(b)0, Xi(b)~0, Xi(bc)+, Xi(bc)~-, (bs), (bs)~, Xi(b)'-, Xi(b)'~+, Xi(b)*-, Xi(b)*~+, Xi(b)'0, Xi(b)'~0, Xi(b)*0, Xi(b)*~0, Omega(b)-, Omega(b)~+, Omega(b)*-, Omega(b)*~+, Omega(bc)0, Omega(bc)~0, (bc), (bc)~, Xi(bc)'0, Xi(bc)'~0, Xi(bc)*0, Xi(bc)*~0, Xi(bc)'+, Xi(bc)'~-, Xi(bc)*+, Xi(bc)*~-, Omega(bc)'0, Omega(bc)'~0, Omega(bc)*0, Omega(bc)*~0, Omega(bcc)+, Omega(bcc)~-, Omega(bcc)*+, Omega(bcc)*~-, (bb), (bb)~, Xi(bb)-, Xi(bb)~+, Xi(bb)*-, Xi(bb)*~+, Xi(bb)0, Xi(bb)~0, Xi(bb)*0, Xi(bb)*~0, Omega(bb)-, Omega(bb)~+, Omega(bb)*-, Omega(bb)*~+, Omega(bbc)0, Omega(bbc)~0, Omega(bbc)*0, Omega(bbc)*~0, Omega(bbb)-, Omega(bbb)~+, vpho, b0, b+, b-, h, D(s0)*+, D(s0)*-, D(s1)+, D(s1)-, chi(c0)(1P), h(c)(1P), B(s0)*0, B(s0)*~0, B(s1)(L)0, B(s1)(L)~0, B(c0)*+, B(c0)*-, B(c1)(L)+, B(c1)(L)-, chi(b0)(1P), h(b)(1P), eta(b2)(1D), D(H)+, D(H)-, chi(c1)(1P), B(H)~0, B(H)+, B(H)-, B(s1)(H)0, B(s1)(H)~0, B(c1)(H)+, B(c1)(H)-, chi(b1)(1P), X(sd), X(sd)~, X(su), X(su)~, X(ss), X(ss)~, psi, Upsilon_1(1D), D(2S)+, D(2S)-, D(2S)0, D(2S)~0, eta(c)(2S), psi(2S), chi(c2), eta(b)(2S), Upsilon(2S), chi(b2)(2P), Upsilon(2D), chi(b0)(2P), h(b)(2P), eta(b2)(2D), chi(b1)(2P), eta(b)(3S), Upsilon(3S), chi(b2)(3P), chi_b0(3P), h_b(3P), chi_b1(3P), Upsilon(4S), pi(tc)0, rho(tc)0, pi(tc)+, pi(tc)-, rho(tc)+, rho(tc)-, pi(tc)'0, omega(tc), nu(e)*0, nu(e)*~0, pi, f(J), Upsilon, rho, Z+, Z-, X

You may need to use your knowledge as a particle physics expert when classifying. For a paper that "measures B_s \\to \\mu^+ \\mu^- and B_0 -> \\mu^+ \\mu^- in pp collisions" the "B_s" would be written as "B(s)0" because B_s (strange B-meson) corresponds to B(s)0 in the provided list of allowed particles.

4. "observable"
Identify all physical quantities being directly measured or derived from experimental data in the current analysis. 

Examples: CKM mixing angles, branching fractions, CP violation parameters, differential cross-sections, particle masses, particle lifetimes

Important:
- Exclude broad physics motivations like lepton universality or matter-antimatter asymmetry.
- If it is not mentioned in the abstract, it is less likely to fit the definition of observable.
- Do not duplicate observables. Create only one entity per observable, regardless of how many kinematic bins it has. If the paper measures a branching fraction ratio then you should NOT make separate observables for the numerator and denominator in the ratio.

Categories:
- "branching_fraction": Explicitly called branching fractions (e.g., B(B^- -> D^0\\pi^-\\pi^+\\pi^-))
- "branching_ratio": Ratio of branching fractions (e.g., B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-)) where both numerator and denominator were measured in this analysis.
- "physical_constant": Fundamental physics parameters (e.g., CP-violating phase γ, CKM angles, wilson coefficients, particle lifetimes)
- "angular_observable": From angular analyses (e.g., asymmetries in angular distributions, polarization fractions, helicity amplitudes)
- "functional_dependence": Measured as functions of kinematic variables (e.g., p_T distributions, differential cross sections vs rapidity, form factors vs q^2)

"relationship_measured_with":
Identify when an "observable" is being measured in the analysis by studying the properties of a "decay" process.

# EXAMPLE - INPUT

Branching fractions of the decays $H_b\\to H_c\\pi^-\\pi^+\\pi^-$ relative to $H_b\\to H_c\\pi^-$ are presented, where $H_b$ ($H_c$) represents B^0-bar($D^+$), $B^-$ ($D^0$), B_s^0-bar ($D_s^+$) and $\\Lambda_b^0$ ($\\Lambda_c^+$). The measurements are performed with the LHCb detector using 35${{\\rm pb^{{-1}}}}$ of data collected at $\\sqrt{{s}}=7$ TeV. The ratios of branching fractions are measured to be B(B^0-bar -> D^+\\pi^-\\pi^+\\pi^-)/ B(B^0-bar -> D^+\\pi^-) = 2.38\\pm0.11\\pm0.21 B(B^- -> D^0\\pi^-\\pi^+\\pi^-) / B(B^- -> D^0\\pi^-) = 1.27\\pm0.06\\pm0.11 B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-) = 2.01\\pm0.37\\pm0.20 B(\\Lambda_b^0->\\Lambda_c^+\\pi^-\\pi^+\\pi^-) / B(\\Lambda_b^0 -> \\Lambda_c^+\\pi^-) = 1.43\\pm0.16\\pm0.13. We also report measurements of partial decay rates of these decays to excited charm hadrons. These results are of comparable or higher precision than existing measurements.


# EXAMPLE - OUTPUT (Format your answer exactly as in the below example output. Do not respond with anything else.)

{{
    "explanation": "The energy is 7 TeV, indicating run1. The paper reports precision measurements of branching fraction ratios, not a search, angular, or amplitude analysis, so strategy is 'other'. Since the production method is not explicitly mentioned, it is likely p-p. This paper studies the decays H_b to H_c pi- pi+ pi- and H_b to H_c pi- for four different values of H_b (H_c). Using names from the provided list of particles: B~0 (D+), B- (D0), B(s)~0 (D(s)+), and Lambda(b)0 (Lambda(c)+). The observables being measured are the branching fraction ratios B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-), B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-), B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-), and B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-). For each of the branching fraction ratios, there are two relevant decay channels that it is measured with.",
    "run": "run1",
    "strategy": "other",
    "decay": [
        {{
            "production": "p-p",
            "parent": "B~0",
            "children": ["D+", "pi-", "pi+", "pi-"],
            "name" : "B~0 -> D+ pi- pi+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "B~0",
            "children": ["D+", "pi-"],
            "name" : "B~0 -> D+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "B-",
            "children": ["D0", "pi-", "pi+", "pi-"],
            "name" : "B- -> D0 pi- pi+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "B-",
            "children": ["D0", "pi-"],
            "name" : "B- -> D0 pi-"
        }},
        {{
            "production": "p-p",
            "parent": "B(s)~0",
            "children": ["D(s)+", "pi-", "pi+", "pi-"],
            "name" : "B(s)~0 -> D(s)+ pi- pi+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "B(s)~0",
            "children": ["D(s)+", "pi-"],
            "name" : "B(s)~0 -> D(s)+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "Lambda(b)0",
            "children": ["Lambda(c)+", "pi-", "pi+", "pi-"],
            "name" : "Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-"
        }},
        {{
            "production": "p-p",
            "parent": "Lambda(b)0",
            "children": ["Lambda(c)+", "pi-"],
            "name" : "Lambda(b)0 -> Lambda(c)+ pi-"
        }}
    ],
    "observable": [
        {{
            "name": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
            "value": "2.38 \\pm 0.11 \\pm 0.21",
            "type": "branching_ratio",
            "relationship_measured_with": [
                {{
                    "target": "B~0 -> D+ pi- pi+ pi-"
                }},
                {{
                    "target": "B~0 -> D+ pi-"
                }}
            ]
        }},
        {{
            "name": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
            "value": "1.27 \\pm 0.06 \\pm 0.11",
            "type": "branching_ratio",
            "relationship_measured_with": [
                {{
                    "target": "B- -> D0 pi- pi+ pi-"
                }},
                {{
                    "target": "B- -> D0 pi-"
                }}
            ]
        }},
        {{
            "name": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
            "value": "2.01 \\pm 0.37 \\pm 0.20",
            "type": "branching_ratio",
            "relationship_measured_with": [
                {{
                    "target": "B(s)~0 -> D(s)+ pi- pi+ pi-"
                }},
                {{
                    "target": "B(s)~0 -> D(s)+ pi-"
                }}
            ]
        }},
        {{
            "name": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
            "value": "1.43 \\pm 0.16 \\pm 0.13",
            "type": "branching_ratio",
            "relationship_measured_with": [
                {{
                    "target": "Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-"
                }},
                {{
                    "target": "Lambda(b)0 -> Lambda(c)+ pi-"
                }}
            ]
        }}
    ]
}}


# INPUT

{abstract}


# OUTPUT

"""

    async def extract_abstract(self, abstract: str):
        response = None
        try:        
            prompt = self.get_abstract_extraction_prompt(abstract)
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
        
    def get_graph_extraction_prompt(self, text, observables):
        return f"""
You are an expert physicist specialized in extracting entities and relationships for a knowledge graph of systematic uncertainties in particle physics publications. Your task is to read the provided paper and perform comprehensive knowledge graph extraction.

# ENTITY TYPES AND PROPERTIES

1. "observable"
The relevant observables have already been extracted, and will be provided to you.

2. "uncertainty_source"
Identify all uncertainty sources affecting the measured observables. This includes the numerical values of physics constants or observables taken from other analyses which are taken as external inputs.

Categories:
- "statistical": From data variability due to random fluctuations or sampling limitations
- "internal": From analysis choices (reconstruction, efficiency modeling, background treatment) - usually LHCb-specific
- "external": From external inputs (theoretical calculations, previous measurements) - cannot be improved by analysis changes

"relationship_affects":
Connects an "uncertainty_source" to an "observable" when it directly contributes to that observable's total uncertainty budget. The "magnitude" is the numerical contribution (percentage, absolute value, etc) exactly as given in the paper to the observable's total uncertainty budget. The "ranking" attribute stores its relative importance in the total uncertainty budget with 1 being the most important source of uncertainty (often described as dominant), 2 less important, etc. If the ranking of an uncertainty source is unknown, then assign it the largest ranking.

3. "method"
Identify all techniques used to quantify systematic uncertainties. These include procedures that measure how much a systematic effect contributes to the total uncertainty, or approaches that assess how analysis choices impact the results. Key distinction: if a technique is used to measure the size of an uncertainty, it's a method; if a technique causes the uncertainty (e.g., a specific selection cut), it's an internal uncertainty source.

"relationship_estimates":
Links a "method" to an "uncertainty_source" when that method is used to quantify or evaluate the magnitude of the systematic uncertainty from that source.

# EXTRACTION PROCESS

## 1. Document Scan
Important Sections:
- Systematic uncertainty summary tables
- Detailed uncertainty sections
- Method descriptions
- Abstract/conclusion for final values

Scan for:
- Locate systematic uncertainty tables and error budget discussions
- Find all mentions of corrections, efficiencies, and uncertainties
- Identify sections with phrases: "systematic uncertainty", "correction factor", "evaluated using"

## 2. Extract Uncertainty Sources
For each uncertainty found:
- Name: Use exact paper terminology
- Description: Physical origin and why it's systematic
- Type: "statistical" (random), "internal" (analysis-specific), or "external" (from other measurements)
- Relationships: Link to observables with magnitude (e.g., "4%", "0.02 MeV")

## 3. Extract Methods
For each evaluation technique:
- Name: Descriptive title of the technique
- Description: What it measures and how it quantifies uncertainty
- Relationships: Which uncertainty sources it estimates

## 4. Key Extraction Rules
- Use uncertainty tables as primary source
- Match observable names exactly from provided list
- Preserve original magnitude notation from paper
- Extract all uncertainties, even if negligible or canceled
- Create separate entries for mode-dependent uncertainties
- If magnitude not specified, use "not specified"


# EXAMPLE - INPUT

Paper:

Branching fractions of the decays $H_b\\to H_c\\pi^-\\pi^+\\pi^-$ relative to $H_b\\to H_c\\pi^-$ are presented, where $H_b$ ($H_c$) represents B^0-bar($D^+$), $B^-$ ($D^0$), B_s^0-bar ($D_s^+$) and $\\Lambda_b^0$ ($\\Lambda_c^+$). The measurements are performed with the LHCb detector using 35${{\\rm pb^{{-1}}}}$ of data collected at $\\sqrt{{s}}=7$ TeV. The ratios of branching fractions are measured to be B(B^0-bar -> D^+\\pi^-\\pi^+\\pi^-)/ B(B^0-bar -> D^+\\pi^-) = 2.38\\pm0.11\\pm0.21 B(B^- -> D^0\\pi^-\\pi^+\\pi^-) / B(B^- -> D^0\\pi^-) = 1.27\\pm0.06\\pm0.11 B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-) = 2.01\\pm0.37\\pm0.20 B(\\Lambda_b^0->\\Lambda_c^+\\pi^-\\pi^+\\pi^-) / B(\\Lambda_b^0 -> \\Lambda_c^+\\pi^-) = 1.43\\pm0.16\\pm0.13. We also report measurements of partial decay rates of these decays to excited charm hadrons. These results are of comparable or higher precision than existing measurements.

Several sources contribute uncertainty to the measured ratios of branching fractions. Because
we are measuring ratios of branching fractions, most, but not all of the potential systematics cancel.
Here, we discuss only the non-cancelling uncertainties. With regard to the reconstruction
of the H_b \to H_c \pi^-\pi^+\pi^- and H_b \to H_c \pi^- decays, the former has two additional pions which need to
pass our selections, and the 3\pi system needs to pass the various vertex-related selection criteria.
The track reconstruction efficiency and uncertainty are evaluated by measuring the ratio of 
fully reconstructed J/\psi's to all J/\psi's obtained from an inclusive single muon trigger,
where only one of the muons is required to be reconstructed.
After reweighting the efficiencies to match the kinematics of the signal tracks, the uncertainty is
found to be 2\fraction ratios. The IP resolution in data is about 20\to (i) a larger efficiency for tracks to pass the IP-related cuts (as well as larger background),
and (ii) a lower efficiency to pass the vertex \chi^2 selections, for data relative to the
value predicted by simulation. The first of these is studied by reducing the IP \chi^2 requirement
in simulation by 20\simulation until it agrees with data. The combined correction is found to be 1.02\pm0.03.
Another potential source of systematic uncertainty is related to the production and decay model for
producing the H_c\pi\pi\pi final state. We have considered that the p_T spectrum of the pions
in the 3\pi system may be different between simulation and data. To estimate the uncertainty, we reweight the
MC simulation to replicate the momentum spectrum of the lowest momentum pion (amongst the pions 
in the 3\pi vertex.) We find that the total efficiency
using the reweighted spectra agrees with the unweighted spectra to within 3\also investigated the effect of differences in the p_T spectra of the charm particle,
and find at most a 1\M(\pi\pi\pi)<3~GeV/c^2. Given that the 
phase space population approaches zero as M(\pi\pi\pi) \to 3.5~GeV/c^2 ( \it i.e., M_B-M_D)
and that the simulation reasonably reproduces the \pi^-\pi^+\pi^- mass spectrum, we
use the simulation to assess the fraction of the \pi\pi\pi mass spectrum beyond 3~GeV/c^2.
We find the fraction of events above 3~GeV/c^2 is (3.5-4.5)\We apply a correction of 1.04\pm0.02, where we have assigned half the correction as 
an estimate of the uncertainty. In total, the correction for production and decay models is 1.04\pm0.04.
As discussed in Sec.~\ref sec:recsel , we choose only one candidate per event. The efficiency of
this selection is estimated by comparing the signal yield in multiple-candidate events before and
after applying the best candidate selection. The selection is estimated to be (75\pm20)\In the H_b \to H_c \pi^-\pi^+\pi^- the multiple candidate rate varies from 4\so we have corrections that vary from 1.01 to 1.03. For H_b \to H_c \pi^-, this effect is negligible.
The corrections for each mode are given in Table~\ref tab:syst .
For the trigger efficiency, we rely on signal MC simulations to emulate the online trigger.
The stability of the relative trigger efficiency was checked by reweighting the b-hadron
p_T spectra for both the signal and normalization modes, and re-evaluating the trigger efficiency ratios. 
We find maximum differences of 2\(2.4\Fitting systematics are evaluated by varying the background shapes and assumptions about the signal
parameterization for both the H_b \to H_c \pi^-\pi^+\pi^- 
and H_b \to H_c \pi^- modes and re-measuring the yield ratios. For the combinatorial background,
using first and second order polynomials leads to a 3\Reflection background uncertainties are negligible, except for 
\kern 0.18em \overline \kern -0.18em B ^0_s \to D_s^+\pi^-\pi^+\pi^- and \kern 0.18em \overline \kern -0.18em B ^0_s \to D_s^+\pi^-, where we find deviations as large as 5\the constraints on the \kern 0.18em \overline \kern -0.18em B ^0 \to D^+\pi^-\pi^+\pi^- and \kern 0.18em \overline \kern -0.18em B ^0 \to D^+\pi^- reflections by \pm1 standard deviation.
We have checked our sensitivity to the signal model by varying the constraints on the width ratio and 
core Gaussian area fraction by one standard deviation (2\We also include a systematic uncertainty of 1\neglecting the small radiative tail in the fit, which is estimated by comparing the
yields between our double Gaussian signal model and the sum of a Gaussian and Crystal Ball~\cite cbal2 
line shape. Taken together, we assign a 4\the relative yields. For the \kern 0.18em \overline \kern -0.18em B ^0_s branching fraction ratio, the total fitting uncertainty is 6.4\Another difference between the H_b \to H_c \pi^- and H_b \to H_c \pi^-\pi^+\pi^- selection is the upper limit on the number of
tracks. The efficiencies of the lower track multiplicity requirements can be evaluated using the samples with higher track 
multiplicity requirements. Using this technique, we find corrections of 0.95\pm0.01 for the B^- and \Lambda_b^0 
branching fraction ratios, and 0.99\pm0.01 for the \kern 0.18em \overline \kern -0.18em B ^0 and \kern 0.18em \overline \kern -0.18em B ^0_s branching fraction ratios.
We have also studied the PID efficiency uncertainty using a D^ *+ calibration sample in data. Since
the PID requirements are either common to the signal and normalization modes, or in the case of the bachelor
pion(s), the selection is very loose, the uncertainty is small and we estimate a correction of 1.01\pm0.01. We have
also considered possible background from H_b \to H_c D_s^- which results in a correction of 0.99\pm0.01.
All of our MC samples have a comparable number of events, from which we incur 3-4\the efficiency ratio determinations. 
The full set of systematic uncertainties and corrections are shown in Table~\ref tab:syst .
In total, the systematic uncertainty is \sim9\\begin table* [ht]
\begin center 
\caption Summary of corrections and systematic uncertainties to the ratio of branching fractions 
\cal B (H_b \to H_c \pi^-\pi^+\pi^-)/ \cal B (H_b \to H_c \pi^-). 
\begin tabular lcccc 
Source & \multicolumn 4 c central value \pm syst. error 
& \raisebox -0.5ex \kern 0.18em \overline \kern -0.18em B ^0 & \raisebox -0.5ex B^- & \raisebox -0.5ex \kern 0.18em \overline \kern -0.18em B ^0_s & \raisebox -0.5ex \Lambda_b 
[0.5ex]
Track Reconstruction & \multicolumn 4 c 1.00\pm0.04 
IP/Vertex Resolution & \multicolumn 4 c 1.02\pm0.03 
Production/Decay Model & \multicolumn 4 c 1.04\pm0.04 
Best Cand. Selection & 1.02\pm0.02 & 1.01\pm0.01 & 1.02\pm0.02 & 1.03\pm0.02
Trigger Efficiency & \multicolumn 4 c 1.00\pm0.02 
Fitting & 1.00\pm0.04 & 1.00\pm0.04 & 1.00\pm0.06 & 1.00\pm0.04
Cut on \#Tracks & 0.99\pm0.01 & 0.95\pm0.01 & 0.99\pm0.01 & 0.95\pm0.01
PID & \multicolumn 4 c 1.01\pm0.01 
H_c D_s^+ background & \multicolumn 4 c 0.99\pm0.01 
MC Statistics & 1.00\pm0.04 & ~~1.00\pm0.03~~ &~~ 1.00\pm0.04~~ & 1.00\pm0.04
Total Correction & 1.07 & 1.01 & 1.07 & 1.03
Total Systematic (\ 
\end tabular 
\end center 
\end table* 

Observables:

{{
    "observable": [
        {{
            "name": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
            "value": "2.38 \\pm 0.11 \\pm 0.21",
            "type": "branching_ratio"
        }},
        {{
            "name": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
            "value": "1.27 \\pm 0.06 \\pm 0.11",
            "type": "branching_ratio"
        }},
        {{
            "name": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
            "value": "2.01 \\pm 0.37 \\pm 0.20",
            "type": "branching_ratio"
        }},
        {{
            "name": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
            "value": "1.43 \\pm 0.16 \\pm 0.13",
            "type": "branching_ratio"
        }}
    ]
}}


# EXAMPLE - OUTPUT (Format your answer exactly as in the below example output. Do not respond with anything else.)

{{
    "uncertainty_source": [
        {{
            "description": "Comes from the three-body decay requiring two extra pions and additional vertex criteria compared to the two-body decay. These different requirements between decay modes lead to efficiency differences that don't cancel in the branching fraction ratio.",
            "name": "Track Reconstruction",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 2
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }}
            ]
        }},
        {{
            "description": "Arises from data having approximately 20% worse impact parameter resolution than predicted by simulation, which affects both the track selection efficiencies (increasing acceptance of both signal and background) and vertex χ² distributions.",
            "name": "IP/Vertex Resolution",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.03",
                    "ranking": 2
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.03",
                    "ranking": 2
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.03",
                    "ranking": 3
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.03",
                    "ranking": 2
                }}
            ]
        }},
        {{
            "description": "Accounts for potential differences between simulation and data in the momentum distributions of particles (pions and charm particles) and the fraction of events beyond the analyzed mass range, as these modeling differences can affect the overall efficiency and event selection.",
            "name": "Production/Decay Model",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 2
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }}
            ]
        }},
        {{
            "description": "Arises from the efficiency loss when selecting a single candidate from events with multiple candidates in the analysis. This correction varies by decay mode due to different multiple candidate rates, with some modes experiencing negligible effects while others require small corrections.",
            "name": "Best Candidate Selection",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 3
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 4
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 3
                }}
            ]
        }},
        {{
            "description": "Accounts for potential variations in the relative trigger efficiency between signal and normalization modes.",
            "name": "Trigger Efficiency",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 3
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.02",
                    "ranking": 3
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 4
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.02",
                    "ranking": 3
                }}
            ]
        }},
        {{
            "description": "Accounts for variation in combinatorial background shapes, signal width and shape parameters, and the treatment of radiative tails.",
            "name": "Fitting",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.06",
                    "ranking": 1
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }}
            ]
        }},
        {{
            "description": "Accounts for differences in selection efficiency when applying requirements on the number of charged tracks in the event.",
            "name": "Cut on #Tracks",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 5
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }}
            ]
        }},
        {{
            "description": "The PID (Particle Identification) efficiency uncertainty.",
            "name": "PID",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 5
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }}
            ]
        }},
        {{
            "description": "Uncertainties due to possible H_b -> H_c D_s background contamination.",
            "name": "H_c D_s background",
            "type": "internal",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 5
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.01",
                    "ranking": 4
                }}
            ]
        }},
        {{
            "description": "Uncertainties in the efficiency ratio determinations due to limited Monte Carlo event statistics.",
            "name": "MC Statistics",
            "type": "statistical",
            "relationship_affects": [
                {{
                    "target": "B(B~0 -> D+ pi- pi+ pi-) / B(B~0 -> D+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }},
                {{
                    "target": "B(B- -> D0 pi- pi+ pi-) / B(B- -> D0 pi-)",
                    "magnitude": "0.03",
                    "ranking": 2
                }},
                {{
                    "target": "B(B(s)~0 -> D(s)+ pi- pi+ pi-) / B(B(s)~0 -> D(s)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 2
                }},
                {{
                    "target": "B(Lambda(b)0 -> Lambda(c)+ pi- pi+ pi-) / B(Lambda(b)0 -> Lambda(c)+ pi-)",
                    "magnitude": "0.04",
                    "ranking": 1
                }}
            ]
        }}
    ],
    "method": [
        {{
            "description": "tracking efficiency is calibrated by measuring the ratio of fully reconstructed J/ψ mesons to single-muon samples from inclusive triggers. Efficiencies are reweighted to match signal track kinematics to determine systematic uncertainty.",
            "name": "J/ψ tracking efficiency calibration",
            "relationship_estimates": [
                {{
                    "target": "Track Reconstruction"
                }}
            ]
        }},
        {{
            "description": "Corrects simulation discrepancies by adjusting IP selection criteria and smearing vertex quality distributions to match data. This addresses both increased track acceptance and reduced vertex efficiency from degraded IP resolution.",
            "name": "IP resolution calibration",
            "relationship_estimates": [
                {{
                    "target": "IP/Vertex Resolution"
                }}
            ]
        }},
        {{
            "description": "Monte Carlo simulations are reweighted to match the data's pT distribution for the lowest momentum pion in three-pion systems, testing if momentum spectra differences affect detection efficiency calculations.",
            "name": "Momentum spectrum reweighting",
            "relationship_estimates": [
                {{
                    "target": "Production/Decay Model"
                }}
            ]
        }},
        {{
            "description": "Measures how effectively the best candidate selection retains true signal events when multiple candidates exist per event, by comparing yields before and after selection.",
            "name": "Multiple candidate yield comparison",
            "relationship_estimates": [
                {{
                    "target": "Best Candidate Selection"
                }}
            ]
        }},
        {{
            "description": "Adjusts momentum distributions in Monte Carlo simulations to test trigger efficiency stability across levels. This evaluates systematic uncertainties by modifying kinematics and recalculating signal-to-normalization efficiency ratios.",
            "name": "b-hadron pT reweighting",
            "relationship_estimates": [
                {{
                    "target": "Trigger Efficiency"
                }}
            ]
        }},
        {{
            "description": "Tests systematic uncertainties by changing background models (polynomial orders, reflection constraints) and signal parameters (width ratios, Gaussian fractions, line shapes), then recalculating yields to quantify the impact.",
            "name": "Fit variation",
            "relationship_estimates": [
                {{
                    "target": "Fitting"
                }}
            ]
        }},
        {{
            "description": "Evaluates efficiency of lower track requirements using higher track data samples. This technique determines correction factors for selection efficiency differences between decay channels with varying final state tracks.",
            "name": "Track multiplicity comparison",
            "relationship_estimates": [
                {{
                    "target": "Cut on #Tracks"
                }}
            ]
        }},
    ]
}}


# INPUT

Paper:

{text}

Observables:

{{
    "observable": {observables}
}}


# OUTPUT

"""

    async def extract_graph(self, text, observables):
        response = None
        try:        
            prompt = self.get_graph_extraction_prompt(text, observables)
            response = (await llm.acomplete(prompt)).text
            graph = eval(response.replace("```python","").replace("```",""))
            return graph
            
        except Exception as e:
            print(f"Exception occurred: {e} \n Paper Text: \n {text} \n Got Response: \n {response}")
            return {}

class Entity():
    def __init__(self, name: str, type: str, description: str, attributes: dict = {}):
        self.name = name
        self.type = type
        self.description = description
        self.attributes = attributes

    def __repr__(self):
        return f"{self.type}: {self.name} \n\t {self.description}"
    
    def add_to_graph(self, graph):
        from py2neo import Node
        embedding = embedding_model.encode([f"{self.name}: {self.description}"])[0].tolist()
        neo4j_node = Node(self.type, name=self.name, description=self.description, embedding=embedding, **{key : value if isinstance(value, str) else repr(value) for key, value in self.attributes.items() if value })
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
        paper_entity = Entity(name=paper.arxiv_id, type="paper", description=paper.abstract, attributes=copy.deepcopy(paper.metadata))
        self.add_entity(paper_entity)

        for type in paper.entities:
            for dict in paper.entities[type]:
                entity_dict = copy.deepcopy(dict)
                entity_dict["arxiv_id"] = paper.arxiv_id
                name = entity_dict.pop("name")
                description = entity_dict.pop("description", "")
                entity = Entity(name=name, type=type, description=description, attributes=entity_dict)
                
                self.add_entity(entity)

                if type in "observable":
                    relationship = Relationship(paper_entity, "determines", entity)
                    self.add_relationship(relationship)
        
        for type in paper.relationships:
            for dict in paper.relationships[type]:
                relationship_dict = copy.deepcopy(dict)
                
                source_name = relationship_dict.pop("source")
                try:  
                    source = self.get_entity(source_name)
                except:
                    print(f"Error loading relationship from {paper.arxiv_id}: \n {dict} \n source {source_name} does not exist")
                    continue
                
                target_name = relationship_dict.pop("target")
                try:
                    target = self.get_entity(target_name)
                except:
                    print(f"Error loading relationship from {paper.arxiv_id}: \n {dict} \n target {target_name} does not exist")
                    continue
            
                relationship = Relationship(source, type, target, relationship_dict)
                self.add_relationship(relationship)
    
    def push_to_neo4j(self, uri, username, password):
        graph = Graph(uri, auth=(username, password))
        graph.delete_all()

        for name, entity in self.entities.items():
            entity.add_to_graph(graph)

        for relationship in self.relationships:
            relationship.add_to_graph(graph)

    def cluster_entities(self, entities, identical_attributes: list[str], max_cluster_size: int, verbose: bool = False):
        """
        Find optimal threshold and cluster entities in one pass.
        Returns clusters with largest cluster <= max_cluster_size.
        """
        # Create text representation for each entity
        texts = [f"{entity.name} {entity.description}" for entity in entities]
        
        # Domain-specific stop words for systematic uncertainties
        stop_words = stopwords.words('english') + [
            'systematic', 'uncertainty', 'uncertainties', 'error', 'errors', 'correction',
            'corrected', 'method', 'methods', 'effect', 'effects', 'source', 'sources'
        ]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.5,
            stop_words=stop_words,
            ngram_range=(1, 3)
        )
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Apply identical attributes constraints
        n_entities = len(entities)
        for i in range(n_entities):
            for j in range(n_entities):
                # If entities came from the same paper, they should not be merged
                arxiv_id_i = entities[i].attributes.get("arxiv_id")
                arxiv_id_j = entities[j].attributes.get("arxiv_id")
                if arxiv_id_i == arxiv_id_j:
                    similarity_matrix[i,j] = 0.0
                # If entities have a different attribute from identical_attributes, they should not be merged
                for attr in identical_attributes:
                    val_i = entities[i].attributes.get(attr)
                    val_j = entities[j].attributes.get(attr)
                    if val_i != val_j:
                        similarity_matrix[i, j] = 0.0
        
        # Search from high to low threshold to find optimal clustering
        for threshold in np.arange(0.0, 1.0, 0.005):
            distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
            
            # Apply Agglomerative Clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1-threshold,
                metric='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            clusters = list(clusters.values())
            
            max_size = max(len(cluster) for cluster in clusters)
            
            if max_size <= max_cluster_size:
                # Found optimal threshold
                if verbose:
                    print(f"Optimal cluster found using threshold = {threshold} with max size = {max_size}")
                return clusters

    async def llm_cluster_entities(self, cluster_entities: List[Entity]) -> List[dict]:
        """
        Uses LLM to make final clustering decisions on TF-IDF suggested clusters.
        """
        
        # Create enumerated entity descriptions for the LLM
        entities_str = ""
        for i, entity in enumerate(cluster_entities):
            entity_attrs = "{" + ", ".join([f'"{k}": "{v}"' for k, v in entity.attributes.items()]) + "}"
            entities_str += f"\n{i+1}. {{\n"
            entities_str += f'   "name": "{entity.name}",\n'
            entities_str += f'   "type": "{entity.type}",\n'
            entities_str += f'   "description": "{entity.description}",\n'
            entities_str += f'   "attributes": {entity_attrs}\n}}\n'
        
        prompt = f"""You are an expert particle physicist in the LHCb collaboration.

# TASK
You have been given a cluster of {len(cluster_entities)} potentially similar entities that were identified by an algorithmic clustering method. Your job is to make the final decision about which entities (if any) should actually be merged together.

# INSTRUCTIONS
1. Carefully examine each entity's name, description, and attributes
2. Decide which entities represent the same or very very similar particle physics concepts
3. You can merge all entities into one, create multiple merged groups, or decide not to merge any entities at all.

# CRITICAL RULES
1. NEVER change attribute names - keep them exactly as they appear. Include ALL attributes
2. For "arxiv_id": combine into comma-separated list 
3. For other attributes: choose most representative value or create appropriate ranges/combinations

# MERGING GUIDELINES
- Entities describing same or very very similar concepts should be merged even if their names differ slightly
- Focus on conceptual similarity NOT wording or semantic similarity
- When merging, create comprehensive descriptions that capture all important aspects from the source entities
- Each source entity can only be assigned to ONE merged entity. If an entity could fit in multiple groups, this implies that your grouping is bad. If unsure, do not merge.
- All merged entities must have more than one source entity. All source entities that should be kept distinct are not included in your output. Ensure that length of "source_indices" is always greater than one.
- Do NOT merge entities just because they share common keywords (e.g., "asymmetry", "K-matrix", "background")
- Different particles or decay channels indicate entities that should remain separate, even if they are otherwise extremely similar.
- General vs specific: a general "fit model" uncertainty is different from a specific "D^- K_S^0 K-matrix model" uncertainty

IMPORTANT: Provide ONLY the Python list in your response. Do not include any explanations, reasoning, or additional text before or after the list. Your entire response should be parseable as a Python list.

# Example Input 1:

1. {{
   "name": "Tracking efficiency",
   "type": "uncertainty_source",
   "description": "Uncertainty due to the presence of prompt hadrons in the signal channel without counterparts in the normalization channel, accounting for hadronic interactions and reconstruction efficiency differences, including downstream track reconstruction precision.",
   "attributes": {{"type": "internal", "arxiv_id": "2501.12779"}}
}}

2. {{
   "name": "Track reconstruction",
   "type": "uncertainty_source",
   "description": "Uncertainty related to the calibration of tracking efficiencies using J/ψ → μ+μ− control samples, with an assigned uncertainty of 0.8% per track that largely cancels in production ratios; residual uncertainty evaluated by sampling methods.",
   "attributes": {{"type": "internal", "arxiv_id": "2501.12611"}}
}}

3. {{
   "name": "Trigger efficiency",
   "type": "uncertainty_source",
   "description": "Systematic uncertainty from residual data-simulation differences and limitations in data-driven methods used to determine trigger efficiencies.",
   "attributes": {{"type": "internal", "arxiv_id": "2412.09414"}}
}}

4. {{
   "name": "Trigger efficiency estimation",
   "type": "uncertainty_source",
   "description": "Systematic uncertainty assigned from a data-driven method determining trigger efficiency from events triggered independently of the signal muons.",
   "attributes": {{"type": "internal", "arxiv_id": "2411.05669"}}
}}

5. {{
   "name": "Additional corrections",
   "type": "uncertainty_source",
   "description": "Systematic uncertainty from residual mismodelling in variables not directly used in the selection after baseline corrections. Assessed by recomputing efficiencies after correcting these additional distributions and taking the difference as uncertainty.",
   "attributes": {{"type": "internal", "arxiv_id": "2501.12779"}}
}}

6. {{
   "name": "Tracking efficiency",
   "type": "uncertainty_source",
   "description": "Systematic uncertainty from residual data-simulation differences and limitations in data-driven methods used to determine tracking efficiencies.",
   "attributes": {{"type": "internal", "arxiv_id": "2412.09414"}}
}}

7. {{
    "name": "Pion reconstruction efficiency",
    "type": "uncertainty_source",
    "description": "Uncertainty in the efficiency for pion reconstruction and selection, including corrections for hadronic interactions derived from control samples and simulation-based track-quality and particle-identification requirements."
    "attributes": {{"type": "internal", "arxiv_id": "2510.09414"}}
}}

8. {{
    "name": "Jet reconstruction efficiency",
    "type": "uncertainty_source",
    "description": "Uncertainty from the efficiency to reconstruct jets given a reconstructed tag, which varies with jet transverse momentum and the fragmentation variable z, decreasing by 10-25% at large z values."
    "attributes": {{"type": "internal", "arxiv_id": "1512.02414"}}
}}

9. {{
    "name": "Choice of lineshape models",
    "type": "uncertainty_source",
    "description": "Systematic uncertainty arising from the choice of lineshape models used to describe resonances, including the difference between the RBW and K-matrix models for the T_{{c\bar{{s}}}} states. This is the dominant uncertainty on the T_{{c\bar{{s}}}} width."
    "attributes": {{"type": "internal", "arxiv_id": "2512.09414"}}
}}

10. {{
    "name": "D^- K_S^0 K-matrix model",
    "type": "uncertainty_source",
    "description": "Systematic uncertainty from using an alternative K-matrix parameterization to describe the spin-1 D^- K_S^0 contributions instead of the nominal model."
    "attributes": {{"type": "internal", "arxiv_id": "1519.02414"}}
}}

# Example Output 1:

[
    {{
        "merged_entity": {{
            "name": "Tracking Efficiency",
            "description": "Systematic uncertainty related to track reconstruction efficiency, including calibration using J/ψ → μ+μ− control samples, residual data-simulation differences, limitations in data-driven methods, hadronic interaction effects, and downstream track reconstruction precision. Includes assigned uncertainties of 0.8% per track evaluated by sampling methods.",
            "attributes": {{
                "arxiv_id": "2501.12779, 2501.12611, 2412.09414",
                "type": "internal"
            }}
        }},
        "source_indices": [1, 2, 6]
    }},
    {{
        "merged_entity": {{
            "name": "Trigger Efficiency",
            "description": "Systematic uncertainty from residual data-simulation differences and limitations in data-driven methods used to determine trigger efficiencies, including uncertainties assigned from events triggered independently of signal muons.",
            "attributes": {{
                "arxiv_id": "2412.09414, 2411.05669",
                "type": "internal"
            }}
        }},
        "source_indices": [3, 4]
    }}
]

# Example Input 2:

1. {{
  "name": "Tracking efficiency",
  "type": "uncertainty_source",
  "description": "Uncertainty due to the presence of prompt hadrons in the signal channel without counterparts in the normalization channel, accounting for hadronic interactions and reconstruction efficiency differences, including downstream track reconstruction precision.",
  "attributes": {{"type": "internal", "arxiv_id": "2501.12779"}}
}}

2. {{
  "name": "Trigger efficiency",
  "type": "uncertainty_source",
  "description": "Systematic uncertainty from imperfect simulation of hardware and software trigger efficiencies, studied separately for muon and dimuon hardware triggers using tag-and-probe and data-driven methods, with the quadratic sum taken as the total uncertainty.",
  "attributes": {{"type": "internal", "arxiv_id": "2501.12611"}}
}}

3. {{
  "name": "PID efficiency",
  "type": "uncertainty_source",
  "description": "Uncertainty on the particle identification efficiency estimated from data using a tag-and-probe method on phi -> K+ K- decays.",
  "attributes": {{"type": "internal", "arxiv_id": "2411.09343"}}
}}

4. {{
  "name": "Selection bias",
  "type": "uncertainty_source",
  "description": "Systematic uncertainty from potential bias in mass measurements originating from event selection, evaluated by comparing reconstructed masses in simulation before and after offline selection, assigned as 0.02 MeV.",
  "attributes": {{"type": "internal", "arxiv_id": "2502.18987"}}
}}

5. {{
  "name": "Neutral kaon system effects",
  "type": "uncertainty_source", 
  "description": "Systematic uncertainty arising from CP violation and mixing in the neutral kaon system and different interaction probabilities of K0 and anti-K0 particles with detector material, affecting the control mode used for asymmetry subtraction.",
  "attributes": {{"type": "internal", "arxiv_id": "2503.02711"}}
}}

6. {{
    "name": "Pion reconstruction efficiency",
    "type": "uncertainty_source",
    "description": "Uncertainty in the efficiency for pion reconstruction and selection, including corrections for hadronic interactions derived from control samples and simulation-based track-quality and particle-identification requirements."
    "attributes": {{"type": "internal", "arxiv_id": "2510.09414"}}
}}

7. {{
    "name": "Jet reconstruction efficiency",
    "type": "uncertainty_source",
    "description": "Uncertainty from the efficiency to reconstruct jets given a reconstructed tag, which varies with jet transverse momentum and the fragmentation variable z, decreasing by 10-25% at large z values."
    "attributes": {{"type": "internal", "arxiv_id": "1512.02414"}}
}}

9. {{
    "name": "Choice of lineshape models",
    "type": "uncertainty_source",
    "description": "Systematic uncertainty arising from the choice of lineshape models used to describe resonances, including the difference between the RBW and K-matrix models for the T_{{c\bar{{s}}}} states. This is the dominant uncertainty on the T_{{c\bar{{s}}}} width."
    "attributes": {{"type": "internal", "arxiv_id": "2512.09414"}}
}}

10. {{
    "name": "D^- K_S^0 K-matrix model",
    "type": "uncertainty_source",
    "description": "Systematic uncertainty from using an alternative K-matrix parameterization to describe the spin-1 D^- K_S^0 contributions instead of the nominal model."
    "attributes": {{"type": "internal", "arxiv_id": "1519.02414"}}
}}

# Example Output 2:

[]

# Input

{entities_str}

# Output

"""
        
        response = None
        try:
            response = (await llm.acomplete(prompt)).text
            
            # Extract the list from the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                return []  # No merging decisions made
                
            list_text = response[start:end]
            merge_decisions = eval(list_text.replace("```python", "").replace("```", ""))
            
            # Convert merge decisions to Entity objects
            results = []
            for decision in merge_decisions:
                merged_data = decision["merged_entity"]
                source_indices = [idx - 1 for idx in decision["source_indices"]]  # Convert to 0-based indexing
                
                merged_entity = Entity(
                    name=merged_data["name"],
                    type=cluster_entities[0].type,  # All entities in cluster should have same type
                    description=merged_data["description"],
                    attributes=merged_data.get("attributes", {})
                )
                
                results.append({
                    "merged_entity": merged_entity,
                    "source_indices": source_indices
                })
                
            return results
            
        except Exception as e:
            print(f"Error in LLM clustering decision: {e}")
            print(f"LLM Response: {response}")
            return []  # Fall back to no merging

    async def merge_entity_type_async(self, type: str, identical_attributes: List[str], stop_ratio: float, max_cluster_size: int, verbose: bool):
        """
        Iterative clustering with dynamic threshold selection.
        Each iteration clusters remaining entities, uses LLM to make merge decisions,
        and continues until no merges are made in an iteration.
        """
        if type not in self.entity_types:
            print(f"ERROR: type {type} must be one of {list(self.entity_types.keys())}")
            return
        
        iteration = 1
        total_merges = 0
        
        while True:
            entities = list(self.entity_types[type])
            
            if len(entities) <= 1:
                if verbose:
                    print(f"Only {len(entities)} entities remaining. Stopping.")
                break
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"ITERATION {iteration}")
                print(f"{'='*50}")
                print(f"Starting with {len(entities)} entities of type '{type}'")
            
            clusters = self.cluster_entities(entities, identical_attributes, max_cluster_size, verbose)
            
            # Filter to only multi-entity clusters
            multi_entity_clusters = [cluster for cluster in clusters if len(cluster) > 1]
            
            # Process each cluster with LLM
            iteration_merges = 0
            entities_to_remove = set()
            
            for cluster_idx, cluster_indices in enumerate(multi_entity_clusters):
                cluster_entities = [entities[i] for i in cluster_indices]
                
                if verbose:
                    print(f"\n--- Processing Cluster {cluster_idx + 1} ---")
                    print(f"Sending {len(cluster_entities)} entities to LLM:")
                    for entity in cluster_entities:
                        print(f"  - {entity.name}: {entity.description}")
                
                # Get LLM's decision on this cluster
                llm_decisions = await self.llm_cluster_entities(cluster_entities)
                
                if verbose:
                    print(f"LLM decided to create {len(llm_decisions)} merged entities")
                
                # Process each merge decision
                for decision in llm_decisions:
                    merged_entity = decision["merged_entity"]
                    source_indices = decision["source_indices"]
                    
                    # Convert local indices back to global entity indices
                    global_source_indices = [cluster_indices[local_idx] for local_idx in source_indices]
                    source_entities = [entities[global_idx] for global_idx in global_source_indices]
                    
                    if len(source_entities) < 2:
                        continue  # Skip if not actually merging multiple entities
                    
                    if verbose:
                        print(f"  Merging {len(source_entities)} entities:")
                        for e in source_entities:
                            print(f"    - {e.name}")
                        print(f"  Into: {merged_entity.name}")
                    
                    # Add the new merged entity
                    self.add_entity(merged_entity)
                    
                    # Mark source entities for removal
                    entities_to_remove.update(source_entities)
                    
                    # Update relationships
                    relationships_to_update = []
                    for relationship in self.relationships.copy():
                        if relationship.source in source_entities:
                            new_relationship = Relationship(
                                source=merged_entity,
                                type=relationship.type,
                                target=relationship.target,
                                attributes=relationship.attributes
                            )
                            relationships_to_update.append((relationship, new_relationship))
                        
                        elif relationship.target in source_entities:
                            new_relationship = Relationship(
                                source=relationship.source,
                                type=relationship.type,
                                target=merged_entity,
                                attributes=relationship.attributes
                            )
                            relationships_to_update.append((relationship, new_relationship))
                    
                    # Apply relationship updates
                    for old_rel, new_rel in relationships_to_update:
                        self.relationships.remove(old_rel)
                        self.relationships.add(new_rel)
                    
                    iteration_merges += 1
            
            
            num_entities = len(entities)
            num_removed = len(entities_to_remove)
            
            # Remove all merged entities
            for entity in entities_to_remove:
                self.entities.pop(entity.name, None)
                self.entity_types[type].discard(entity)
            
            if verbose:
                print(f"\n--- Iteration {iteration} Results ---")
                print(f"Merges performed: {iteration_merges}")
                print(f"Entities removed: {num_removed}")
                print(f"Remaining entities: {len(self.entity_types[type])}")
            
            total_merges += iteration_merges
            
            # Stop if no merges were made in this iteration
            if num_removed < stop_ratio * num_entities:
                if verbose:
                    print("Too few merges made in this iteration. Stopping.")
                break
            
            iteration += 1
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"FINAL RESULTS")
            print(f"{'='*50}")
            print(f"Total iterations: {iteration}")
            print(f"Total merges: {total_merges}")
            print(f"Final number of {type} entities: {len(self.entity_types[type])}")

    def merge_entity_type(self, type: str, identical_attributes: List[str] = [], stop_ratio: float = 0.1, max_cluster_size: int = 100, verbose: bool = False):
        """
        Main entry point for iterative entity merging with dynamic threshold selection.
        """
        asyncio.run(self.merge_entity_type_async(type, identical_attributes, stop_ratio, max_cluster_size, verbose))