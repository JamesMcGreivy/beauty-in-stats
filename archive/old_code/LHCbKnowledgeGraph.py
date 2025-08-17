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


def get_lhcb_entity_extraction_prompt(text):
    return f"""
    ## INSTRUCTIONS ##

    You are an expert in high energy physics, particularly in LHCb experiments at the Large Hadron Collider. Your task is to extract entities from the provided text to construct a knowledge graph of systematic uncertainties and related analysis procedures in LHCb papers.

    Extract entities that belong only to these entity types:

    Type: measurement_quantity
    - The specific physical quantity being measured. Examples include branching fraction ratio, cross section, mass measurement, asymmetries, lifetime or decay width, form factors, angular observables, CKM matrix elements, mixing and oscillation parameters, upper limits or signal exclusion.

    Type: decay_process
    - Any primary, reference, control, etc decay process or decay channel studied or mentioned in the analysis.

    Type: data_period
    - Specific data-taking periods with defined conditions. Examples include Run 1, Run 2, specific years, specific luminosities (1 fb^-1), etc.

    Type: analysis_technique
    - Specific methods, models, or algorithms used in the analysis. This includes signal/background modeling functions (like Crystal Ball or Breit-Wigner), form factor models (like ISGW2, LLSW), efficiency determination methods, kinematic corrections, and other technical approaches used in the analysis pipeline.

    Type: systematic_uncertainty_source
    - Specific sources of systematic uncertainties in LHCb analyses. Make sure to include any quantitative information about the magnitude of this systematic uncertainty contribution.

    Type: systematic_uncertainty_evaluation
    - Techniques for evaluating the impact of systematic uncertainties.

    Type: systematic_uncertainty_correlations
    - Information about correlations between different systematic uncertainties. Captures statements about which uncertainties are considered correlated or uncorrelated.

    Type: systematic_handling_method
    - How systematic uncertainties are handled in the final result (e.g., added in quadrature, treated as nuisance parameters, etc.)

    Only extract entities that represent established techniques, methods, or concepts in LHCb analyses. Do not extract:
    - Paper-specific variables or notation
    - Generic mathematical expressions
    - Overly broad concepts
    - Paper-specific parameter names without wider significance
    - One-off examples used purely for illustration

    Input: Raw text from an LHCb paper describing analysis procedures.

    Output: A list of Python dictionaries with the following structure:
    [
        {{"entity_name" : "name of the entity", "entity_type" : "one of the types defined above", "magnitude" : "only include for systematic_uncertainty_source", "justification" : "include direct summary of the parts of the text that describe this entity"}},
        ...
    ]

    If the provided text doesn't include any relevant entities, return an empty list: []

    Do not include anything else in your output, including ```python ... ``` or ```json ... ```. Do not make any information up that does not exist within the text.

    ## EXAMPLE ##

    Input:
    The signal yield is compared to the yield of the normalisation channel to determine the branching fraction ratio $\\kappa_{{D_{{1,2}}^{{**0}}}}$, defined as
    \\begin{{equation}}
    \\label{{eqr}}
    \\kappa_{{D_{{1,2}}^{{**0}}}} \\equiv \\frac{{{{\\cal B}}(B^- \\to D_{{1,2}}^{{**0}} \\tau^- \\overline{{\\nu}}_\\tau)}}{{{{\\cal B}}(B^- \\to D_{{1,2}}^{{**0}} D_s^{{(*)-}})}} = \\frac{{N_{{\\text{{signal}}}}}}{{N_{{\\text{{norm}}}}}} \\times \\frac{{{{\\cal B}}(D_s^- \\to \\pi^- \\\\pi^- \\pi^+)}}{{{{\\cal B}}(\\tau^- \\to \\pi^- \\pi^- \\pi^+ (X)) \\times \\epsilon_R}},
    \\end{{equation}}
    where $N_{{\\text{{signal}}}}$ and $N_{{\\text{{norm}}}}$ are the signal and normalisation yields, respectively, and $\\epsilon_R$ is the ratio of selection efficiencies between the two channels. Systematic uncertainties on $\\kappa_{{D_{{1,2}}^{{**0}}}}$ are assessed through various studies. To verify fit stability and potential bias, the same fitting procedure is applied to simulated samples of inclusive $b$-hadron decays producing $D^* \\pi^- \\pi^+ \\pi^-$ final states, with signal-to-background ratios matching data. The fitted $D_{{1,2}}^{{**0}}$ yields agree within $\\pm1\\sigma$ of the true simulated yields. This agreement holds even when reducing the signal yield in the simulation. In the extreme case of no signal events, the fit returns $35 \\pm 80$, consistent with zero. The $B^- \\to D^{{**0}} \\tau^- \\overline{{\\nu}}_\\tau$ decays are simulated using the ISGW2 model~\\cite{{ISGW}}. Alternative models, LLSW~\\cite{{LLSW}} and BLR~\\cite{{BLR}}, are implemented using the \\textsc{{Hammer}} package~\\cite{{Hammer}} to reweight the $q^2$ distribution for the $B^- \\to D_1(2420)^0 \\tau^- \\overline{{\\nu}}_\\tau$ component. The resulting yields differ by 3.7\\% from the ISGW2 baseline, which is assigned as the systematic uncertainty due to limited form-factor knowledge. The fit assumes a fixed ratio of $D_1(2420)^0$ to $D_2^*(2460)^0$ contributions. Varying this ratio between 0 and 1 leads to a 4.4\\% systematic uncertainty. Fits with large $D_2^*(2460)^0$ contributions result in worse $\\chi^2$, excluding a pure $D_2^*(2460)^0$ component at $2.7\\sigma$, and disfavoring a 50\\% $D_2^*(2460)^0$ fraction at $2\\sigma$. The systematic uncertainty from limited simulated sample size is evaluated using the bootstrap method~\\cite{{efron}}, following Ref.~\\cite{{LHCb-PAPER-2022-052}}, yielding a 4.1\\% uncertainty. Changing the fit variables, e.g., replacing $q^2$ with the $D^*$ helicity angle~\\cite{{LHCb-PAPER-2023-020}}, and altering binning schemes, results in a 5.0\\% uncertainty, taken as half the largest observed deviation. Background from $B^- \\to D^{{**0}} \\overline{{D}}^0 X$ and $B^- \\to D^{{**0}} D^- (X)$ is estimated to contribute a 3.6\\% uncertainty, assuming half of the background yield is signal-like. This analysis requires reconstructing an additional track compared to the $\\mathcal{{R}}(D^*)$ analysis~\\cite{{LHCb-PAPER-2022-052}}. The associated efficiency uncertainty largely cancels due to the same requirement in the normalisation channel. A control sample of $B^- \\to D^{{*+}} \\pi^- \\pi^- \\pi^- \\pi^+$ decays is used to compare selection efficiencies in data and simulation. Relative efficiencies are evaluated near the known $B^-$ mass~\\cite{{PDG2024}}. Good agreement is found except for the vertex detachment requirement, which differs by 10\\%. After correcting for this, a 2\\% systematic uncertainty is assigned for selection efficiency, consistent with the $\\mathcal{{R}}(D^*)$ analysis. Prompt background suppression via the detachment requirement must be well modelled. Suppression factors in data and simulation are compared by varying the vertex separation requirement from 2 to 4$\\sigma$. A 4\\% difference is observed and assigned as the related systematic uncertainty. The analysis uses proton-proton ( {{ p }} {{ p }} ) collision data recorded by the LHCb experiment at a center-of-mass energy of 13 {{TeV}} in the years 2016, 2017 and 2018, which correspond to an integrated luminosity of 5.4 {{{{ fb}}^{{-1}}}}.

    # Type: measurement_quantity
    # Type: decay_process
    # Type: data_period
    # Type: analysis_technique
    Type: systematic_uncertainty_source
    Type: systematic_uncertainty_evaluation
    Type: systematic_uncertainty_correlations
    Type: systematic_handling_method

    Output:
    [ 
    {{"entity_name": "branching fraction ratio", "entity_type": "measurement_quantity", "justification": "The branching fraction ratio is used to determine the ratio of signal yields to normalization yields, specifically for the decay processes involving $B^- \to D_{{{{1,2}}}}^{{{{0}}}} \tau^- \overline{{{{\nu}}}}\tau$ and $B^- \to D{{{{1,2}}}}^{{{{0}}}} D_s^{{{{()-}}}}$."}}, {{"entity_name": "B^- \to D_{{{{1,2}}}}^{{{{**0}}}} \tau^- \overline{{{{\nu}}}}\tau", "entity_type": "decay_process", "justification": "This decay process is part of the formula for the branching fraction ratio and is used to define the signal yield."}}, {{"entity_name": "B^- \to D{{{{1,2}}}}^{{{{**0}}}} D_s^{{{{()-}}}}", "entity_type": "decay_process", "justification": "This decay process serves as the denominator in the branching fraction ratio formula, providing the normalization yield."}}, {{"entity_name": "D_s^- \to \pi^- \\pi^- \pi^+", "entity_type": "decay_process", "justification": "This decay process contributes to the calculation of the branching fraction ratio, specifically in relation to the $\mathcal{{B}}(D_s^- \to \pi^- \pi^- \pi^+)$ term."}}, {{"entity_name": "\tau^- \to \pi^- \pi^- \pi^+ (X)) \times \epsilon_R", "entity_type": "decay_process", "justification": "This decay process appears as part of the denominator in the branching fraction ratio, factoring in the selection efficiencies between channels."}}, {{"entity_name": "B^- \to D_1(2420)^0 \tau^- \overline{{{{\nu}}}}\tau", "entity_type": "decay_process", "justification": "This decay process is one of the components modeled in simulations to assess uncertainties and is included in the systematic uncertainty due to the limited form-factor knowledge."}}, {{"entity_name": "B^- \to D^{{{{**0}}}} \overline{{{{D}}}}^0 X", "entity_type": "decay_process", "justification": "This decay process contributes to the background estimation, and the uncertainty arising from its background yield is evaluated as part of systematic uncertainties."}}, {{"entity_name": "B^- \to D^{{{{*+}}}} \pi^- \pi^- \pi^- \pi^+", "entity_type": "decay_process", "justification": "This decay process is used as a control sample to compare selection efficiencies in data and simulation, contributing to the analysis of systematic uncertainties."}}, {{"entity_name": "2016 2017 and 2018, 13 TeV center of mass energy, integrated luminosity of 5.4 fb^-1.", "entity_type": "data_period", "justification": "The text specifies the data periods used for analysis, with the proton-proton collisions recorded by the LHCb experiment at a center-of-mass energy of 13 TeV."}}, {{"entity_name": "ISGW2", "entity_type": "analysis_technique", "justification": "The ISGW2 model is used to simulate $B^- \to D^{{{{**0}}}} \tau^- \overline{{{{\nu}}}}\tau$ decays and contributes to the systematic uncertainty due to model differences."}}, {{"entity_name": "LLSW", "entity_type": "analysis_technique", "justification": "LLSW is another model used to reweight the $q^2$ distribution for the $B^- \to D_1(2420)^0 \tau^- \overline{{{{\nu}}}}\tau$ component, contributing to systematic uncertainty analysis."}}, {{"entity_name": "BLR", "entity_type": "analysis_technique", "justification": "BLR is a model used similarly to LLSW to account for differences in the simulation of $q^2$ distributions, contributing to the evaluation of systematic uncertainty."}}, {{"entity_name": "Hammer", "entity_type": "analysis_technique", "justification": "The Hammer package is used for the reweighting of the $q^2$ distribution for the $B^- \to D_1(2420)^0 \tau^- \overline{{{{\nu}}}}\tau$ component, assisting in the evaluation of systematic uncertainties."}}, {{"entity_name": "bootstrap method", "entity_type": "analysis_technique", "justification": "The bootstrap method is applied to evaluate the uncertainty from limited simulated sample size as part of systematic uncertainty estimation."}}, {{"entity_name": "q^2 variable substitution", "entity_type": "analysis_technique", "justification": "Changing the $q^2$ variable to the $D^$ helicity angle is one of the methods used to assess systematic uncertainty, contributing to a 5% uncertainty."}}, {{"entity_name": "binning scheme variation", "entity_type": "analysis_technique", "justification": "Varying binning schemes is another technique used to evaluate systematic uncertainty, contributing to a 5% uncertainty."}}, {{"entity_name": "control sample method", "entity_type": "analysis_technique", "justification": "The use of a control sample method helps compare selection efficiencies in data and simulation, which is part of the systematic uncertainty evaluation."}}, {{"entity_name": "vertex detachment requirement", "entity_type": "analysis_technique", "justification": "The vertex detachment requirement plays a role in comparing data and simulation, and differences in this requirement lead to a 2% systematic uncertainty."}}, {{"entity_name": "prompt background suppression", "entity_type": "analysis_technique", "justification": "Prompt background suppression is evaluated by varying the vertex separation requirement and contributes to a 4% systematic uncertainty."}}, {{"entity_name": "relative efficiency evaluation", "entity_type": "analysis_technique", "justification": "Relative efficiency evaluation is performed to compare selection efficiencies, and this process contributes to the systematic uncertainty estimation."}}, {{"entity_name": "limited form-factor knowledge", "entity_type": "systematic_uncertainty_source", "magnitude": "3.7%", "justification": "The uncertainty due to limited form-factor knowledge is estimated by comparing different models (ISGW2, LLSW, BLR) and is assigned a 3.7% uncertainty."}}, {{"entity_name": "fixed D₁(2420)⁰ to D₂(2460)⁰ ratio assumption", "entity_type": "systematic_uncertainty_source", "magnitude": "4.4%", "justification": "The assumption of a fixed ratio between $D_1(2420)^0$ and $D_2^(2460)^0$ contributions introduces a 4.4% systematic uncertainty."}}, {{"entity_name": "limited simulated sample size", "entity_type": "systematic_uncertainty_source", "magnitude": "4.1%", "justification": "The limited simulated sample size introduces a 4.1% uncertainty, evaluated using the bootstrap method."}}, {{"entity_name": "fit variable change (q² to D helicity angle)", "entity_type": "systematic_uncertainty_source", "magnitude": "5.0%", "justification": "The change of the fit variable from $q^2$ to the $D^*$ helicity angle results in a 5.0% uncertainty."}}, {{"entity_name": "background from B⁻ → D⁰ D̅⁰ X and B⁻ → D⁰ D⁻(X)", "entity_type": "systematic_uncertainty_source", "magnitude": "3.6%", "justification": "The background from $B^- \to D^{{{{**0}}}} \overline{{{{D}}}}^0 X$ and $B^- \to D^{{{{0}}}} D^- (X)$ is estimated to contribute a 3.6% uncertainty."}}, {{"entity_name": "vertex detachment efficiency difference", "entity_type": "systematic_uncertainty_source", "magnitude": "2%", "justification": "The efficiency difference in vertex detachment, between data and simulation, contributes a 2% uncertainty."}}, {{"entity_name": "prompt background suppression modeling", "entity_type": "systematic_uncertainty_source", "magnitude": "4%", "justification": "The modeling of prompt background suppression introduces a 4% systematic uncertainty based on the comparison of different vertex separation requirements."}}, {{"entity_name": "bootstrap method", "entity_type": "systematic_uncertainty_evaluation", "justification": "The bootstrap method is applied to evaluate uncertainty arising from limited simulated sample sizes, contributing to the overall uncertainty evaluation."}}, {{"entity_name": "fit to simulated inclusive b-hadron decays", "entity_type": "systematic_uncertainty_evaluation", "justification": "The fit to simulated inclusive $b$-hadron decays is one of the methods used to assess the systematic uncertainties in the analysis."}}, {{"entity_name": "fit to zero-signal simulation", "entity_type": "systematic_uncertainty_evaluation", "justification": "The fit to zero-signal simulation helps verify the stability of the fit and potential bias in the results."}}, {{"entity_name": "varying D₁(2420)⁰ to D₂(2460)⁰ ratio", "entity_type": "systematic_uncertainty_evaluation", "justification": "Varying the ratio of $D_1(2420)^0$ to $D_2^(2460)^0$ contributions is used to assess the systematic uncertainty and contributes to a 4.4% uncertainty."}}, {{"entity_name": "varying fit variables and binning schemes", "entity_type": "systematic_uncertainty_evaluation", "justification": "Varying the fit variables and binning schemes is used to assess uncertainties in the analysis and contributes to a 5.0% uncertainty."}}, {{"entity_name": "comparison of efficiency using control sample", "entity_type": "systematic_uncertainty_evaluation", "justification": "Comparing the efficiency using a control sample helps evaluate systematic uncertainties arising from the selection criteria."}}, {{"entity_name": "comparing data/simulation for vertex separation", "entity_type": "systematic_uncertainty_evaluation", "justification": "The comparison of data and simulation for vertex separation requirements helps assess systematic uncertainties in background suppression modeling."}}
    ]

    ## INPUT ##

    Input:
    {text}

    Output:
    """



# def get_lhcb_relationship_extraction_prompt(text, entities):
#     entity_list = "\n".join([f"- {e['entity_name']} (Type: {e['entity_type']})" for e in entities])
    
#     return f"""
#     ## INSTRUCTIONS ##

#     You are an expert in high energy physics, particularly in LHCb experiments. Your task is to identify relationships between entities extracted from an LHCb paper section.

#     Below is a text section from an LHCb paper, followed by a list of entities that have been extracted from it. Identify meaningful relationships between these entities based on the text.

#     Use only these relationship types:
    
#     - USES: Entity A employs Entity B as part of its implementation or process
#     - DEPENDS_ON: Entity A requires Entity B or is calculated using Entity B
#     - IMPLEMENTED_WITH: Entity A is implemented using software/framework B
#     - EVALUATES: Entity A is used to evaluate or assess Entity B
#     - CONSTRAINS: Entity A places constraints on or restricts Entity B
#     - APPLIED_TO: Entity A is applied to or used specifically in the context of Entity B

#     Provide a list of relationship dictionaries with the following structure:
#     [
#         {{"source": "source entity name", "relation": "RELATIONSHIP_TYPE", "target": "target entity name", "evidence": "direct text evidence supporting this relationship"}},
#         ...
#     ]
#     If you cannot identify any relationships between these entities, return an empty list: []

#     Important guidelines:
#     1. Only identify relationships that are explicitly supported by the text.
#     2. Only create relationships between entities in the provided list.
#     3. Provide a brief text excerpt as evidence for each relationship. This must be a direct quotation from the text.
#     4. Do not create relationships based on general knowledge or assumptions.
#     5. Some entities might have no relationships - that's perfectly fine.
    
#     Do not include anything else in your output, including ```python ... ``` or ```json ... ```.

#     ## EXAMPLE ##
        
#     Text:
#     Unbinned maximum-likelihood mass fits are performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples, and for each data-taking period as the detector operating-conditions varied for each interval. For both the signal and normalization channels, the B^+ peak is modeled by a double-sided Crystal Ball function (DSCB), while the background is described by an exponential function. The DSCB tail parameters are obtained from simulation and are fixed in the fits to the data. The yields of signal and background, the signal peak position M_{{B^+}} and width σ_{{B^+}}, and the parameters describing the exponential backgrounds, are floated in the fits. In these fits, background is statistically subtracted by means of the sPlot technique, using the results of the fit to the B^+ mass distributions. Particle-identification efficiencies are obtained from calibration data samples. The remaining efficiencies, for both the signal and normalization decay modes, are obtained from simulated samples and calculated separately for each data-taking period. Results from the different data-taking periods are then combined using the Best Linear Unbiased Estimator (BLUE) method, which accounts for correlations among systematic effects.

#     Entities:
#     - B^+ → ψ(2S)φK^+ (Type: physics_process)
#     - B^+ → J/ψφK^+ (Type: physics_process)
#     - Run 1 (Type: data_period)
#     - Run 2 (Type: data_period)
#     - Unbinned maximum-likelihood fits (Type: statistical_method)
#     - Double-sided Crystal Ball function (Type: background_modeling_function)
#     - Exponential function (Type: background_modeling_function)
#     - sPlot technique (Type: background_subtraction_method)
#     - MC-based efficiency calculations (Type: efficiency_determination)
#     - Best Linear Unbiased Estimator (BLUE) (Type: combination_method)
#     - Branching fraction ratio (Type: measurement_quantity)
    
#     Output:
#     [
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "APPLIED_TO", "target": "B^+ → ψ(2S)φK^+", "evidence": "Unbinned maximum-likelihood mass fits are performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples"}},
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "APPLIED_TO", "target": "B^+ → J/ψφK^+", "evidence": "Unbinned maximum-likelihood mass fits are performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples"}},
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "USES", "target": "Double-sided Crystal Ball function", "evidence": "For both the signal and normalization channels, the B^+ peak is modeled by a double-sided Crystal Ball function (DSCB)"}},
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "USES", "target": "Exponential function", "evidence": "while the background is described by an exponential function"}},
#     {{"source": "sPlot technique", "relation": "DEPENDS_ON", "target": "Unbinned maximum-likelihood fits", "evidence": "In these fits, background is statistically subtracted by means of the sPlot technique, using the results of the fit to the B^+ mass distributions"}},
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "APPLIED_TO", "target": "Run 1", "evidence": "performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples, and for each data-taking period"}},
#     {{"source": "Unbinned maximum-likelihood fits", "relation": "APPLIED_TO", "target": "Run 2", "evidence": "performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples, and for each data-taking period"}},
#     {{"source": "MC-based efficiency calculations", "relation": "APPLIED_TO", "target": "B^+ → ψ(2S)φK^+", "evidence": "The remaining efficiencies, for both the signal and normalization decay modes, are obtained from simulated samples"}},
#     {{"source": "MC-based efficiency calculations", "relation": "APPLIED_TO", "target": "B^+ → J/ψφK^+", "evidence": "The remaining efficiencies, for both the signal and normalization decay modes, are obtained from simulated samples"}},
#     {{"source": "Best Linear Unbiased Estimator (BLUE)", "relation": "DEPENDS_ON", "target": "Run 1", "evidence": "Results from the different data-taking periods are then combined using the Best Linear Unbiased Estimator (BLUE) method"}},
#     {{"source": "Best Linear Unbiased Estimator (BLUE)", "relation": "DEPENDS_ON", "target": "Run 2", "evidence": "Results from the different data-taking periods are then combined using the Best Linear Unbiased Estimator (BLUE) method"}},
#     {{"source": "Branching fraction ratio", "relation": "DEPENDS_ON", "target": "Unbinned maximum-likelihood fits", "evidence": "Unbinned maximum-likelihood mass fits are performed separately for the B^+ → ψ(2S)φK^+ and B^+ → J/ψφK^+ samples"}},
#     {{"source": "Branching fraction ratio", "relation": "DEPENDS_ON", "target": "MC-based efficiency calculations", "evidence": "The remaining efficiencies, for both the signal and normalization decay modes, are obtained from simulated samples"}},
#     {{"source": "Branching fraction ratio", "relation": "DEPENDS_ON", "target": "Best Linear Unbiased Estimator (BLUE)", "evidence": "Results from the different data-taking periods are then combined using the Best Linear Unbiased Estimator (BLUE) method"}}
#     ]

#     ## INPUT ##

#     Text:
#     {text}

#     Entities:
#     {entity_list}

#     Output:
#     """


# def get_prompt(text):
#     return f"""# Systematic Uncertainty Knowledge Graph Extraction

# You are an expert physicist specialized in extracting structured knowledge about systematic uncertainties from particle physics publications. Your task is to construct a comprehensive knowledge graph by analyzing the provided text and identifying entities, properties, and relationships related to systematic uncertainties.

# # ENTITY TYPES AND PROPERTIES

# 1. Uncertainty Sources
# Categorize each uncertainty source using these precise definitions:
# - **Experimental**: Uncertainties from measurement apparatus, detector performance, calibration, or data collection. Examples: detector efficiency, resolution effects, instrument calibration.
# - **Methodological**: Uncertainties from analysis techniques, fitting procedures, binning strategies, or data processing choices. Examples: model selection, bin migration, background subtraction.
# - **Theoretical**: Uncertainties from theoretical assumptions, external input parameters, or modeling choices. Examples: branching ratios, hadronization models, polarization assumptions.

# For each uncertainty source, extract:
# - `name`: The specific uncertainty source (standardized terminology)
# - `type`: Experimental/Methodological/Theoretical
# - `description`: Concise explanation of the uncertainty's origin and nature

# 2. Measurements/Observables
# For each measurement affected by systematic uncertainties:
# - `name`: Standardized measurement name
# - `description`: What is being measured and what are the relevant kinematic ranges
# - `value`: Numerical value given in the text with units included
# - `statistical_uncertainty`: Numerical value of complete statistical uncertainty with units included
# - `systematic_uncertainty`: Numerical value of complete systematic uncertainty with units included
# - `additional_uncertainty`: Numerical value of other uncertainties with units included (e.g., polarization)

# 3. Estimation Methods
# Extract all methods used to reduce, estimate, or otherwise evaluate the impact of uncertainties:
# - `name`: Method name (use standardized terminology)
# - `description`: Detailed explanation of the method
# - `validation`: How the method was validated (if mentioned, otherwise say 'not mentioned')

# 4. Detector Components
# Extract all of the relevant detector components:
# - `name`: Component name
# - `description`: Component function and characteristics
# - `resolution`: Performance metrics (if provided)
# - `calibration_method`: How the component is calibrated

# # RELATIONSHIP TYPES

# 1. **AFFECTS** (Uncertainty Source affects Measurement/Observable)
#    - type: "AFFECTS"
#    - source: Uncertainty Sources - The name of the uncertainty source that impacts the measurement
#    - target: Measurements/Observables - The specific measurement being affected
#    - description: Detailed explanation of the mechanism by which this uncertainty influences the measurement result
#    - magnitude: The quantitative impact expressed as a percentage, absolute value, or relative contribution (if specified in the paper)

# 2. **DOMINATES** (Uncertainty Source dominates Measurement/Observable)
#    - type: "DOMINATES"
#    - source: Uncertainty Sources - The primary uncertainty source with largest impact
#    - target: Measurements/Observables - The measurement for which this uncertainty is dominant
#    - description: Explanation of why this uncertainty has the largest impact, including any relevant physics or methodological reasons
#    - magnitude: N/A (implied to be the largest contribution)

# 3. **CORRELATED_WITH** (Uncertainty Source is correlated with Uncertainty Source)
#    - type: "CORRELATED_WITH"
#    - source: Uncertainty Sources - The first uncertainty source in the correlation
#    - target: Uncertainty Sources - The second uncertainty source in the correlation
#    - description: Physical or methodological reason why these uncertainties are not independent
#    - magnitude: Correlation coefficient (numerical value if provided) or qualitative strength description (e.g., "strong", "weak")

# 4. **ESTIMATED_WITH** (Uncertainty Source is estimated with Estimation Method)
#    - type: "ESTIMATED_WITH"
#    - source: Uncertainty Sources - The uncertainty being evaluated
#    - target: Estimation Methods - The specific method used to quantify the uncertainty
#    - description: Step-by-step explanation of how the method was applied to this specific uncertainty
#    - magnitude: The resulting precision or uncertainty on the uncertainty (e.g., "±0.5%")

# 5. **CONTRIBUTES_TO** (Detector Component contributes to Uncertainty Source)
#    - type: "CONTRIBUTES_TO"
#    - source: Detector Components - The specific detector element or subsystem
#    - target: Uncertainty Sources - The uncertainty that arises from this component
#    - description: The physical mechanism by which this detector component introduces or affects the uncertainty
#    - magnitude: The relative or absolute contribution to the total uncertainty (e.g., "30% of total detector uncertainty" or "0.5% absolute")

# 6. **REDUCED_BY** (Uncertainty Source is reduced by Estimation Method)
#    - type: "REDUCED_BY"
#    - source: Uncertainty Sources - The uncertainty being mitigated
#    - target: Estimation Methods - The method used to reduce the uncertainty
#    - description: Specific details of how the method was implemented to minimize this uncertainty
#    - magnitude: Quantitative reduction factor (e.g., "reduced by factor of 2") or before/after values (e.g., "from 3.2% to 1.1%")

# # OUTPUT FORMAT

# Provide your extraction as a Python dictionary with this structure:

# {{
# "Uncertainty Sources": [
# {{
# "name": str,
# "type": str,
# "description": str
# }},
# ...
# ],
# "Measurements/Observables": [
# {{
# "name": str,
# "description": str,
# "value": str,
# "statistical_uncertainty": str,
# "systematic_uncertainty": str,
# "additional_uncertainty": str
# }},
# ...
# ],
# "Estimation Methods": [
# {{
# "name": str,
# "description": str,
# "validation": str
# }},
# ...
# ],
# "Detector Components": [
# {{
# "name": str,
# "description": str,
# "resolution": str,
# "calibration_method": str
# }},
# ...
# ],
# "Relationships": [
# {{
# "type": str,
# "source": str,
# "target": str,
# "description": str,
# "magnitude": str
# }},
# ...
# ]
# }}

# # EXTRACTION PRINCIPLES

# 1. **Comprehensive Coverage**: Extract ALL entities and relationships, even if only implied. When in doubt, include it.

# 2. **Precision**: Maintain exact numerical values with proper units and distinguish between absolute/relative uncertainties.

# 3. **Completeness**: Capture the full uncertainty propagation chain, from detector components through to final measurements.

# 4. **Rich Connectivity**: Identify all relationships between entities, especially correlation structures critical for error propagation.

# 5. **Context Preservation**: Note when uncertainties apply only to specific kinematic regions or conditions.

# Focus particularly on:
# - Less obvious estimation methods embedded in technical descriptions
# - Complete uncertainty breakdowns with proper mathematical notation
# - Complex correlation structures between different uncertainty sources
# - Detector-specific contributions to systematic uncertainties
# - Temporal or kinematic dependence of uncertainties

# IMPORTANT: ensure that every relationship's "source" and "target" fields actually exist as entities

# # TEXT TO ANALYZE

# {text}
# """