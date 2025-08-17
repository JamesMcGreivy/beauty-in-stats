"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""


def get_abstract_extraction_prompt(abstract):

    default = {
        "run": "<UNK>",
        "strategy": "<UNK>",
        "decay": [],
        "observable": [],
    }

    prompt = f"""
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
    return prompt, default



def get_graph_extraction_prompt(text, observables):
    
    default = {

    }    
    
    prompt = f"""You are an expert physicist specialized in extracting entities and relationships for a knowledge graph of systematic uncertainties in particle physics publications. Your task is to read the provided paper and perform comprehensive knowledge graph extraction.

# ROLE & EXPERTISE
You have deep expertise in:
- LHCb experimental analysis methodologies
- Systematic uncertainty quantification and categorization
- Statistical vs systematic error sources
- Detector effects and reconstruction algorithms
- Physics modeling and theoretical uncertainties

# ENTITY TYPES AND PROPERTIES

## 1. "observable" (Pre-extracted)
The relevant observables have already been extracted and will be provided to you.

## 2. "uncertainty_source"
Identify ALL uncertainty sources affecting the measured observables, including:
- Statistical uncertainties from limited sample sizes
- Systematic uncertainties from analysis choices and external inputs
- Numerical values of physics constants or observables from external analyses used as inputs

### Categories (EXACTLY one per source):
- "statistical": Random fluctuations, sampling limitations, finite data/MC statistics, statistical precision of control samples
- "internal": Analysis-specific choices (reconstruction, efficiency modeling, background treatment, fitting procedures) - typically LHCb-controllable
- "external": External inputs (theoretical calculations, PDG values, previous measurements) - cannot be improved by changing this analysis

### Quality Standards for Descriptions:
✅ INCLUDE: Physical mechanism, why it's systematic, quantitative details when available
✅ EXPLAIN: How it affects the measurement, what causes the uncertainty
❌ AVOID: Generic statements, circular definitions, pure name restatements
❌ NEVER: "Uncertainty in X" without explaining the physical origin

### "relationship_affects":
Links uncertainty_source → observable with:
- "magnitude": Exact numerical contribution as stated in paper (preserve units: "4%", "0.02 MeV", "2.1×10⁻³")
- "ranking": Relative importance BY MAGNITUDE within each observable (1=largest, 2=second largest, etc.)
  - If magnitude unclear or equal → assign largest available ranking number
  - Rank independently for each observable

## 3. "method" 
Identify techniques used to QUANTIFY/EVALUATE systematic uncertainties.

### Focus on TRANSFERABLE TECHNIQUES:
✅ EXTRACT: Evaluation procedures applicable to similar systematics in other analyses
✅ EXTRACT: Distinct methodological approaches with clear procedural steps
❌ SKIP: One-off corrections, simple parameter variations, analysis-specific fixes
❌ SKIP: Techniques that CAUSE uncertainty (those become uncertainty_sources)

### "relationship_estimates":
Links method → uncertainty_source when that method quantifies the uncertainty magnitude.

# EXTRACTION PROCESS

## Phase 1: Strategic Document Scan
Priority Sections (scan in order):
1. Systematic uncertainty summary tables (highest priority)
2. Detailed systematic sections 
3. Method/procedure descriptions
4. Abstract/conclusions for final values

## Phase 2: Systematic Uncertainty Extraction
For each uncertainty source found:

1. Name: Use EXACT paper terminology (preserve capitalization, abbreviations)
2. Description: Write 2-3 sentences explaining:
   - Physical/technical origin of the uncertainty
   - Why it's systematic (not random)
   - How it impacts the measurement
   - Key quantitative aspects when available
3. Type: Apply strict categorization rules
4. Relationships: Link to ALL affected observables with exact magnitudes

### Type Classification Rules:
- Statistical: Limited sample size, random fluctuations, MC statistics, finite control sample sizes, statistical precision of calibrations
- Internal: Detector modeling, reconstruction choices, analysis procedures, fitting methods, selection criteria
- External: PDG constants, previous measurements, theoretical predictions, world average values

## Phase 3: Method Extraction
For each evaluation technique:

1. Name: Descriptive, technique-focused title
2. Description: Explain the methodological approach:
   - What systematic effect it measures
   - How it quantifies uncertainty magnitude
   - Key procedural steps for replication
3. Relationships: Link to ALL uncertainty sources it evaluates

# CRITICAL EXTRACTION RULES

## Completeness Requirements:
- Extract ALL uncertainties mentioned, including negligible ones
- Create separate entries for mode-dependent uncertainties (different magnitudes per observable)
- Include cancelled uncertainties if explicitly discussed
- If magnitude not specified → use "not specified"

## Precision Requirements:
- Match observable names EXACTLY from provided list
- Preserve ALL original magnitude notation (units, scientific notation, ranges)
- Maintain exact paper terminology for uncertainty names
- Use precise technical language in descriptions

## Quality Assurance:
- Every uncertainty_source must have meaningful description explaining physics
- Every method must describe a replicable evaluation technique
- Rankings must be consistent with magnitude ordering within each observable
- No circular definitions or generic placeholder text

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
            "description": "Uncertainty in particle identification efficiency for distinguishing between different particle types (kaons, pions, muons). Evaluated using calibration samples in data with loose selection requirements that largely cancel between signal and normalization modes, resulting in small residual systematic uncertainty.",
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

    return prompt, default



def get_entity_canonicalization_prompt(similar_entities, instructions):
    """This prompt is used by the LLM in order to pass final judgement on how to canonicalize potentially similar entities"""
    
    default = []
    
    # Create enumerated entity descriptions for the LLM
    entities_str = ""
    for i, entity in enumerate(similar_entities):
        entity_attrs = "{" + ", ".join([f'"{k}": "{v}"' for k, v in entity.attributes.items()]) + "}"
        entities_str += f"\n{i+1}. {{\n"
        entities_str += f'   "name": "{entity.name}",\n'
        entities_str += f'   "type": "{entity.type}",\n'
        entities_str += f'   "description": "{entity.description}",\n'
        entities_str += f'   "attributes": {entity_attrs}\n}}\n'

    prompt = f"""You are an expert particle physicist in the LHCb collaboration.

# TASK
You have been given a cluster of {len(similar_entities)} potentially similar entities that were identified by an algorithmic clustering method. Your job is to make the final decision about which entities (if any) should actually be merged together.

# INSTRUCTIONS
- Carefully examine each entity's name, description, and attributes
- Decide which entities represent the exact same concepts and ONLY merge those.
- You can merge all entities into one, create multiple merged groups, or decide not to merge any entities at all.

# CRITICAL RULES
- NEVER change attribute names - keep them exactly as they appear. Include ALL attributes
- For "arxiv_id": combine, de-duplicate, and sort numerically as a comma-separated string.
- For other attributes: choose most representative value or create appropriate ranges/combinations
- "source_indices" is an integer list of the input indices for input entities that were merged to create the "merged_entity"  

# MERGING GUIDELINES
{instructions}

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

    return prompt, default


