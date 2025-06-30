"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

from py2neo import Graph
import regex as re

from SystematicsGraph import llm, embedding_model

class GraphQueryAgent:
    """
    Agent that converts natural language queries about particle physics systematic uncertainties
    into Cypher queries, executes them against a Neo4j knowledge graph, and synthesizes the results.
    """
    
    def __init__(self, uri, username, password):
        self.graph = Graph(uri, auth=(username, password))
        
    def get_cypher_query_prompt(self, query):
        return f"""
You are an expert in particle physics and database queries, specializing in systematic uncertainties in LHCb analyses. Your task is to translate natural language questions about systematic uncertainties into Cypher queries for our Neo4j knowledge graph.

# Overview
You translate natural language questions about systematic uncertainties in LHCb analyses into Cypher queries. When asked about systematic uncertainties, observable measurements, or analysis methods, you construct precise Cypher queries to retrieve the relevant information from our knowledge graph.

# Graph Schema

Our neo4j knowledge graph contains the following node types and relationships:

## Node Types & Properties

1. (n:paper)
name: str # The ArXiV ID of the paper
description: str # The full abstract of the paper
embedding: list[float] # The sentence embedding vector for the full abstract of the paper
run: str
- "run1": Data collected during 2011-2012 at center-of-mass energies of 7-8 TeV
- "run2": Data collected during 2015-2018 at a center-of-mass energy of 13 TeV
- "run1,run2": Combined dataset from both Run 1 and Run 2 periods (2011-2018)
- "run3": Data collected during 2022-2025 at a center-of-mass energy of 13.6 TeV
- "ift": Any paper which does not use p-p as its production method.
strategy: str
- "angular_analysis": Studies that examine angular distributions or asymmetries in particle decays. These include forward-backward asymmetries, angular coefficients, or observables related to Wilson coefficients. These analyses typically perform multi-dimensional fits to angular variables to extract physics parameters and test Standard Model predictions.
- "amplitude_analysis": Studies focused on decay amplitude structures, including Dalitz plot analyses, determination of resonance properties, spin-parity assignments, or relative phase measurements. These analyses model interfering amplitudes and extract quantities such as fit fractions, strong phases, or resonance parameters. 
- "search": Analyses aimed at discovering previously unobserved states, processes, or symmetry-breaking effects. This includes first observations of rare decay modes, searches for new particles, or forbidden processes. Often results in significance measurements for new signals or limit setting at confidence levels (typically 90% or 95%) when no significant signal is observed. 
- "other": Any analysis that does not fall into the above three categories. This includes analyses primarily focused on precision measurements of established quantities, such as branching fraction determinations, lifetime measurements, production cross-sections, mass measurements.
description: str # The raw abstract of the paper     

2. (n:observable)
name: str  # The raw name of the observable
arxiv_id: str # The ArXiV ID of the relevant paper
type: str
- "branching_fraction": Explicitly called branching fractions (e.g., B(B^- -> D^0\\pi^-\\pi^+\\pi^-))
- "branching_ratio": Ratio of branching fractions (e.g., B(B_s^0-bar -> D_s^+\\pi^-\\pi^+\\pi^-) / B(B_s^0-bar -> D_s^+\\pi^-)) where both numerator and denominator were measured in this analysis.
- "physical_constant": Fundamental physics parameters (e.g., CP-violating phase γ, CKM angles, wilson coefficients, particle lifetimes)
- "angular_observable": From angular analyses (e.g., asymmetries in angular distributions, polarization fractions, helicity amplitudes)
- "functional_dependence": Measured as functions of kinematic variables (e.g., p_T distributions, differential cross sections vs rapidity, form factors vs q^2)

3. (n:decay)
name: str # The raw name of the decay
arxiv_id: str # The ArXiV ID of the relevant paper
production: str # The production method (what was being smashed together) of the decay
- p-p
- Pb-Pb
- p-Pb
- Xe-Xe
- O-O
- Pb-Ar
- p-O
parent: str # The initial state particle in the decay
children: str # The final state particles in the decay

IMPORTANT: When querying for parent or children particles in the decay you MUST phrase particles exactly as in the list below, otherwise the query will fail.
- Always use exact particle names from the provided list
- For antiparticles, use the tilde notation (e.g., K~0 for anti-K0)
- For excited states, include parentheses (e.g., D(s)+)
particles:
d, d~, u, u~, s, s~, c, c~, b, b~, t, t~, e-, e+, nu(e), nu(e)~, mu-, mu+, nu(mu), nu(mu)~, tau-, tau+, nu(tau), nu(tau)~, g, gamma, Z0, W+, W-, H0, R0, R~0, X(u)0, X(u)+, X(u)-, pi0, rho0, a0, K(L)0, B(L)0, pi+, pi-, rho+, rho-, a+, a-, eta, omega, f, K(S)0, K0, K~0, K*0, K*~0, K+, K-, K*+, K*-, eta', phi, f', B(sL)0, D+, D-, D*+, D*-, D0, D~0, D*0, D*~0, D(s)+, D(s)-, D(s)*+, D(s)*-, D(s2)*+, D(s2)*-, eta(c)(1S), J/psi(1S), chi(c2)(1P), B(H)0, B0, B~0, B*0, B*~0, B+, B-, B*+, B*-, B(sH)0, B(s)0, B(s)~0, B(s)*0, B(s)*~0, B(s2)*0, B(s2)*~0, B(c)+, B(c)-, B(c)*+, B(c)*-, B(c2)*+, B(c2)*-, eta(b)(1S), Upsilon(1S), chi(b2)(1P), Upsilon(1D), (dd), (dd)~, Delta-, Delta~+, Delta0, Delta~0, N0, N~0, (ud), (ud)~, n, n~, Delta+, Delta~-, N+, N~-, (uu), (uu)~, p, p~, Delta++, Delta~--, (sd), (sd)~, Sigma-, Sigma~+, Lambda, Lambda~, (su), (su)~, Sigma0, Sigma~0, Sigma+, Sigma~-, (ss), (ss)~, Xi-, Xi~+, Xi0, Xi~0, Omega-, Omega~+, (cd), (cd)~, Sigma(c)0, Sigma(c)~0, Lambda(c)+, Lambda(c)~-, Xi(c)0, Xi(c)~0, (cu), (cu)~, Sigma(c)+, Sigma(c)~-, Sigma(c)++, Sigma(c)~--, Xi(c)+, Xi(c)~-, (cs), (cs)~, Xi(c)'0, Xi(c)'~0, Xi(c)'+, Xi(c)'~-, Omega(c)0, Omega(c)~0, (cc), (cc)~, Xi(cc)+, Xi(cc)~-, Xi(cc)*+, Xi(cc)*~-, Xi(cc)++, Xi(cc)~--, Xi(cc)*++, Xi(cc)*~--, Omega(cc)+, Omega(cc)~-, Omega(cc)*+, Omega(cc)*~-, Omega(ccc)*++, Omega(ccc)*~--, (bd), (bd)~, Sigma(b)-, Sigma(b)~+, Sigma(b)*-, Sigma(b)*~+, Lambda(b)0, Lambda(b)~0, Xi(b)-, Xi(b)~+, Xi(bc)0, Xi(bc)~0, (bu), (bu)~, Sigma(b)0, Sigma(b)~0, Sigma(b)*0, Sigma(b)*~0, Sigma(b)+, Sigma(b)~-, Sigma(b)*+, Sigma(b)*~-, Xi(b)0, Xi(b)~0, Xi(bc)+, Xi(bc)~-, (bs), (bs)~, Xi(b)'-, Xi(b)'~+, Xi(b)*-, Xi(b)*~+, Xi(b)'0, Xi(b)'~0, Xi(b)*0, Xi(b)*~0, Omega(b)-, Omega(b)~+, Omega(b)*-, Omega(b)*~+, Omega(bc)0, Omega(bc)~0, (bc), (bc)~, Xi(bc)'0, Xi(bc)'~0, Xi(bc)*0, Xi(bc)*~0, Xi(bc)'+, Xi(bc)'~-, Xi(bc)*+, Xi(bc)*~-, Omega(bc)'0, Omega(bc)'~0, Omega(bc)*0, Omega(bc)*~0, Omega(bcc)+, Omega(bcc)~-, Omega(bcc)*+, Omega(bcc)*~-, (bb), (bb)~, Xi(bb)-, Xi(bb)~+, Xi(bb)*-, Xi(bb)*~+, Xi(bb)0, Xi(bb)~0, Xi(bb)*0, Xi(bb)*~0, Omega(bb)-, Omega(bb)~+, Omega(bb)*-, Omega(bb)*~+, Omega(bbc)0, Omega(bbc)~0, Omega(bbc)*0, Omega(bbc)*~0, Omega(bbb)-, Omega(bbb)~+, vpho, b0, b+, b-, h, D(s0)*+, D(s0)*-, D(s1)+, D(s1)-, chi(c0)(1P), h(c)(1P), B(s0)*0, B(s0)*~0, B(s1)(L)0, B(s1)(L)~0, B(c0)*+, B(c0)*-, B(c1)(L)+, B(c1)(L)-, chi(b0)(1P), h(b)(1P), eta(b2)(1D), D(H)+, D(H)-, chi(c1)(1P), B(H)~0, B(H)+, B(H)-, B(s1)(H)0, B(s1)(H)~0, B(c1)(H)+, B(c1)(H)-, chi(b1)(1P), X(sd), X(sd)~, X(su), X(su)~, X(ss), X(ss)~, psi, Upsilon_1(1D), D(2S)+, D(2S)-, D(2S)0, D(2S)~0, eta(c)(2S), psi(2S), chi(c2), eta(b)(2S), Upsilon(2S), chi(b2)(2P), Upsilon(2D), chi(b0)(2P), h(b)(2P), eta(b2)(2D), chi(b1)(2P), eta(b)(3S), Upsilon(3S), chi(b2)(3P), chi_b0(3P), h_b(3P), chi_b1(3P), Upsilon(4S), pi(tc)0, rho(tc)0, pi(tc)+, pi(tc)-, rho(tc)+, rho(tc)-, pi(tc)'0, omega(tc), nu(e)*0, nu(e)*~0, pi, f(J), Upsilon, rho, Z+, Z-, X

5. (n:uncertainty_source)
name: str # The name of the uncertainty source
arxiv_id: str # The ArXiV ID of the relevant paper
description: str # A description of the uncertainty source
embedding: list[float] # The sentence embedding vector for the description of the uncertainty source
type: str
- "statistical": From data variability due to random fluctuations or sampling limitations
- "internal": From analysis choices (reconstruction, efficiency modeling, background treatment) - usually LHCb-specific
- "external": From external inputs (theoretical calculations, previous measurements) - cannot be improved by analysis changes

6. (n:method)
name: str # The name of the method
arxiv_id: str # The ArXiV ID of the relevant paper
description: str # A description of the method
embedding: list[float] # The sentence embedding vector for the description of the method

## Relationship Types & properties

(n:uncertainty_source)-[r:affects]-(m:observable)
Connects uncertainty sources to observables, indicating how a specific systematic uncertainty impacts the measurement precision
ranking: int # The importance rank of (n:uncertainty_source) relative to all other uncertainty sources affecting (m:observable) (1 is most important, 2 is less important, etc).
magnitude: str # The numerical contribution (percentage, absolute value, etc) exactly as given in the paper to the observable's total uncertainty budget. You should NOT query on this.

(n:paper)-[r:determines]-(m:observable)
Links papers to the observables they measure, showing which analysis determined which physical quantities

(n:method)-[r:estimates]-(m:uncertainty_source)
Shows which methods were used to quantify the impact of specific systematic uncertainties

(n:observable)-[r:measured_with]-(m:decay)
Connects observables to the decay channels used in their measurement

# General Principles

1. The results of the cypher queries will be synthesized by an llm in combination with the natural language queries in order to answer the question. Thus, you should always aim to return more information (descriptions, etc) in order to provide context to the results of the cypher query. 

2. Always limit the number of results returned to no more than 20, as not to overload the context of the synthesization llm.

3. Always check the content of a string using o.type CONTAINS "type". DO NOT use o.type = "type" for string comparison.

4. You are FORBIDDEN from querying directly on the n.description attribute of a node. Instead, you MUST use the n.embedding attribute alongside the function vector.similarity.cosine(embedding, embedding) where the query embedding is wrapped in $("query string..."). You should order results with larger similarity being more relevant.
Template: 
- "WITH vector.similarity.cosine(n.embedding, $("describe what you are looking for inside of the description...")) AS similarity"

5. Always try to order the results according to some metric of "best" or "most common". 
Examples: 
- If appropriate given the query, when returning systematic uncertainties you might order them by their average importance ranking:
(...rest of query...) MATCH (u:uncertainty_source)-[r]-(o:observable) WITH u.name as name, u.description as description, AVG(DISTINCT r.ranking) as importance RETURN name, importance ORDER BY importance ASC LIMIT 20
- You might also order them by how many observables they affect:
(...rest of query...) MATCH (u:uncertainty_source)-[r]-(o:observable) WITH u.name as name, u.description as description, COUNT(DISTINCT r) as frequency RETURN name, frequency ORDER BY frequency DESC LIMIT 20

6. Without losing relational information necessary to answer the query, always aggregate as many of the results as you can using functions such as COLLECT(DISTINCT ...).
Example:
- If a query asks about methods to handle an uncertainty under some conditions, you would not want to clutter the results by returning the same method on multiple rows. To fix this:
(...rest of query given conditions...) MATCH (u:uncertainty_source)-[]-(m:method) RETURN u.name AS name, u.description AS description, COLLECT(DISTINCT [m.name, m.description]) AS methods LIMIT 20
- If a query asked about methods to handle dominant uncertainties, it would make sense to return one uncertainty per row with multiple methods on the same row:
(...rest of query given conditions...) MATCH (o:observable)-[r:affects]-(u:uncertainty_source)-[]-(m:method) RETURN u.name as name, u.description as description, COLLECT(DISTINCT [m.name, m.description]) as methods, AVG(DISTINCT r.ranking) as avg_ranking ORDER BY avg_ranking ASC LIMIT 10

7. Always return the arxiv_id of relevant entities in order to provide citations for the user to look in for more detailed information answering their query. You MUST return arxiv_id in every single query without fail.

8. Always return n.description attributes when possible. This will give more context to the user.

# EXAMPLES

## Example 1

Query: How can I correct for particle misidentification uncertainty when studying decays with muons in the final state

Cypher:
{{
"explanation": "This Cypher query identifies methods that address particle misidentification uncertainties in decays containing muons by first finding all decays with μ+ or μ- particles, then traversing the graph to find related observables, uncertainty sources, and correction methods. It ranks the results by semantic similarity between uncertainty sources and particle identification keywords, returning the top 20 correction methods along with their descriptions, associated uncertainty sources, and relevant decay processes."
"cypher": "MATCH (d:decay) WHERE d.children CONTAINS "mu-" OR d.children CONTAINS "mu+" WITH d MATCH (d)-[]-(o:observable)-[]-(u:uncertainty_source)-[]-(m:method) WITH d, o, u, m, vector.similarity.cosine(u.embedding, $("particle misidentification, identification, PID")) AS similarity RETURN m.name AS method, m.description AS method_description, m.arxiv_id AS arxiv_id, COLLECT(DISTINCT [u.name, u.description]) AS uncertainty_sources, COLLECT(DISTINCT d.name) AS decays, similarity ORDER BY similarity DESC LIMIT 20", 
}}

## Example 2

Query: What is the most common source of uncertainties when studying decays with pions in the final state

Cypher:
{{
"explanation": "This Cypher query identifies uncertainty sources that affect measurements involving pion-containing decays by first finding all relevant decay processes, then connecting them to their associated observables and uncertainty sources. For each (uncertainty, observable) pair, it counts how many unique decay processes are involved and collects their names, returning the top 20 uncertainty sources ranked by the number of distinct pion decays they affect.",
"cypher": "MATCH (d:decay) WHERE d.children CONTAINS "pi" WITH d MATCH (d)-[]-(o:observable)-[]-(u:uncertainty_source) WITH u, o, COUNT(DISTINCT d) AS frequency, COLLECT(DISTINCT d.name) as decays RETURN u.name as name, u.description as description, u.type as type, u.arxiv_id as arxiv_id, frequency, decays ORDER BY frequency DESC LIMIT 20"
}}

## Example 3

Query: What are the top uncertainties I should consider when studying CP-violation with B-meson decays? What methods can I use to correct for them?

Cypher:
{{
"explanation": "This Cypher query identifies key uncertainty sources affecting CP violation measurements in B-meson decays by first finding physical constant observables from B-meson decay processes, sorted by semantic similarity to 'CP violation'. For each matching uncertainty source, it collects the associated correction methods, relevant observables, and decay channels, providing a comprehensive view of systematic uncertainties impacting CP violation studies in B-mesons and how they can be addressed.",
"cypher": "MATCH (o:observable)-[]-(d:decay) WHERE d.parent CONTAINS "B" AND o.type CONTAINS "physical_constant" ORDER BY vector.similarity.cosine(o.embedding, $("CP violation")) DESC WITH o, d MATCH (m:method)-[]-(u:uncertainty_source)-[r:affects]-(o) RETURN u.name AS uncertainty_source, u.type AS uncertainty_type, u.description AS uncertainty_description, u.arxiv_id as arxiv_id, COLLECT(DISTINCT [o.name, o.type]) AS observables, COLLECT(DISTINCT [m.name, m.description]) AS methods, COLLECT(DISTINCT d.name) AS22 decays LIMIT 20"
}}

# Input 

Query: {query}

Cypher:
"""

    def process_cypher_query(self, cypher_query):
        pattern = r'\$\(["\'](.*?)["\']\)'
        
        def replacement_function(match):
            text = match.group(1)
            embedding = list(embedding_model.encode(text))
            return str(embedding)
        
        processed_cypher = re.sub(pattern, replacement_function, cypher_query)
        return processed_cypher
    
    def cursor_to_formatted_results(self, cursor):
        if not cursor:
            return {"columns": [], "results": []}
        
        # Extract column names and data from cursor
        columns = cursor.keys()
        results = []
        
        # Convert to a list of dictionaries for easier processing
        for record in cursor:
            row = {}
            for col in columns:
                row[col] = record[col]
            results.append(row)
            
        return {
            "columns": columns,
            "results": results
        }

    def generate_synthesis_prompt(self, query, cypher, results):
        import json
        # Convert results to a readable format for the LLM
        results_str = json.dumps(results, indent=2)
        
        return f"""
You are an expert in particle physics, particularly in LHCb analyses and systematic uncertainties. A researcher has run a database query about systematic uncertainties in particle physics experiments and needs help understanding the results. You will be given the original query, the cypher translation of the query, as well as the cypher query results. 

Your response should:
1. Focus exclusively on information present in the query results - do not introduce external knowledge or make assumptions beyond what's explicitly shown in the results.
2. Identify the most important pieces of the query result if any ranking or frequency information is included.
3. Use a natural language and tone while maintaining scientific accuracy.
4. If certain parts of the query are unanswered by the results, acknowledge this rather than filling gaps with assumed information.
6. Cite the sources in your answer whenever possible using the arxiv ids returned from the query. For example: "...trigger efficiency corrections are estimated via the TISTOS method [2307.09427, 2404.19510, 2404.03375], a technique used to..."

Respond with a brief summary of the key points which answer the query (including in-text arxiv_id citations). If relevant, you may also include a discussion of any parts of the query that are unanswered by the results. Do not respond with anything else. Your answer should not be longer than a couple of short paragraphs. You should incorporate the description fields to build context, rather than just listing off query results without context.

Original Query:
{query}

Graph Database Query (Cypher):
{cypher}

Query Results:
{results_str}

Answer:
"""

    def query(self, query_text):
        # Step 1: Generate Cypher query from natural language
        cypher_prompt = self.get_cypher_query_prompt(query_text)
        llm_response = llm.complete(cypher_prompt).text
        
        # Extract cypher query from the response
        try:
            response_dict = eval(llm_response)
            cypher_query = response_dict["cypher"]
            explanation = response_dict["explanation"]
        except:
            # Fallback if parsing fails
            error_msg = "Failed to parse LLM response to extract Cypher query."
            return {
                "cypher_query": None,
                "raw_results": None,
                "explanation": error_msg,
                "synthesized_answer": error_msg
            }
        
        # Step 2: Process and execute the Cypher query
        processed_cypher = self.process_cypher_query(cypher_query)
        
        try:
            cursor = self.graph.query(processed_cypher)
            formatted_results = self.cursor_to_formatted_results(cursor)
        except Exception as e:
            error_msg = f"Error executing Cypher query: {str(e)}"
            return {
                "cypher_query": processed_cypher,
                "raw_results": None,
                "explanation": explanation,
                "synthesized_answer": error_msg
            }
        
        # Step 3: Synthesize results with LLM
        synthesis_prompt = self.generate_synthesis_prompt(query_text, cypher_query, formatted_results)
        synthesized_answer = llm.complete(synthesis_prompt).text
        
        # Return comprehensive results
        return {
            "cypher_query": cypher_query,
            "raw_results": formatted_results,
            "explanation": explanation,
            "synthesized_answer": synthesized_answer
        }