import gradio as gr
from abstract_analysis import OllamaClassifier, ClassificationConfig
import random
import os
import glob
import json

# Initialize classifier
classifier = OllamaClassifier(model_name="mistral")

# Store prompts in a persistent way
class PromptManager:
    def __init__(self):
        self.system_prompt = ClassificationConfig.SYSTEM_PROMPT
        self.user_prompt_template = ClassificationConfig.get_prompt_template()
        
        # Try to load saved prompts if they exist
        try:
            with open('saved_prompts.json', 'r') as f:
                saved = json.load(f)
                self.system_prompt = saved.get('system_prompt', self.system_prompt)
                self.user_prompt_template = saved.get('user_prompt', self.user_prompt_template)
        except FileNotFoundError:
            pass
    
    def save_prompts(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt
        with open('saved_prompts.json', 'w') as f:
            json.dump({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt
            }, f)
        return "Prompts saved successfully!"

prompt_manager = PromptManager()

def reset_to_default():
    """Reset prompts to their default values from ClassificationConfig"""
    try:
        os.remove('saved_prompts.json')
    except FileNotFoundError:
        pass
        
    return (
        ClassificationConfig.SYSTEM_PROMPT,
        ClassificationConfig.get_prompt_template(),
        "Prompts reset to original defaults!"
    )

def load_random_abstract():
    """Load a random abstract from the specified directory"""
    abstract_dir = "/work/submit/blaised/beauty-in-stats/data/abstracts"
    abstract_files = glob.glob(os.path.join(abstract_dir, "*.tex"))
    
    if not abstract_files:
        return "No abstract files found in directory.", ""
    
    random_file = random.choice(abstract_files)
    try:
        with open(random_file, 'r') as f:
            content = f.read()
        filename = os.path.basename(random_file).replace('.tex', '')
        return content, f"## arXiv ID: `{filename}`"
    except Exception as e:
        return f"Error loading abstract: {str(e)}", ""

def classify_abstract(abstract: str, system_prompt: str, user_prompt: str):
    """Classify the given abstract using current prompts"""
    temp_classifier = OllamaClassifier(model_name="mistral")
    temp_classifier.config.SYSTEM_PROMPT = system_prompt
    temp_classifier.prompt_template = user_prompt
    
    result = temp_classifier.classify_abstract(abstract)
    if result:
        try:
            # Split into lines
            lines = result.strip().split('\n')
            
            # Find the focus, run, and strategy from the response text
            focus = run = strategy = "Error parsing response"
            explanation = ""
            
            # Track whether we're past the classification block
            passed_classification = False
            explanation_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if line == '}}}}':  # End of classification block
                    passed_classification = True
                    continue
                    
                if passed_classification:  # Collect all lines after classification as explanation
                    explanation_lines.append(line)
                elif '"focus":' in line:
                    focus = line.split('"focus":', 1)[1].strip().strip('",')
                elif '"run":' in line:
                    run = line.split('"run":', 1)[1].strip().strip('",')
                elif '"strategy":' in line:
                    strategy = line.split('"strategy":', 1)[1].strip().strip('",')
            
            # Join all explanation lines
            explanation = ' '.join(explanation_lines)
            
            return focus, run, strategy, explanation
            
        except Exception as e:
            print(f"Debug - Error parsing: {str(e)}")
            print(f"Debug - Raw result: {result}")
            return str(e), str(e), str(e), str(e)
    
    return "No response", "No response", "No response", "No response from classifier"

def save_prompts(system_prompt: str, user_prompt: str):
    """Save the current prompts"""
    return prompt_manager.save_prompts(system_prompt, user_prompt)

# Create the interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Main Interface"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("""
                    # AnalysisCopilot: LHCb Abstract Classifier
                    Using the `mistral` open-source LLM

                    ## Instructions
                    1. Enter your LHCb paper abstract or load a random one
                    2. Click 'Classify Abstract' to analyze
                    3. Review the classification and explanation below
                    """)
                    arxiv_id = gr.Markdown("")
                    
                    with gr.Row():
                        abstract_input = gr.Textbox(
                            label="Enter abstract text",
                            placeholder="Paste your LHCb paper abstract here...",
                            lines=8
                        )
                    
                    with gr.Row():
                        random_btn = gr.Button("Load Random Abstract")
                        classify_btn = gr.Button("Classify Abstract", variant="primary")
                    
                    with gr.Row():
                        physics_output = gr.Textbox(label="Physics Focus")
                        run_output = gr.Textbox(label="Run Period")
                        strategy_output = gr.Textbox(label="Analysis Strategy")
                    
                    # Add the new explanation output box
                    explanation_output = gr.Textbox(
                        label="Classification Reasoning",
                        lines=3,
                        show_label=True
                    )

                with gr.Column(scale=1):
                    gr.Markdown("""# Classification schema information
### Classification Categories

#### Physics Focus
- `b->sll`: Loop-suppressed beauty decays with leptons
- `c->sll`: Loop-suppressed charm decays with leptons
- `radiative_decays`: Heavy-flavor decays with photons
- `spectroscopy`: Exotic multi-quark states
- `semileptonic`: B→D(*)ℓν decays with neutrinos
- `lifetime`: Heavy-flavored hadron lifetimes
- `electoweak`: W, Z, and Higgs physics
- `dark_sector`: Dark Matter/Dark-Sector searches
- `forbidden_decays`: Baryon/lepton number violation
- `jet_physics`: Jet production and QCD
- `heavy_ions`: Non-pp collision analyses
- `charm`: Charm physics and CP violation
- `CPV`: Beauty CP violation measurements

#### Run Period
- `Run1`: 2011-2012 (3 fb⁻¹, 7-8 TeV)
- `Run2`: 2015-2018 (5.4/6 fb⁻¹, 13 TeV)
- `Run1+2`: Combined datasets (~9 fb⁻¹, 13 TeV)
- `Run3`: 2022-2025 (13.6 TeV, expected ~15 fb⁻¹)

#### Analysis Strategy
- `angular_analysis`: Angular distributions and WC extraction
- `amplitude_analysis`: Decay amplitudes and resonances
- `search`: New state searches with limits
- `direct_measurement`: Observable measurements (BF, mass, etc.)
""")
        
        with gr.Tab("Prompt Editor"):
            with gr.Column():
                system_prompt_input = gr.Textbox(
                    label="System Prompt",
                    value=prompt_manager.system_prompt,
                    lines=15
                )
                user_prompt_input = gr.Textbox(
                    label="User Prompt Template",
                    value=prompt_manager.user_prompt_template,
                    lines=10
                )
                with gr.Row():
                    save_btn = gr.Button("Save Prompts", variant="primary")
                    reset_btn = gr.Button("Reset to Default", variant="secondary")
                save_status = gr.Textbox(label="Save Status", interactive=False)
    
    # Event handlers
    random_btn.click(
        fn=load_random_abstract,
        outputs=[abstract_input, arxiv_id]
    )
    
    classify_btn.click(
        fn=classify_abstract,
        inputs=[abstract_input, system_prompt_input, user_prompt_input],
        outputs=[physics_output, run_output, strategy_output, explanation_output]  # Added explanation_output
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[system_prompt_input, user_prompt_input],
        outputs=[save_status]
    )
    
    reset_btn.click(
        fn=reset_to_default,
        outputs=[system_prompt_input, user_prompt_input, save_status]
    )

# Run the app
demo.launch(share=True)