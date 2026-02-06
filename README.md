# Project Overview
## Title
GeneAgent: Self-verification Language Agent for Gene Set Analysis using Domain Databases
## Abstract
GeneAgent is a first-of-kinds language agent built upon GPT-4 to automatically interact with domain-specific databases to annotate functions for gene sets. GeneAgent generates interpretable and contextually accurate biological process names for user-provided gene sets, either aligning with significant enrichment analyses or introducing novel terms. At the core of GeneAgent’s functionality is a self-verification setting. This mechanism autonomously interacts with various expert-curated biological databases through Web APIs. By utilizing relevant domain-specific information, GeneAgent performs fact verification and provides objective evidence to support or refute the raw LLM output, reducing hallucination and enabling reliable, evidence-based insights into gene function.
<p align="center" width="50%">
  <img width="80%" src="https://github.com/ncbi-nlp/GeneAgent/blob/main/workflow.geneagent.svg">
</p>

# Requirement
- Python >= 3.12
- See `pyproject.toml` for complete dependency list

## Core Dependencies
- `baseagent` - Agent framework (installed from https://github.com/BinglanLi/BaseAgent)
- `openai >= 1.0.0` - OpenAI API client
- `langchain-core >= 0.1.0` - LangChain core functionality
- `langchain-openai >= 0.1.0` - OpenAI integration for LangChain
- `langchain-ollama >= 0.1.0` - Ollama integration for LangChain
- `langgraph >= 0.1.0` - Workflow orchestration
- `pydantic >= 2.0.0` - Data validation
- `pandas >= 2.1.4` - Data manipulation
- `numpy >= 1.26.3` - Numerical computing
- `requests >= 2.31.0` - HTTP library
- `tiktoken >= 0.11.0` - Token counting
- `python-dotenv >= 1.0.0` - Environment variable management

## Optional Dependencies
- `langchain-anthropic >= 0.1.0` - For Anthropic Claude models
- `langchain-aws >= 0.1.0` - For AWS Bedrock models

# Datasets
- Gene Ontology: contain 1000 gene sets from the GO:BP branch of the gene ontology database
- MsigDB: contain 56 gene sets including the hallmark gene sets
- NeST: contain 50 gene sets sampled from the human cancer proteomic data
>[!TIP]
>The original datasets could be found at
>* https://github.com/idekerlab/llm_evaluation_for_gene_set_interpretation/blob/main/data/
>* https://github.com/monarch-initiative/talisman-paper/tree/main/genesets/human

# Configuration:
## Installation 

### Method 1: Using UV (Recommended)

1. Clone the repository:
   ```bash
   git clone git@github.com:BinglanLi/GeneAgent.git
   cd GeneAgent
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create environment and install all dependencies (including BaseAgent from GitHub):
   ```bash
   uv sync
   ```

4. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

5. (Optional) Install with additional LLM provider support:
   ```bash
   uv sync --extra anthropic  # For Anthropic Claude models
   uv sync --extra bedrock    # For AWS Bedrock models
   uv sync --extra all        # For all optional providers
   ```

### Method 2: Using pip

1. Clone the repository:
   ```bash
   git clone git@github.com:BinglanLi/GeneAgent.git
   cd GeneAgent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt  # If available
   # Or install from pyproject.toml
   pip install .
   ```

4. (Optional) Install with additional LLM provider support:
   ```bash
   # For Anthropic Claude models
   pip install ".[anthropic]"
   
   # For AWS Bedrock models
   pip install ".[bedrock]"
   
   # For all optional providers
   pip install ".[all]"
   ```

### Note on BaseAgent

BaseAgent is automatically installed from https://github.com/BinglanLi/BaseAgent as specified in `pyproject.toml`. The installation will fetch the latest version from GitHub.

## Configure LLM Credentials

Before running GeneAgent, you need to configure your LLM provider credentials.

1. Set environment variables (recommended): 

   **For Public OpenAI:**
   - `OPENAI_API_KEY`

   **For Azure OpenAI (optional):**
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_VERSION`
   - Use model names with `azure-` prefix (e.g., `azure-gpt-4`)

   **For Ollama (optional):**
   - Ollama must be running locally on `http://localhost:11434`
   - Use model names containing `gpt-oss` (e.g., `gpt-oss:20b`)
   - Special handling of the Ollama server is attached below.

   The code automatically detects the provider based on the model name:
- Models starting with `gpt-` or `o1-` → OpenAI API
- Models starting with `azure-` → Azure OpenAI
- Models containing `gpt-oss` → Ollama (OpenAI-compatible endpoint)
- Models starting with `claude-` → Anthropic (via BaseAgent)
- Other models → Auto-detected by BaseAgent
   
   >[!NOTE]
   >GeneAgent now uses BaseAgent's unified LLM infrastructure (`llm_utils.py`), which provides support for multiple LLM providers (OpenAI, Azure OpenAI, Ollama, etc.) through a single interface.

## Set up Ollama on HPC

1. Download and extract the Ollama Linux binary in a folder of your choice.
   ```bash
   # create the directory
   mkdir -p ~/ollama
   cd ~/ollama
   # install
   curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst
   # extract
   mkdir -p ~/ollama
   zstd -d ollama-linux-amd64.tar.zst -c | tar -xf - -C ~/ollama
   ```

2. Configure the environment variables by adding the following lines in `~/.bashrc` or` ~/.zshrc`.
   ```bash
   # Add the binary to your PATH so you can type 'ollama' instead of the full path
   export PATH=$HOME/ollama/bin:$PATH

   # (Optional) Set a custom path for models if you want them somewhere specific
   # If you skip this, they will default to ~/.ollama/models (which is fine for most users)
   export OLLAMA_MODELS=$HOME/ollama/models
   ```

3. Run the Ollama server
   ```bash
   # & symbol runs the Ollama server in the background
   ollama serve &
   ```

4. (Optional) Create a simple script to manage the Ollama server without `sudo`.
   Create a new file named `ollama-manager.sh` with the following content under the `~/ollama/` folder. 
   ```bash
   #!/bin/bash

   # --- CONFIGURATION ---
   INSTALL_DIR="$HOME/ollama"
   export OLLAMA_MODELS="$INSTALL_DIR/models"
   export PATH="$INSTALL_DIR/bin:$PATH"

   PID_FILE="$INSTALL_DIR/ollama.pid"
   LOG_FILE="$INSTALL_DIR/server.log"
   OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
   # ---------------------

   # Helper: Check if a PID is actually an ollama process
   is_ollama_running() {
      local pid="$1"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
         # Verify it's actually an ollama process, not a recycled PID
         if ps -p "$pid" -o comm= 2>/dev/null | grep -q "ollama"; then
               return 0
         fi
      fi
      return 1
   }

   case "$1" in
      start)
         if [ -f "$PID_FILE" ]; then
            OLD_PID=$(cat "$PID_FILE")
            if is_ollama_running "$OLD_PID"; then
               echo "Ollama is already running (PID: $OLD_PID)."
               exit 0
            else
               echo "Removing stale PID file..."
               rm -f "$PID_FILE"
            fi
         fi
         
         echo "Starting Ollama..."
         nohup ollama serve > "$LOG_FILE" 2>&1 &
         NEW_PID=$!
         echo $NEW_PID > "$PID_FILE"
         
         # Wait briefly and verify it actually started
         sleep 2
         if is_ollama_running "$NEW_PID"; then
            echo "Ollama started! (PID: $NEW_PID)"
            echo "Logs are being written to: $LOG_FILE"
         else
            echo "ERROR: Ollama failed to start. Check logs: $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
         fi
         ;;
      stop)
         if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if is_ollama_running "$PID"; then
               echo "Stopping Ollama (PID: $PID)..."
               kill "$PID"
               # Wait for graceful shutdown (up to 10 seconds)
               for i in {1..10}; do
                     if ! kill -0 "$PID" 2>/dev/null; then
                        break
                     fi
                     sleep 1
               done
               # Force kill if still running
               if kill -0 "$PID" 2>/dev/null; then
                     echo "Force killing..."
                     kill -9 "$PID" 2>/dev/null
               fi
               rm -f "$PID_FILE"
               echo "Ollama stopped."
            else
               echo "PID file exists but process is not running. Cleaning up..."
               rm -f "$PID_FILE"
            fi
         else
            echo "Ollama is not running (no PID file)."
            # Fallback cleanup
            pkill -u "$USER" -x ollama 2>/dev/null && echo "Cleaned up stray ollama processes."
         fi
         ;;
      restart)
         "$0" stop
         sleep 2
         "$0" start
         ;;
      status)
         if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if is_ollama_running "$PID"; then
               echo "✅ Ollama is RUNNING (PID: $PID)"
               # Optionally check if the server is responding
               if curl -s --max-time 2 "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
                     echo "   API is responding at http://${OLLAMA_HOST}"
               else
                     echo "   ⚠️  Process running but API not responding yet"
               fi
            else
               echo "🔴 Ollama is STOPPED (stale PID file found)"
               rm -f "$PID_FILE"
            fi
         else
            echo "🔴 Ollama is STOPPED"
         fi
         ;;
      *)
         echo "Usage: $0 {start|stop|restart|status}"
         exit 1
         ;;
   esac
   ```

   Make the file executable:
   ```bash
   chmod +x ~/ollama/ollama-manager.sh
   ```

   Use the file:
   ```bash
   ~/ollama/ollama-manager.sh start
   ~/ollama/ollama-manager.sh status
   ~/ollama/ollama-manager.sh stop
   ```

# Execute
## Running
Type following command in your virtual environment.
```
python main_cascade.py
```
The results will be stored accordingly.

## Architecture
GeneAgent uses a unified LLM infrastructure built on BaseAgent:
- **`llm_utils.py`**: Unified LLM client supporting multiple providers (OpenAI, Azure, Ollama)
- **`main_cascade.py`**: Main cascade workflow for gene set analysis
- **`worker.py`**: AgentPhD class for claim verification using function calling
- **`apis/`**: Domain-specific API functions for biological databases

 >[!TIP]
  >If you want to evaluate your own gene sets, save them to **Datasets** directory and change the directory path in **main_cascade.py**
>Also, the output path can be changed according to your preference.

## Example outputs
```
Process: MAPK Signaling Pathway
The proteins encoded by the genes ERBB2, ERBB4, FGFR2, FGFR4, HRAS, and KRAS are all integral components of the MAPK signaling pathway, which is crucial for cell growth, differentiation, and survival.
ERBB2 and ERBB4 are members of the epidermal growth factor receptor (EGFR) family of receptor tyrosine kinases (RTKs). ERBB2 is unique in that it has no known ligands, and it prefers to form heterodimers with other EGFR family members, enhancing their kinase activity. ERBB4 is activated by neuregulins and other factors and induces a variety of cellular responses including mitogenesis and differentiation.
FGFR2 and FGFR4 are part of the fibroblast growth factor receptor (FGFR) family of RTKs. They are activated by fibroblast growth factors, leading to receptor dimerization and autophosphorylation. This triggers downstream signaling pathways that regulate cellular processes such as proliferation, differentiation, and migration.
HRAS and KRAS are GTPases that act as molecular switches in RTK signaling. They are activated by guanine nucleotide exchange factors (GEFs) that catalyze the exchange of GDP for GTP. Once activated, RAS proteins can interact with a variety of effector proteins to propagate the signal downstream.
The interaction between these proteins forms a complex network of signaling events that regulate key cellular processes. Dysregulation of this system, such as mutations that lead to constitutive activation of RTKs or RAS proteins, can result in uncontrolled cell growth and cancer. Therefore, understanding the precise mechanisms of MAPK signaling and its regulation is crucial for the development of targeted cancer therapies.
```
## Evaluate the outputs
Open **evaluate.ipynb** to run the corresponding cells based on your requirements.

# Demonstration website
A demonstration website with an open-access permissions is available at https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/GeneAgent/.
<p align="center" width="50%">
  <img width="80%" src="https://github.com/ncbi-nlp/GeneAgent/blob/main/homepage.geneagent.jpg">
</p>

# Acknowledgements
This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

# Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical or genomics professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NLM's disclaimer policy is available.

# Zenodo identifier
[DOI: 10.5281/zenodo.15008591](https://zenodo.org/records/15008591)


