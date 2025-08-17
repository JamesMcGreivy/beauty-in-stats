# LHCb Scraper Setup

Critical to the development of the LHCb analysis copilot is apt data curation. This starts with retrieving from the LHCb paper archive the metadata of all papers, focusing on the `Working Group` and `Years/Run` fields.

An agent has been developed to scrape the LHCb paper archive and extract the metadata. This README provides instructions for running the scraper using Docker or Apptainer.

The SubMIT node with GPU access does not have a browrser. To interface the agent with the LHCb paper archive, a headless browser is used, installing it a dedicated container to avoid conflicts with the SubMIT system.

The scraper can be run using Docker or Apptainer. This README provides instructions for both setups, with a preference for using Apptainer to inherit GPU access from the host.



## Prerequisites

- **Docker**: Ensure Docker is installed and running on your machine.
- **Apptainer**: Ensure Apptainer is installed. You can find installation instructions [here](https://apptainer.org/docs/user/quickstart.html).
- **NVIDIA GPU**: If you want to use GPU acceleration, ensure you have an NVIDIA GPU and the necessary drivers installed.

## Using Docker

1. **Build the Docker Image**:
   Navigate to the directory containing the `Apptainer.def` file and run the following command:

   ```bash
   docker build -t lhcb_scraper .
   ```

2. **Run the Docker Container**:
   To run the scraper, execute the following command:

   ```bash
   docker run --rm lhcb_scraper
   ```

   This will execute the `lhcb_scraper.py` script inside the container.

## Using Apptainer

1. **Build the Apptainer Image**:
   Navigate to the directory containing the `Apptainer.def` file and run the following command:

   ```bash
   apptainer build lhcb-scraper-container.sif Apptainer.def
   ```

2. **Run the Apptainer Container with GPU Access**:
   To run the scraper with GPU access, use the following command:

   ```bash
   apptainer shell --nv lhcb-scraper-container.sif
   ```

   The `--nv` flag allows the container to access the GPU resources of the host.

## Verifying the Setup

1. **Check GPU Access**:
   If you are using GPU acceleration, you can verify that the GPU is accessible within the Apptainer container by running:

   ```bash
   apptainer exec --nv lhcb_scraper.sif nvidia-smi
   ```

   This command should display the GPU information if the setup is correct.

2. **Check the Output**:
   The scraper should generate a JSON file named `lhcb_metadata.json` in the `/app/data/` directory. You can check the contents of this file to ensure that the metadata was fetched correctly.

   ```bash
   cat /app/data/lhcb_metadata.json
   ```
