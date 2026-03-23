# Differential Gene Expression Analysis of GSE47363 Dataset Using limma

**Investigating the transcriptomic impact of miR-542-3p mimics in human cell lines.**

### Project Overview
This repository contains a reproducible pipeline for the differential expression analysis of the GSE47363 dataset. The study compares a miR-542-3p treatment group against a negative control group to identify gene expression changes induced by the miRNA mimic.

The analysis is performed using the limma framework (implemented via InMoose) to handle Illumina microarray intensities.

### Key Features
Data Processing: Loading and processing of non-normalized Illumina intensity files.

Gene Deduplication: Utilization of the MaxMean method to handle multiple probes mapping to the same gene symbol.

Statistical Modeling: Linear modeling and empirical Bayes moderation via limma.

Multiple Testing Correction: Implementation of the Benjamini-Hochberg (BH) procedure to control the False Discovery Rate (FDR).

### Repository Structure
├── .devcontainer/      # Docker & DevContainer configuration
├── data/               # (User-created) Input directory for GEO and validation files
├── notebooks/          # Jupyter notebooks for DE analysis and interpretation
├── results/            # Output: iPathway Guide files, CDF, and Volcano plots
└── README.md

### Getting Started

#### Prerequisites
* Python 3.10+
* InMoose (for limma implementation)
* pandas, numpy, and matplotlib/seaborn for visualization

#### Data Acquisition
This analysis requires three specific files located in the data/ directory.

1. Expression Data (GSE47363)
Source: NCBI GEO GSE47363

File: GSE47363_non-normalized.txt.gz

Instructions: Download and extract using gunzip so that data/GSE47363_non-normalized.txt is available.

2. Platform Annotations (GPL10558)
Source: GPL10558 - Illumina HumanHT-12 V4.0

File: GPL10558_annot.txt

Instructions: This file is used to map Illumina Probe IDs to Gene Symbols. Ensure it is placed in the data/ folder.

3. TargetScan Validation (Private)
Source: Provided for validation (not public).

File: targetscan_validation_results.csv

Instructions: This file contains the predicted targets for miR-542-3p. It must be manually placed in the data/ folder for the validation step of the pipeline to run.

#### Installation
```bash
git clone https://github.com/adevlen/GSE47363_analysis.git
cd /GSE47363_analysis
```
#### Development Environment
This project is configured for easy reproducibility using Docker. You can either use the automated VS Code setup or build the environment manually.

**Option 1: VS Code DevContainers**
If you have the Dev Containers extension installed in VS Code:

Open the project folder in VS Code.

When prompted with "Reopen in Container," click Reopen.

The extension will automatically build the image and install all dependencies (Python, InMoose, etc.).

**Option 2: Manual Docker Build**
If you prefer using the command line, you can build and run the container manually:

```bash
docker build -t gse47363-analysis .
docker run -it \
  -v $(pwd)/data:/work/data \
  -v $(pwd)/results:/work/results \
  -p 8888:8888 \
  gse47363-analysis
```
### Usage
To run the analysis from scratch, execute the main processing notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Results
The analysis identifies key downstream targets of miR-542-3p. Summary plots (Volcano and MA plots) can be found in the results/ directory.

### Biological Context