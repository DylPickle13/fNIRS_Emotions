# Neural mechanisms in processing of emotion in real and virtual faces using functional-near infrared spectroscopy (fNIRS)

## Overview

This repository contains the code and data processing pipeline for a Master's thesis investigating neural mechanisms involved in emotion processing when viewing real and virtual faces using functional near-infrared spectroscopy (fNIRS). The study explores how the brain differentially processes emotional expressions in real human faces versus computer-generated virtual avatars.

## Abstract

As avatars permeate social media, gaming, and telecommunications, understanding how the brain reads emotions from virtual faces is increasingly important. 
We recorded functional near-infrared spectroscopy (fNIRS) data from adults viewing real photographs and matched computer-generated faces expressing Anger, Disgust, Fear, Joy, Sadness, Surprise, or Neutral (control). 
General-linear-model mapping revealed higher activation in virtual faces in the left occipital region, and higher activation in Neutral and Surprise compared to the other emotions in parietal and occipital regions. 
Functional-connectivity analysis revealed higher connectivity in real faces across the brain, and higher connectivity across the brain in Anger and Fear compared to the other emotions. 
Collectively, the results demonstrate differences in activation in occipital areas, and differential processing of face and emotion types across the whole brain.  
These neural signatures provide quantitative targets for refining the realism and emotional efficacy of digital characters in virtual and augmented environments.

## Project Structure

```
fNIRS_Emotions/
├── data/                           # Raw fNIRS data from participants (P_1 to P_91)
│   ├── P_1/
│   │   ├── 2024-09-19_002/        # Session data
│   │   │   ├── *.snirf            # SNIRF format fNIRS data
│   │   │   ├── *.nirs             # NIRx format data
│   │   │   └── ...
│   │   └── *.csv                   # Behavioral data
│   └── ...
├── processed_data/                 # Processed and analyzed data
│   ├── behavioural_responses/      # Behavioral analysis results
│   ├── epochs/                     # Epoched fNIRS data
│   ├── glm/                        # General Linear Model results
│   ├── mappings/                   # Channel mappings and configurations
│   ├── models/                     # Statistical models and results
│   ├── raw_haemos/                 # Processed hemoglobin concentration data
│   ├── raw_ods/                    # Optical density data
│   ├── raws/                       # Preprocessed raw data
│   ├── roi_timeseries_activity/    # Region of interest time series
│   ├── spectral_connectivity_time/ # Connectivity analysis results
│   └── windows/                    # Windowed data analysis
├── plots/                          # Generated figures and visualizations
├── Configurations/                 # NIRx configuration files
├── UIBVFED_cropped/               # Virtual face stimuli database
│   ├── ANGER/
│   ├── DISGUST/
│   ├── FEAR/
│   ├── JOY/
│   ├── NEUTRAL/
│   ├── SADNESS/
│   └── SURPRISE/
├── RADIATE Color Faces/           # Real face stimuli database
├── writing/                       # Thesis manuscript and documentation
├── *.ipynb                        # Jupyter notebooks for analysis
├── *.py                           # Python analysis scripts
└── requirements.txt               # Python dependencies
```

## Stimuli

### Real Faces
- **RADIATE Color Faces Database**: High-quality photographs of real human faces displaying various emotional expressions
- 10 different identities across 7 emotion categories

### Virtual Faces  
- **UIBVFED Database**: Computer-generated 3D virtual avatars
- Standardized virtual faces with identical facial features displaying the same emotional expressions
- 7 emotion categories: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

## Data Collection

- **Participants**: 91 participants (P_1 to P_91)
- **fNIRS System**: NIRx system with multiple wavelengths
- **Sampling Rate**: ~6.1 Hz
- **Data Format**: SNIRF (Shared Near Infrared Spectroscopy Format) standard

## Installation

### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook/Lab
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/fNIRS_Emotions.git
cd fNIRS_Emotions
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Scripts and Notebooks

- `preprocess_nirs.py`: Main preprocessing pipeline
- `nirs_functions.py`: Core analysis functions and utilities
- `all_datasets processing.ipynb`: Complete analysis workflow
- `generate_tables.ipynb`: Results visualization and table generation
- `EmotionVR_task.py`: PsychoPy experimental paradigm
- `coherence_demo.py`: Coherence analysis example

## Publication and Data Availability

- **Thesis**: "Neural mechanisms in processing of emotion in real and virtual faces using functional-near infrared spectroscopy (fNIRS)"
- **Author**: Dylan Rapanan, Ontario Tech University
- **Degree**: Master of Science in Computer Science
- **Year**: 2025
- **Data Repository**: [Open Science Framework (OSF)](https://osf.io/d7bzp/?view_only=f5a96f051edb4e768c5e4461699ef1ce)