# Neural Mechanisms in Processing of Emotion in Real and Virtual Faces using fNIRS

## Overview

This repository contains the code and data processing pipeline for a Master's thesis investigating neural mechanisms involved in emotion processing when viewing real and virtual faces using functional near-infrared spectroscopy (fNIRS). The study explores how the brain differentially processes emotional expressions in real human faces versus computer-generated virtual avatars.

## Abstract

This research examines the neural correlates of emotion recognition across different stimulus modalities (real vs. virtual faces) using fNIRS neuroimaging. The study provides insights into how the human brain processes emotional information in both natural and virtual environments, with implications for virtual reality applications, human-computer interaction, and understanding of emotion processing mechanisms.

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
- **UIBVFED (University of Bridgeport Virtual Face Expression Database)**: Computer-generated 3D virtual avatars
- Standardized virtual faces with identical facial features displaying the same emotional expressions
- 7 emotion categories: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

## Data Collection

- **Participants**: 91 participants (P_1 to P_91)
- **fNIRS System**: NIRx system with multiple wavelengths
- **Sampling Rate**: ~6.1 Hz
- **Experimental Paradigm**: Block design with emotional face viewing tasks
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

## Key Dependencies

- **MNE-Python** (1.9.0): Neurophysiological data analysis
- **MNE-NIRS** (0.7.1): fNIRS-specific analysis tools
- **NumPy** & **SciPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib** & **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning and statistical analysis
- **Nilearn**: Neuroimaging analysis

## Key Scripts and Notebooks

- `preprocess_nirs.py`: Main preprocessing pipeline
- `nirs_functions.py`: Core analysis functions and utilities
- `all_datasets processing.ipynb`: Complete analysis workflow
- `generate_tables.ipynb`: Results visualization and table generation
- `EmotionVR_task.py`: PsychoPy experimental paradigm
- `coherence_demo.py`: Coherence analysis example

## Results and Outputs

The analysis generates:
- **Activation Maps**: Brain regions showing differential responses to real vs. virtual faces
- **Time Series Analysis**: Hemodynamic response functions for different emotion categories
- **Statistical Reports**: GLM results and connectivity analyses
- **Behavioral Correlations**: Relationships between neural activity and behavioral responses
- **Quality Control Reports**: Data quality assessments and preprocessing summaries

## Publication and Data Availability

- **Thesis**: "Neural mechanisms in processing of emotion in real and virtual faces using functional-near infrared spectroscopy (fNIRS)"
- **Author**: Dylan Rapanan, Ontario Tech University
- **Degree**: Master of Science in Computer Science
- **Year**: 2025
- **Data Repository**: [Open Science Framework (OSF)](https://osf.io/d7bzp/?view_only=f5a96f051edb4e768c5e4461699ef1ce)