# Ontology-guided Ensemble Learning for WSD Diagnosis

This repository contains code for the paper:

> Integrating Ontology and Machine Learning for Diagnosing White Spot Disease in Shrimp Aquaculture.

The project integrates a shrimp disease ontology (ShrimpDisease_Full.owl) with
Logistic Regression, SVM, and a 1D-CNN to create an ontology-guided ensemble
for White Spot Disease (WSD) risk prediction.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
