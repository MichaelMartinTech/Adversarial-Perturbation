# Adversarial-Perturbation
## Overview
This repository contains the code, experiments, and analysis for a project investigating how diffusion-based image models interpret poisoned artworks (e.g., Glaze, Nightshade) using Explainable AI (XAI) techniques.
The project focuses on evaluating:
- How protected images differ from clean ones in latent space
- What LightShed’s encoder and decoder respond to
- How entropy influences poison detectability
- What visual patterns activate poison-sensitive neurons

## System Setup & Experimental Reproducibility
### MAC
- To be added (G)
### Windows
- To be added (M)

## Research Questions
### RQ1 — Latent Representation
How does LightShed represent clean vs. Glazed vs. Shaded images in feature space?
### RQ2 — Poison Detection Behavior
Which visual patterns activate LightShed’s poison-sensitive neurons?
### RQ3 — Resistance to Detection
Which synthetic high-entropy noise patterns can avoid reconstruction/detection?

## Current Pipeline
Curated 7 diverse images (personal + public domain).
Images span: watercolor, oil, digital, pixel art, stylized illustration.
Organized unified dataset can be found in ``training_data``

## Poison Generation
Applied Glaze and Nightshade locally
Generated three variants per image:
- Glaze-only
- Nightshade-only
- Nightshade → Glaze (Recommended ordering according to [NightShade/Glaze authors](https://nightshade.cs.uchicago.edu/userguide.html#:~:text=If%20you%20are%20planning%20to,JPG%20when%20it's%20all%20done.))

## LightShed Processing
Imported [LightShed](https://www.usenix.org/conference/usenixsecurity25/presentation/foerster)’s pretrained encoder–decoder.
Extracted:
- Encoder latent embeddings
- Decoder reconstructions
- Reconstructed poison masks

## Progress by Research Question

### **RQ1 – Latent Clustering**
- Extracted last-conv embeddings from LightShed encoder  
- Reduced dimensionality using **t-SNE**  
- Generated preliminary cluster maps  
- Clean, Glaze, and Nightshade form distinct clusters  
- **Nightshade → Glaze** overlaps strongly with **Glaze-only** (early finding)

---

### **RQ2 – Neuron Activation Maps (Feature Visualization)**
> Implemented feature visualization on LightShed decoder  
> Identified neurons consistently responding to:
- High-frequency perturbation textures  
- Glaze-specific patterns  
- Nightshade-specific patterns  
- Visualizations highlight recurring poison signatures

---

### **RQ3 – Entropy Experiments**
(Revising)
