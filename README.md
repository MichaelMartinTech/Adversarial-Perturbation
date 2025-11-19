# Adversarial-Perturbation
## Overview
This repository contains the code, experiments, and analysis for a project investigating how diffusion-based image models interpret poisoned artworks (e.g., Glaze, Nightshade) using Explainable AI (XAI) techniques.
The project investigates latent representations of clean and poisoned images, how models view them, and how future poisoning techniques can evade detection.

## Setup

This project requires two files not published to this repository in compliance with the [LightShed](https://www.usenix.org/conference/usenixsecurity25/presentation/foerster) terms of use:
- A PyTorch checkpoint file, `*.pth`, which can be placed anywhere and will be passed as a command line argument
- LightShed's model architecture, `lightshed_model.py`, which must be in the root directory of this repository

1. In the terminal, navigate to the root directory:
    ```
    cd Adversarial-Perturbation
    ```

2. Set up a Python virtual environment:

    On Mac:
    ```
    python -m venv venv
    source venv/bin/activate
    ```

    On Windows:
    ```
    TODO
    ```

    (We used Python 3.13.0)

3. Install packages:
    ```
    pip install requirements.txt
    ```

## Running

- RQ1 - Visualizing Latent Clustering

    _How do image models view original images compared to their Glazed or Shaded counterparts?_
    ```
    python lightshed_xai.py --pth <*.pth>  --mode tsne --folder <directory>
    ```
    We used `./tsne_data` for the `--folder` argument. Images in this directory must be in `jpg`, `jpeg`, or `png` format.

    This displays a t-SNE plot of all images in `--folder`, color coded by poisoning technique.

    For proper color coding, file names should contain the substring `glazed` for Glazed images, `shaded` for Shaded images, and both substrings if both poisoning techniques are used.

- RQ2 - Visualizing Feature and Latent Activations

    _What do poison detection models look for?_
    ```
    python lightshed_xai.py --pth <*.pth> --mode activation --image <filename>
    ```
    We used individual images from the `./tsne_data` folder for the `--image` argument. Images must be in `jpg`, `jpeg`, or `png` format.

    This visualizes activations of the first 10 channels of each of the 5 encoding convolutional layers of LightShed.

- RQ3 - Improving Perturbation Techniques

    _What poisoning techniques, if any, can reliably avoid detection?_
    
    To reduce the size of the repository, the set of images that are eventually passed through LightShed are omitted. They are constructed by combining images in the directories `./noise_data/procedurals`, `./noise_data/masks`, and `./noise_data/noises`.

    Images in `./noise_data/procedurals` must be in 16-bit Grayscale. This repository comes with outputs from Substance Designer.\
    Images in `./noise_data/masks` must be in 8-bit Grayscale.\
    Images in `./noise_data/noises` must be in 8-bit RGB. This repository comes with outputs from Adobe Photoshop, Glaze, and Nightshade.

    All images must be in `jpg`, `jpeg`, or `png` format.\

    1. Generate masks:
        ```
        python generate_masks.py --folder {./noise_data/procedurals} --output {./noise_data/masks}
        ```
        `--folder` is a directory containing starter images from which to create masks. Masks are formed by adjusting the gamma of the starter images such that the average pixel value over the resulting image is a target value $\mathcal{L}$ accurate to some tolerance $\epsilon$. The variable `targets` contains the list of $\mathcal{L}$ values that we used. Resulting images are stored in `--output`.

        For the analysis script to function properly, file names in `--folder` must not contain underscores (`_`).

    2. Poison an image:
        ```
        python permute_noises_masks.py --image <filename> --noises {./noise_data/noises} --masks {./noise_data/masks} --output {./noise_data/results}
        ```
        `--output` will contain all combinations $c$ between images $n$ in `--noises` and $m$ `--masks` placed over `--image` $b$ according to this formula:
        $$
        c = \text{Clip}(b + n \odot (0.15 * m), 0, 255)
        $$
        where $c, b, n$ are in the range [0, 255] and $m$ is in the range [0.0, 1.0].

        For the analysis script to function properly, file names in `--noises` must not contain underscores (`_`).

        Though not required, `--masks` should be the same directory as `--output` from the previous step.

    3. Process the output of the previous step with LightShed.
    
        This requires access to LightShed, which is not part of this repository. However, we have provided a sample CSV output (`detection_analytics.csv`) to use in the next step.

    4. Analyze LightShed output:
        ```
        python lightshed_analysis.py --csv <filename>
        ```


<!-- ## Current Pipeline
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

### **RQ3 – Entropy Experiments (Poison Detectability)**
> Created custom synthetic poison patterns to test LightShed’s detectability  
> Evaluated entropy using lossless compression as an approximate randomness metric

- Generated multiple noise types (Gaussian, Clouds 2, Perlin, binary variants)  
- Applied procedural masks at varying densities and lightness values  
- Overlaid all noise–mask combinations onto a clean base image at 15% opacity  
- Computed entropy for each composite image and measured LightShed reconstruction strength  
- Higher-entropy, spatially irregular patterns showed reduced detectability  
- Low-frequency or uniform patterns were more easily reconstructed by LightShed -->
