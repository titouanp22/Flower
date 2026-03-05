# Flower

This project contains two experiments based on **FLOWER**:

1. A **2D toy experiment** with a conditional GMM prior (`gmm.ipynb`)
2. A **face image super-resolution experiment** using a pretrained model (`faces.ipynb`)

The goal is to **explain and visualize the 3-step FLOWER scheme**:

1. **Flow-consistent destination estimation**
2. **Refining this destination using the measurements**
3. **Updating the trajectory over time**

The results are then compared with simple baselines.

![Faces Example](results_examples/faces/flower_pretrained_superres_steps_dense_with_refs.png)

---

# Repository Structure

* `gmm.ipynb`: 2D GMM experiment, flow matching training, trajectory visualization
* `faces.ipynb`: face super-resolution experiment using pretrained FLOWER weights

### Scripts

* `scripts/create_gmm_data.py`: generation of the toy GMM prior and posterior
* `scripts/train_gmm_flow_matching.py`: training of the flow matching model on posterior samples
* `scripts/gmm_flow_model.py`: time-conditioned MLP architecture + Euler sampling
* `scripts/flower_steps.py`: factorized implementation of the FLOWER steps (GMM and image inverse problems)
* `scripts/flower_plotting.py`: plotting utilities for the GMM experiment
* `scripts/faces_pipeline.py`: utilities for images, seeds, PSNR computation, and local super-resolution

### Other folders

* `results_examples/`: example images already generated to illustrate outputs
* `mehrsapo_Flower/`: FLOWER source code used by the `faces.ipynb` experiment

---

# Experiment 1: 2D GMM (`gmm.ipynb`)

1. Generates a **2D GMM prior** and its **posterior conditioned on a noisy linear observation**
2. Trains a **flow matching model** on posterior samples
3. Applies the **three FLOWER steps** and saves **step-by-step visualizations**

---

# Experiment 2: Face Super-Resolution (`faces.ipynb`)

1. Loads a **pretrained FLOWER model trained on CelebA** (weights are downloaded if missing)
2. Takes a **custom face image** and generates a **noisy low-resolution observation**
3. Reconstructs the **high-resolution image using the FLOWER steps**
4. Compares results with simple **baselines (adjoint, bicubic)** using **PSNR**

