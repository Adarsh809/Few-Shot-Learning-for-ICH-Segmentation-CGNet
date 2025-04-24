# CGNet: Few-Shot Learning for Intracranial Hemorrhage Segmentation

![Project Logo or Illustration](path_to_image_if_any)

## Overview

This repository contains the implementation of **CGNet**, a novel few-shot deep learning model for **Intracranial Hemorrhage (ICH) segmentation** in medical images, based on the paper:

> Gong et al., "CGNet: Few-shot learning for Intracranial Hemorrhage Segmentation", 2024.

CGNet addresses the challenge of segmenting hemorrhages with very limited annotated data by leveraging a **Cross Feature Module (CFM)** and **Support Guide Query (SGQ)** to effectively fuse support and query features at multiple scales.

---

## Features

- Few-shot segmentation setup (1-way K-shot).
- Multi-scale feature fusion with CFM and SGQ modules.
- Supports 2D slice-based segmentation from 3D medical volumes.
- Training and evaluation on BHSD and IHSAH datasets.
- Implements Dice and IoU metrics for evaluation.
- Modular PyTorch codebase for easy extension.



