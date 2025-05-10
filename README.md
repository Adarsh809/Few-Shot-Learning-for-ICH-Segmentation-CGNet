# 🧠 Intracranial Hemorrhage Segmentation (ICH-Segmentation)

This project focuses on segmenting **Intracranial Hemorrhage (ICH)** regions from 3D CT volumes using a **few-shot learning** approach built on a lightweight **CGNet** architecture. Our preprocessing pipeline slices volumes, extracts informative 2D patches, and feeds them into the network for robust segmentation, even with limited data.

---

## 🛠️ Preprocessing

We preprocess volumetric CT scans into 2D patches using:

- **HU windowing**: `[-40, 120]` for brain CT
- **Fixed slicing**: (512×512) axial slices
- **Patch extraction**: 4 non-overlapping (256×256) regions per slice
- **Filtering**: Ignore pure-background patches (i.e., zero-mask)

Run it:
```bash
python slice_and_save_patches.py
```

## 🚀 Model + Training (Few-shot CGNet)
Base model: CGNet

Setup: K-shot segmentation with a tiny support set

Loss: Dice + BCE combo

Training is handled in the k-shot_CGNet_ICH.ipynb notebook

## 🧪 Results (Sample)
Coming soon™ — working on metrics and visualizations.
Expect: DSC, IoU, and qualitative overlays of predictions.

