# Representation Matters: Rethinking Domain Generalization in Polyp Segmentation

Official implementation of **"Representation Matters: Rethinking Domain Generalization in Polyp Segmentation"**.

> **Accepted at [MICCAI 2026 MLMI]**

Colonoscopy images exhibit substantial variability across clinical centers due to differences in acquisition devices, patient populations, imaging protocols, illumination, and other center-specific factors. Rather than addressing this variability through increasingly complex, task-specific segmentation architectures, this work investigates whether representation quality itself can be the main driver of cross-domain generalization.

We systematically evaluate DINOv1, DINOv2, DINOv2 with register tokens, and DINOv3 for generalizable polyp segmentation, considering both frozen and supervised fine-tuned feature extractors. Across the multi-center PolypGen dataset, stronger pretrained representations consistently translate into better generalization performance.

Notably, the latest DINO variants outperform UM-Net, a specialized architecture specifically designed for polyp segmentation, despite relying on a comparatively simple segmentation framework. These results suggest that, particularly under severe data scarcity and domain shift, the quality of the learned representation can matter more than architectural specialization.

---

## Main Contributions

- We investigate DINO ViT key features as dense descriptors for fully automatic polyp segmentation.

- We evaluate both frozen and fine-tuned representations under **Domain Generalization (DG)** and **Extreme Single Domain Generalization (ESDG)**.

- We systematically compare **DINOv1, DINOv2, DINOv2 with register tokens, and DINOv3** to study how advances in self-supervised representation learning translate to downstream medical image segmentation.

- We compare token, query, key, and value representations and investigate their suitability for dense prediction.

---

# Experimental Settings

We evaluate two cross-center generalization protocols.

## Domain Generalization (DG)

Models are trained using **five clinical centers** and evaluated on the remaining unseen center.

```text
Train: C1 + C2 + C3 + C4 + C5
Test:  C6

Train: C1 + C2 + C3 + C4 + C6
Test:  C5

...
```

The experiment is repeated until every center has served as the held-out test domain.

## Extreme Single Domain Generalization (ESDG)

Models are trained using data from **a single clinical center** and evaluated jointly on the remaining five unseen centers.

```text
Train: C1
Test:  C2 + C3 + C4 + C5 + C6

Train: C2
Test:  C1 + C3 + C4 + C5 + C6

...
```

This setting simulates deployment under severe data scarcity and substantial distribution shift.

---

# Models

We evaluate the following DINO representations:

| Model | Registers | Training |
|---|:---:|---|
| DINOv1 | – | Frozen / Fine-tuned |
| DINOv2 | No | Frozen / Fine-tuned |
| DINOv2-R | Yes | Frozen / Fine-tuned |
| DINOv3 | Yes | Frozen / Fine-tuned |

We use the notation:

- **A** - DINO feature extractor is **frozen**
- **B** - DINO feature extractor is **fine-tuned**

For example:

```text
V3(A) → DINOv3 with frozen encoder
V3(B) → DINOv3 with fine-tuned encoder
```

---
# Results
<p align="center">
  <img width="708" height="501" alt="image"
       src="https://github.com/user-attachments/assets/546d1942-743b-41db-a5f2-6bb4c6f14133" />
</p>

<p align="center">
  <img width="1652" height="631" alt="image"
       src="https://github.com/user-attachments/assets/3d6ae3f8-c443-47b9-81dd-0abc484f1c23" />
</p>

# Training

Training behavior can be selected through the experiment configuration.

For a frozen feature extractor:

```yaml
train:
  case: A
```

For supervised fine-tuning:

```yaml
train:
  case: B
```

# Running Experiments

> Run main.py, changing the config.yaml to your wished configuration.

# Acknowledgements

This project builds upon several excellent open-source projects and datasets, including:

- [DINO](https://github.com/facebookresearch/dino)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [PolypGen](https://github.com/PRIS-CV/PolypGen)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [UM-Net]( https://github.com/dxqllp/UM-Net)

We thank their authors for making their work publicly available.

---

<!--
# Citation

If you find this work or code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{monteiro2026representation,
    title     = {Representation Matters: Rethinking Domain Generalization in Polyp Segmentation},
    author    = {<AUTHOR LIST>},
    booktitle = {<MICCAI 2026 WORKSHOP / PROCEEDINGS NAME>},
    year      = {2026}
}

---
-->
# Contact
Carla Monteiro (carla.s.monteiro@inesctec.pt)

For questions or issues regarding the code, please open a GitHub issue.
