Our paper, **CTTA-T: Continual Test-Time Adaptation for Text Understanding via Teacher-Student with a Domain-aware and Generalized Teacher**, has been accepted by **ACL 2026 Main**!

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Obtain the Source Pre-trained Models

In this paper, we employed two backbone models: **base** and **xtune**. You can obtain the following models:

- **Base model**: 
- **XTune model**: 
After the paper is published, the link will be provided

## Obtain the Datasets
Click here (After the paper is published, the link will be provided) to download the dataset.

## Usage

To run the model, use the following command:

```bash
bash ./scripts/continual_short_1.sh xlm-roberta-base tasks [device] [model_path] [method] [gamma] [tau] [learning_rate]
```

### Available [method]:

- "CTTA-T"
- "cotta"
- "OIL"
- "PL"
- "Tent"

Regarding the codes for SAR, SoTTA, and REM, we are still in the process of sorting them out.

### Hyperparameters:

**gamma** and **tau** are hyperparameters. For the **base** model, gamma is 0.4, and tau is 1.2. For the **xTune** model, gamma is 0.4 and tau is 0.6.

### Learning Rate:

 For **[learning_rate]**, we use 1e-5.

### Sequence Types:

We provide scripts of standard short sequences (Order 1) and standard long sequences (Order 4). If you wish to test other task orders, you can modify the sequence in the `continual_short_1.sh` or `continual_long_1.sh` scripts.

## Citation
```
@inproceedings{liu-etal-2026-ctta,
    title = "{CTTA}-{T}: Continual Test-Time Adaptation for Text Understanding via Teacher-Student with a Domain-aware and Generalized Teacher",
    author = "Liu, Tianlun  and
      Tian, Zhiliang  and
      Huang, Zhen  and
      Zhou, Xingzhi  and
      Yu, Wanlong  and
      Liu, Tianle  and
      Liu, Feng  and
      Li, Dongsheng",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.187/",
    doi = "10.18653/v1/2026.acl-long.187",
    pages = "4058--4078",
    ISBN = "979-8-89176-390-6",
    abstract = "Text understanding application often suffers from domain shifts. To handle testing domains, domain adaptation (DA) is trained to adapt to a fixed and observed testing domain; a more challenging paradigm, test-time adaptation (TTA), cannot access the testing domain during training and online adapts to the testing samples during testing, where the samples are from a fixed domain. We aim to explore a more practical and underexplored scenario, continual test-time adaptation (CTTA) for text understanding, which involves a sequence of testing (unobserved) domains in testing. Current CTTA methods struggle in reducing error accumulation over domains and enhancing generalization to handle unobserved domains: 1) Noise-filtering reduces accumulated errors but discards useful information, and 2) accumulating historical domains enhances generalization, but it is hard to achieve adaptive accumulation. In this paper, we propose a CTTA-T (continual test-time adaptation for text understanding) framework adaptable to evolving target domains: CTTA-T adopts a teacher-student framework, where the teacher is equipped with domain awareness and generalization for evolving domains. To improve teacher predictions, we propose a refine-then-filter based on dropout-driven consistency, which calibrates predictions and removes unreliable guidance. For the adaptation{--}generalization trade-off, we construct a domain-aware teacher by dynamically accumulating cross-domain semantics via incremental PCA, which continuously tracks domain shifts. Experiments show CTTA-T excels baselines."
}
```
