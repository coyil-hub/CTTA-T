# Continual Test-Time Adaptation for Domain Shift Text Understanding

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Obtain the Source Pre-trained Models

In this paper, we employed two backbone models: **base** and **xtune**. You can obtain the following models:

- **Base model**: [XTREME](https://github.com/google-research/xtreme)
- **XTune model**: [xTune](https://github.com/bozheng-hit/xTune)

## Obtain the Datasets
Click [here](https://drive.google.com/file/d/1Zpi1doMIg09ODWYRNhEKz-Pfs7TNm94c/view?usp=drive_link) to download the dataset.

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

### Hyperparameters:

**gamma** and **tau** are hyperparameters. For the **base** model, gamma is 0.4 and tau is 1.2. For the **xTune** model, gamma is 0.4 and tau is 0.6.

### Learning Rate:

 For **[learning_rate]**, we use 1e-5.

### Sequence Types:

We provide scripts of standard short sequences (Order 1) and standard long sequences (Order 4). If you wish to test other task orders, you can modify the sequence in the `continual_short_1.sh` or `continual_long_1.sh` scripts.
