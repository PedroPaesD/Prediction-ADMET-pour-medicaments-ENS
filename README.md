# ADMET Prediction for Drug Discovery

This repository contains the solution to a challenge focused on predicting the ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of chemical compounds based on their SMILES (Simplified Molecular Input Line Entry System) representations. The goal is to predict whether three specific properties of a drug are within an optimal range using machine learning techniques.

## Problem Overview

The challenge consists of predicting the profiles of three ADMET properties from chemical compounds encoded as SMILES strings. For each property, the task is a binary classification problem (1 for optimal and 0 for non-optimal). This task is approached as a multi-label classification problem where the labels are binary values for each property. 

### Properties to Predict:
- **Y1**: Represents the first ADMET property.
- **Y2**: Represents the second ADMET property.
- **Y3**: Represents the third ADMET property.

## Data

The data consists of the following files:

- `X_train.csv`: Training set with molecule IDs and SMILES representations.
- `y_train.csv`: Labels for the training set, containing the binary values for the three properties.
- `X_test.csv`: Test set with molecule IDs and SMILES representations for prediction.
- `random_submission_example.csv`: Example of the submission format.
- `supplementary_files`: Contains a Jupyter notebook introducing the challenge and the reference score.

### Columns:
- **ID**: Unique identifier for each molecule.
- **SMILES**: SMILES representation of the molecule.
- **Y1, Y2, Y3**: Binary labels for the three ADMET properties.

## Methodology

The solution involves the following steps:

### 1. Data Preprocessing:
The SMILES strings are converted into molecular fingerprints using RDKit's `RDKFingerprint` method. This step encodes the chemical structure into a binary vector that captures the presence of specific chemical features, such as functional groups, atom types, and bonds.

### 2. Model Development:
We applied multiple machine learning models and evaluated them for the ADMET prediction:

#### Classical Models:
- **Decision Trees**: We trained separate decision tree classifiers for each property (Y1, Y2, Y3).
- **Support Vector Machines (SVM)**: Using `BinaryRelevance` to apply SVM for multi-label classification.
- **K-Nearest Neighbors (KNN)**: Used in conjunction with `MultiOutputClassifier` for multi-label prediction.

#### Deep Learning:
- **Neural Network**: A fully connected feedforward neural network implemented using PyTorch was trained for the prediction of all three properties simultaneously. The network was trained using a binary cross-entropy loss function.

### 3. Model Evaluation:
The performance of the models was evaluated using:
- **F1 Score**: The micro-averaged F1 score was used to measure the balance between precision and recall.
- **Precision and Recall**: These metrics were computed for each of the three properties.

### 4. Hyperparameters and Training:
- For the neural network model, the optimizer used was Stochastic Gradient Descent (SGD), and training was carried out for 400 epochs.
- The `BCEWithLogitsLoss` was used for training the deep learning model, which is suitable for multi-label classification.

### 5. Submission:
The final predictions for the test set were generated and saved in a CSV file following the format specified in the challenge.

## Installation

You can install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Required Libraries:
- `numpy`
- `pandas`
- `rdkit`
- `scikit-learn`
- `skmultilearn`
- `torch`
- `matplotlib`
- `torchmetrics`
- `torcheval`

## Running the Code

To train and evaluate the models, simply run the Jupyter notebook or Python script provided. The script will load the data, preprocess it, train the models, evaluate their performance, and generate the predictions for the test set.

### Example usage:
```python
python train_model.py
```

## Results

The models were evaluated using a micro-average F1 score across all three properties. The deep learning model with PyTorch performed best in this challenge, achieving the highest F1 score on the test set.

## Conclusion

This project demonstrates how machine learning, especially deep learning, can be used to predict important drug properties (ADMET) efficiently. By transforming SMILES strings into molecular fingerprints and applying a variety of models, we can significantly accelerate the drug discovery process and reduce the associated costs.
