# AEJaya-DE-main
AEJaya+DE: Feature Selection for Intrusion Detection Systems

This repository contains the source code and datasets used in the paper "AEJaya+DE: An Adaptive Optimization Technique for Feature Selection in Intrusion Detection Systems."
Files and Folders

    AEJaya+DE_NSLKDD.py: Implementation of AEJaya+DE for the NSL-KDD dataset.
    AEJaya+DE_UNSW.py: Implementation of AEJaya+DE for the UNSW-NB15 dataset.
    classification.py: Classification script for evaluating the selected features.
    KDDTrain+.txt and KDDTest+.txt: Training and testing subsets of the NSL-KDD dataset.
    UNSW_NB15_training-set.csv: Training set of the UNSW-NB15 dataset.
    dataset.zip: Zipped folder containing the full datasets used for training and testing.
    README.md: This file.

Dependencies

    Python 3.8+
    Required libraries:
        numpy
        pandas
        scikit-learn
        matplotlib
        xgboost
        lightgbm
        catboost

Install dependencies using:

pip install -r requirements.txt

Usage

    Clone the repository:

git clone https://github.com/fawzia-omer/AEJaya-DE-main.git
cd AEJaya-DE-main

Run the feature selection scripts:

    For NSL-KDD:

python AEJaya+DE_NSLKDD.py

For UNSW-NB15:

    python AEJaya+DE_UNSW.py

Evaluate the classification results:

    python classification.py

Datasets

    NSL-KDD: Provided in KDDTrain+.txt and KDDTest+.txt.
    UNSW-NB15: Provided in UNSW_NB15_training-set.csv and included in dataset.zip.
