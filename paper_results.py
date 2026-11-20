"""
Numbers reported in Bai et al., IEEE Trans. Cybernetics 44(10), 2014,
for the two datasets reproduced here (Iris, Wine).

Source: open-access author manuscript, https://pmc.ncbi.nlm.nih.gov/articles/PMC4883115
  Table II  - dataset sizes          (Iris 75 train / 75 test, Wine 89 / 89)
  Table III - selected parameters    (5-fold CV on the training set)
  Tables IV-VII - training / testing accuracy (%) over 20 trials
  Table VIII - number of support vectors
"""

SPLIT = {"iris": 75, "wine": 89}  # number of training samples (rest = test)

# Table III. gaussian -> (C, sigma); poly -> (C, m); sigmoid / sinusoid -> C
PAPER_PARAMS = {
    "iris": {
        ("SVM", "gaussian"): (10, 1),     ("SVM", "poly"): (1, 3),
        ("UELM", "gaussian"): (500, 2),   ("UELM", "poly"): (10, 3),
        ("UELM", "sigmoid"): (1000, None), ("UELM", "sinusoid"): (500, None),
        ("SELM", "gaussian"): (1, 0.5),   ("SELM", "poly"): (2, 2),
        ("SELM", "sigmoid"): (1000, None), ("SELM", "sinusoid"): (1000, None),
    },
    "wine": {
        ("SVM", "gaussian"): (5, 1),      ("SVM", "poly"): (1, 3),
        ("UELM", "gaussian"): (1, 2),     ("UELM", "poly"): (0.5, 1),
        ("UELM", "sigmoid"): (2, None),   ("UELM", "sinusoid"): (1, None),
        ("SELM", "gaussian"): (5, 0.5),   ("SELM", "poly"): (10, 2),
        ("SELM", "sigmoid"): (1000, None), ("SELM", "sinusoid"): (5, None),
    },
}

# Tables IV-VII: (train acc, test acc mean, test acc std) in %
PAPER_ACC = {
    "iris": {
        ("SVM", "gaussian"): (100.0, 93.33, 0.0),
        ("UELM", "gaussian"): (100.0, 97.33, 0.0),
        ("SELM", "gaussian"): (98.40, 97.27, 1.60),
        ("SVM", "poly"): (100.0, 96.00, 0.0),
        ("UELM", "poly"): (100.0, 97.33, 0.0),
        ("SELM", "poly"): (99.00, 97.47, 1.66),
        ("UELM", "sigmoid"): (98.67, 97.20, 0.00),
        ("SELM", "sigmoid"): (97.00, 97.40, 0.66),
        ("UELM", "sinusoid"): (98.60, 96.20, 0.01),
        ("SELM", "sinusoid"): (97.27, 97.33, 0.42),
    },
    "wine": {
        ("SVM", "gaussian"): (100.0, 97.75, 0.0),
        ("UELM", "gaussian"): (100.0, 98.89, 0.0),
        ("SELM", "gaussian"): (100.0, 97.92, 0.82),
        ("SVM", "poly"): (100.0, 97.75, 0.0),
        ("UELM", "poly"): (100.0, 98.88, 0.0),
        ("SELM", "poly"): (99.44, 97.46, 1.89),
        ("UELM", "sigmoid"): (100.0, 99.16, 0.01),
        ("SELM", "sigmoid"): (100.0, 99.16, 0.70),
        ("UELM", "sinusoid"): (100.0, 99.10, 0.01),
        ("SELM", "sinusoid"): (100.0, 99.10, 0.84),
    },
}

# Table VIII: number of support vectors (SVM and sparse ELM only;
# unified ELM is dense and uses all training samples)
PAPER_NSV = {
    "iris": {("SVM", "gaussian"): 29, ("SVM", "poly"): 23,
             ("SELM", "gaussian"): 54.45, ("SELM", "poly"): 38.4,
             ("SELM", "sigmoid"): 46.8, ("SELM", "sinusoid"): 47.4},
    "wine": {("SVM", "gaussian"): 72, ("SVM", "poly"): 46,
             ("SELM", "gaussian"): 149.05, ("SELM", "poly"): 49.9,
             ("SELM", "sigmoid"): 54.55, ("SELM", "sinusoid"): 53.25},
}
