"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os, sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
from classifiers.core_functions import extract_preprocessed_datas, score_classifier, SEED

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 500
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 19
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Load, preprocess and split the dataset
    datas_path = ".\\data\\nba_logreg.csv"
    df_features, labels = extract_preprocessed_datas(datas_path, 'TARGET_5Yrs', ['Name'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_features.values, labels.values, test_size=0.33, random_state=SEED)
    
    # Scaled the data
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Set the torch seed
    torch.manual_seed(SEED)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X_train, dtype=torch.float32))
        loss = nn.BCELoss()(output, torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_test = model(torch.tensor(X_test, dtype=torch.float32))
            output_test_binary = (output_test >= 0.5).float().flatten()
            acc = accuracy_score(y_test, output_test_binary)

        trial.report(acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return acc


if __name__ == "__main__":
    current_path = Path.cwd()
    db_file = current_path / 'optuna_study.db'
    if not os.path.exists(db_file):
        open(db_file, 'a').close()
    db_url = f'sqlite:///{db_file}'

    study = optuna.create_study(direction="maximize", storage=db_url)
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))