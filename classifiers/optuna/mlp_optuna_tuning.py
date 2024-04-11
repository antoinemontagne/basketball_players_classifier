import os, sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
from classifiers.core_functions import extract_preprocessed_datas, SEED
from classifiers.mlp_classifier import MLPClassifier_torch

import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


def define_model(trial, input_dim):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    out_features = []
    dropouts = []
    activations = []

    for i in range(n_layers):
        out_feature = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        activation = trial.suggest_categorical("activation_l{}".format(i), ["ReLU", "Tanh", "LeakyReLU", "ELU"])

        out_features.append(out_feature)
        dropouts.append(dropout)
        activations.append(activation)

    model = MLPClassifier_torch(input_dim, n_layers, out_features, dropouts, activations)

    return model


def objective(trial):
    # Load, preprocess and split the dataset
    datas_path = ".\\data\\nba_logreg.csv"
    df_features, labels = extract_preprocessed_datas(datas_path, 'TARGET_5Yrs', ['Name'])

    # Create the model path
    current_path = Path.cwd()
    model_path = current_path / "model"
    if not model_path.exists():
        model_path.mkdir(parents=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_features.values, labels.values, test_size=0.33, random_state=SEED)
 
    # Scale the data
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Generate the model
    torch.manual_seed(SEED)
    model = define_model(trial, X_train.shape[1])

    # Define the optimzer, batch size and loss function
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_size = trial.suggest_int("batch_size", 32, 128)
    epochs = trial.suggest_int("epochs", 50, 1000)
    criterion = nn.BCELoss()

    # Train the model
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
    random_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            output_test = model(torch.tensor(X_test, dtype=torch.float32))
            loss_test = criterion(output_test, torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1))

        trial.report(loss_test.item(), epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return loss_test.item()


if __name__ == "__main__":
    current_path = Path.cwd()
    db_path = current_path / "classifiers" / "optuna" / "study"
    if not db_path.exists():
        db_path.mkdir(parents=True)
    db_file = db_path / 'mlp_optu_100_tpe_500_loss.db'
    if not os.path.exists(db_file):
        open(db_file, 'a').close() 
    db_url = f'sqlite:///{db_file}'

    study = optuna.create_study(direction="minimize", storage=db_url)
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