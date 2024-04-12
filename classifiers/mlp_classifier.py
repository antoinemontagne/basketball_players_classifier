from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from classifiers.core_functions import extract_preprocessed_datas, score_classifier, SEED


parser = argparse.ArgumentParser(
    prog="MLP classifier for binary classification"
)
parser.add_argument(
    "--cross_validation",
    action="store_true",
    help="Choose to perform cross-validation or not",
)


class MLPClassifier_torch(nn.Module):
    '''MLP classifier with modulable size using PyTorch'''
    def __init__(self, in_features, nb_hidden_layers, hidden_layer_sizes, hidden_dropout_probas, hidden_activations):
        '''Initialize the model
        Args:
            in_features (int): number of input features
            nb_hidden_layers (int): number of hidden layers
            hidden_layer_sizes (list): size of the hidden layers
            hidden_dropout_probas (list): dropout probability for each hidden layer
            hidden_activations (list): activation function for each hidden layer
        '''
        super(MLPClassifier_torch, self).__init__()
        self.layers = []
        self.nb_hidden_layers = nb_hidden_layers
        self.hidden_activations = hidden_activations
        self.hidden_dropout_probas = hidden_dropout_probas
        torch.manual_seed(SEED)
        for i in range(nb_hidden_layers):
            new_linear_layer = nn.Linear(in_features, hidden_layer_sizes[i])
            nn.init.xavier_uniform_(new_linear_layer.weight)
            nn.init.zeros_(new_linear_layer.bias)
            self.layers.append(new_linear_layer)
            in_features = hidden_layer_sizes[i]
        
        last_layer = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        self.layers.append(last_layer)
        self.layers = nn.ParameterList(self.layers)  # To grant the optimizer access to the model's layers.

    def forward(self, x):
        '''Forward pass of the model
        Args:
            x (torch.Tensor): input data
        '''
        for i in range(self.nb_hidden_layers):
            x = self.layers[i](x)
            activation_layer = getattr(nn, self.hidden_activations[i])
            x = activation_layer()(x)
            x = nn.Dropout(self.hidden_dropout_probas[i])(x)

        x = self.layers[-1](x)
        x = nn.Sigmoid()(x)
        return x
    

class MLPClassifier():
    '''Training and evaluation for MLP classifier'''
    def __init__(self, 
                 in_features, 
                 nb_hidden_layers, 
                 hidden_layer_sizes, 
                 hidden_dropout_probas, 
                 hidden_activations, 
                 optimizer_name, 
                 lr, 
                 weight_decay, 
                 nb_epochs, 
                 batch_size, 
                 path, 
                 X_test, 
                 y_test,
    ):
        '''Initialize the model
        Args:
            in_features (int): number of input features
            nb_hidden_layers (int): number of hidden layers
            hidden_layer_sizes (list): size of the hidden layers
            hidden_dropout_probas (list): dropout probability for each hidden layer
            hidden_activations (list): activation function for each hidden layer
            optimizer_name (str): optimizer name
            lr (float): learning rate
            nb_epochs (int): number of epochs
            path (Path): path to save the model
        '''
        self.mlp = MLPClassifier_torch(in_features, nb_hidden_layers, hidden_layer_sizes, hidden_dropout_probas, hidden_activations)
        torch.save(self.mlp.state_dict(), path / "init_mlp_model.pth")
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.optimizer = getattr(optim, self.optimizer_name)(self.mlp.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.path = path
        self.X_test = X_test
        self.y_test = y_test

    def reset_model(self):
        '''Resetting the model weights and the state of the optimizer'''
        init_weights = torch.load(self.path / "init_mlp_model.pth")
        self.mlp.load_state_dict(init_weights)
        self.optimizer = getattr(optim, self.optimizer_name)(self.mlp.parameters(), lr=self.lr)

    def set_nb_epochs(self, nb_epochs):
        '''Set the number of epochs
        Args:
            nb_epochs (int): number of epochs
        '''
        self.nb_epochs = nb_epochs

    def fit(self, X_train, y_train):
        '''Train the model
        Args:
            X_train (np.array): input data
            y_train (np.array): target data
        '''
        losses = [[], []]
        best_valid_loss = float('inf')
        best_epoch = 0
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
        random_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=random_sampler)
        for epoch in range(self.nb_epochs):
            self.mlp.train()
            batch_losses = []
            torch.manual_seed(SEED + epoch)
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.mlp(X_batch)
                loss = self.criterion(output, y_batch)
                batch_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            losses[0].append(sum(batch_losses) / len(batch_losses))

            # Evaluate the model on the test set
            self.mlp.eval()
            with torch.no_grad():
                output_test = self.mlp(torch.tensor(self.X_test, dtype=torch.float32))
                loss_test = self.criterion(output_test, torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1))
                losses[1].append(loss_test.item())

            if loss_test < best_valid_loss:
                best_valid_loss = loss_test
                best_epoch = epoch
                # Save the best model => early-stopping
                self.save()
            
            if epoch % (self.nb_epochs // 10) == 0:
                print(f"Epoch {epoch} - Train loss: {losses[0][-1]} - Test loss: {losses[1][-1]}")

        plt.clf()
        plt.plot(losses[0], label='Train loss')
        plt.plot(losses[1], label='Test loss')
        plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best epoch {best_epoch}')
        plt.legend()
        plt.savefig(self.path / 'losses.png')

    def predict(self, X):
        '''Predict the label of the input data
        Args:
            X (np.array): input data
        Returns:
            output_binary (np.array): binary output
        '''
        self.mlp.eval()
        with torch.no_grad():
            output = self.mlp(torch.tensor(X, dtype=torch.float32))
            output_binary = (output >= 0.5).float()
            return output_binary.flatten().numpy()
        
    def predict_proba(self, X):
        '''Predict the probability of the input data to belong to the positive class
        Args:
            X (np.array): input data
        Returns:
            new_2_output (np.array): probability output
        '''
        self.mlp.eval()
        with torch.no_grad():
            output = self.mlp(torch.tensor(X, dtype=torch.float32))
            zeros_column = torch.zeros(output.size(0), 1)
            new_2_output = torch.cat((zeros_column, output), dim=1)
            return new_2_output.float().numpy()
        
    def save(self):
        '''Save the model for inference'''
        torch.save(self.mlp.state_dict(), self.path / "mlp_model.pth")

    def load(self):
        '''Load the model for inference'''
        self.mlp.load_state_dict(torch.load(self.path / "mlp_model.pth"))


def read_parameters(parameters_path):
    '''Read the parameters from the parameters file
    Args:
        parameters_path (Path): path to the parameters file
    Returns:
        parameters (dict): parameters
    '''
    parameters = {}
    with open(parameters_path, "r") as file:
        for line in file.readlines():
            key, value = line.split(":")
            if key == "optimizer_name":
                parameters[key] = value.strip()
            elif key in ["lr", "weight_decay"]:
                parameters[key] = float(value)
            elif key in ["nb_epochs", "batch_size"]:
                parameters[key] = int(value)
            elif "activations" == key:
                parameters[key] = [x.strip() for x in value.split(",")]
            elif key == "n_neurons":
                parameters[key] = [int(x) for x in value.split(",")]
            else:
                parameters[key] = [float(x) for x in value.split(",")]
    return parameters
    

def main():
    # Parse the arguments
    args = parser.parse_args()
    cross_validation = args.cross_validation

    # Load, preprocess and split the dataset
    datas_path = ".\\data\\nba_logreg.csv"
    df_features, labels = extract_preprocessed_datas(datas_path, 'TARGET_5Yrs', ['Name'])

    # Create the model path
    current_path = Path.cwd()
    model_path = current_path / "classifiers" / "model_weights"
    if not model_path.exists():
        model_path.mkdir(parents=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df_features.values, labels.values, test_size=0.33, random_state=SEED)
    
    # Get the parameters from the parameters file
    parameters = read_parameters(current_path / "classifiers" / "parameters.txt")

    # Create the model
    scaler = MinMaxScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp = MLPClassifier(X_train.shape[1], 
                            len(parameters["n_neurons"]), 
                            parameters["n_neurons"], 
                            parameters["dropouts"], 
                            parameters["activations"],
                            parameters["optimizer_name"], 
                            parameters["lr"], 
                            parameters["weight_decay"],
                            parameters["nb_epochs"], 
                            parameters["batch_size"],
                            model_path,
                            X_test_scaled,
                            y_test,
    )

    pipeline = Pipeline([
        ('scaler', scaler),
        ('mlp_classifier', mlp)
    ])

    # Train & Evaluate the model
    if not cross_validation:
        # Train the pipeline
        print("Training the model")
        pipeline.fit(X_train, y_train)

        # Test the best model
        # Load the best model via early-stopping
        pipeline[1].load()
        output_binary = pipeline.predict(X_test)
        print(confusion_matrix(y_test, output_binary))
        print(recall_score(y_test, output_binary))

        # Train with reset model
        print("Training the model with reset model")
        pipeline[1].reset_model()
        pipeline.fit(X_train, y_train)
        
        # Test the best model
        # Load the best model via early-stopping
        pipeline[1].load()
        output_binary = pipeline.predict(X_test)
        print(confusion_matrix(y_test, output_binary))
        print(recall_score(y_test, output_binary))

    
    else:
        print("Cross-validation")
        print("Testing for several epochs")
        epochs = [(i+1)*50 for i in range(8, 15)]
        all_recalls_epochs = []
        for epoch in epochs:
            print(f"Testing for {epoch} epochs")
            pipeline[1].set_nb_epochs(epoch)  # Set the new epoch number before the next CV
            _, recalls = score_classifier(df_features.values, pipeline, labels, 3, False, True, False)
            all_recalls_epochs.append(sum(recalls) / len(recalls))
            print(f"Mean recall score: {all_recalls_epochs[-1]}\n")
        plt.clf()
        plt.plot(epochs, all_recalls_epochs)
        plt.xlabel("Number of epochs")
        plt.ylabel("Mean recall score")
        plt.savefig(model_path / 'epochs.png')
        # pipeline[1].set_nb_epochs(550)  # Set the new epoch number before the next CV
        # _, _ = score_classifier(df_features.values, pipeline, labels, 3, True, True, True)


if __name__ == '__main__':
    main()