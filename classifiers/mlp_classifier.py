from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from classifiers.core_functions import extract_preprocessed_datas, score_classifier, SEED


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
        for i in range(nb_hidden_layers):
            self.layers.append(nn.Linear(in_features, hidden_layer_sizes[i]))
            in_features = hidden_layer_sizes[i]
        
        self.layers.append(nn.Linear(in_features, 1))
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
    def __init__(self, in_features, nb_hidden_layers, hidden_layer_sizes, hidden_dropout_probas, hidden_activations, optimizer_name, lr, nb_epochs, path, X_test, y_test):
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
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.optimizer = getattr(optim, self.optimizer_name)(self.mlp.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.nb_epochs = nb_epochs
        self.path = path
        self.X_test = X_test
        self.y_test = y_test

    def reset_model(self):
        '''Resetting the model weights and the state of the optimizer'''
        for name, modules in self.mlp.named_children():
            for module in modules:
                module.reset_parameters()
        self.optimizer = getattr(optim, self.optimizer_name)(self.mlp.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        '''Train the model
        Args:
            X (np.array): input data
            y (np.array): target data
        '''
        print("Training the model")
        losses = [[], []]
        best_valid_loss = float('inf')
        best_epoch = 0
        for epoch in range(self.nb_epochs):
            self.mlp.train()
            self.optimizer.zero_grad()
            output = self.mlp(torch.tensor(X_train, dtype=torch.float32))
            loss = self.criterion(output, torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
            loss.backward()
            losses[0].append(loss.item())
            self.optimizer.step()

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
                print(f"Epoch {epoch} loss: {loss.item()}")

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
    

def main():
    torch.manual_seed(SEED)
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
    
    # Set the parameters by reading the parameters file
    parameters = {}
    with open(current_path / "classifiers" / "parameters.txt", "r") as file:
        for line in file.readlines():
            key, value = line.split(":")
            if "optimizer_name" == key:
                parameters[key] = value.strip()
            elif key == "lr":
                parameters[key] = float(value)
            elif key == "nb_epochs":
                parameters[key] = int(value)
            elif "activations" == key:
                parameters[key] = [x.strip() for x in value.split(",")]
            elif key == "n_neurons":
                parameters[key] = [int(x) for x in value.split(",")]
            else:
                parameters[key] = [float(x) for x in value.split(",")]

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
                             parameters["nb_epochs"], 
                             model_path,
                             X_test_scaled,
                             y_test,
    )

    pipeline = Pipeline([
        ('scaler', scaler),
        ('mlp_classifier', mlp)
    ])
    
    # Train the pipeline    
    pipeline.fit(X_train, y_train)

    # Test the model
    best_mlp = MLPClassifier(X_train.shape[1], 
                             len(parameters["n_neurons"]), 
                             parameters["n_neurons"], 
                             parameters["dropouts"], 
                             parameters["activations"], 
                             parameters["optimizer_name"], 
                             parameters["lr"], 
                             parameters["nb_epochs"], 
                             model_path,
                             X_test_scaled,
                             y_test,
    )
    best_mlp.load()
    best_pipeline = Pipeline([
        ('scaler', scaler),
        ('best_mlp_classifier', best_mlp)
    ])
    output_binary = best_pipeline.predict(X_test)
    print(confusion_matrix(y_test, output_binary))
    print(recall_score(y_test, output_binary))
        
    # # Cross-validation
    # print("Cross-validation")

    # # Create the model
    # scaler = MinMaxScaler().fit(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # mlp = MLPClassifier(X_train.shape[1], 
    #                          len(parameters["n_neurons"]), 
    #                          parameters["n_neurons"], 
    #                          parameters["dropouts"], 
    #                          parameters["activations"], 
    #                          parameters["optimizer_name"], 
    #                          parameters["lr"], 
    #                          parameters["nb_epochs"], 
    #                          model_path,
    #                          X_test_scaled,
    #                          y_test,
    # )

    # pipeline = Pipeline([
    #     ('scaler', scaler),
    #     ('mlp_classifier', mlp)
    # ])

    # _, _ = score_classifier(df_features.values, pipeline, labels, 3, True, True)


if __name__ == '__main__':
    main()