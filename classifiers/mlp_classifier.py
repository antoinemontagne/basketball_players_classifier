from pathlib import Path
import os, sys
import numpy as np
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

LEARNING_RATE = 0.006844259372316053
NB_EPOCHS = 400
HIDDEN_LAYER_SIZE = 26
DROPOUT = 0.46489983932774215


class MLPClassifier_torch(nn.Module):
    '''MLP classifier using PyTorch'''
    def __init__(self, hidden_layer_size=HIDDEN_LAYER_SIZE, dropout_proba=DROPOUT):
        '''Initialize the model
        Args:
            hidden_layer_size (int): size of the hidden layer
        '''
        super(MLPClassifier_torch, self).__init__()
        self.fc1 = nn.Linear(19, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)
        self.dropout = nn.Dropout(dropout_proba)

    def forward(self, x):
        '''Forward pass of the model
        Args:
            x (torch.Tensor): input data
        '''
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    

class MLPClassifier():
    '''Training and evaluation for MLP classifier'''
    def __init__(self, lr, nb_epochs, path, X_test=None, y_test=None, hidden_layer_size=HIDDEN_LAYER_SIZE, dropout_proba=DROPOUT):
        '''Initialize the model
        Args:
            lr (float): learning rate
            nb_epochs (int): number of epochs
            path (Path): path to save the model
            X_test (np.array): test data
            y_test (np.array): test target
            hidden_layer_size (int): size of the hidden layer
        '''
        self.mlp = MLPClassifier_torch(hidden_layer_size, dropout_proba)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.nb_epochs = nb_epochs
        self.path = path
        self.lr = lr
        if X_test is not None:
            self.X_test = MinMaxScaler().fit_transform(X_test)
        else:
            self.X_test = X_test
        self.y_test = y_test

    def reset_model(self):
        '''Reset the model to its initial state'''
        self.mlp = MLPClassifier_torch()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)

    def fit(self, X, y):
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
            output = self.mlp(torch.tensor(X, dtype=torch.float32))
            loss = self.criterion(output, torch.tensor(y, dtype=torch.float32).reshape(-1, 1))
            loss.backward()
            losses[0].append(loss.item())
            self.optimizer.step()

            if self.X_test is not None:
                self.mlp.eval()
                with torch.no_grad():
                    output_test = self.mlp(torch.tensor(self.X_test, dtype=torch.float32))
                    loss_test = self.criterion(output_test, torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1))
                    losses[1].append(loss_test.item())

                if loss_test < best_valid_loss:
                    best_valid_loss = loss_test
                    best_epoch = epoch
                    self.save()

            if epoch % (self.nb_epochs // 10) == 0:
                print(f"Epoch {epoch} loss: {loss.item()}")

        if self.X_test is not None:
            plt.plot(losses[0], label='Train loss')
            plt.plot(losses[1], label='Test loss')
            plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best epoch {best_epoch}')
            plt.legend()
            plt.show()

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
        '''Save the model to the disk'''
        torch.save(self.mlp.state_dict(), self.path / "mlp_model.pth")

    def load(self):
        '''Load the model from the disk'''
        self.mlp.load_state_dict(torch.load(self.path / "mlp_model.pth"))
    

def main():
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
    
    # Create the model and the pipeline
    torch.manual_seed(SEED)
    mlp = MLPClassifier(LEARNING_RATE, NB_EPOCHS, model_path, X_test, y_test, HIDDEN_LAYER_SIZE, DROPOUT)
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('mlp_classifier', mlp)
    ])
    
    # Train the pipeline    
    pipeline.fit(X_train, y_train)

    # Test the model
    best_mlp = MLPClassifier(LEARNING_RATE, NB_EPOCHS, model_path, X_test, y_test, HIDDEN_LAYER_SIZE, DROPOUT)
    best_mlp.load()
    best_pipeline = Pipeline([
        ('scaler', MinMaxScaler().fit(X_train)),
        ('best_mlp_classifier', best_mlp)
    ])
    output_binary = best_pipeline.predict(X_test)
    print(confusion_matrix(y_test, output_binary))
    print(recall_score(y_test, output_binary))
        
    # Cross-validation
    print("Cross-validation")
    X = MinMaxScaler().fit_transform(df_features.values)
    mlp = MLPClassifier(LEARNING_RATE, 200, model_path, X_test=None, y_test=None)
    _, _ = score_classifier(X, mlp, labels, 3, True, True)


if __name__ == '__main__':
    main()