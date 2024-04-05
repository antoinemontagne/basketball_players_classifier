import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

SEED = 0


def score_classifier(X, classifier, labels, n_splits, confusion=True, mlp=False, printing=True):
    '''Score the classifier using stratified stratified k-fold cross-validation
    Args:
        X (np.array): features
        classifier (object): classifier object
        labels (pd.Series): target
        n_splits (int): number of splits
        confusion (bool): display confusion matrix
        mlp (bool): use mlp
        printing (bool): display the results
    Returns:
        predictions (pd.DataFrame): predictions
        recalls (list): recall scores
    '''
    kf = StratifiedKFold(n_splits=n_splits,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    predictions = []
    recalls = []
    for training_ids,test_ids in kf.split(X, labels):
        training_set = X[training_ids]
        training_labels = labels[training_ids]
        test_set = X[test_ids]
        test_labels = labels[test_ids]

        if mlp:
            classifier.reset_model()
        classifier.fit(training_set,training_labels.values)
        predicted_labels = classifier.predict(test_set) 
        predicted_proba = classifier.predict_proba(test_set)[:, 1]
        confusion_mat+=confusion_matrix(test_labels, predicted_labels)
        predictions.append(pd.DataFrame({'labels':test_labels, 'predictions':predicted_labels, 'predictions_proba':predicted_proba}))
        recalls.append(recall_score(test_labels, predicted_labels))

    predictions = pd.concat(predictions).sort_index()
    if confusion:
        print(confusion_mat)
        
    if printing:
        print(f"All recall scores: {recalls}")
        print(f"Mean recall score: {sum(recalls) / len(recalls)}")
    return predictions, recalls


def extract_preprocessed_datas(datas_path, label_name, features_to_drop):
    '''Extract and preprocess the datas
    Args:
        datas_path (str): path to the datas
        label_name (str): name of the target
        features_to_drop (list): list of features to drop
    Returns:
        df_features (pd.DataFrame): features
        labels (pd.Series): target
    '''
    # Load dataset
    df = pd.read_csv(datas_path)

    # extract labels, and features
    labels = df[label_name]
    columns_to_drop = features_to_drop + [label_name]
    df_features = df.drop(columns=columns_to_drop)

    # Check column with nan values
    is_nans = df_features.isnull().any()
    columns_with_nan = []
    for i, is_nan in enumerate(is_nans):
        if is_nan:
            columns_with_nan.append(i)

    # Fill nan values with mean in columns with nan values
    dataframe_columns = list(df_features.columns)
    for idx_column in columns_with_nan:
        column = dataframe_columns[idx_column]
        df_features[column].fillna(value=df_features[column].mean(), inplace=True) 

    return df_features, labels


def plot_roc_auc(predictions):
    '''Plot the roc curve
    Args:
        predictions (pd.DataFrame): predictions
    '''
    fpr, tpr, _ = roc_curve(predictions['labels'], predictions['predictions_proba'])
    auc_score = roc_auc_score(predictions['labels'], predictions['predictions_proba'])

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def main():
    datas_path = "../data/nba_logreg.csv"

    # Extract and normalize datas
    df_features, labels = extract_preprocessed_datas(datas_path, 'TARGET_5Yrs', ['Name'])

    # Scale the datas
    X = MinMaxScaler().fit_transform(df_features)

    #example of scoring with support vector classifier
    svm = SVC()
    neigh = KNeighborsClassifier(n_neighbors=27)


if __name__ == "__main__":
    main()


