import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import metrics
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

def format_decimal(value):
    return "{:.4f}".format(value)

def calculate_far(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    far = fp.sum() / (fp.sum() + tn.sum())
    return far

def result(y_pred, y_test, y_pred_proba, algo, time_to_predict, time_to_train):
    with open('detailed_results.csv', 'a') as outfile:
        outfile.write(f"{algo},")
        outfile.write(f"{format_decimal(time_to_train)},")
        outfile.write(f"{format_decimal(time_to_predict)},")
        outfile.write(f"{format_decimal(metrics.accuracy_score(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.precision_score(y_test, y_pred, average='weighted'))},")
        outfile.write(f"{format_decimal(metrics.recall_score(y_test, y_pred, average='weighted'))},")
        outfile.write(f"{format_decimal(metrics.f1_score(y_test, y_pred, average='weighted'))},")
        outfile.write(f"{format_decimal(metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))},")
        outfile.write(f"{format_decimal(metrics.matthews_corrcoef(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.jaccard_score(y_test, y_pred, average='weighted'))},")
        outfile.write(f"{format_decimal(metrics.cohen_kappa_score(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.hamming_loss(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.zero_one_loss(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.mean_absolute_error(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.mean_squared_error(y_test, y_pred))},")
        outfile.write(f"{format_decimal(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))},")
        outfile.write(f"{format_decimal(metrics.balanced_accuracy_score(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.explained_variance_score(y_test, y_pred)*100)},")
        outfile.write(f"{format_decimal(calculate_far(y_test, y_pred))},")
        outfile.write(f"{format_decimal(metrics.roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'))}\n")

def save_results(accuracy, precision, recall, f1, cm, y_test, y_pred_proba, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # Calculate and plot ROC curve for multiclass
    n_classes = y_pred_proba.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'auc_roc': {str(i): roc_auc[i] for i in range(n_classes)}
    }
    with open(f'evaluation_metrics_{model_name}.json', 'w') as f:
        json.dump(metrics, f)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    predict_start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    predict_time = time.time() - predict_start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    save_results(accuracy, precision, recall, f1, cm, y_test, y_pred_proba, model_name)
    result(y_pred, y_test, y_pred_proba, model_name, predict_time, train_time)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {metrics.roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'):.4f}")
    print(f"False Alarm Rate: {calculate_far(y_test, y_pred):.4f}")
    print(f"Evaluation metrics saved to 'evaluation_metrics_{model_name}.json'")
    print(f"Confusion matrix saved as 'confusion_matrix_{model_name}.png'")
    print(f"ROC curve saved as 'roc_curve_{model_name}.png'")

def main():
    train_data = pd.read_csv('UNSW_NB_15_selected_features_train.csv')
    test_data = pd.read_csv('UNSW_NB_15_selected_features_test.csv')

    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    # CatBoost
    catboost_model = CatBoostClassifier(random_state=42, verbose=False)
    train_and_evaluate(catboost_model, X_train, X_test, y_train, y_test, "CatBoost")

    # LightGBM
    lgbm_model = LGBMClassifier(random_state=42)
    train_and_evaluate(lgbm_model, X_train, X_test, y_train, y_test, "LightGBM")

    # XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42)
    train_and_evaluate(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")

    # Extra Trees
    et_model = ExtraTreesClassifier(random_state=42)
    train_and_evaluate(et_model, X_train, X_test, y_train, y_test, "ExtraTrees")

    print("\nDetailed results saved to 'detailed_results.csv'")

if __name__ == "__main__":
    # Create or clear the detailed_results.csv file with headers
    with open('detailed_results.csv', 'w') as outfile:
        outfile.write("Algorithm,Train Time,Predict Time,Accuracy,Precision,Recall,F1 Score,F0.5 Score,")
        outfile.write("Matthews Correlation Coefficient,Jaccard Score,Cohen's Kappa,Hamming Loss,")
        outfile.write("Zero-One Loss,Mean Absolute Error,Mean Squared Error,Root Mean Squared Error,")
        outfile.write("Balanced Accuracy,Explained Variance Score,False Alarm Rate,AUC-ROC\n")

    main()
