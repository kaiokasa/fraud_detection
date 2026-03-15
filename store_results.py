from sklearn.metrics import classification_report, roc_auc_score

def store_and_print_results(results_df, model_name, y_test, y_pred, y_proba):
   

    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"=== {model_name} Classification Report ===")
    print(f"Class 0 (Non-Fraud): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}")
    print(f"Class 1 (Fraud)    : Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}\n")
    
    fraud_precision = report['1']['precision']
    fraud_recall = report['1']['recall']
    fraud_f1 = report['1']['f1-score']
    roc_auc = roc_auc_score(y_test, y_proba)
    non_fraud_precision = report['0']['precision']
    non_fraud_recall = report['0']['recall']
    non_fraud_f1 = report['0']['f1-score']
    
    results_df.loc[len(results_df)] = [
        model_name,
        non_fraud_precision, non_fraud_recall, non_fraud_f1,
        fraud_precision, fraud_recall, fraud_f1,
        roc_auc
    ]
    
    return results_df