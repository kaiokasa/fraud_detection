from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

def print_crossval_results(model, X_train, y_train, cv=5):

    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)

    report_cv = classification_report(y_train, y_pred_cv, output_dict=True)

    print("=== Cross-Validation Classification Report ===")
    print(f"Class 0 (Non-Fraud): Precision={report_cv['0']['precision']:.3f}, "
          f"Recall={report_cv['0']['recall']:.3f}, F1={report_cv['0']['f1-score']:.3f}")
    print(f"Class 1 (Fraud)    : Precision={report_cv['1']['precision']:.3f}, "
          f"Recall={report_cv['1']['recall']:.3f}, F1={report_cv['1']['f1-score']:.3f}")
