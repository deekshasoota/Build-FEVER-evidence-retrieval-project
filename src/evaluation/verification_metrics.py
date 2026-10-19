"""Classification metrics wrapper."""
def classification_metrics(gold: list[str], predicted: list[str]) -> dict:
    if not gold or len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must be non-empty and have equal lengths")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    return {"accuracy": float(accuracy_score(gold, predicted)),
            "classification_report": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(), "labels": labels}
