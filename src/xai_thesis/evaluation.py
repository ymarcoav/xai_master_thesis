from sklearn.metrics import classification_report


def evaluate_classifier(model, x_test, y_test) -> dict:
    """Return common classification metrics."""
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions, output_dict=True)
