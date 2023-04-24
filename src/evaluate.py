from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
import numpy as np

# Train and evaluate the model using leave-one-out cross-validation
def train_and_evaluate(model_builder, input_shape, num_classes, data, labels, epochs, batch_size):
    loo = LeaveOneOut()
    metrics = []

    for train_idx, test_idx in loo.split(data):

        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        train_data, train_labels = [data[i] for i in train_idx], [labels[i] for i in train_idx]
        test_data, test_labels = [data[i] for i in test_idx], [labels[i] for i in test_idx]

        train_data = np.stack(train_data, axis=0)
        train_labels = np.stack(train_labels, axis=0)


        model = model_builder(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)
        predictions = np.argmax(model.predict(test_data), axis=-1)

        # Calculate performance metrics
        report = classification_report(test_labels.ravel(), predictions.ravel(), output_dict=True, zero_division=0)
        metrics.append(report)

    # Calculate average performance metrics
    avg_metrics = {}
    metric_keys = metrics[0].keys()
    for key in metric_keys:
        if isinstance(metrics[0][key], dict):
            avg_metrics[key] = {}
            for sub_key in metrics[0][key].keys():
                avg_metrics[key][sub_key] = np.mean([m[key][sub_key] for m in metrics])
        else:
            avg_metrics[key] = np.mean([m[key] for m in metrics])

    return avg_metrics