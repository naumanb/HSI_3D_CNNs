from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np

# Define a custom loss function that ignores the padding in the labels (unlabeled pixels)
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, 0)
    y_true_adjusted = y_true - 1
    y_true_adjusted = tf.where(mask, y_true_adjusted, 0)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_adjusted, y_pred, from_logits=False)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

# Define a custom accuracy metric that ignores the padding in the labels (unlabeled pixels)
def masked_sparse_categorical_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, 0)
    y_true_adjusted = y_true - 1
    y_true_adjusted = tf.where(mask, y_true_adjusted, 0)

    matches = tf.cast(tf.equal(y_true_adjusted, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.float32)
    mask = tf.cast(mask, matches.dtype)
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)

# Train and evaluate the model using leave-one-out cross-validation
def train_and_evaluate(model_builder, input_shape, num_classes, data, labels, epochs, batch_size):
    loo = LeaveOneOut()
    scores = []
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(data):

        train_data = [data[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_data = data[test_idx[0]]
        test_labels = labels[test_idx[0]]

        # Stack and reshape data
        train_data = np.stack(train_data, axis=0)
        test_data = np.expand_dims(test_data, axis=0)

        # Stack and reshape labels
        train_labels = np.stack(train_labels, axis=0)
        test_labels = np.expand_dims(test_labels, axis=0)

        model = model_builder(input_shape, num_classes)
        model.compile(optimizer='adam', loss= masked_sparse_categorical_crossentropy, metrics=[masked_sparse_categorical_accuracy])

        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)
        score = model.evaluate(test_data, test_labels, verbose=0)
        scores.append(score)

        print("test_labels shape:", test_labels.shape)
        print("Predicted labels shape:", np.argmax(model.predict(test_data), axis=1).shape)

        # Store true labels and predictions for classification report
        y_true.extend(np.ravel(test_labels))
        y_pred.extend(np.argmax(model.predict(test_data), axis=-1).ravel())

    # Calculate average performance metrics
    avg_score = np.mean(scores, axis=0)

    # Calculate classification report
    report = classification_report(y_true, y_pred)

    return avg_score, report, y_true, y_pred