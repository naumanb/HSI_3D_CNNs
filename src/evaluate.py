
from sklearn.metrics import classification_report
from data_processing.data_splitting import get_leave_one_out_splits
import tensorflow as tf
import numpy as np

# Define a custom loss function that ignores the padding in the labels (unlabeled pixels)
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, 0)
    y_true_adjusted = y_true - 1
    y_true_adjusted = tf.where(mask, y_true_adjusted, 0)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_adjusted, y_pred, from_logits=False, axis=-1)
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
def train_and_evaluate(model_builder, num_classes, X, y, epochs, batch_size, reshape=False):
    
    scores = []
    y_true = []
    y_pred = []

    num_samples = X.shape[0]
    input_shape = (X.shape[1], X.shape[2], X.shape[3]) if len(X.shape) > 3 else (X.shape[1], X.shape[2])

    
    train_indices, test_indices = get_leave_one_out_splits(X,y)

    for train_index, test_index in zip(train_indices, test_indices):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_builder(input_shape=input_shape, num_classes=num_classes)

        model.compile(optimizer='adam', loss= masked_sparse_categorical_crossentropy, metrics=[masked_sparse_categorical_accuracy])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        score = model.evaluate(X_test, y_test, verbose=0)
        scores.append(score)

        # Store true labels and predictions for classification report
        y_true.extend(np.ravel(y_test))
        y_pred.extend(np.argmax(model.predict(X_test), axis=-1).ravel())

    # Calculate average performance metrics
    avg_score = np.mean(scores, axis=0)

    # Calculate classification report only for labelled pixels
    filtered_y_true = [value for value in y_true if value != 0]
    filtered_y_pred = [pred for pred, true in zip(y_pred, y_true) if true != 0]
    report = classification_report(filtered_y_true, filtered_y_pred)

    return avg_score, report, y_true, y_pred