import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import tensorflow as tf
from Utils import preprocessutils


# create a plot function
def plot_predictions(train_data=None,
                     train_labels=None,
                     test_data=None,
                     test_labels=None,
                     predictions=None):
    """
    Plots training data
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return:
    """

    plt.figure(figsize=(10, 7))
    # plot training data in blue
    plt.scatter(train_data, train_labels, c='b', label="Train data")
    # plot testing data in green
    plt.scatter(test_data, test_labels, c='g', label="Test data")
    # plot models pred in red
    plt.scatter(test_data, predictions, c='r', label="predictions")
    plt.legend()
    plt.show()


def data_visualizing(x_train, y_train, X_test, y_test):
    # Visualizing the data
    print("X_train", x_train)
    plt.figure(figsize=(10, 7))
    # training data in blue
    plt.scatter(x_train, y_train, c="b", label="Training data")
    # training data in blue
    plt.scatter(X_test, y_test, c="g", label="Testing data")
    plt.legend()
    plt.show()


def plot_decision_boundary(model, X, y):
    # Define the axis boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (make predictions o these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # ckeck for multiclass
    if len(y_pred[0]) > 1:
        print("multiclass classification")
        # reshape our prediction to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape((xx.shape))
    else:
        print("binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # plot the decision boundaries
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_confusion_matrix(confusion_matrix, classes=None, figsize=(15, 15), text_size=10):
    # normalize confusion matrix
    cm_norm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print(cm_norm)

    n_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    # create a matrix plot
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(confusion_matrix.shape[0])

    # Label the axis
    ax.set(title="Confusion matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # set the threshold for diff colors
    threshold = (confusion_matrix.max() + confusion_matrix.min()) / 2.

    # plot the text
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, f"{confusion_matrix[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > threshold else "black",
                 size=text_size)

    # set x-axis label to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    plt.savefig("conf_mat.png")
    plt.show()


def plot_random_image(model, images, true_labels, classes):
    """
    Plot random images and labels it with prediction and truth label

    :param model:
    :param images:
    :param true_labels:
    :param classes:
    :return:
    """
    i = random.randint(0, len(images))
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # plot image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # change the color of title depending
    # on if pred is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100 * tf.reduce_max(pred_probs),
                                                     true_label),
               color=color)


def plot_predicted_images_random(test_dir, model, class_names=None, number_of_images=3):
    if class_names is None:
        class_names = []
    plt.figure(figsize=(17, 10))
    for i in range(number_of_images):
        # choose random images from random classes
        class_name = random.choice(class_names)
        filename = random.choice((os.listdir(test_dir + "/" + class_name)))
        filepath = test_dir + class_name + "/" + filename
        # print(filepath)
        img = preprocessutils.load_and_prep_image(filepath, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]

        plt.subplot(int(number_of_images/3), 3, i + 1)
        plt.imshow(img / 255.)
        if class_name == pred_class:
            title_color = "g"
        else:
            title_color = "r"
        plt.title(f"Actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():2f}", c=title_color)
        plt.axis(False)

    plt.show()

def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # plot loss
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.show()


def compare_histories(original_history, new_history, initial_epochs=5):
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Compare hist
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training accuracy")
    plt.plot(total_val_acc, label="val accuracy")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="start fine tuning")
    plt.legend(loc="lower right")
    plt.title("Train and val accuracy")
    # plt.show()

    # Make plots for loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training loss")
    plt.plot(total_val_loss, label="Validation loss")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="start fine tuning")
    plt.legend(loc="upper right")
    plt.title("Train and val loss")
    plt.show()
