
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import tensorflow as tf

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
    x_min, x_max = X[:,0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (make predictions o these)
    x_in = np.c_[xx.ravel(), yy.ravel()] #stack 2D arrays together

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
    plt.scatter(X[:,0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_confusion_matrix(confusion_matrix, classes=None, text_size=10):
    figsize = (15, 15)
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
    threshold = (confusion_matrix.max() + confusion_matrix.min())/2.

    # plot the text
    for i,j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j,i, f"{confusion_matrix[i,j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i,j]>threshold else "black",
                 size=text_size)

    # set x-axis label to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

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
                                                     100*tf.reduce_max(pred_probs),
                                                     true_label),
               color=color)


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











