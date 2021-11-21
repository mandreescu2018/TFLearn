
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

def plot_confusion_matrix(confusion_matrix, classes=None):
    figsize = (10, 10)
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
                 size=15)

    # set x-axis label to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(20)











