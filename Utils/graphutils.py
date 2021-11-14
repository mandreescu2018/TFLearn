
import matplotlib.pyplot as plt


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