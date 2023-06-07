import torch
import cv.core.perception as net
import torch.nn as nn
from torch import optim
import cv.dataset.loader as cv
import numpy as np
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history = []


def accuracy_score(y: np.ndarray, t: np.ndarray):
    y = np.argmax(y, axis=0)
    t = np.argmax(t, axis=0)
    return metrics.accuracy_score(y, t)


def print_progress(fold, epoch, n_epochs, avg_train_accuracy, avg_train_loss, avg_test_accuracy, avg_test_loss):
    """
  Print training and testing performance
  per epoch
  """
    print("Fold: %d, Epoch: %d/%d" % (fold + 1, epoch + 1, n_epochs))
    print("Train accuracy: %.2f%%" % (avg_train_accuracy * 100))
    print("Train loss: %.3f" % (avg_train_loss))
    print("Test accuracy: %.2f%%" % (avg_test_accuracy * 100))
    print("Test loss: %.3f" % (avg_test_loss))
    print("")


def test(x_batch, y_batch, model, criterion):
    """
  Perform a single forward propagation step
  on the testing data
  """
    # forward propagate
    output = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # compute model loss
    loss = criterion(output, y_truth)

    # compute validation accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # mean validation accuracy
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
    # predicted and ground truth values converted to a list
    y_pred = y_pred.tolist()
    y_truth = y_truth.tolist()

    return accuracy, loss, y_pred, y_truth


def train(x_batch, y_batch, model, criterion, model_optimizer):
    """
  Perform a single forward and back propagation step
  on the training data
  """

    model_optimizer.zero_grad()  # remove any existing gradients

    # forward propagate
    output = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # compute model loss
    loss = criterion(output, y_truth)

    # backpropagate the gradients
    loss.backward()

    # update parameters based on backprop
    model_optimizer.step()

    # accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # average accuracy
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
    return accuracy, loss


def fit(fold, model, x_train, y_train, x_test, y_test, batch_size=32, n_epochs=25, learning_rate=0.001):
    global history

    # Initialize criterion and optimizers
    criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        running_train_accuracy = []
        running_train_loss = 0.0

        running_test_accuracy = []
        running_test_loss = 0.0

        n_iters_train = len(x_train) / batch_size
        n_iters_test = len(x_test) / batch_size
        print(n_iters_test)

        # Perform training
        model.train()
        for index in range(0, len(x_train), batch_size):
            x_batch = x_train[index: index + batch_size]
            y_batch = y_train[index: index + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            train_accuracy, train_loss = train(
                x_batch, y_batch, model, criterion, model_optimizer
            )
            running_train_accuracy.append(train_accuracy.data)
            print(running_train_accuracy)
            running_train_loss += train_loss.item()

        # Perform testing
        with torch.no_grad():
            model.eval()

            test_pred, test_truth = [], []

            for index in range(0, len(x_test), batch_size):
                x_batch = x_test[index: index + batch_size]
                y_batch = y_test[index: index + batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                test_accuracy, test_loss, pred, truth = test(
                    x_batch, y_batch, model, criterion
                )

                running_test_accuracy.append(test_accuracy)
                print(running_test_accuracy)
                running_test_loss += test_loss.item()
                test_pred.extend(pred)
                test_truth.extend(truth)

        # Track metrics
        avg_train_accuracy = np.mean(running_train_accuracy)
        avg_train_loss = running_train_loss / n_iters_train

        avg_test_accuracy = np.mean(running_test_accuracy)
        avg_test_loss = running_test_loss / n_iters_test

        history.append(
            {
                "fold": fold + 1,
                "epoch": epoch + 1,
                "avg_train_accuracy": avg_train_accuracy * 100,
                "avg_test_accuracy": avg_test_accuracy * 100,
                "avg_train_loss": avg_train_loss,
                "avg_test_loss": avg_test_loss,
                "test_pred": test_pred,
                "test_truth": test_truth,
            }
        )

        # Print progress
        print_progress(
            fold,
            epoch,
            n_epochs,
            avg_train_accuracy,
            avg_train_loss,
            avg_test_accuracy,
            avg_test_loss,
        )


def run():
    ck = cv.CKPLoader()
    ck.load()
    k_fold = ck.dataset.kfold()

    for fold, (training_set, validation_set) in enumerate(k_fold):
        x_train, y_train, x_test, y_test = convert_to_torch(training_set.data, training_set.labels, validation_set.data,
                                                            validation_set.labels)

        model = net.DeXpression()
        model = model.to(device)

        fit(fold, model, x_train, y_train, x_test, y_test)
        print("fine")


def convert_to_torch(x_train, y_train, x_test, y_test):
    """Converts train and test data into torch tensors."""

    # converting training images into torch tensor
    x_train = torch.from_numpy(x_train)
    x_train = x_train.type(torch.FloatTensor)

    # converting the label into torch tensor
    y_train = y_train.astype(int)
    y_train = torch.from_numpy(y_train)

    # converting test images into torch tensor
    x_test = torch.from_numpy(x_test)
    x_test = x_test.type(torch.FloatTensor)

    # converting the label into torch tensor
    y_test = y_test.astype(int)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    run()
