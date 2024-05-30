import torch
from regression import fit_regression_model, get_train_data

def test_fit_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    assert loss < 1e-3, "The loss is too high"

def test_fit_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    assert loss < 1e-3, "The loss is too high"

def test_fit_and_predict_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[6.0], [7.0], [8.0]])
    y_pred = model(X_test)
    assert ((y_pred - torch.tensor([[17.0], [19.0], [21.0]])).abs() < 1e-1).all(), " y_pred is not correct"

def test_fit_and_predict_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]])
    y_pred = model(X_test)
    assert ((y_pred - torch.tensor([[16.0], [18.0], [20.0]])).abs() < 1e-1).all(), " y_pred is not correct"

if __name__ == "__main__":
    test_fit_regression_model_1d()
    test_fit_regression_model_2d()
    test_fit_and_predict_regression_model_1d()
    test_fit_and_predict_regression_model_2d()
    print("All tests passed!")
