import csv

import numpy as np
import torch
from torch import nn as torch_nn


_EPS = 1e-12


def _as_2d_numpy(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def _add_bias(a):
    return np.hstack([np.ones((a.shape[0], 1)), a])


def _add_bias_torch(a):
    ones = torch.ones((a.shape[0], 1), dtype=a.dtype, device=a.device)
    return torch.cat([ones, a], dim=1)


def _one_hot(y, n_classes):
    y_int = y.astype(int)
    out = np.zeros((len(y_int), n_classes), dtype=float)
    out[np.arange(len(y_int)), y_int] = 1.0
    return out


def _normalize_activations(activations, n_layers, default_hidden, default_output):
    if activations is None:
        if n_layers == 1:
            return [default_output]
        return [default_hidden] * (n_layers - 1) + [default_output]

    if isinstance(activations, str):
        if n_layers == 1:
            return [default_output]
        return [activations] * (n_layers - 1) + [default_output]

    values = list(activations)
    if len(values) != n_layers:
        raise ValueError("activations must specify one entry per layer")
    return values


def _init_weight_matrices(layer_sizes, random_state):
    rng = np.random.default_rng(random_state)
    weights = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights.append(rng.uniform(-limit, limit, size=(n_in + 1, n_out)))
    return weights


def _activation_forward_np(name, z):
    if name == "sigmoid":
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "linear":
        return z
    raise ValueError(f"Unsupported activation: {name}")


def _activation_derivative_np(name, z, a):
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "relu":
        return (z > 0.0).astype(float)
    if name == "linear":
        return np.ones_like(z)
    raise ValueError(f"Unsupported activation: {name}")


def _activation_forward_torch(name, z):
    if name == "sigmoid":
        return torch.sigmoid(z)
    if name == "relu":
        return torch.relu(z)
    if name == "linear":
        return z
    raise ValueError(f"Unsupported activation: {name}")


def _regularization_penalty_np(weights, lambda_, m):
    if lambda_ == 0.0:
        return 0.0
    reg = 0.0
    for weight in weights:
        reg += np.sum(weight[1:, :] ** 2)
    return lambda_ * reg / (2.0 * m)


def _regularization_penalty_torch(weights, lambda_, m):
    if lambda_ == 0.0:
        return torch.tensor(0.0, dtype=weights[0].dtype, device=weights[0].device)
    reg = torch.zeros((), dtype=weights[0].dtype, device=weights[0].device)
    for weight in weights:
        reg = reg + torch.sum(weight[1:, :] ** 2)
    return (lambda_ / (2.0 * m)) * reg


def _prepare_targets_classification(y):
    classes = np.unique(y)
    class_to_index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_index[c] for c in y], dtype=int)
    Y = _one_hot(y_idx, len(classes))
    return classes, y_idx, Y


class _BaseANNModel:
    def __init__(self, weights):
        self._weights = [w.copy() for w in weights]

    def weights(self):
        return [w.copy() for w in self._weights]


class _ANNClassifierModel(_BaseANNModel):
    def __init__(self, weights, classes_, activations):
        super().__init__(weights)
        self.classes_ = classes_.copy()
        self.activations_ = list(activations)

    def _forward(self, X):
        X = _as_2d_numpy(X)
        a = X
        for weight, activation_name in zip(self._weights, self.activations_):
            z = _add_bias(a) @ weight
            a = _activation_forward_np(activation_name, z)
        return a

    def predict(self, X):
        return self._forward(X)


class _ANNRegressionModel(_BaseANNModel):
    def __init__(self, weights, activations):
        super().__init__(weights)
        self.activations_ = list(activations)

    def _forward(self, X):
        X = _as_2d_numpy(X)
        a = X
        for weight, activation_name in zip(self._weights, self.activations_):
            z = _add_bias(a) @ weight
            a = _activation_forward_np(activation_name, z)
        return a

    def predict(self, X):
        return self._forward(X).ravel()


class _TorchANNClassifierModel(_BaseANNModel):
    def __init__(self, network, classes_, activations, dtype, device):
        self._network = network
        self.classes_ = classes_.copy()
        self.activations_ = list(activations)
        self._dtype = dtype
        self._device = device

    def weights(self):
        return [param.detach().cpu().numpy().copy() for param in self._network.weights]

    def predict(self, X):
        X = _as_2d_numpy(X)
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=self._dtype, device=self._device)
            out = self._network(X_t)
        return out.detach().cpu().numpy()


class _TorchANNRegressionModel(_BaseANNModel):
    def __init__(self, network, activations, dtype, device):
        self._network = network
        self.activations_ = list(activations)
        self._dtype = dtype
        self._device = device

    def weights(self):
        return [param.detach().cpu().numpy().copy() for param in self._network.weights]

    def predict(self, X):
        X = _as_2d_numpy(X)
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=self._dtype, device=self._device)
            out = self._network(X_t)
        return out.detach().cpu().numpy().ravel()


class _NumpyANNBase:
    default_hidden_activation = "sigmoid"

    def __init__(
        self,
        units,
        lambda_=0.0,
        learning_rate=0.1,
        epochs=15000,
        optimizer="adam",
        random_state=0,
        activations=None,
        gradient_check_eps=1e-5,
    ):
        self.units = list(units)
        self.lambda_ = float(lambda_)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.random_state = random_state
        self.activations = activations
        self.gradient_check_eps = float(gradient_check_eps)

    def _init_weights(self, layer_sizes):
        return _init_weight_matrices(layer_sizes, self.random_state)

    def _forward_full(self, X, weights, activations):
        activations_out = [X]
        pre_activations = []
        a = X
        for weight, activation_name in zip(weights, activations):
            z = _add_bias(a) @ weight
            a = _activation_forward_np(activation_name, z)
            pre_activations.append(z)
            activations_out.append(a)
        return activations_out, pre_activations

    def _apply_optimizer(self, weights, grads, state, t):
        if self.optimizer not in {"adam", "gd", "sgd"}:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.optimizer in {"gd", "sgd"}:
            return [w - self.learning_rate * g for w, g in zip(weights, grads)]

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        new_weights = []
        for i, (w, g) in enumerate(zip(weights, grads)):
            state["m"][i] = beta1 * state["m"][i] + (1.0 - beta1) * g
            state["v"][i] = beta2 * state["v"][i] + (1.0 - beta2) * (g * g)
            m_hat = state["m"][i] / (1.0 - beta1**t)
            v_hat = state["v"][i] / (1.0 - beta2**t)
            new_weights.append(w - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps))
        return new_weights

    def _backprop(self, X, Y, weights, activations):
        m = X.shape[0]
        activations_out, pre_activations = self._forward_full(X, weights, activations)
        output = activations_out[-1]
        output_z = pre_activations[-1]

        loss, delta = self._loss_and_output_delta(Y, output, output_z, activations[-1], weights, m)

        grads = [np.zeros_like(weight) for weight in weights]
        for l in reversed(range(len(weights))):
            a_prev_bias = _add_bias(activations_out[l])
            grad = (a_prev_bias.T @ delta) / m
            if self.lambda_ != 0.0:
                grad[1:, :] += (self.lambda_ / m) * weights[l][1:, :]
            grads[l] = grad
            if l > 0:
                w_no_bias = weights[l][1:, :]
                delta = (delta @ w_no_bias.T) * _activation_derivative_np(
                    activations[l - 1], pre_activations[l - 1], activations_out[l]
                )

        loss = loss + _regularization_penalty_np(weights, self.lambda_, m)
        return grads, output, loss, activations_out, pre_activations

    def _loss_and_output_delta(self, Y, output, output_z, output_activation, weights, m):
        raise NotImplementedError

    def _fit_loop(self, X, Y, layer_sizes, activations):
        weights = self._init_weights(layer_sizes)
        state = {
            "m": [np.zeros_like(w) for w in weights],
            "v": [np.zeros_like(w) for w in weights],
        }

        best_weights = [w.copy() for w in weights]
        best_loss = np.inf

        for t in range(1, self.epochs + 1):
            grads, _, loss, _, _ = self._backprop(X, Y, weights, activations)
            weights = self._apply_optimizer(weights, grads, state, t)
            if loss < best_loss:
                best_loss = loss
                best_weights = [w.copy() for w in weights]

        return best_weights


class ANNClassification(_NumpyANNBase):
    def _loss_and_output_delta(self, Y, output, output_z, output_activation, weights, m):
        output_clipped = np.clip(output, _EPS, 1.0 - _EPS)
        loss = -np.sum(Y * np.log(output_clipped) + (1.0 - Y) * np.log(1.0 - output_clipped)) / m
        dL_da = -(Y / output_clipped) + (1.0 - Y) / (1.0 - output_clipped)
        delta = dL_da * _activation_derivative_np(output_activation, output_z, output)
        return loss, delta

    def fit(self, X, y):
        X = _as_2d_numpy(X)
        y = np.asarray(y)
        classes, y_idx, Y = _prepare_targets_classification(y)
        activations = _normalize_activations(
            self.activations,
            len(self.units) + 1,
            self.default_hidden_activation,
            "sigmoid",
        )
        layer_sizes = [X.shape[1]] + self.units + [len(classes)]
        weights = self._fit_loop(X, Y, layer_sizes, activations)
        return _ANNClassifierModel(weights, classes, activations)

    def _fit_loop(self, X, Y, layer_sizes, activations):
        return super()._fit_loop(X, Y, layer_sizes, activations)

    def gradient_check(self, X, y, max_params=50, random_state=123):
        X = _as_2d_numpy(X)
        y = np.asarray(y)
        classes, y_idx, Y = _prepare_targets_classification(y)
        activations = _normalize_activations(
            self.activations,
            len(self.units) + 1,
            self.default_hidden_activation,
            "sigmoid",
        )
        layer_sizes = [X.shape[1]] + self.units + [len(classes)]
        weights = self._init_weights(layer_sizes)
        analytical_grads, _, _, _, _ = self._backprop(X, Y, weights, activations)

        params = []
        grads = []
        for wi, weight in enumerate(weights):
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    params.append((wi, i, j))
                    grads.append(analytical_grads[wi][i, j])

        grads = np.array(grads)
        n_params = len(params)
        if max_params is None or max_params >= n_params:
            chosen = np.arange(n_params)
        else:
            local_rng = np.random.default_rng(random_state)
            chosen = local_rng.choice(n_params, size=max_params, replace=False)

        numerical = np.zeros(len(chosen), dtype=float)
        analytical = np.zeros(len(chosen), dtype=float)

        for k, idx in enumerate(chosen):
            wi, i, j = params[idx]
            analytical[k] = grads[idx]
            w_plus = [w.copy() for w in weights]
            w_minus = [w.copy() for w in weights]
            w_plus[wi][i, j] += self.gradient_check_eps
            w_minus[wi][i, j] -= self.gradient_check_eps

            forward_plus = self._forward_full(X, w_plus, activations)
            forward_minus = self._forward_full(X, w_minus, activations)
            out_plus = forward_plus[0][-1]
            out_minus = forward_minus[0][-1]
            z_plus = forward_plus[1][-1]
            z_minus = forward_minus[1][-1]
            loss_plus, _ = self._loss_and_output_delta(Y, out_plus, z_plus, activations[-1], w_plus, X.shape[0])
            loss_minus, _ = self._loss_and_output_delta(Y, out_minus, z_minus, activations[-1], w_minus, X.shape[0])
            loss_plus += _regularization_penalty_np(w_plus, self.lambda_, X.shape[0])
            loss_minus += _regularization_penalty_np(w_minus, self.lambda_, X.shape[0])
            numerical[k] = (loss_plus - loss_minus) / (2.0 * self.gradient_check_eps)

        abs_diff = np.abs(numerical - analytical)
        rel_diff = abs_diff / np.maximum(1.0, np.abs(numerical), np.abs(analytical))
        return {
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "mean_rel_diff": float(np.mean(rel_diff)),
            "checked_parameters": int(len(chosen)),
        }


class ANNRegression(_NumpyANNBase):
    def _loss_and_output_delta(self, Y, output, output_z, output_activation, weights, m):
        y_col = Y.reshape(-1, 1)
        loss = 0.5 * np.sum((output - y_col) ** 2) / m
        dL_da = output - y_col
        delta = dL_da * _activation_derivative_np(output_activation, output_z, output)
        return loss, delta

    def fit(self, X, y):
        X = _as_2d_numpy(X)
        y = np.asarray(y, dtype=float)
        Y = y.reshape(-1, 1)
        activations = _normalize_activations(
            self.activations,
            len(self.units) + 1,
            self.default_hidden_activation,
            "linear",
        )
        layer_sizes = [X.shape[1]] + self.units + [1]
        weights = self._fit_loop(X, Y, layer_sizes, activations)
        return _ANNRegressionModel(weights, activations)

    def _fit_loop(self, X, Y, layer_sizes, activations):
        return super()._fit_loop(X, Y, layer_sizes, activations)


class _TorchANNBase:
    default_hidden_activation = "sigmoid"

    def __init__(
        self,
        units,
        lambda_=0.0,
        learning_rate=0.1,
        epochs=15000,
        optimizer="adam",
        random_state=0,
        activations=None,
        device="cpu",
        dtype=torch.float64,
    ):
        self.units = list(units)
        self.lambda_ = float(lambda_)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.random_state = random_state
        self.activations = activations
        self.device = torch.device(device)
        self.dtype = dtype

    def _init_weights(self, layer_sizes):
        arrays = _init_weight_matrices(layer_sizes, self.random_state)
        return [torch.tensor(array, dtype=self.dtype, device=self.device) for array in arrays]


class _TorchMLP(torch_nn.Module):
    def __init__(self, weights, activations, dtype, device):
        super().__init__()
        self.weights = torch_nn.ParameterList(
            [torch_nn.Parameter(weight.clone().to(device=device, dtype=dtype)) for weight in weights]
        )
        self.activations = list(activations)
        self.dtype = dtype
        self.device = device

    def forward(self, X):
        a = X
        for weight, activation_name in zip(self.weights, self.activations):
            z = _add_bias_torch(a) @ weight
            a = _activation_forward_torch(activation_name, z)
        return a


class ANNClassificationTorch(_TorchANNBase):
    def fit(self, X, y):
        X = _as_2d_numpy(X)
        y = np.asarray(y)
        classes, y_idx, Y_np = _prepare_targets_classification(y)
        activations = _normalize_activations(
            self.activations,
            len(self.units) + 1,
            self.default_hidden_activation,
            "sigmoid",
        )

        torch.manual_seed(self.random_state)
        layer_sizes = [X.shape[1]] + self.units + [len(classes)]
        weights = self._init_weights(layer_sizes)
        network = _TorchMLP(weights, activations, self.dtype, self.device)

        X_t = torch.tensor(X, dtype=self.dtype, device=self.device)
        Y_t = torch.tensor(Y_np, dtype=self.dtype, device=self.device)

        state = {
            "m": [torch.zeros_like(param) for param in network.weights],
            "v": [torch.zeros_like(param) for param in network.weights],
        }

        best_state = [param.detach().clone() for param in network.weights]
        best_loss = float("inf")

        for t in range(1, self.epochs + 1):
            for param in network.weights:
                if param.grad is not None:
                    param.grad.zero_()

            output = network(X_t)
            output_clipped = torch.clamp(output, _EPS, 1.0 - _EPS)
            loss = -torch.sum(Y_t * torch.log(output_clipped) + (1.0 - Y_t) * torch.log(1.0 - output_clipped)) / X_t.shape[0]
            loss = loss + _regularization_penalty_torch(network.weights, self.lambda_, X_t.shape[0])
            loss.backward()

            grads = [param.grad.detach().clone() for param in network.weights]
            if self.optimizer not in {"adam", "gd", "sgd"}:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            with torch.no_grad():
                if self.optimizer in {"gd", "sgd"}:
                    for param, grad in zip(network.weights, grads):
                        param -= self.learning_rate * grad
                else:
                    beta1 = 0.9
                    beta2 = 0.999
                    eps = 1e-8
                    for i, (param, grad) in enumerate(zip(network.weights, grads)):
                        state["m"][i] = beta1 * state["m"][i] + (1.0 - beta1) * grad
                        state["v"][i] = beta2 * state["v"][i] + (1.0 - beta2) * (grad * grad)
                        m_hat = state["m"][i] / (1.0 - beta1**t)
                        v_hat = state["v"][i] / (1.0 - beta2**t)
                        param -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + eps)

            current_loss = float(loss.detach().cpu().item())
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = [param.detach().clone() for param in network.weights]

        with torch.no_grad():
            for param, best in zip(network.weights, best_state):
                param.copy_(best)

        return _TorchANNClassifierModel(network, classes, activations, self.dtype, self.device)


class ANNRegressionTorch(_TorchANNBase):
    def fit(self, X, y):
        X = _as_2d_numpy(X)
        y = np.asarray(y, dtype=float)
        Y_np = y.reshape(-1, 1)
        activations = _normalize_activations(
            self.activations,
            len(self.units) + 1,
            self.default_hidden_activation,
            "linear",
        )

        torch.manual_seed(self.random_state)
        layer_sizes = [X.shape[1]] + self.units + [1]
        weights = self._init_weights(layer_sizes)
        network = _TorchMLP(weights, activations, self.dtype, self.device)

        X_t = torch.tensor(X, dtype=self.dtype, device=self.device)
        Y_t = torch.tensor(Y_np, dtype=self.dtype, device=self.device)

        state = {
            "m": [torch.zeros_like(param) for param in network.weights],
            "v": [torch.zeros_like(param) for param in network.weights],
        }

        best_state = [param.detach().clone() for param in network.weights]
        best_loss = float("inf")

        for t in range(1, self.epochs + 1):
            for param in network.weights:
                if param.grad is not None:
                    param.grad.zero_()

            output = network(X_t)
            loss = 0.5 * torch.sum((output - Y_t) ** 2) / X_t.shape[0]
            loss = loss + _regularization_penalty_torch(network.weights, self.lambda_, X_t.shape[0])
            loss.backward()

            grads = [param.grad.detach().clone() for param in network.weights]
            if self.optimizer not in {"adam", "gd", "sgd"}:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            with torch.no_grad():
                if self.optimizer in {"gd", "sgd"}:
                    for param, grad in zip(network.weights, grads):
                        param -= self.learning_rate * grad
                else:
                    beta1 = 0.9
                    beta2 = 0.999
                    eps = 1e-8
                    for i, (param, grad) in enumerate(zip(network.weights, grads)):
                        state["m"][i] = beta1 * state["m"][i] + (1.0 - beta1) * grad
                        state["v"][i] = beta2 * state["v"][i] + (1.0 - beta2) * (grad * grad)
                        m_hat = state["m"][i] / (1.0 - beta1**t)
                        v_hat = state["v"][i] / (1.0 - beta2**t)
                        param -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + eps)

            current_loss = float(loss.detach().cpu().item())
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = [param.detach().clone() for param in network.weights]

        with torch.no_grad():
            for param, best in zip(network.weights, best_state):
                param.copy_(best)

        return _TorchANNRegressionModel(network, activations, self.dtype, self.device)


# Backwards-compatible aliases for the PyTorch implementation.
ANNClassificationPT = ANNClassificationTorch
ANNRegressionPT = ANNRegressionTorch


# data reading helpers preserved for assignment convenience

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y
