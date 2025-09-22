from __future__ import annotations

from collections.abc import Callable
from abc import ABC
from typing import Tuple, List
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

np.random.seed(1337)
random.seed(1337)

important_vars = []

class Layer:
    def __init__(self, activation_fn: Callable[[Var], Var], weights: List[List[Var]], biases: List[Var]):
        self.activation_fn = activation_fn
        self.weights = weights
        self.biases = biases
        self.input_size = len(weights[0])
        self.output_size = len(weights)
        # self.output_size, self.input_size = weights.shape

    def forward(self, inputs: List[Var]) -> List[Var]:
        # print("forward", len(inputs), self.input_size, self.output_size)
        # assert len(inputs) == self.input_size
        return np.vectorize(self.activation_fn)(np.dot(self.weights, inputs) + self.biases)
        outputs = []
        for i in range(self.output_size):
            # Manual dot product with Var operations
            weighted_sum = Var(0)
            for j in range(self.input_size):
                weighted_sum += self.weights[i][j] * inputs[j]
            # Add bias
            weighted_sum += self.biases[i]
            # Apply activation
            output = self.activation_fn(weighted_sum)
            outputs.append(output)
        return outputs

    def __repr__(self):
        return f"Layer({self.weights}, {self.biases})"

class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def infer(self, inputs: List[Var]) -> List[Var]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def loss(self, training_data: List[Tuple[Var, Var]]) -> Var:
        """
        SUM over all i in I (t_i - f(x_i))^2
        """
        loss = Var(0)
        for x, t in training_data:
            # y = self.infer(np.array([x])).T
            y = self.infer([x])
            diff = y[0] - t
            # diff = t - y[0]
            loss += diff ** 2
        return loss

    def __repr__(self):
        return f"Network({self.layers})"

class Op(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    POWER = "**"
    RELU = "ReLU"

class Dependency(ABC):
    def __init__(self, vars: Tuple[Var, Var], op: Op):
        self.vars = vars
        self.op = op

    def __repr__(self):
        return f"({self.vars[0]} {self.op.value} {self.vars[1]})"

"""
Expr := Expr op Expr | Var

Backprop on Expr:

"""

class Var:
    def __init__(self, value: float, dep: Dependency | None = None, label: str = ""):
        self.label = label
        self.value = value
        self.dep = dep
        self.grad: float | None = None

    def __add__(self, other: Var):
        return Var(self.value + other.value,
            Dependency((self, other), Op.ADD), label=f"{self.label} + {other.label}")

    def __sub__(self, other: Var):
        return Var(self.value - other.value,
            Dependency((self, other), Op.SUBTRACT), label=f"{self.label} - {other.label}")

    def __mul__(self, other: Var):
        """DO NOT USE THIS IF YOU CAN USE POWER. IDIOT!"""
        return Var(self.value * other.value,
            Dependency((self, other), Op.MULTIPLY), label=f"{self.label} * {other.label}")

    def __pow__(self, exponent: float):
        assert exponent > 0, "Exponent must be positive"
        return Var(self.value ** exponent,
            Dependency((self, Var(exponent, label=str(exponent))), Op.POWER), label=f"{self.label} ** {exponent}")

    def __sqrt__(self):
        return Var(self.value ** 0.5,
            Dependency((self, Var(0.5, label="0.5")), Op.POWER), label=f"sqrt({self.label})")

    def __repr__(self) -> str:
        if self.dep:
            return repr(self.dep)
        elif self.label:
            return f"Var({self.label}={self.value}, grad={self.grad})"
        else:
            return f"Var({self.value})"

    def __ge__(self, other: Var):
        return self.value >= other.value

    def __le__(self, other: Var):
        return self.value <= other.value

    def __lt__(self, other: Var):
        return self.value < other.value

    def __gt__(self, other: Var):
        return self.value > other.value

    def backprop(self):
        """
        Pre: grad is calculated on parent already
        """
        # Calculates grad for self.vars
        self.calc_dep_grads()

        if self.dep:
            self.dep.vars[0].backprop()
            self.dep.vars[1].backprop()

    def calc_dep_grads(self):
        if not self.dep:
            return

        assert self.grad is not None, "Grad must be calculated before calculating dependency grads"

        match self.dep.op:
            case Op.ADD:
                self.dep.vars[0].grad = self.grad
                self.dep.vars[1].grad = self.grad
            case Op.SUBTRACT:
                self.dep.vars[0].grad = self.grad
                self.dep.vars[1].grad = -self.grad
            case Op.MULTIPLY:
                self.dep.vars[0].grad = self.dep.vars[1].value * self.grad
                self.dep.vars[1].grad = self.dep.vars[0].value * self.grad
            case Op.POWER:
                n = self.dep.vars[1].value
                self.dep.vars[0].grad = n * self.dep.vars[0].value ** (n - 1) * self.grad
            case Op.RELU:
                self.dep.vars[0].grad = self.grad if self.dep.vars[0].value > 0 else 0
            case _:
                raise ValueError(f"Unsupported operation: {self.dep.op}")


# def relu(x: Var) -> Var:
#     return x if x > Var(0) else Var(0, label="relued!") * x
def relu(x: Var) -> Var:
    # Don't modify x in place! Create a new Var with the relu'd value
    relu_value = max(x.value, 0)
    idiot = Var(relu_value, dep=Dependency((x, Var(0)), op=Op.RELU))
    important_vars.append(idiot)
    x = idiot
    return idiot

def identity(x: Var) -> Var:
    return x

# def varify(x: np.ndarray) -> np.ndarray:
#     return (np.vectorize(Var))(x)

def split_into_batches(data: List[Tuple[Var, Var]], batches: int, randomize: bool=True) -> List[List[Tuple[Var, Var]]]:
    if randomize:
        random.shuffle(data)
    return [data[i:i+batches] for i in range(0, len(data), batches)]

if __name__ == "__main__":
    learn_rate = 1E-4
    EPOCHS = 500
    NUM_BATCHES = 10
    hidden_units = 1

    x = np.linspace(1, 10, num=100)
    t = np.vectorize(lambda x: (x-5)**2 if x > 5 else 0)(x)
    training_data = [(Var(x_val, label=f"x_{i}"), Var(t_val, label=f"t_{i}"))
                     for i, (x_val, t_val) in enumerate(zip(x, t))]


    w1 = [[Var(32, label=f"w1_{i}")] for i in range(hidden_units)]
    b1 = [Var(-266, label=f"b1_{i}") for i in range(hidden_units)]

    wy = [[Var(0.1, label=f"wy_{i}") for i in range(hidden_units)]]
    by = [Var(0, label="by")]

    network: Network | None = None
    network_predictions = []
    loss_values = []

    training_data_copy = training_data.copy()
    for i in range(EPOCHS):
        print(f"\nEPOCH {i+1}")
        batches = split_into_batches(training_data_copy, NUM_BATCHES, randomize=False)
        for j, batch in enumerate(batches):
            activation = lambda x: relu(x)
            network = Network([
                Layer(activation_fn=activation, weights=w1, biases=b1),
                Layer(activation_fn=identity, weights=wy, biases=by),
            ])

            loss = network.loss(training_data)
            loss.grad = 1.0
            loss.backprop()

            # for idiot in important_vars:
            #     print(f"val {idiot.value}. grad {idiot.grad}. child val: {idiot.dep.vars[0].value}. child grad: {idiot.dep.vars[0].grad}")

            important_vars = []

            def step(vals: List[Var] | List[List[Var]], learn_rate):
                res = []
                if isinstance(vals[0], Var):
                    for var in vals:
                        res.append(Var(var.value - learn_rate * var.grad, label=var.label))
                else:
                    for row in vals:
                        new_row = []
                        for var in row:
                            new_row.append(Var(var.value - learn_rate * var.grad, label=var.label))
                        res.append(new_row)
                return res

            w1 = step(w1, learn_rate)
            # b1 = step(b1, learn_rate)

            # wy = step(wy, learn_rate)
            by = step(by, learn_rate)

        print(network)
        # Track predictions for this iteration
        iteration_predictions = []
        for x, t in training_data:
            infer = network.infer([Var(x.value)])  # Keep as Var
            iteration_predictions.append(infer[0].value)
        network_predictions.append(iteration_predictions)

        loss = network.loss(training_data)
        print(f"loss.value: {loss.value}")
        loss_values.append(loss.value)

        # w1 = np.vectorize(lambda x: Var(x.value - learn_rate * x.grad, label=x.label))(w1)
        # b1 = np.vectorize(lambda x: Var(x.value - bias_learn_rate * x.grad, label=x.label))(b1)

        # w2 = np.vectorize(lambda x: Var(x.value - learn_rate * x.grad, label=x.label))(w2)
        # b2 = np.vectorize(lambda x: Var(x.value - bias_learn_rate * x.grad, label=x.label))(b2)

        # w3 = np.vectorize(lambda x: Var(x.value - learn_rate * x.grad, label=x.label))(w3)
        # b3 = np.vectorize(lambda x: Var(x.value - bias_learn_rate * x.grad, label=x.label))(b3)

    # root cause: relued! -> w1 was not part of expression, so not backpropped to update grad.
    #


    if network:
        x_vals = []
        y_actual = []

        for x, t in training_data:
            x_vals.append(x.value)
            y_actual.append(t.value)

        # Create subplots for both predictions and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot predictions vs actual values
        ax1.plot(x_vals, y_actual, 'bo--', label='Actual', markersize=8, linewidth=2)

        # Plot predictions from selected iterations
        iterations_to_plot = np.linspace(0, EPOCHS-1, num=EPOCHS//10, dtype=int)

        # Create a colormap for gradient colors
        colors = cm.viridis(np.linspace(0, 1, len(iterations_to_plot)))

        for idx, iter_num in enumerate(iterations_to_plot):
            if iter_num < len(network_predictions):
                ax1.plot(x_vals, network_predictions[iter_num], 'o-',
                        label=f'Iteration {iter_num+1}',
                        markersize=1, alpha=0.7, color=colors[idx])

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Neural Network Training Progress: Predictions vs Actual Values')
        # ax1.legend()
        ax1.grid(True)

        # Plot loss over time with gradient colors
        from matplotlib.collections import LineCollection
        epochs = np.array(range(1, len(loss_values) + 1))
        loss_array = np.array(loss_values)

        # Create line segments
        points = np.array([epochs, loss_array]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create gradient colors for loss line
        loss_colors = cm.viridis(np.linspace(0, 1, len(segments)))

        lc = LineCollection(segments, colors=loss_colors, linewidth=2)
        ax2.add_collection(lc)
        ax2.set_xlim(epochs.min(), epochs.max())
        ax2.set_ylim(loss_array.min(), loss_array.max())
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Over Time')
        ax2.grid(True)
        # ax2.set_yscale('log')  # Use log scale for better visualization of loss decay

        plt.tight_layout()
        plt.show()

        # Print final predictions
        print("\nFinal predictions:")
        for i, (x, t) in enumerate(training_data):
            print(f"x: {x.value}, Prediction: {network_predictions[-1][i]:.4f}, Actual: {t.value}")
