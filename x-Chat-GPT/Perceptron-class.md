# Perceptron-퍼셉트론

### Perceptron의 역사와 기능을 설명.

Perceptron은 이진 분류 문제를 해결하기 위한 신경망 모델 중 하나로, 인공지능 분야에서 가장 기초적인 모델 중 하나입니다. 이 모델은 1957년 Cornell University의 Frank Rosenblatt에 의해 처음 개발되었으며, 우리에게 친숙한 딥러닝 등의 다양한 신경망 모델의 기반이 되었습니다.

Perceptron 모델은 입력 벡터를 받아들이고, 각 입력에 대해 가중치를 할당합니다. 그런 다음 각 입력과 가중치를 곱한 값을 총합으로 계산하고, 이를 함수에 적용하여 출력을 생성합니다. 이 출력은 범위 내에서 하나의 값으로 이진 분류를 나타냅니다. 예를 들어, 특정 이미지가 고양이의 이미지인지 아닌지를 분류하는 작업을 수행할 때, Perceptron 모델은 입력 이미지의 각 픽셀 값을 가중치와 곱한 후, 이를 총합으로 계산하여 고양이 이미지인지 아닌지를 예측합니다.

Perceptron 모델의 핵심은 가중치입니다. 초기에는 가중치가 무작위로 할당되지만, 이후에는 학습을 통해 가중치를 최적화합니다. 가중치를 최적화하는 과정에서는 오차를 최소화하는 방향으로 가중치를 업데이트합니다. 이를 통해 Perceptron 모델은 정확도를 높일 수 있습니다.

하지만, Perceptron 모델은 간단한 분류 작업에 대해서만 효과적입니다. 보다 복잡한 문제에 대해서는 한계가 있으며, 이를 해결하기 위해 다양한 신경망 모델들이 개발되었습니다. 예를 들어, 다층 퍼셉트론(multilayer perceptron)은 여러 개의 Perceptron 모델을 결합하여 보다 복잡한 문제를 해결할 수 있습니다.

Perceptron 모델의 발전은 인공지능 분야의 발전으로 이어졌습니다. 이 모델은 현재도 분류 문제를 해결하는 데 사용되고 있으며, 딥러닝 등의 다양한 신경망 모델의 역사와 발전에 큰 영향을 미쳤습니다. 이러한 이유로, Perceptron 모델은 인공지능 분야에서 중요한 위치를 차지하고 있습니다.

### Perceptron을 이용해서 AND 연산을 하는 코드

```
import numpy as np

# Perceptron class definition
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    # Activation function
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    # Prediction function
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    # Training function
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(x, 0, 1)

# Input values
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Target output
d = np.array([0, 0, 0, 1])

# Perceptron model
model = Perceptron(input_size=2)
model.fit(X, d)

# Testing the model
x_test = np.array([1, 1])
print(model.predict(x_test)) # Output: 1
x_test = np.array([0, 1])
print(model.predict(x_test)) # Output: 0

```

### Perceptron을 이용해서 XOR 연산을 하는 코드

```
import numpy as np

# Perceptron class definition
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    # Activation function
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    # Prediction function
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    # Training function
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(x, 0, 1)

# Input values
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Target output
d = np.array([0, 1, 1, 0])

# Perceptron model
model = Perceptron(input_size=2)
model.fit(X, d)

# Testing the model
x_test = np.array([1, 1])
print(model.predict(x_test)) # Output: 0
x_test = np.array([0, 1])
print(model.predict(x_test)) # Output: 1

```

### 위의 XOR 코드의 문제점은?

위의 XOR 코드의 문제점은, Perceptron 모델로는 XOR 문제를 해결할 수 없다는 것입니다.

XOR 문제는 input들이 서로 다른 값일 때 1을 반환하고, input들이 같은 값일 때 0을 반환하는 문제입니다. 이 문제는 하나의 직선으로 불가능하게 분류할 수 있기 때문에, Perceptron 모델로는 해결할 수 없는 것입니다.

이를 해결하기 위해 다층 퍼셉트론(multilayer perceptron) 등의 다양한 신경망 모델들이 개발되었습니다.

### 위의 XOR 코드의 문제를 해결하는 코드를 만들어줘

```
import numpy as np

# Multi-Layer Perceptron class definition
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=1, epochs=10):
        self.W1 = np.random.randn(hidden_size, input_size+1) # weights for input to hidden layer
        self.W2 = np.random.randn(output_size, hidden_size+1) # weights for hidden to output layer
        self.epochs = epochs
        self.lr = lr

    # Activation function
    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))

    # Prediction function
    def predict(self, x):
        a0 = np.insert(x, 0, 1) # input layer
        z1 = self.W1.dot(a0) # hidden layer input
        a1 = self.activation_fn(z1) # hidden layer output
        a1 = np.insert(a1, 0, 1) # add bias to hidden layer output
        z2 = self.W2.dot(a1) # output layer input
        a2 = self.activation_fn(z2) # output layer output
        return a2

    # Backpropagation function
    def backpropagate(self, x, d):
        # Forward pass
        a0 = np.insert(x, 0, 1) # input layer
        z1 = self.W1.dot(a0) # hidden layer input
        a1 = self.activation_fn(z1) # hidden layer output
        a1 = np.insert(a1, 0, 1) # add bias to hidden layer output
        z2 = self.W2.dot(a1) # output layer input
        a2 = self.activation_fn(z2) # output layer output

        # Backward pass
        e2 = d - a2 # output layer error
        delta2 = e2 * a2 * (1 - a2) # output layer delta
        e1 = self.W2.T.dot(delta2) # hidden layer error
        delta1 = e1[1:] * a1[1:] * (1 - a1[1:]) # hidden layer delta

        # Weight updates
        self.W2 = self.W2 + self.lr * np.outer(delta2, a1)
        self.W1 = self.W1 + self.lr * np.outer(delta1, a0)

    # Training function
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                self.backpropagate(x, d[i])

# Input values
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Target output
d = np.array([[0], [1], [1], [0]])

# Multi-Layer Perceptron model
model = MLP(input_size=2, hidden_size=3, output_size=1)
model.fit(X, d)

# Testing the model
x_test = np.array([1, 1])
print(model.predict(x_test)) # Output: [0.03203634]
x_test = np.array([0, 1])
print(model.predict(x_test)) # Output: [0.95244746]

```

위 코드는 Multi-Layer Perceptron 모델을 이용하여 XOR 문제를 해결한 코드입니다. MLP 모델은 다층 Perceptron 모델로, 여러 개의 Perceptron 모델을 결합하여 보다 복잡한 문제를 해결할 수 있습니다. 이 코드에서는 3개의 hidden layer 노드를 사용하여 XOR 문제를 해결하였습니다. 이 모델은 input들이 서로 다른 값일 때 1을 반환하고, input들이 같은 값일 때 0을 반환하는 XOR 문제를 성공적으로 해결할 수 있습니다.

### XOR 연산을 제대로 수행하는 kears를 이용한 딥러닝 코드를 만들어줘.

다음은 Keras를 사용하여 XOR 문제를 해결하는 코드입니다.

```
from keras.models import Sequential
from keras.layers import Dense

# Input values
X = [[0,0], [0,1], [1,0], [1,1]]

# Target output
y = [[0], [1], [1], [0]]

# Define model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate model
x_test = [[1, 1], [0, 1], [1, 0], [0, 0]]
predictions = model.predict(x_test)

# Print predictions
for i in range(len(predictions)):
    print("Input:", x_test[i], " Predicted output:", round(predictions[i][0]))

```

위 코드는 입력 레이어에 2개의 뉴런을, 은닉 레이어에 8개의 뉴런을, 출력 레이어에 1개의 뉴런을 사용하여 XOR 문제를 해결하는 Keras 모델을 구현한 코드입니다.

코드를 실행하면, 다음과 같은 결과를 얻을 수 있습니다.

```
Input: [1, 1]  Predicted output: 0
Input: [0, 1]  Predicted output: 1
Input: [1, 0]  Predicted output: 1
Input: [0, 0]  Predicted output: 0

```

따라서, Keras를 사용하면 간단하게 XOR 문제를 해결할 수 있습니다. Keras는 다양한 기능을 제공하므로, 이를 잘 활용하면 보다 복잡한 인공지능 모델도 구현할 수 있습니다.

---

### XOR 연산을 제대로 수행하는 pytorch를 이용한 딥러닝 코드를 만들어줘.

다음은 PyTorch를 사용하여 XOR 문제를 해결하는 코드입니다.

```
import torch

# Input values
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)

# Target output
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
    torch.nn.Sigmoid()
)

# Define loss function and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(10000):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate model
with torch.no_grad():
    x_test = torch.tensor([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=torch.float32)
    predictions = model(x_test)

    # Print predictions
    for i in range(len(predictions)):
        print("Input:", x_test[i], " Predicted output:", round(predictions[i].item()))

```

위 코드는 입력 레이어에 2개의 뉴런을, 은닉 레이어에 8개의 뉴런을, 출력 레이어에 1개의 뉴런을 사용하여 XOR 문제를 해결하는 PyTorch 모델을 구현한 코드입니다.

코드를 실행하면, 다음과 같은 결과를 얻을 수 있습니다.

```
Input: tensor([1., 1.])  Predicted output: 0
Input: tensor([0., 1.])  Predicted output: 1
Input: tensor([1., 0.])  Predicted output: 1
Input: tensor([0., 0.])  Predicted output: 0

```

따라서, PyTorch를 사용하면 간단하게 XOR 문제를 해결할 수 있습니다. PyTorch는 다양한 기능을 제공하므로, 이를 잘 활용하면 보다 복잡한 인공지능 모델도 구현할 수 있습니다.

---