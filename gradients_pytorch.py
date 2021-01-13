
# ********************************************** #

#  Training Pipeline proceducte in pytorch

# 1 ) Design model (input, output size, fowrd pass)
# 2 ) construct loss optimizer
# 3 ) Training loop
#     - forward pass: copute prediction
#     - backward pass: gradients
#     - update weights

# ********************************************** #

import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
#X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
#Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

# weights
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# model prediction
#def forward(x):
    # return w * x


# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x -y)
#def gradient(x, y, y_predicted):
    # return np.dot(2*x, y_predicted-y).mean()


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


#Training
learning_rate = 0.01
#number of iteration
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    # y_pred = forward(X)
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backword pass
    l.backward() # will calculate dl/dw

    # update weights
    #with torch.no_grad():
       # w -= learning_rate * w.grad
    optimizer.step()

    # zero gradients
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [ w, b] = model.parameters()
        print(f'epoch+1 {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
