import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

# weights
w = 0.0

# model prediction
def forward(x):
    return w * x


# loass = MSE
def loss(y, y_predicted):
    """
    calculate loass
    """
    return ((y_predicted-y)**2).mean()


# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x -y)
def gradient(x,y, y_predicted):
    """
    calculate gradient
    """
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')


#Training
learning_rate = 0.01
#number of iteration
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient
    dw = gradient(X,Y,y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch+1 {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
