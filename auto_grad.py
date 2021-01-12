import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
z = z.mean()

print(z)

z.backward()
print(x.grad)


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad) #this increments the weights every iteration

    # to solve the issue we have to empy the grad after each iteration

    weights.grad.zero_()