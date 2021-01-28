import torch

x = torch.rand(2,2)

y = torch.rand(2,2)

print(x)
print(y)

z = x + y
print(z)

z = torch.add(x,y)


# reshaping a tensor
x = torch.rand(4, 4)

y = x.view(-1, 8)

print(y.size())

import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))


a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

# if we add 1 on a it will add it on b as well, as they share same memory on CPU
a +=1
print(a)
print(b)