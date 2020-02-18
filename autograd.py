# coding=utf-8
'''
    The `autograd` package provides automatic differentiation for all operations
    on Tensors. It is a define-by-run framework, which means that your backprop 
    is defined by how your code is run, and that every single iteration can be 
    different.
'''

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

# `y` was created as a result of an operation, so it has a `grad_fn`.
print(y.grad_fn)

# do more operation on `y`
z = y * y * 3
out = z.mean()

print(z, out)

print('------------------------------------------------------')
# `.requires_grad_(...)` changes an existing Tensor's requires_grad flag in-place.
# The input flag defaults to `False` if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


print('------------------------------------------------------')
'''Gradients
    Let's backprop now. Because `out` contains a single scalar, `out.backward()`
    is equivalent to `out.backward(torch.tensor(1.)).
'''
out.backward()

# Print gradients d(out)/dx
print(x.grad)

print('------------------------------------------------------')
# Generally speaking, `torch.autograd` is an engine for computing vector-Jacobian 
# product.
x = torch.randn(3, requires_grad=True)
y = x + 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

print('------------------------------------------------------')
# Now in this case `y` is no longer a scalaar. `torch.autograd` could not 
# compute the full Jacobian directly, but if we just want the vector-Jacobian
# product, simply pass the vector to `backward` as argument
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print('------------------------------------------------------')
# you can also stop autograd from tracking histroy on Tensors with 
# `.requires_grad=True` either by wrapping the code block in `with torch.no_grad():`
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    
print('------------------------------------------------------')
# Or by using `.detach()` to get a new Tensor with the same content but that 
# does not require gradients:
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
