import jax.numpy as jnp


class Transform:
    def forward(self, x) -> jnp.DeviceArray:
        ...
    def inverse(self, x) -> jnp.DeviceArray:
        ...


class Bounded(Transform):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def forward(self, x):
        return self.low + (self.high - self.low) / (1 + jnp.exp(-x))
    def inverse(self, x):
        return jnp.log(x - self.low) - jnp.log(self.high - x)


class Exponential(Transform):
    def forward(self, x):
        return jnp.exp(x)
    def inverse(self, x):
        return jnp.log(x)


class Union(Transform):
    def __init__(self, *transforms):
        self.transforms = transforms
    def forward(self, x):
        for t in self.transforms:
            x = t.forward(x)
        return x
    def inverse(self, x):
        for t in self.transforms[::-1]:
            x = t.inverse(x)
        return x


class Scale(Transform):
    def __init__(self, scale):
        self.scale = scale
    def forward(self, x):
        return self.scale * x
    def inverse(self, x):
        return x / self.scale


class Shift(Transform):
    def __init__(self, shift):
        self.shift = shift
    def forward(self, x):
        return x + self.shift
    def inverse(self, x):
        return x - self.shift


class Sin(Transform):
    def __init__(self, amplitude):
        self.amplitude = amplitude
    def forward(self, x):
        return self.amplitude * jnp.sin(x)
    def inverse(self, x):
        return jnp.arcsin(x) / self.amplitude   
