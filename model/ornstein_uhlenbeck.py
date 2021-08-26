import numpy as np


class OrnsteinUhlenbeckActionNoise:
    """Source: https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab """

    def __init__(self, mu, sigma=.3, theta=.15, delta_t=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.delta_t = delta_t

        self.x = np.zeros_like(self.mu)

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.delta_t + self.sigma * np.sqrt(self.delta_t) \
                 * np.random.normal(size=self.mu.shape)
        return self.x
