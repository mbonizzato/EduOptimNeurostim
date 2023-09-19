import random
import numpy as np
import torch
import gpytorch


class GreedySearch(object):
    """
    Algorithm class for greedy search. In greedy search, one dimension or direction is optimized
    at any given time. For example, if you have a 2D rectangle search space with width A and
    length B, this algorithm would take the following steps:
    1. Randomly pick between the width (A) and length (B) directions
    2. Given an initial position (a, b) and the direction from step 1, get all responses
    3. Given all responses in the chosen row (A) or column (B) depending on step 1,
       find the position that gives the highest response.
    4. Repeat steps 1, 2, and 3 sequentially as long as the algorithm is called
    """
    def __init__(self, ch2xy, shape):
        self.ch2xy = ch2xy
        self.dims = list(range(len(shape)))
        self.current_dim = random.choice(self.dims)
        self.next_dims = [dim for dim in self.dims if dim != self.current_dim]
        self.next_channels = []
        self.responses = {}

    def curate_next_channels(self, channel):
        """Function to curate next channels given a channel and the current dimension direction"""
        # Make sure the given channel is of correct shape
        assert len(channel) == len(self.dims)
        # Sample all viable next channels, i.e., channels with same the same value on the given dim
        self.next_channels = self.ch2xy[self.ch2xy[:, self.current_dim] == channel[self.current_dim]].tolist()
        # Reset responses for the next round
        self.responses = {}

    def get_random_channel(self):
        """Function to sample a random channel from the given list of viable next channels"""
        # Make sure that the list of viable next channels is not empty
        assert self.next_channels
        return random.choice(self.next_channels)

    def record_response(self, channel, response):
        """Function to keep a record of responses each channel receives"""
        # Update the `responses` dictionary and remove the channel from next channels to visit
        self.responses[tuple(channel)] = response
        self.next_channels.remove(channel)

        # If no next channel is left
        if not self.next_channels:
            # If no next dim to visit is left, re-generate next dims list
            if not self.next_dims:
                self.next_dims = self.dims.copy()
                self.next_dims.remove(self.current_dim)
                # NOTE: We still have to make sure the prev. dim will not be repeated

            # Compute the best channel given the recorded responses in the current round
            best_channel = list(max(self.responses, key=lambda x: self.responses.get(x)))

            # Select the dim to traverse for the next round and update `next_dims` accordingly
            self.current_dim = random.choice(self.next_dims)
            self.next_dims.remove(self.current_dim)

            # Curate next channels given the best channel and `current_dim`
            self.curate_next_channels(channel=list(best_channel))


class PriorMeanGPy(object):
    """Class that represents a probability mean function (PMF) for the `GPy` library"""
    def __init__(self, prior, ch2xy, scale):
        self.prior = prior
        self.ch2xy = ch2xy
        self.scale = scale

    def __call__(self, x):
        y = np.array([self.prior[self.ch2xy.tolist().index(xi.tolist())] for xi in x])
        y = np.reshape(y, (len(x), 1)) * self.scale
        return y


class PriorMeanGPytorch(gpytorch.means.Mean):
    """Class that represents a probability mean function (PMF) for the `gpytorch` library"""
    def __init__(self, prior_map):
        super().__init__()
        self.register_parameter('prior_map', torch.nn.Parameter(prior_map, requires_grad=False))

    def forward(self, x):
        # Subtracting 1 from input is not conventional -- we just observed better results like this
        x = x - 1

        # Adjust (reshape, chunk, etc.) based on the shape
        if x.shape[0] == 1:
            prior_map_ = self.prior_map[tuple(x[0].long())].reshape(1, )
        else:
            x = x.long()
            prior_map_ = self.prior_map[x.t().chunk(chunks=6, dim=0)].flatten()

        return prior_map_


class GP(gpytorch.models.ExactGP):
    """Class that extends ExactGP for GPU-based training of Gaussian Process models"""
    def __init__(self, train_x, train_y, likelihood, kernel, prior_map=None, system_shape=None):
        super(GP, self).__init__(train_x, train_y, likelihood)

        # Initialize mean module dependent on whether prior map is specified or not
        if prior_map is not None:
            # Reshape and assign the prior map as the mean module
            prior_map = torch.tensor(prior_map, dtype=torch.float).view(*system_shape)
            self.mean_module = PriorMeanGPytorch(prior_map)
            self.mean_module.requires_grad = False
        else:
            self.mean_module = gpytorch.means.ZeroMean()

        # Initialize covariance module
        self.covariance_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covariance_x = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)


def optimize(model, likelihood, training_iter, train_x, train_y, learning_rate=0.01):
    """Optimizes GP model with the Adam optimizer and marginal log likelihood loss"""
    # Setup optimizer which will optimize GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss criterion for GPs is chosen as the marginal log likelihood (MLL)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass to get output from model
        output = model(train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, train_y).mean()
        loss.backward()
        optimizer.step()

    return model, likelihood
