import numpy as np
import jax.numpy as jnp
from jax import random
import os
from glob import glob
import json

from deai.logging import PrintLogger, FileLogger

class Dataset:
    def __init__(self, silo_id, n_silos):
        self.silo_id = silo_id
        self.n_silos = n_silos
        self.logger = FileLogger(f"Silo: {silo_id}", log_file_path=f"logs/silo-{silo_id}.log")

    @property
    def shapes(self):
        raise NotImplementedError

    def load_data(self, key, silo_id):
        raise NotImplementedError

    def data_generator(self, key, batch_size):
        raise NotImplementedError
    
    def all_silo_data_generator(self, key, batch_size):
        for silo_id in range(self.n_silos):
            self.logger.info(f"Loading data for silo {silo_id}")
            x, y = self.load_data(key, silo_id)
            n = len(x)
            assert n > batch_size
            for i in range(0, len(x), batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size]
    
    def infinite_data_generator(self, key, batch_size):
        self.logger.info("Starting infinite data generator")
        # BUG i think its possible to get stuck in this loop if no data is generated
        counter = 0
        while True:
            key, subkey = random.split(key)
            data_generated = False

            for data in self.data_generator(subkey, batch_size):
                yield data
                data_generated = True

            if not data_generated:
                raise ValueError("No data generated")
            
            self.logger.info(f"Finished epoch {counter}. Looping back to the start of the data")
            counter += 1
            
    def data_generator(self, key, batch_size):
        x, y = self.load_data(key, self.silo_id)
        n = len(x)
        assert n > batch_size
        for i in range(0, len(x), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]

class SimpleDataset(Dataset):
    """
    A very simple dataset for testing purposes.
    y = mx + c
    """
    def __init__(self, silo_id, n_silos=2, n_data=200, *args, **kwargs):
        super().__init__(silo_id, n_silos)
        self.silo_id = silo_id
        self.n_data = n_data
        self.weights = {i: n_data for i in range(n_silos)}

    @property
    def shapes(self):
        # shape of the input and output
        return ((4,), 1)
    
    def load_data(self, data_key, silo_id):
        silo_key = random.PRNGKey(silo_id)
        key = silo_key + data_key
        key, subkey = random.split(key)
        x = random.normal(subkey, (self.n_data, 4))
        y = jnp.sum(2*x + 1, axis=1).reshape((self.n_data, 1))
        return x, y

class FeSimple(Dataset):
    """
    A simple federaged dataset generator.
    """
    def __init__(self, silo_id, n_silos=2, n_components=3, dims=4, *args, **kwargs):
        super().__init__(silo_id, n_silos)
        self.n_silos = n_silos
        self.n_components = n_components
        self.dims = dims
        
        weight_vals = random.randint(random.PRNGKey(n_silos), n_silos, minval=100, maxval=5000)
        self.weights = {i: w for i, w in enumerate(weight_vals)}
        
    def load_data(self, key, silo_id):
        data_key = random.PRNGKey(silo_id)
        k1, k2, k3, k4, k5 = random.split(key, 5)
        
        self.mixture_weights = random.dirichlet(
            k1, alpha=jnp.ones(self.n_components), shape=(self.n_silos,)
        )
        
        # Component parameters (unique per mixture)
        self.component_means = random.normal(k2, (self.n_components, self.dims)) * 2
        self.component_stds = random.uniform(
            k3, (self.n_components, self.dims), minval=0.5, maxval=3
        )
        
        # Silo-specific label noise
        self.w_noise = 1 + random.uniform(k4, (1,), minval=-0.1, maxval=0.1)
        self.b_noise = random.normal(k4, (1,))

        # data heterogeneity
        n_data = self.weights[silo_id]

        # Sample mixture components
        choice_key, x_key = random.split(data_key)
        choices = random.categorical(
            choice_key,
            logits=jnp.log(self.mixture_weights[silo_id]),
            shape=(n_data,)
        )
        
        means = self.component_means[choices]
        stds = self.component_stds[choices]
        
        # Generate x with component-specific std
        e = random.normal(x_key, (n_data, self.dims))
        x = means + e * stds
        
        # Labels with silo-specific noise
        y = jnp.sum((2 * self.w_noise) * x + (1 + self.b_noise), axis=1)
        y = y.reshape((n_data, 1))

        return x, y

    @property
    def shapes(self):
        return ((self.dims,), 1)

class FeMNIST(Dataset):
    # using the femnist dataset from https://github.com/TalwalkarLab/leaf

    # TODO i believe that if we set datadir = train / test
    # the silo_id's should match up to each other.
    # silo_id=0==train.user = silo_id=0==test.user
    def __init__(self, silo_id, datadir, train=True, *args, **kwargs):
        if train:
            self.datadir = os.path.join(datadir, 'train')
        else:
            self.datadir = os.path.join(datadir, 'test')

        self.n_classes = 62  # TODO ? is this correct?

        self.user2path, self.weights, self.users = self.parse_data(self.datadir)

        self.n_users = len(self.users)  # 196
        super().__init__(silo_id, n_silos=self.n_users)

    def load_data(self, key, silo_id):
        user = self.users[silo_id]
        # Load the data for this worker during initialization
        self.logger.info(f"Loading data for user {user}")
        fname = self.user2path[user]
        with open(fname, "r") as f:
            data = json.load(f)
        
        images = jnp.array(data['user_data'][user]['x']).reshape(-1, 28, 28, 1)
        labels = jnp.array(data['user_data'][user]['y'])

        return images, labels

    @property
    def shapes(self):
        return ((28, 28, 1), 1)

    @staticmethod
    def parse_data(datadir):
        user2path = dict()
        weights = dict()
        users = []
        for fname in glob(os.path.join(datadir, "*.json")):
            with open(fname, "r") as f:
                data = json.load(f)
            del data['user_data']

            data['num_samples']
            data['users']

            user2path.update({u: fname for u in data['users']})
            # weights.update({u: w for u, w in zip(data['users'], data['num_samples'])})
            weights.update({i: w for i, w in enumerate(data['num_samples'])})
            # weights should be indexed by silo_id not user_id
            users += data['users']

        return user2path, weights, users

    def data_generator(self, key, batch_size, silo_id=None):
        """Yields shuffled batches of data using a JAX PRNG key."""
        # Shuffle the data using the provided key
        if silo_id is None:
            silo_id = self.silo_id
            
        images, labels = self.load_data(key, silo_id)

        key, subkey = random.split(key)
        perm = random.permutation(subkey, len(images))

        images = images[perm]
        labels = labels[perm]

        num_samples = len(images)
        # Generate batches
        for i in range(0, num_samples, batch_size):
            yield images[i:i+batch_size], labels[i:i+batch_size]

class GroupedFeMNIST(FeMNIST):
    # FeMNIST, but we merge silos to make fewer silos
    def __init__(self, silo_id, datadir, n_silos, train=True, *args, **kwargs):
        super().__init__(silo_id, datadir, train=train, *args, **kwargs)
        self.n_silos = n_silos  # overwrite the number of silos

        if n_silos > self.n_users:
            raise ValueError("Too many silos for the number of users")

        self.silo_map = np.array_split(np.arange(self.n_users), n_silos)

    def data_generator(self, key, batch_size):
        for silo_id_ in self.silo_map[self.silo_id]:
            yield from super().data_generator(key, batch_size, silo_id_)