import datetime
import jax
import jax.numpy as jnp
from jax import jit, random, pmap, tree
from flax.training import checkpoints
import optax

from deai.types import TrainState

def count_params(params):
    flat_params, _ = tree.flatten(params)
    return sum(map(lambda x: x.size, flat_params))

def clip_grad_norm(grad, max_norm):
    norms = tree.map(lambda g: jnp.linalg.norm(g.ravel()), grad)
    return tree.map(lambda g, n: jnp.where(n < max_norm, g, g * max_norm / n), grad, norms)

class Trainer:
    """
    A Learner class that supports training a `Model`.
    The model must define;
        - a `dldparams` method that returns the loss and the gradients.
        - an `initial_params` method that returns the initial params of the model.
    """
    def __init__(self, model, opt_name, learning_rate, decay_steps, decay_rate, **kwargs):
        self.model = model

        self.scheduler = optax.exponential_decay(learning_rate, decay_steps, decay_rate)

        if opt_name == "adam":
            self.optimizer = optax.adam(self.scheduler)
        elif opt_name == "sgd":
            self.optimizer = optax.sgd(self.scheduler)#, nesterov=True)
        else:
            raise ValueError("Optimizer not supported")

        self.apply_updates = jit(self.apply_updates)
        self.update = jit(self.update)
        self._pupdate = pmap(self._pupdate, axis_name='devices')
        self.pupdate = jit(self.pupdate)
        self.model.b_eval_fn = jit(self.model.b_eval_fn)
    
    def initial_train_state(self, key) -> TrainState:
        params = self.model.initial_params(key)
        opt_state = self.optimizer.init(params)
        return TrainState(
            params=params,
            opt_state=opt_state)

    def update(self, train_state: TrainState, batch):
        loss, grads = self.model.dldparams(train_state.params, *batch)
        # grads = tree_map(jnp.nan_to_num, grads)
        return self.apply_updates(train_state, grads), loss

    def apply_updates(self, train_state, grads):
        grads = clip_grad_norm(grads, 1.0)
        updates, opt_state = self.optimizer.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)
        return TrainState(
            params=new_params,
            opt_state=opt_state)

    def replicate_train_state(self, train_state):
        n_devices = jax.local_device_count()
        return TrainState(
            params=jax.tree_util.tree_map(lambda x: jnp.stack([x]*n_devices), train_state.params),
            opt_state=jax.tree_util.tree_map(lambda x: jnp.stack([x]*n_devices), train_state.opt_state))
    
    def replicate_batch(self, batch):
        n_devices = jax.local_device_count()
        return jax.tree_map(lambda x: jnp.stack([x]*n_devices), batch)
    
    def _pupdate(self, train_state: TrainState, batch):
        # needs train_state and batch to be replicated across devices
        loss, grads = self.model.dldparams(train_state['params'], batch)
        loss = jax.lax.pmean(loss, axis_name='devices')
        grads = jax.lax.pmean(grads, axis_name='devices')
        return self._apply_grads(grads, train_state), loss
    
    def pupdate(self, train_state: TrainState, batch):
        train_state = self.replicate_train_state(train_state)
        batch = self.replicate_batch(batch)
        return self._pupdate(train_state, batch)

    def load(self, model_load_path, train_state=None):
        train_state = checkpoints.restore_checkpoint(
            ckpt_dir=model_load_path, 
            target=train_state)
        return train_state
        
    def save(self, model_save_path, train_state):
        i = int(datetime.datetime.now().timestamp()*1000)
        checkpoints.save_checkpoint(
            ckpt_dir=model_save_path,
            target=train_state,
            step=i,
            overwrite=True,
            keep=1)
        
    def eval(self, train_state, data_generator):
        metrics = []
        for batch in data_generator:
            metrics.append(self.model.b_eval_fn(train_state.params, *batch))
        return sum(metrics)/len(metrics)