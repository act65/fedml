import pytest
import datetime
import jax
import jax.numpy as jnp
from jax import random, tree
from flax.training import checkpoints
import optax

from deai.types import TrainState
from deai.trainer import count_params, clip_grad_norm, Trainer # Assuming the code is in your_module.py

# Dummy Model for testing Trainer
class DummyModel:
    def initial_params(self, key):
        return {'dense1': {'kernel': random.normal(key, (2, 3)), 'bias': jnp.zeros((3,))},
                'dense2': {'kernel': random.normal(key, (3, 1)), 'bias': jnp.zeros((1,))}}

    def dldparams(self, params, x, y):
        def loss_fn(params):
            y_pred = self.apply(params, x, method='apply') # Assuming apply exists, if not adjust
            return jnp.mean((y_pred - y)**2)
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        return loss, grads

    def b_eval_fn(self, params, x, y):
        y_pred = self.apply(params, x, method='apply') # Assuming apply exists, if not adjust
        return jnp.mean((y_pred - y)**2)

    def apply(self, params, x, method='apply'): # Dummy apply method
        x = x @ params['dense1']['kernel'] + params['dense1']['bias']
        x = jax.nn.relu(x)
        x = x @ params['dense2']['kernel'] + params['dense2']['bias']
        return x


def test_count_params():
    key = random.PRNGKey(0)
    params = DummyModel().initial_params(key)
    assert count_params(params) == 13 


class TestTrainer:
    def test_initial_train_state(self):
        model = DummyModel()
        trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
        key = random.PRNGKey(0)
        train_state = trainer.initial_train_state(key)
        assert isinstance(train_state, TrainState)
        assert isinstance(train_state.params, dict)

    def test_update_and_apply_updates(self):
        model = DummyModel()
        trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
        key = random.PRNGKey(0)
        train_state = trainer.initial_train_state(key)

        x_batch = random.normal(key, (2, 2)) # Batch size 2, input dim 2
        y_batch = random.normal(key, (2, 1)) # Batch size 2, output dim 1
        batch = (x_batch, y_batch)

        initial_params = train_state.params
        updated_train_state, loss = trainer.update(train_state, batch)

        assert isinstance(updated_train_state, TrainState)
        for l1, l2 in zip(tree.leaves(updated_train_state.params), tree.leaves(initial_params)):
            assert not jnp.array_equal(l1, l2)
        assert loss.shape == () # Loss should be a scalar

#     def test_replicate_methods(self):
#         model = DummyModel()
#         trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
#         key = random.PRNGKey(0)
#         train_state = trainer.initial_train_state(key)
#         x_batch = random.normal(key, (2, 2))
#         y_batch = random.normal(key, (2, 1))
#         batch = (x_batch, y_batch)

#         replicated_train_state = trainer.replicate_train_state(train_state)
#         replicated_batch = trainer.replicate_batch(batch)

#         n_devices = jax.local_device_count()
#         assert jax.tree_util.tree_leaves(replicated_train_state.params)[0].shape[0] == n_devices
#         assert jax.tree_util.tree_leaves(replicated_train_state.opt_state)[0].shape[0] == n_devices
#         assert jax.tree_util.tree_leaves(replicated_batch)[0].shape[0] == n_devices

#     def test_pupdate_and_apply_grads(self): # Combined test for pupdate and _apply_grads
#         model = DummyModel()
#         trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
#         key = random.PRNGKey(0)
#         train_state = trainer.initial_train_state(key)

#         x_batch = random.normal(key, (2, 2))
#         y_batch = random.normal(key, (2, 1))
#         batch = (x_batch, y_batch)

#         initial_params = train_state.params
#         updated_train_state, loss = trainer.pupdate(train_state, batch)

#         assert isinstance(updated_train_state, TrainState)
#         assert updated_train_state.params != initial_params # Params should be updated
#         assert loss.shape == () # Loss should be a scalar

    def test_eval_method(self):
        model = DummyModel()
        trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
        key = random.PRNGKey(0)
        train_state = trainer.initial_train_state(key)

        def dummy_data_generator():
            for _ in range(2): # Generate 2 batches
                x_batch = random.normal(key, (2, 2))
                y_batch = random.normal(key, (2, 1))
                yield (x_batch, y_batch)

        avg_metric = trainer.eval(train_state, dummy_data_generator())
        assert avg_metric.shape == () # Metric should be scalar


    def test_save_load_methods(self, tmpdir): # Use tmpdir fixture for temporary directory
        model = DummyModel()
        trainer = Trainer(model, opt_name="adam", learning_rate=0.01, decay_steps=100, decay_rate=0.9)
        key = random.PRNGKey(0)
        train_state = trainer.initial_train_state(key)

        save_path = tmpdir.mkdir("model_save")
        trainer.save(str(save_path), train_state)

        loaded_train_state = trainer.load(str(save_path), train_state)
        # loaded_train_state = TrainState(
        #     params=loaded_train_state['params'],
        #     opt_state=loaded_train_state['opt_state']
        # )

        assert loaded_train_state is not None
        assert tree.all(jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), loaded_train_state.params, train_state.params))
        assert tree.all(jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), loaded_train_state.opt_state, train_state.opt_state))