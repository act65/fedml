import pytest
from unittest.mock import MagicMock
import random

from deai.worker import FederatedEvaluator, BaseWorker  # Replace your_module
from deai.utils import build_trainer, build_dataset # Replace your_module
from deai.types import TrainState # Replace your_module

class MockMessage:
    def __init__(self, var):
        self.var = var

def test_basic_check():
    assert True

@pytest.fixture
def mock_build_dataset(mocker):
    return mocker.patch("deai.utils.build_dataset", autospec=True) # Replace your_module

@pytest.fixture
def mock_build_trainer(mocker):
    return mocker.patch("deai.utils.build_trainer", autospec=True) # Replace your_module

@pytest.fixture
def mock_base_worker(mocker):
    return mocker.patch("deai.worker.BaseWorker.__init__", autospec=True) # Replace your_module

@pytest.fixture
def mock_comms_manager(mocker):
    return mocker.Mock()

@pytest.fixture
def mock_trainer(mocker):
    trainer_mock = mocker.Mock()
    trainer_mock.initial_train_state.return_value = MagicMock(params="initial_params") # Example initial state
    trainer_mock.eval.return_value = 0.5 # Example metric value
    return trainer_mock

@pytest.fixture
def mock_test_dataset(mocker):
    dataset_mock = mocker.Mock()
    dataset_mock.all_silo_data_generator.return_value = range(2) # Example data generator
    return dataset_mock

@pytest.fixture
def mock_logger(mocker):
    return mocker.patch("your_module.FederatedEvaluator.logger") # Replace your_module


class TestFederatedEvaluator:

    def test_init(self, mock_base_worker, mock_build_dataset, mock_build_trainer, mock_test_dataset, mock_trainer):
        comms_config = {}
        dataset_config = {'original': 'value'}
        model_config = {}
        trainer_config = {}
        algo_config = {}

        mock_build_dataset.return_value = mock_test_dataset
        mock_build_trainer.return_value = mock_trainer

        evaluator = FederatedEvaluator(
            worker_id='evaluator_1',
            comms_config=comms_config,
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_config=trainer_config,
            algo_config=algo_config
        )

        mock_base_worker.assert_called_once_with(evaluator, 'evaluator_1', comms_config)
        mock_build_dataset.assert_called_once()
        call_args = mock_build_dataset.call_args.kwargs
        assert call_args['train'] is False
        assert call_args.get('silo_id') == -1
        assert evaluator.test_dataset == mock_test_dataset # Check if dataset is assigned
        mock_build_trainer.assert_called_once_with(mock_test_dataset, model_config, trainer_config)
        assert evaluator.trainer == mock_trainer # Check if trainer is assigned


    def test_run_training_no_messages(self,
                                     mock_comms_manager,
                                     mock_trainer,
                                     mock_test_dataset,
                                     mock_logger):
        mock_comms_manager.get_available_messages.return_value = [] # No messages available
        dataset_config = {}
        model_config = {}
        trainer_config = {}
        algo_config = {}
        comms_config = {'comms_manager': mock_comms_manager}

        evaluator = FederatedEvaluator(
            worker_id='evaluator_1',
            comms_config=comms_config,
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_config=trainer_config,
            algo_config=algo_config
        )
        evaluator.trainer = mock_trainer
        evaluator.test_dataset = mock_test_dataset


        evaluator.run_training(
            n_steps=10, # Doesn't matter as it's message driven
            model_save_path='/tmp/model', # Doesn't matter for this test
            batch_size=32,
            seed=0
        )

        mock_comms_manager.get_available_messages.assert_called()
        mock_trainer.eval.assert_not_called() # Eval should not be called if no messages
        mock_logger.info.assert_not_called() # No logging if no messages


    def test_run_training_with_messages(self,
                                        mock_comms_manager,
                                        mock_trainer,
                                        mock_test_dataset,
                                        mock_logger):
        # Simulate a message being available
        fake_params = {'layer1': [1, 2, 3]}
        mock_comms_manager.get_available_messages.side_effect = [
            [MockMessage(var=fake_params)], # First call returns a message
            [],                             # Second call returns no message, loop exits
        ]
        dataset_config = {}
        model_config = {}
        trainer_config = {}
        algo_config = {}
        comms_config = {'comms_manager': mock_comms_manager}


        evaluator = FederatedEvaluator(
            worker_id='evaluator_1',
            comms_config=comms_config,
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_config=trainer_config,
            algo_config=algo_config
        )
        evaluator.trainer = mock_trainer
        evaluator.test_dataset = mock_test_dataset


        evaluator.run_training(
            n_steps=10, # Doesn't matter as it's message driven
            model_save_path='/tmp/model', # Doesn't matter for this test
            batch_size=32,
            seed=0
        )

        mock_comms_manager.get_available_messages.assert_called()
        mock_trainer.eval.assert_called_once() # Eval should be called once a message is available
        mock_test_dataset.all_silo_data_generator.assert_called_once()
        mock_logger.info.assert_called_once() # Logging should happen when eval is run

        # Check if train_state.params was updated from the message
        assert evaluator.trainer.eval.call_args[0][0].params == fake_params
        mock_logger.info.assert_called_with(f"Server, Step 0, Metric 0.50000") # Check log output and metric value which is mocked to 0.5