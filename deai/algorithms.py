from deai.aggregators import Average, LatestPerWorker, WeightedAverage, Latest
from deai.types import Message, TrainState, WorkerState

class FedAlgo:
    def __init__(self, server_agg, worker_agg):
        self.server_agg = server_agg
        self.worker_agg = worker_agg
        self.filter = LatestPerWorker()

    def init_worker_state(self, train_state: TrainState):
        return WorkerState(train_state=train_state, extras=None)
    
    def init_server_state(self, train_state: TrainState):
        return WorkerState(train_state=train_state, extras=None)
    
    def update_server_state(self, server_state: WorkerState, messages: list[Message]) -> WorkerState:
        raise NotImplementedError
    
    def update_worker_state(self, worker_state: WorkerState, messages: list[Message]) -> WorkerState:
        raise NotImplementedError
    
    def send_server_var(self, server_state: WorkerState) -> any:
        raise NotImplementedError
    
    def send_worker_var(self, worker_state: WorkerState) -> any:
        raise NotImplementedError

class FedAvg(FedAlgo):
    def __init__(self, server_agg, worker_agg):
        super().__init__(server_agg, worker_agg)
        # TODO move to config to pick these. Average or WeightedAverage...
        self.server_agg = server_agg
        self.worker_agg = worker_agg
        self.filter = LatestPerWorker() # Server usually filters to get latest from each worker

    def update_server_state(self, server_state: WorkerState, messages: list[Message]):
        """
        Update the server state using the FedAvg algorithm.
        """
        messages = self.filter(messages) # Ensure latest message from each worker
        params = self.server_agg(messages) # Average worker parameters
        return WorkerState(train_state=TrainState(params=params, 
                                                  opt_state=server_state.train_state.opt_state),
                           extras=server_state.extras)
    
    def update_worker_state(self, worker_state: WorkerState, messages: list[Message]) -> WorkerState:
        """
        Update the worker state using the FedAvg algorithm.
        """
        return WorkerState(train_state=TrainState(params=self.worker_agg(messages),
                                                    opt_state=worker_state.train_state.opt_state),
                            extras=worker_state.extras)

    def send_server_var(self, server_state: WorkerState) -> any:
        """
        What the server should send to the worker in FedAvg.
        In FedAvg, the server sends the updated model parameters.
        """
        return server_state.train_state.params

    def send_worker_var(self, worker_state: WorkerState) -> any:
        """
        What the worker should send to the server in FedAvg.
        In FedAvg, workers send their model parameters.
        """
        return worker_state.train_state.params
    

# class FedDiff(FedAvg):
#     def __init__(self, trainer_config):
#         super().__init__()
#         self.trainer = build_trainer(trainer_config)

#     def init_server_state(self, train_state):
#         opt_state = self.trainer.init(train_state.params)
#         return WorkerState(train_state=train_state, extras=opt_state)

#     def update_server_state(self, server_state, messages):
#         messages = self.filter(messages)  
#         # make sure we only use the latest message from each worker
#         # NOTE maybe strange behavior if we dont filter? (when paired w weighted average)
#         grads = self.avg_agg(messages)
#         train_state = TrainState(params=server_state.train_state.params, opt_state=server_state.extras)
#         return self.trainer.apply_updates(train_state, grads)

#     def send_worker_var(self, worker_state):
#         if hasattr(self, '_old_params'):
#             old_params = self._old_params
#         else:
#             old_params = worker_state.train_state.params

#         diff = tree.map(lambda params, old_params: tree.map(lambda p, op: p-op, params, old_params), train_state.params, old_params)
#         self._old_params = copy.deepcopy(train_state.params)
#         return diff