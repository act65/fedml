
class Decentralized(Base):
    """
    This class handles training in a decentralized setting.
    """
    def __init__(self, worker_id, ip, port, peer_ips, peer_ports, dataset_config, network_config, trainer_config):
        super().__init__(worker_id, ip, port, peer_ips, peer_ports)
        self.comms_policy = PeriodicPolicy(trainer_config.send_interval)
        self.aggregator = Average()

        dataset_config.update({'silo_id': None})
        self.dataset = build_dataset(dataset_config)

        self.inner_trainer = build_trainer(self.dataset.shapes, network_config, trainer_config.inner)
        self.outer_trainer = build_trainer(self.dataset.shapes, network_config, trainer_config.outer)

    def run_training(self,
                     n_steps,
                     batch_size,
                     model_save_path,
                     seed=0):
        
        key = random.PRNGKey(seed)
        data_generator = self.dataset.data_generator(key, batch_size)

        # TODO support pulling params from elsewhere
        # since we initialised with the same seed, params are the same
        inner_train_state = self.inner_trainer.initial_train_state(key)
        outer_train_state = self.outer_trainer.initial_train_state(key)
        
        for i in range(n_steps):
            # Local training
            batch = next(data_generator)
            inner_train_state, loss = self.inner_trainer.update(inner_train_state, batch)
            print(f"Worker {self.worker_id}, Step {i}, Loss {loss:.2f}")#, end="\r", flush=True)
            # everything below should be run not on the GPU

            if self.comms_policy.should_send(i):
                # outer train state still holds the t-1 params
                # n_step_grad = model(t) - model(t-1)
                n_step_inner_grad = tree.map(lambda params_t, params_tm1: params_t-params_tm1, inner_train_state.params, outer_train_state.params)
                message = Message(
                    var=n_step_inner_grad, 
                    metadata={'port': self.comms_manager.port})
                
                # Push to peers (non-blocking)
                # add noise locally, before we send, for differential privacy
                # NOTE if you add noise seperately for each peer this introduces a secutiry vulnerability
                # since a malicious actor could be running multiple peers and average out the noise
                # TODO use a different, local key here
                key, subkey = random.split(key)
                noisy_n_step_inner_grad = add_noise(subkey, n_step_inner_grad, 0.01)
                for peer_ip, peer_port in zip(self.comms_manager.peer_ips, self.comms_manager.peer_ports):
                    noisy_messsage = Message(
                        var=noisy_n_step_inner_grad, 
                        metadata={'port': self.comms_manager.port})
                    self.comms_manager.send(peer_ip, peer_port, noisy_messsage)

            if self.comms_policy.should_receive(i):
                if self.comms_manager.n_available_messages > 0:
                    # Aggregate received updates
                    grads = self.aggregator(message, self.comms_manager.get_available_messages())
                    # NOTE we can use the non-noisy local grad here since this update remains local (we never send the model params)
                    
                    # Update outer model
                    outer_train_state = self.outer_trainer.apply_updates(outer_train_state, grads)

                    # Update inner params
                    inner_train_state.replace(params=outer_train_state.params)
                    # NOTE should we reset the inner opt state here?
