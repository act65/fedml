Goal: build a decentralised federated learner for the IBM AML dataset.

1. Using jax / pytorch to hangle inter-node communication outside of a cluster seems like too much of a sharp edge atm.
Instead, use flask / http to handle communication between nodes.
Not the fastest / most efficient, but it's simple. 
(maybe move to gRPC later)

2. Use a low communication variant of federated learning.
Similar to DiLoCo. Train inner loop x 100, then communicate, and update outer loop.
6. Sync v async?
7. Send grads v diffs v models?

3. Use an AWS virtual private cloud (VPC) to simulate the network.
This way the network is private.
We don't have to worry about others meddling with our network.
Should also give low latency.

4. Use AWS EC2 ??? instances in the same region to host the nodes.
Decentralised could support geographically distributed, pre-emptible / spot instances, and / or hetrogenous hardware.
But for now, just use the same instance type, and don't worry about pre-emptible instances.

5. Privacy is achieved by adding noise to the gradients.
This is a form of differential privacy, it should suffice.

8. Use json to communicate between nodes.
This is simple, but not the most efficient.
(Can move to jax.serialize?)

9. 