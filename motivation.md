Why do we need deAI?

> Each block needs to store model weights, thus not suitable for training of large models.

#1 scaling cluster infrastructure is hard
i have heard that building really large clusters is hard / does not scale well. (i am partly skeptical of this claim of scaling poorly. from what I heard its more about cooling and power issues than compute)
decentralised training allows you to train across multiple (smaller) clusters.

#2 synchronicity doesn't scale well.
maybe you have 10,000 GPUs in a cluster. all you need is one GPU failure (or consistently late) and your training is broken / suboptimal.
and synchronicity is complex / costly. it requires specialized high-speed in-terconnects.

#3 leveraging hetrogenous devices
current distributed training requires all devices to be the same.

(all technical answers I've borrowed from papers)
i have my own (non technical, rather political) motivation which relates to democratising the control of AI. i need a sec to draft that clearly.


***

Let’s break down how blockchain addresses trustless coordination, auditability, and incentive alignment in decentralized ML systems, along with practical examples and trade-offs:

1. Trustless Coordination
What it solves:
In decentralized networks, participants (workers/nodes) may not trust each other. Without a central authority, ensuring honest collaboration (e.g., valid model updates) is critical. Blockchain replaces trust in a single entity with cryptographic guarantees and consensus mechanisms.

How it works:

Consensus Algorithms:

Proof-of-Stake (PoS): Nodes "stake" tokens to validate updates. Malicious actors lose their stake.

Practical Byzantine Fault Tolerance (PBFT): Nodes vote on the validity of updates.

Example: In ML training, workers compute gradients and submit them to the blockchain. Validators check for consistency (e.g., via zk-proofs or outlier detection).

Smart Contracts:

Encode rules for collaboration (e.g., "gradients must pass validation checks before aggregation").

Example: A smart contract on Ethereum automates gradient aggregation and penalizes workers submitting invalid updates.

Practical Use Cases:

Decentralized Federated Learning: Hospitals collaboratively train a model. A blockchain enforces that only cryptographically signed, validated updates (e.g., using homomorphic encryption) are aggregated.

Open-AI Networks: Public contributors train a model (e.g., Bittensor), with consensus ensuring no single party dominates the training process.

Limitations:

Overhead: Consensus and validation add latency (unsuitable for real-time training).

Scalability: Most blockchains (e.g., Ethereum) can’t handle high-frequency ML updates.

2. Auditability
What it solves:
Regulated industries (e.g., healthcare, finance) require traceability of model behavior. Blockchain provides an immutable ledger to track data provenance, model updates, and participant contributions.

How it works:

Hashing: Store hashes of model checkpoints, data batches, or gradients on-chain.

Zero-Knowledge Proofs (ZKPs): Prove compliance (e.g., "data was preprocessed correctly") without revealing raw data.

Example: A hospital logs a hash of its local dataset on-chain. During federated training, hashes of model updates are recorded, allowing auditors to verify data usage.

Practical Use Cases:

GDPR Compliance: A pharma company proves patient data was anonymized and used only for approved purposes.

Model Governance: Trace which training data caused a biased prediction (e.g., IBM Watson).

Limitations:

Storage Costs: Storing large ML artifacts (e.g., model weights) on-chain is impractical.

Privacy Risks: Even hashes can leak metadata; solutions like ZKPs add complexity.

3. Incentive Alignment
What it solves:
Participants need motivation to contribute resources (data, compute). Blockchain enables tokenized economies where contributions are rewarded programmatically.

How it works:

Token Rewards: Workers earn tokens for submitting valid updates (e.g., Ocean Protocol rewards data providers).

Slashing: Malicious actors lose staked tokens (e.g., Bittensor penalizes low-quality model updates).

Decentralized Marketplaces: Platforms like SingularityNET let users buy/sell AI services with tokens.

Practical Use Cases:

Federated Learning Marketplaces: A self-driving car company pays token rewards to edge devices contributing sensor data.

Compute Sharing: Users rent idle GPUs for training LLMs, earning tokens (e.g., Gensyn).

Limitations:

Freeloading: Participants might submit low-effort work; sybil attacks require mitigation (e.g., proof-of-work).

Valuation Complexity: Quantifying the "value" of data/compute contributions is subjective.