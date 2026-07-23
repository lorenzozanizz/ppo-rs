use tch::{ nn, nn::Module, Kind, Tensor };

/// Agent trait: abstraction for any policy-value function.
/// Allows different implementations (neural nets, linear, etc.)
pub trait Agent: Send {
    /// Get value estimate for observation batch.
    /// Input shape: [batch_size, obs_dim, ...]
    /// Output shape: [batch_size]
    fn get_value(&self, obs: &Tensor) -> Tensor;

    /// Get action, log probability, entropy, and value.
    /// Input: obs shape [batch_size, obs_dim, ...]
    ///        action (optional) shape [batch_size] for discrete actions
    /// Output: (action [batch_size], logprob [batch_size], entropy [batch_size], value [batch_size])
    fn get_action_and_value(
        &self,
        obs: &Tensor,
        action: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor, Tensor);

    /// Returns the device (CPU or CUDA) the agent is on.
    fn device(&self) -> tch::Device;
}

/// Neural network agent with actor-critic architecture.
/// - Critic: observation -> scalar value
/// - Actor: observation -> logits over actions
pub struct NeuralNetAgent {
    critic: nn::Sequential,
    actor: nn::Sequential,
    device: tch::Device,
    num_actions: usize,
}
