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

impl NeuralNetAgent {
    /// Create a new neural network agent.
    ///
    /// # Arguments
    /// - `vs`: Variable store (nn path)
    /// - `obs_dim`: Observation dimension (flattened)
    /// - `num_actions`: Number of discrete actions
    /// - `hidden_size`: Hidden layer width
    /// - `device`: Device to run on (CPU or CUDA)
    pub fn new(
        vs: &nn::Path,
        obs_dim: usize,
        num_actions: usize,
        hidden_size: usize,
        device: tch::Device,
    ) -> Self {
        let critic = nn::seq()
            .add(layer_init(
                nn::linear(vs / "critic_layer_1", obs_dim as i64, hidden_size as i64, Default::default()),
                std::f64::consts::SQRT_2,
                0.0,
            ))
            .add_fn(|x| x.tanh())
            .add(layer_init(
                nn::linear(vs / "critic_layer_2", hidden_size as i64, hidden_size as i64, Default::default()),
                std::f64::consts::SQRT_2,
                0.0,
            ))
            .add_fn(|x| x.tanh())
            .add(layer_init(
                nn::linear(vs / "critic_layer_3", hidden_size as i64, 1, Default::default()),
                1.0,
                0.0,
            ));

        let actor = nn::seq()
            .add(layer_init(
                nn::linear(vs / "actor_layer_1", obs_dim as i64, hidden_size as i64, Default::default()),
                std::f64::consts::SQRT_2,
                0.0,
            ))
            .add_fn(|x| x.tanh())
            .add(layer_init(
                nn::linear(vs / "actor_layer_2", hidden_size as i64, hidden_size as i64, Default::default()),
                std::f64::consts::SQRT_2,
                0.0,
            ))
            .add_fn(|x| x.tanh())
            .add(layer_init(
                nn::linear(vs / "actor_layer_3", hidden_size as i64, num_actions as i64, Default::default()),
                0.01,
                0.0,
            ));

        Self {
            critic: critic.to(device),
            actor: actor.to(device),
            device,
            num_actions,
        }
    }
}

impl Agent for NeuralNetAgent {
    fn get_value(&self, obs: &Tensor) -> Tensor {
        self.critic.forward(obs).squeeze_dim(-1)
    }

    fn get_action_and_value(
        &self,
        obs: &Tensor,
        action: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let logits = self.actor.forward(obs);

        // Compute log probabilities and sample action if not provided
        let (action_result, logprob, entropy) = {
            let logprobs_full = logits.log_softmax(-1, Kind::Float);
            let probs = logits.softmax(-1, Kind::Float);

            let act = match action {
                Some(a) => a.shallow_clone(),
                None => {
                    // Sample from categorical distribution
                    let sampled = Tensor::multinomial(&probs, 1, false);
                    sampled.squeeze_dim(-1)
                }
            };

            // Gather log probabilities for taken actions
            let act_long = act.to_kind(Kind::Int64);
            let logprob = logprobs_full.gather(1, &act_long.unsqueeze(-1), false).squeeze_dim(-1);

            // Compute entropy: -sum(p * log(p))
            let entropy = -(probs * logprobs_full).sum_dim_intlist(
                &[-1i64],
                false,
                Kind::Float,
            );

            (act, logprob, entropy)
        };

        let value = self.get_value(obs);

        (action_result, logprob, entropy, value)
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

/// Initialize a linear layer with orthogonal weights and constant bias.
///
/// This follows the initialization scheme from [1]
fn layer_init(
    mut layer: nn::Linear,
    std: f64,
    bias_const: f64,
) -> nn::Linear {
    // Orthogonal initialization for weights
    let shape = layer.ws.size();
    let mut vs_init = tch::nn::VarStore::new(tch::Device::Cpu);
    let weights = tch::nn::init::ortho(&vs_init.root(), &[shape[0], shape[1]], std);
    let _ = tch::no_grad(|| {
        layer.ws.copy(&weights);
    });

    // Constant initialization for bias
    tch::no_grad(|| {
        layer.bs.fill_(bias_const);
    });

    layer
}
