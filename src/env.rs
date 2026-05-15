mod agent;

use std::fmt::Debug;
use std::collections::HashMap;

/// Core Environment trait.
/// Users implement this for their specific environment.
///
pub trait Environment: Send {

    /// The observation type returned by reset() and step()
    type Observation: Send + Sync + Clone + Debug;

    /// Action type (can be an integer for discrete actions, or an enum)
    type Action: Send + Sync + Clone + Debug;

    /// The action space, used to obtain info about its shape
    type ActionSpace;

    /// The observation space, used to obtain info about its shape
    type ObservationSpace;

    /// Reset the environment and return initial observation.
    fn reset(&mut self) -> Self::Observation;

    /// Execute one step. Returns (next_obs, reward, done, info).
    ///
    /// Information returned by environment after a step.
    /// Can be extended by users for custom metrics (episode return, length, etc.)
    fn step(&mut self, action: Self::Action) -> (Self::Observation, f32, bool, HashMap<String, String>);

    /// Return the number of action choices (for discrete spaces)
    fn num_actions(&self) -> usize;


    ///
    ///
    ///
    ///
    ///
    ///
    fn action_space(&self) -> &Self::ActionSpace;

    ///
    ///
    ///
    ///
    ///
    ///
    ///
    fn observation_space(&self) -> &Self::ObservationSpace;


    ///
    ///
    ///
    ///
    ///
    ///
    ///
    fn close(&mut self);

}

/// Runtime configuration for PPO training.
#[derive(Debug, Clone)]
pub struct PpoConfig {

    // Environment setup
    pub num_envs: usize,
    pub num_steps: usize,

    // Network architecture
    pub hidden_size: usize,

    // Learning (for the Adam optimizer)
    pub learning_rate: f32,
    pub anneal_lr: bool,
    pub max_grad_norm: f32,

    // PPO algorithm
    pub gamma: f32,                   // Discount factor
    pub gae_lambda: f32,              // GAE lambda
    pub use_gae: bool,                // Use GAE or n-step returns
    pub num_minibatches: usize,
    pub num_update_epochs: usize,
    pub clip_coeff: f32,              // Policy clip coefficient
    pub clip_v_loss: bool,            // Clip value function loss
    pub ent_coeff: f32,               // Entropy coefficient for the loss function
    pub vf_coeff: f32,                // Value function coefficient
    pub target_kl: Option<f32>,       // Early stopping threshold

    // Training
    pub total_timesteps: usize,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {

        Self {
            num_envs: 4,
            num_steps: 128,
            hidden_size: 64,
            learning_rate: 2.5e-4,
            anneal_lr: true,
            max_grad_norm: 0.5,
            gamma: 0.99,
            gae_lambda: 0.95,
            use_gae: true,
            num_minibatches: 4,
            num_update_epochs: 4,
            clip_coeff: 0.2,
            clip_v_loss: true,
            ent_coeff: 0.01,
            vf_coeff: 0.5,
            target_kl: None,
            total_timesteps: 25000,
            seed: 1,
        }
    }
}

impl PpoConfig {

    /// Derive batch size from environment and step configuration.
    /// As reported in [1],  this is the amount of items that are grouped, shuffled
    /// and used for a minibatch.
    pub fn batch_size(&self) -> usize {
        self.num_envs * self.num_steps
    }

    /// Derive minibatch size from batch size and num_minibatches.
    pub fn minibatch_size(&self) -> usize {
        self.batch_size() / self.num_minibatches
    }

    /// Total number of policy updates during training.
    pub fn num_updates(&self) -> usize {
        self.total_timesteps / self.batch_size()
    }

}
