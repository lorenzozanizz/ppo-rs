
use num::{ Float };

/// Single environment interface
///
///
pub trait Environment: Send {

    type Scalar: Float;
    type Action: Clone + Send;
    type Observation: Clone + Send;
    type ActionSpace;
    type ObservationSpace;

    ///
    ///
    ///
    ///
    ///
    ///
    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation, Self::Scalar>;

    ///
    ///
    ///
    ///
    ///
    ///
    fn reset(&mut self) -> Self::Observation;

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

///
///
///
pub struct StepResult<ObsType, Scalar: Float> {
    pub observation: ObsType,
    pub reward: Scalar,
    pub done: bool,
    /// Metadata information about the step, used for debugging and logging
    pub info: std::collections::HashMap<String, String>,
}

/// Vectorized environment (multiple envs running)
pub struct VectorEnv<E: Environment> {
    envs: Vec<E>,
}

impl<E: Environment> VectorEnv<E> {

    pub fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }

    pub fn step(&mut self, actions: Vec<E::Action>) -> VectorStepResult<E::Observation, E::Scalar> {
        let mut observations = Vec::new();
        let mut rewards = Vec::new();
        let mut dones = Vec::new();

        for (env, action) in self.envs.iter_mut().zip(actions) {
            let result = env.step(action);
            observations.push(result.observation);
            rewards.push(result.reward);
            dones.push(result.done);
        }

        VectorStepResult {
            observations,
            rewards,
            dones,
        }
    }

    pub fn reset(&mut self) -> Vec<E::Observation> {
        self.envs.iter_mut().map(|e| e.reset()).collect()
    }

}

pub struct VectorStepResult<ObsType, Scalar: Float> {

    pub observations: Vec<ObsType>,
    pub rewards: Vec<Scalar>,
    pub dones: Vec<bool>,
}


pub trait Agent<Scalar: Float = f32> {
    type Action: Clone + Send;


}


fn main() {


}
