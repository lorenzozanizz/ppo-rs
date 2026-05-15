
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_config_defaults() {
        let cfg = PpoConfig::default();
        assert_eq!(cfg.batch_size(), 4 * 128);
        assert_eq!(cfg.minibatch_size(), 128);
        assert_eq!(cfg.num_updates(), 25000 / 512);
    }
}