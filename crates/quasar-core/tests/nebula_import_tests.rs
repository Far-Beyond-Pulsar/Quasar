//! Tests for the Nebula integration bridge.
//! These tests verify that we can deserialize nebula-audio's bincode format
//! and convert it to Quasar's probe grid.

#[cfg(feature = "nebula-import")]
mod nebula_import_tests {
    use quasar_core::nebula_import::nebula_bytes_to_probe_grid;

    /// Helper: create minimal valid nebula-format bincode bytes for testing.
    fn create_minimal_nebula_bytes() -> Vec<u8> {
        use serde::Serialize;

        #[derive(Serialize)]
        struct TestIR {
            listener_position: [f32; 3],
            sample_rate: u32,
            bands: [Vec<f32>; 8],
            t60_per_band: [f32; 8],
            broadband_t60: f32,
            early_late_split_secs: f32,
        }

        #[derive(Serialize)]
        struct TestOutput {
            impulse_responses: Vec<TestIR>,
            reverb_zones: Vec<()>,
            config_json: String,
        }

        let output = TestOutput {
            impulse_responses: vec![TestIR {
                listener_position: [1.0, 2.0, 3.0],
                sample_rate: 44100,
                bands: [
                    vec![0.1, 0.05],
                    vec![0.2, 0.04],
                    vec![0.3, 0.03],
                    vec![0.4, 0.02],
                    vec![0.5, 0.01],
                    vec![0.6, 0.005],
                    vec![0.7, 0.003],
                    vec![0.8, 0.001],
                ],
                t60_per_band: [0.5; 8],
                broadband_t60: 0.5,
                early_late_split_secs: 0.08,
            }],
            reverb_zones: vec![],
            config_json: "{}".to_string(),
        };

        bincode::serde::encode_to_vec(&output, bincode::config::standard()).unwrap()
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_valid() {
        let bytes = create_minimal_nebula_bytes();
        let grid = nebula_bytes_to_probe_grid(&bytes);
        assert!(grid.is_ok(), "should successfully parse valid nebula bytes: {:?}", grid.err());
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_probes() {
        let bytes = create_minimal_nebula_bytes();
        let grid = nebula_bytes_to_probe_grid(&bytes).unwrap();
        assert_eq!(grid.len(), 1, "should have 1 probe");
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_probe_data() {
        let bytes = create_minimal_nebula_bytes();
        let grid = nebula_bytes_to_probe_grid(&bytes).unwrap();
        let probe = &grid.probes[0];
        assert_eq!(probe.position, [1.0, 2.0, 3.0]);
        assert_eq!(probe.sample_rate, 44100);
        assert_eq!(probe.broadband_t60, 0.5);
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_rir_samples() {
        let bytes = create_minimal_nebula_bytes();
        let grid = nebula_bytes_to_probe_grid(&bytes).unwrap();
        let probe = &grid.probes[0];
        assert_eq!(probe.rir_samples.len(), 2, "should have 2 RIR samples");
        assert_eq!(probe.rir_samples[0].0.len(), 8);
        assert!((probe.rir_samples[0].0[0] - 0.1).abs() < 1e-6);
        assert!((probe.rir_samples[1].0[0] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_empty_ir_list() {
        #[derive(serde::Serialize)]
        struct EmptyOutput {
            impulse_responses: Vec<()>,
            reverb_zones: Vec<()>,
            config_json: String,
        }
        let output = EmptyOutput {
            impulse_responses: vec![],
            reverb_zones: vec![],
            config_json: "{}".to_string(),
        };
        let bytes = bincode::serde::encode_to_vec(&output, bincode::config::standard()).unwrap();
        let result = nebula_bytes_to_probe_grid(&bytes);
        assert!(result.is_err(), "empty IR list should produce error");
    }

    #[test]
    fn test_nebula_bytes_to_probe_grid_corrupt() {
        let corrupt_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let result = nebula_bytes_to_probe_grid(&corrupt_bytes);
        assert!(result.is_err(), "corrupt bytes should produce error");
    }

    #[test]
    fn test_nebula_bytes_empty() {
        let result = nebula_bytes_to_probe_grid(&[]);
        assert!(result.is_err(), "empty bytes should produce error");
    }
}
