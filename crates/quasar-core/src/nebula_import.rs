//! Nebula audio import bridge.
//! Deserializes nebula-audio's bincode-serialized AcousticOutput into Quasar's
//! AcousticProbeGrid. No compile-time dependency on nebula-audio.

use serde::{Deserialize, Serialize};
use crate::bands::{Band8, FREQ_BAND_COUNT};
use crate::error::SpatialAudioError;
use crate::probe_grid::{AcousticProbe, AcousticProbeGrid};

// ── Mirror types matching nebula-audio's binary layout ────────────────────

/// Mirrors nebula_audio::output::AcousticOutput
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NebulaAcousticOutput {
    pub impulse_responses: Vec<NebulaImpulseResponse>,
    pub reverb_zones:      Vec<NebulaReverbZone>,
    pub config_json:       String,
}

/// Mirrors nebula_audio::output::ImpulseResponse
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NebulaImpulseResponse {
    pub listener_position:     [f32; 3],
    pub sample_rate:           u32,
    pub bands:                 [Vec<f32>; FREQ_BAND_COUNT],
    pub t60_per_band:          [f32; FREQ_BAND_COUNT],
    pub broadband_t60:         f32,
    pub early_late_split_secs: f32,
}

/// Mirrors nebula_audio::output::ReverbZone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NebulaReverbZone {
    pub aabb_min:     [f32; 3],
    pub aabb_max:     [f32; 3],
    pub t60:          f32,
    pub edt:          f32,
    pub c80:          f32,
    pub d50:          f32,
    pub room_gain_db: f32,
    pub drr_db:       f32,
    pub absorption:   [f32; FREQ_BAND_COUNT],
}

// ── Conversion logic ─────────────────────────────────────────────────────

/// Deserialize nebula-audio's bincode-serialized AcousticOutput and convert
/// to Quasar's AcousticProbeGrid.
///
/// The probe grid is constructed from the impulse response list.
/// If the nebula bake used regularly-spaced listener points, the probes
/// are arranged into a grid. Otherwise, they are treated as an irregular
/// set (grid_dims = [n, 1, 1], grid_spacing = [0, 0, 0]).
///
/// # Arguments
/// * `bytes` — Bincode-serialized nebula_audio::AcousticOutput
///
/// # Returns
/// * `AcousticProbeGrid` — Quasar runtime probe grid
///
/// # Errors
/// * `SpatialAudioError::Deserialize` if bincode decoding fails
/// * `SpatialAudioError::ProbeGrid` if the data is invalid
pub fn nebula_bytes_to_probe_grid(bytes: &[u8]) -> Result<AcousticProbeGrid, SpatialAudioError> {
    let nebula_output: NebulaAcousticOutput = bincode::serde::decode_from_slice(
        bytes,
        bincode::config::standard(),
    )
    .map_err(|e| SpatialAudioError::Deserialize(e.to_string()))?
    .0;

    let probes: Vec<AcousticProbe> = nebula_output
        .impulse_responses
        .into_iter()
        .map(|ir| {
            let n_samples = ir.bands[0].len();
            let mut rir_samples: Vec<Band8> = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut values = [0.0_f32; FREQ_BAND_COUNT];
                for b in 0..FREQ_BAND_COUNT {
                    values[b] = ir.bands[b].get(i).copied().unwrap_or(0.0);
                }
                rir_samples.push(Band8(values));
            }

            AcousticProbe {
                position: ir.listener_position,
                rir_samples,
                sample_rate: ir.sample_rate,
                t60: Band8(ir.t60_per_band),
                broadband_t60: ir.broadband_t60,
                early_late_split_secs: ir.early_late_split_secs,
            }
        })
        .collect();

    let n_probes = probes.len();

    if n_probes == 0 {
        return Err(SpatialAudioError::ProbeGrid(
            "nebula audio data contains zero impulse responses".to_string(),
        ));
    }

    let grid_dims: [u32; 3] = [n_probes as u32, 1, 1];
    let grid_origin: [f32; 3] = probes.first()
        .map(|p| p.position)
        .unwrap_or([0.0; 3]);
    let grid_spacing: [f32; 3] = [0.0, 0.0, 0.0];

    AcousticProbeGrid::new(probes, grid_origin, grid_spacing, grid_dims)
}

/// Deserialize nebula-audio's reverb zones from baked data.
///
/// # Arguments
/// * `bytes` — Bincode-serialized nebula_audio::AcousticOutput
///
/// # Returns
/// * `Vec<NebulaReverbZone>` — Deserialized reverb zones
///
/// # Errors
/// * `SpatialAudioError::Deserialize` if bincode decoding fails
pub fn nebula_bytes_to_reverb_zones(bytes: &[u8]) -> Result<Vec<NebulaReverbZone>, SpatialAudioError> {
    let nebula_output: NebulaAcousticOutput = bincode::serde::decode_from_slice(
        bytes,
        bincode::config::standard(),
    )
    .map_err(|e| SpatialAudioError::Deserialize(e.to_string()))?
    .0;

    Ok(nebula_output.reverb_zones)
}

/// Extract the config JSON string from nebula-audio baked data.
///
/// # Arguments
/// * `bytes` — Bincode-serialized nebula_audio::AcousticOutput
///
/// # Returns
/// * `String` — The config JSON string
///
/// # Errors
/// * `SpatialAudioError::Deserialize` if bincode decoding fails
pub fn nebula_bytes_to_config_json(bytes: &[u8]) -> Result<String, SpatialAudioError> {
    let nebula_output: NebulaAcousticOutput = bincode::serde::decode_from_slice(
        bytes,
        bincode::config::standard(),
    )
    .map_err(|e| SpatialAudioError::Deserialize(e.to_string()))?
    .0;

    Ok(nebula_output.config_json)
}
