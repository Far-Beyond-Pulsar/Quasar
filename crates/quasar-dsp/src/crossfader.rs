use quasar_core::param_exchange::SpatialCoefficients;

/// Manages smooth transitions between spatial coefficient sets.
///
/// The compute thread publishes new params; the audio thread reads them
/// and crossfades from current → target over a fixed window.
///
/// Crossfade curve: g₀(t) = cos(πt/2), g₁(t) = sin(πt/2) for t ∈ [0,1]
/// This ensures constant perceived power during transition (g₀² + g₁² = 1).
pub struct EqualPowerCrossfader {
    current: SpatialCoefficients,
    target: SpatialCoefficients,
    fade_frames: u32,
    frame_counter: u32,
}

impl EqualPowerCrossfader {
    /// Create a new crossfader.
    ///
    /// `fade_ms`: fade duration in milliseconds (typically 10-20ms).
    /// `sample_rate`: audio sample rate in Hz.
    /// `initial`: starting spatial coefficients.
    pub fn new(fade_ms: f32, sample_rate: f32, initial: SpatialCoefficients) -> Self {
        let fade_frames = ((fade_ms / 1000.0) * sample_rate).round() as u32;
        Self {
            current: initial.clone(),
            target: initial,
            fade_frames: fade_frames.max(1),
            frame_counter: fade_frames,
        }
    }

    /// Set a new target. If the source_id differs, snap immediately.
    /// Otherwise resets the fade counter for a smooth blend.
    ///
    /// Called from the audio thread after picking up new triple-buffer data.
    pub fn set_target(&mut self, target: SpatialCoefficients) {
        if self.current.source_id != target.source_id {
            // Source identity changed — instant switch to avoid stale panning
            self.current.source_id = target.source_id;
            self.current.direct_gain = target.direct_gain;
            self.current.direct_delay_samples = target.direct_delay_samples;
            self.current.direct_azimuth = target.direct_azimuth;
            self.current.direct_elevation = target.direct_elevation;
            self.current.late_t60 = target.late_t60;
            self.current.late_gain_db = target.late_gain_db;
            self.current.early_reflections = target.early_reflections.clone();
            self.target = target;
            self.frame_counter = self.fade_frames; // already at target
        } else {
            self.target = target;
            self.frame_counter = 0;
        }
    }

    /// Returns true if the crossfade is complete (target reached).
    pub fn is_complete(&self) -> bool {
        self.frame_counter >= self.fade_frames
    }

    /// Returns the current blend factor t ∈ [0,1].
    pub fn blend_factor(&self) -> f32 {
        if self.fade_frames == 0 {
            return 1.0;
        }
        (self.frame_counter as f32 / self.fade_frames as f32).min(1.0)
    }

    /// Get the current active coefficients (blended current→target).
    pub fn current_coefficients(&self) -> &SpatialCoefficients {
        &self.current
    }

    /// Get the target coefficients.
    pub fn target_coefficients(&self) -> &SpatialCoefficients {
        &self.target
    }

    /// Advance the crossfade by one frame. Call once per sample block.
    /// Returns the current blend factor.
    pub fn advance(&mut self) -> f32 {
        if self.frame_counter < self.fade_frames {
            self.frame_counter += 1;
            let t = self.blend_factor();
            let gain_cur = (std::f32::consts::PI * t / 2.0).cos();
            let gain_tgt = (std::f32::consts::PI * t / 2.0).sin();

            // Crossfade per-band direct gain
            for i in 0..8 {
                self.current.direct_gain.0[i] =
                    gain_cur * self.current.direct_gain.0[i]
                        + gain_tgt * self.target.direct_gain.0[i];
            }
            // Direct delay
            self.current.direct_delay_samples = gain_cur * self.current.direct_delay_samples
                + gain_tgt * self.target.direct_delay_samples;
            // Azimuth / elevation
            self.current.direct_azimuth = gain_cur * self.current.direct_azimuth
                + gain_tgt * self.target.direct_azimuth;
            self.current.direct_elevation = gain_cur * self.current.direct_elevation
                + gain_tgt * self.target.direct_elevation;
            // Late reverb T60 per band
            for i in 0..8 {
                self.current.late_t60.0[i] =
                    gain_cur * self.current.late_t60.0[i] + gain_tgt * self.target.late_t60.0[i];
            }
            self.current.late_gain_db =
                gain_cur * self.current.late_gain_db + gain_tgt * self.target.late_gain_db;

            // Crossfade early reflection parameters
            let num_refs = self.current.early_reflections.len().min(self.target.early_reflections.len());
            for i in 0..num_refs {
                let src = &self.target.early_reflections[i];
                let dst = &mut self.current.early_reflections[i];
                dst.azimuth = gain_cur * dst.azimuth + gain_tgt * src.azimuth;
                dst.elevation = gain_cur * dst.elevation + gain_tgt * src.elevation;
                dst.delay_samples = gain_cur * dst.delay_samples + gain_tgt * src.delay_samples;
                for b in 0..8 {
                    dst.gain.0[b] = gain_cur * dst.gain.0[b] + gain_tgt * src.gain.0[b];
                }
            }

            t
        } else {
            1.0
        }
    }

    /// Reset to a new starting point instantly (no crossfade).
    pub fn snap_to(&mut self, coefficients: SpatialCoefficients) {
        self.current = coefficients.clone();
        self.target = coefficients;
        self.frame_counter = self.fade_frames;
    }
}
