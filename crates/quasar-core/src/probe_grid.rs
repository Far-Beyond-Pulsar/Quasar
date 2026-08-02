use crate::bands::Band8;
use crate::error::SpatialAudioError;

/// A single acoustic probe (baked impulse response at a point in space).
#[derive(Clone, Debug)]
pub struct AcousticProbe {
    /// World-space position of the probe.
    pub position: [f32; 3],
    /// Time-series of 8-band energy samples (the room impulse response).
    pub rir_samples: Vec<Band8>,
    /// Sample rate of the RIR data (Hz).
    pub sample_rate: u32,
    /// Per-band RT60 reverberation time (seconds).
    pub t60: Band8,
    /// Broadband (omnibus) RT60 (seconds).
    pub broadband_t60: f32,
    /// Early / late split time (seconds).
    pub early_late_split_secs: f32,
}

/// An interpolated sample from an `AcousticProbeGrid`.
#[derive(Clone, Debug)]
pub struct AcousticProbeSample {
    /// Per-band RT60 interpolated at the sample position.
    pub t60: Band8,
    /// Interpolation quality: 1.0 = fully inside the grid, 0.0 = at boundary or outside.
    pub interpolation_quality: f32,
    /// Index of the nearest probe in the probe list.
    pub nearest_probe_index: usize,
}

/// A 3D grid of acoustic probes for spatial lookup and trilinear interpolation.
#[derive(Clone, Debug)]
pub struct AcousticProbeGrid {
    /// All probes in the grid, stored in row-major order (x, then y, then z).
    pub probes: Vec<AcousticProbe>,
    /// World-space origin (minimum corner) of the grid.
    pub grid_origin: [f32; 3],
    /// Spacing between adjacent probes along each axis.
    pub grid_spacing: [f32; 3],
    /// Number of probes along each axis.
    pub grid_dims: [u32; 3],
}

impl AcousticProbeGrid {
    /// Create a new grid from a flat probe list with grid metadata.
    ///
    /// `probes.len()` must equal `grid_dims[0] * grid_dims[1] * grid_dims[2]`.
    pub fn new(
        probes: Vec<AcousticProbe>,
        grid_origin: [f32; 3],
        grid_spacing: [f32; 3],
        grid_dims: [u32; 3],
    ) -> Result<Self, SpatialAudioError> {
        let expected = (grid_dims[0] as usize)
            * (grid_dims[1] as usize)
            * (grid_dims[2] as usize);
        if probes.len() != expected {
            return Err(SpatialAudioError::ProbeGrid(format!(
                "expected {} probes for grid dimensions {:?}, got {}",
                expected,
                grid_dims,
                probes.len()
            )));
        }
        Ok(Self {
            probes,
            grid_origin,
            grid_spacing,
            grid_dims,
        })
    }

    /// Sample the grid at a world position using trilinear interpolation.
    ///
    /// Returns `None` if the position is outside the grid bounds.
    pub fn sample(&self, position: &[f32; 3]) -> Option<AcousticProbeSample> {
        let cell = self.cell_index(position)?;

        let wx = (position[0] - self.grid_origin[0]) / self.grid_spacing[0] - cell[0] as f32;
        let wy = (position[1] - self.grid_origin[1]) / self.grid_spacing[1] - cell[1] as f32;
        let wz = (position[2] - self.grid_origin[2]) / self.grid_spacing[2] - cell[2] as f32;

        let weights = [wx, wy, wz];
        let corner_indices = self.cell_probe_indices(cell);
        let t60 = self.trilinear_interpolate(weights, corner_indices);

        // Interpolation quality: 1.0 at cell centre, 0.0 at cell boundary.
        let quality = (1.0 - (wx - 0.5).abs() * 2.0)
            * (1.0 - (wy - 0.5).abs() * 2.0)
            * (1.0 - (wz - 0.5).abs() * 2.0);
        let quality = quality.clamp(0.0, 1.0);

        // Index of the nearest probe (round to nearest corner).
        let nx = if wx < 0.5 { cell[0] } else { cell[0] + 1 };
        let ny = if wy < 0.5 { cell[1] } else { cell[1] + 1 };
        let nz = if wz < 0.5 { cell[2] } else { cell[2] + 1 };
        let nearest = (nz * self.grid_dims[1] * self.grid_dims[0] + ny * self.grid_dims[0] + nx)
            as usize;

        Some(AcousticProbeSample {
            t60,
            interpolation_quality: quality,
            nearest_probe_index: nearest,
        })
    }

    /// Get the grid cell index for a world position.
    ///
    /// Returns `None` if outside the grid bounds.
    pub fn cell_index(&self, position: &[f32; 3]) -> Option<[u32; 3]> {
        if self.grid_dims[0] == 0 || self.grid_dims[1] == 0 || self.grid_dims[2] == 0 {
            return None;
        }

        let max_x = self.grid_origin[0] + (self.grid_dims[0] as f32 - 1.0) * self.grid_spacing[0];
        let max_y = self.grid_origin[1] + (self.grid_dims[1] as f32 - 1.0) * self.grid_spacing[1];
        let max_z = self.grid_origin[2] + (self.grid_dims[2] as f32 - 1.0) * self.grid_spacing[2];

        if position[0] < self.grid_origin[0]
            || position[1] < self.grid_origin[1]
            || position[2] < self.grid_origin[2]
            || position[0] > max_x
            || position[1] > max_y
            || position[2] > max_z
        {
            return None;
        }

        let cx = ((position[0] - self.grid_origin[0]) / self.grid_spacing[0]) as u32;
        let cy = ((position[1] - self.grid_origin[1]) / self.grid_spacing[1]) as u32;
        let cz = ((position[2] - self.grid_origin[2]) / self.grid_spacing[2]) as u32;

        // Clamp to the last valid cell (dims - 2) because cell [d - 1] has no +1 corner.
        let cx = cx.min(self.grid_dims[0].saturating_sub(2));
        let cy = cy.min(self.grid_dims[1].saturating_sub(2));
        let cz = cz.min(self.grid_dims[2].saturating_sub(2));

        Some([cx, cy, cz])
    }

    /// Get probe indices at the 8 corners of a grid cell.
    fn cell_probe_indices(&self, cell: [u32; 3]) -> [usize; 8] {
        let sx = self.grid_dims[0] as usize;
        let sy = self.grid_dims[1] as usize;

        let x0 = cell[0] as usize;
        let y0 = cell[1] as usize;
        let z0 = cell[2] as usize;
        let x1 = (cell[0] + 1) as usize;
        let y1 = (cell[1] + 1) as usize;
        let z1 = (cell[2] + 1) as usize;

        [
            z0 * sy * sx + y0 * sx + x0,
            z0 * sy * sx + y0 * sx + x1,
            z0 * sy * sx + y1 * sx + x0,
            z0 * sy * sx + y1 * sx + x1,
            z1 * sy * sx + y0 * sx + x0,
            z1 * sy * sx + y0 * sx + x1,
            z1 * sy * sx + y1 * sx + x0,
            z1 * sy * sx + y1 * sx + x1,
        ]
    }

    /// Trilinear interpolation of `t60` values from 8 corner probes.
    fn trilinear_interpolate(&self, weights: [f32; 3], corner_indices: [usize; 8]) -> Band8 {
        let [wx, wy, wz] = weights;
        let ix = 1.0 - wx;
        let iy = 1.0 - wy;
        let iz = 1.0 - wz;

        let c00 = self.probes[corner_indices[0]]
            .t60
            .scale(ix * iy * iz)
            .add(&self.probes[corner_indices[1]].t60.scale(wx * iy * iz));
        let c01 = self.probes[corner_indices[2]]
            .t60
            .scale(ix * wy * iz)
            .add(&self.probes[corner_indices[3]].t60.scale(wx * wy * iz));
        let c10 = self.probes[corner_indices[4]]
            .t60
            .scale(ix * iy * wz)
            .add(&self.probes[corner_indices[5]].t60.scale(wx * iy * wz));
        let c11 = self.probes[corner_indices[6]]
            .t60
            .scale(ix * wy * wz)
            .add(&self.probes[corner_indices[7]].t60.scale(wx * wy * wz));

        let c0 = c00.add(&c01);
        let c1 = c10.add(&c11);

        // Final interpolation along z.
        let iz_wz = iz;
        let wz = wz;
        c0.scale(iz_wz).add(&c1.scale(wz))
    }

    /// Number of probes in the grid.
    pub fn len(&self) -> usize {
        self.probes.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }
}

impl Default for AcousticProbeGrid {
    fn default() -> Self {
        Self {
            probes: Vec::new(),
            grid_origin: [0.0; 3],
            grid_spacing: [1.0; 3],
            grid_dims: [0; 3],
        }
    }
}
