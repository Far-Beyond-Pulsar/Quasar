/// Simplified runtime geometry for acoustic ray tracing.
#[derive(Clone, Debug)]
pub struct AcousticScene {
    /// The meshes that make up the scene.
    pub meshes: Vec<AcousticMesh>,
}

impl Default for AcousticScene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
        }
    }
}

impl AcousticScene {
    /// Create a new empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh to the scene and return its index.
    pub fn add_mesh(&mut self, mesh: AcousticMesh) -> usize {
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        idx
    }

    /// Total number of triangles across all meshes in the scene.
    pub fn total_triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// Whether the scene contains no meshes.
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
}

/// A single mesh in the acoustic scene (proxy / low-poly geometry).
#[derive(Clone, Debug)]
pub struct AcousticMesh {
    /// Unique identifier for this mesh.
    pub id: u64,
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Triangle index buffer (triplets of vertex indices).
    pub indices: Vec<u32>,
    /// Handle of the acoustic material assigned to this mesh.
    pub material_handle: u32,
    /// 4×4 column-major transform matrix from local to world space.
    pub transform: [f32; 16],
}

impl AcousticMesh {
    /// Create a new `AcousticMesh` with an identity transform.
    pub fn new(
        id: u64,
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        material_handle: u32,
    ) -> Self {
        Self {
            id,
            positions,
            indices,
            material_handle,
            transform: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Number of triangles in this mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}
