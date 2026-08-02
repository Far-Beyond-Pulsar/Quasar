use glam::Vec3A;

/// A single ray for acoustic scene intersection.
#[derive(Clone, Debug)]
pub struct Ray {
    /// Origin of the ray in world space.
    pub origin: [f32; 3],
    /// Direction of the ray (must be normalized).
    pub direction: [f32; 3],
    /// Minimum distance along the ray to consider intersections.
    pub min_distance: f32,
    /// Maximum distance along the ray to consider intersections.
    pub max_distance: f32,
}

impl Ray {
    /// Create a new ray with `min_distance = 0.0` and `max_distance = f32::MAX`.
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self {
            origin,
            direction,
            min_distance: 0.0,
            max_distance: f32::MAX,
        }
    }

    /// Get position along ray: `origin + t * direction`.
    pub fn point_at(&self, t: f32) -> [f32; 3] {
        let o = Vec3A::from_array(self.origin);
        let d = Vec3A::from_array(self.direction);
        (o + d * t).to_array()
    }
}

/// Result of a ray-scene intersection.
#[derive(Clone, Debug)]
pub struct RayHit {
    /// Distance along the ray to the intersection point.
    pub distance: f32,
    /// World-space position of the intersection.
    pub point: [f32; 3],
    /// Surface normal at the intersection point.
    pub normal: [f32; 3],
    /// Handle of the material at the intersection point.
    pub material_handle: u32,
    /// Whether the ray hit anything.
    pub hit: bool,
}

impl RayHit {
    /// Create a "miss" result where `hit` is false.
    pub fn miss() -> Self {
        Self {
            distance: f32::MAX,
            point: [0.0; 3],
            normal: [0.0; 3],
            material_handle: 0,
            hit: false,
        }
    }
}

/// Context for evaluating a material interaction at a ray hit point.
#[derive(Clone, Debug)]
pub struct RayInteractionContext {
    /// Angle between the negated ray direction and the surface normal (radians).
    pub incident_angle_rad: f32,
    /// Surface normal at the hit point.
    pub surface_normal: [f32; 3],
    /// Incoming ray direction (pointing toward the surface).
    pub ray_direction: [f32; 3],
    /// Air temperature in Celsius at the interaction point.
    pub temperature_celsius: f32,
    /// Relative humidity percentage at the interaction point.
    pub humidity_percent: f32,
}

impl Default for RayInteractionContext {
    fn default() -> Self {
        Self {
            temperature_celsius: 20.0,
            humidity_percent: 50.0,
            incident_angle_rad: 0.0,
            surface_normal: [0.0, 1.0, 0.0],
            ray_direction: [0.0, -1.0, 0.0],
        }
    }
}
