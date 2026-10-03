//! The acoustic debug overlay: what is drawn and how the window title summarises it.
//!
//! Keys (handled in `main.rs`): V pauses / resumes trace capture, C cycles the emitter, B shows all
//! emitters, N rejected reflection candidates, M the solver's probe rays. The overlay replays the
//! retained [`AcousticDebugFrame`] through Helio's world-space debug lines every display frame (the
//! renderer clears debug geometry each frame), so a paused capture stays visible.

use helio::Renderer;
use quasar_backends::debug_capture::{AcousticDebugFrame, CaptureDetail, DebugDirect, DebugRayKind, RejectReason};

/// What the acoustic overlay shows. Keys: C cycles the emitter (auto = nearest to the
/// listener -> each emitter -> auto), B shows all emitters, N rejected reflection
/// candidates, M the solver's probe rays (occlusion, diffraction, path validation).
#[derive(Default)]
pub struct AcousticView {
    /// `None` follows the emitter nearest to the listener.
    pub emitter: Option<u32>,
    pub all_emitters: bool,
    pub show_rejected: bool,
    pub show_probes: bool,
    /// The scene is NOT watertight, so the solver cannot tell inside from outside and reports every
    /// point as "inside"; the title says the side is unknown instead.
    pub side_unknown: bool,
}

/// Valid reflection paths drawn emphasised (brightest, with bounce markers) per emitter.
const OVERLAY_STRONGEST_PATHS: usize = 16;

impl AcousticView {
    /// The capture detail the solver must store for what is currently drawn.
    pub fn detail(&self) -> CaptureDetail {
        CaptureDetail {
            occlusion_probes: self.show_probes,
            diffraction_probes: self.show_probes,
            reflection_validation: self.show_probes,
            rejected_paths: self.show_rejected,
        }
    }

    fn emitter_ids(frame: &AcousticDebugFrame) -> Vec<u32> {
        let mut ids: Vec<u32> = frame.directs.iter().map(|d| d.source_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// auto -> first emitter -> ... -> last emitter -> auto.
    pub fn cycle_emitter(&mut self, frame: Option<&AcousticDebugFrame>) {
        let ids = frame.map(Self::emitter_ids).unwrap_or_default();
        self.all_emitters = false;
        self.emitter = match self.emitter {
            None => ids.first().copied(),
            Some(cur) => ids.iter().copied().find(|&id| id > cur),
        };
    }

    /// Emitters drawn: all, the chosen one, or the one nearest to the listener.
    pub fn active_emitters(&self, frame: &AcousticDebugFrame) -> Vec<u32> {
        if self.all_emitters {
            return Self::emitter_ids(frame);
        }
        if let Some(id) = self.emitter {
            return vec![id];
        }
        let dist = |d: &DebugDirect| (glam::Vec3::from_array(d.source) - glam::Vec3::from_array(d.listener)).length();
        frame
            .directs
            .iter()
            .min_by(|a, b| dist(a).total_cmp(&dist(b)))
            .map(|d| vec![d.source_id])
            .unwrap_or_default()
    }

    pub fn label(&self) -> &'static str {
        if self.all_emitters {
            "all emitters"
        } else if self.emitter.is_some() {
            "chosen emitter"
        } else {
            "nearest emitter"
        }
    }
}

/// Window-title statistics of the retained acoustic frame.
pub fn acoustic_title(frame: &AcousticDebugFrame, view: &AcousticView, status: &str) -> String {
    let active = view.active_emitters(frame);
    let selected = frame.paths.iter().filter(|p| p.selected).count();
    let stored = frame.rays.len();
    let side = match frame.directs.iter().find(|d| active.contains(&d.source_id)) {
        _ if view.side_unknown => "inside/outside unknown: scene not watertight",
        Some(d) if d.reflections_skipped => "outside: reflections skipped",
        Some(d) if d.listener_outside => "listener outside",
        Some(_) => "listener inside",
        None => "no query",
    };
    format!(
        "Quasar | {status} | {} rays traced | {stored} stored / {} dropped | {} valid paths, {selected} selected | {} {active:?} | {side} | C emitter, B all, N rejected {}, M probes {} | V pauses with last trace visible",
        frame.ray_count,
        frame.rays_dropped,
        frame.paths.len(),
        view.label(),
        if view.show_rejected { "on" } else { "off" },
        if view.show_probes { "on" } else { "off" },
    )
}

pub fn hsl_to_rgba(h: f32, s: f32, l: f32, a: f32) -> [f32; 4] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h * 6.0).floor() as i32 {
        0 => (c, x, 0.),
        1 => (x, c, 0.),
        2 => (0., c, x),
        3 => (0., x, c),
        4 => (x, 0., c),
        _ => (c, 0., x),
    };
    [r + m, g + m, b + m, a]
}

/// Replay the retained acoustic trace through Helio's world-space debug API.
/// The renderer clears debug geometry every frame, so paused captures are redrawn here.
///
/// Per active emitter: the direct segment (green clear, yellow partial, red blocked) and
/// its valid reflection paths coloured by order and faded by gain (the strongest
/// [`OVERLAY_STRONGEST_PATHS`] emphasised, with bounce markers and surface normals).
/// Rejected candidates and probe rays appear only when their keys are on.
pub fn draw_acoustic_snapshot(renderer: &mut Renderer, frame: &AcousticDebugFrame, view: &AcousticView) {
    let active = view.active_emitters(frame);

    if view.show_probes {
        for sample in frame.rays.iter().filter(|s| active.contains(&s.source_id)) {
            let ray = &sample.ray;
            let end = sample.hit.as_ref().map(|hit| hit.point).unwrap_or_else(|| {
                ray.point_at(if ray.max_distance < 1.0e6 { ray.max_distance } else { 60.0 })
            });
            let color = match (sample.kind, sample.hit.is_some()) {
                (DebugRayKind::DiffractionProbe, _) => [1.0, 0.6, 0.1, 0.22],
                (DebugRayKind::ReflectionValidation, _) => [0.6, 0.6, 0.95, 0.2],
                (_, true) => [1.0, 0.18, 0.12, 0.28],
                (_, false) => [0.1, 0.65, 1.0, 0.22],
            };
            renderer.debug_line(ray.point_at(ray.min_distance), end, color);
        }
    }

    if view.show_rejected {
        for r in frame.rejected.iter().filter(|r| active.contains(&r.source_id)) {
            let color = match r.reason {
                RejectReason::Blocked => [1.0, 0.45, 0.1, 0.35],
                RejectReason::OutsideSurface => [0.55, 0.55, 0.6, 0.25],
                RejectReason::EdgeFade => [0.95, 0.9, 0.2, 0.3],
                RejectReason::BelowEnergy => [0.3, 0.35, 0.9, 0.3],
            };
            // OutsideSurface candidates are partial (listener side bounces only).
            let mut from = r.listener;
            for &bounce in r.bounces.iter().rev() {
                renderer.debug_line(from, bounce, color);
                from = bounce;
            }
            if r.reason != RejectReason::OutsideSurface {
                renderer.debug_line(from, r.source, color);
            }
        }
    }

    for &id in &active {
        let energy = |p: &quasar_backends::debug_capture::DebugReflectionPath| p.reflection.gain.0.iter().map(|g| g * g).sum::<f32>();
        let mut paths: Vec<_> = frame.paths.iter().filter(|p| p.source_id == id).collect();
        paths.sort_by(|a, b| energy(b).total_cmp(&energy(a)));
        let strongest = paths.first().map(|p| energy(p)).unwrap_or(0.0).max(1.0e-20);
        // Weakest first so the strongest paths are drawn on top.
        for (rank, path) in paths.iter().enumerate().rev() {
            let emphasised = rank < OVERLAY_STRONGEST_PATHS;
            let t = (1.0 + 10.0 * (energy(path) / strongest).max(1.0e-20).log10() / 40.0).clamp(0.2, 1.0);
            let base = match path.reflection.order {
                1 => [0.2, 1.0, 0.4],
                2 => [0.2, 0.8, 1.0],
                _ => [0.85, 0.45, 1.0],
            };
            let color = [base[0] * t, base[1] * t, base[2] * t, if emphasised { 0.5 + 0.5 * t } else { 0.12 + 0.18 * t }];
            let mut from = path.source;
            for &bounce in &path.bounces {
                renderer.debug_line(from, bounce, color);
                from = bounce;
            }
            renderer.debug_line(from, path.listener, color);
            if !emphasised {
                continue;
            }
            for (&point, &normal) in path.bounces.iter().zip(&path.normals) {
                let p = glam::Vec3::from_array(point);
                let mark = [1.0, 0.85, 0.05, 0.4 + 0.6 * t];
                for axis in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z] {
                    renderer.debug_line((p - axis * 0.07).to_array(), (p + axis * 0.07).to_array(), mark);
                }
                renderer.debug_line(point, (p + glam::Vec3::from_array(normal) * 0.35).to_array(), mark);
            }
        }
    }

    // Direct segments last: green clear, yellow partially occluded, red blocked.
    for d in frame.directs.iter().filter(|d| active.contains(&d.source_id)) {
        let color = if !d.occluded {
            [0.2, 1.0, 0.3, 1.0]
        } else if d.occlusion_factor > 0.3 {
            [1.0, 0.85, 0.1, 1.0]
        } else {
            [1.0, 0.15, 0.1, 1.0]
        };
        renderer.debug_line(d.source, d.listener, color);
    }
}

/// Speaker markers: a sphere, a cone along the aim axis and a ground circle per speaker, plus the
/// listener marker. `aim` is the audience point every speaker points at.
pub fn draw_stage(renderer: &mut Renderer, speakers: &[glam::Vec3], aim: glam::Vec3, listener: glam::Vec3, forward: glam::Vec3) {
    for (i, &pos) in speakers.iter().enumerate() {
        let hue = i as f32 / speakers.len() as f32;
        let color = hsl_to_rgba(hue, 0.9, 0.6, 1.0);
        renderer.debug_sphere(pos.into(), 0.25, color, 16);
        let dir = (aim - pos).normalize();
        renderer.debug_cone((pos + dir * 0.3).into(), dir.into(), 1.5, 0.8, [color[0], color[1], color[2], 0.3], 12);
        renderer.debug_circle(pos.into(), 2.0, [color[0], color[1], color[2], 0.12], 24);
    }
    renderer.debug_sphere(listener.into(), 0.2, [0.0, 1.0, 0.3, 1.0], 12);
    renderer.debug_cone((listener + forward * 0.2).into(), forward.into(), 0.4, 0.15, [0.0, 0.8, 0.0, 0.4], 8);
}

/// A world-space AABB as 12 debug lines.
pub fn draw_aabb(renderer: &mut Renderer, min: [f32; 3], max: [f32; 3], color: [f32; 4]) {
    let c = |i: usize| [if i & 1 == 0 { min[0] } else { max[0] }, if i & 2 == 0 { min[1] } else { max[1] }, if i & 4 == 0 { min[2] } else { max[2] }];
    for a in 0..8usize {
        for bit in [1usize, 2, 4] {
            let b = a ^ bit;
            if a < b {
                renderer.debug_line(c(a), c(b), color);
            }
        }
    }
}

/// The probe grid as small spheres joined by faint lines, restricted to the probe layers within
/// `max_height` of the floor (the full grid reaches the vault and would be mostly unseen clutter).
pub fn draw_probe_grid(renderer: &mut Renderer, origin: [f32; 3], spacing: [f32; 3], dims: [u32; 3], max_height: f32) {
    let at = |x: u32, y: u32, z: u32| {
        [origin[0] + x as f32 * spacing[0], origin[1] + y as f32 * spacing[1], origin[2] + z as f32 * spacing[2]]
    };
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            if at(0, y, 0)[1] - origin[1] > max_height {
                continue;
            }
            for x in 0..dims[0] {
                let p = at(x, y, z);
                renderer.debug_sphere(p, 0.2, [0.3, 0.6, 1.0, 0.7], 6);
                if x + 1 < dims[0] {
                    renderer.debug_line(p, at(x + 1, y, z), [0.3, 0.6, 1.0, 0.15]);
                }
                if z + 1 < dims[2] {
                    renderer.debug_line(p, at(x, y, z + 1), [0.3, 0.6, 1.0, 0.15]);
                }
                if y + 1 < dims[1] && at(0, y + 1, 0)[1] - origin[1] <= max_height {
                    renderer.debug_line(p, at(x, y + 1, z), [0.3, 0.6, 1.0, 0.15]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(id: u32, source: [f32; 3]) -> DebugDirect {
        DebugDirect {
            source,
            listener: [0.0; 3],
            source_id: id,
            query_index: id,
            occluded: false,
            occlusion_factor: 1.0,
            source_outside: false,
            listener_outside: false,
            reflections_skipped: false,
            rays_traced: 0,
        }
    }

    #[test]
    fn acoustic_view_picks_nearest_cycles_and_requests_detail() {
        let frame = AcousticDebugFrame {
            directs: vec![direct(4, [10.0, 0.0, 0.0]), direct(2, [3.0, 0.0, 0.0]), direct(9, [5.0, 0.0, 0.0])],
            ..Default::default()
        };
        let mut view = AcousticView::default();
        assert_eq!(view.active_emitters(&frame), vec![2], "default = nearest to the listener");
        view.cycle_emitter(Some(&frame));
        assert_eq!(view.active_emitters(&frame), vec![2]);
        view.cycle_emitter(Some(&frame));
        assert_eq!(view.active_emitters(&frame), vec![4]);
        view.cycle_emitter(Some(&frame));
        assert_eq!(view.active_emitters(&frame), vec![9]);
        view.cycle_emitter(Some(&frame));
        assert!(view.emitter.is_none(), "wraps back to nearest");
        view.all_emitters = true;
        assert_eq!(view.active_emitters(&frame), vec![2, 4, 9]);
        assert_eq!(view.detail(), CaptureDetail::NONE, "probes and rejected paths are off by default");
        view.show_probes = true;
        view.show_rejected = true;
        assert_eq!(view.detail(), CaptureDetail::ALL);
        assert!(acoustic_title(&frame, &view, "capturing").contains("all emitters"));
    }

    #[test]
    fn acoustic_title_reports_outside_listener() {
        let mut d = direct(1, [1.0, 0.0, 0.0]);
        d.listener_outside = true;
        d.reflections_skipped = true;
        let frame = AcousticDebugFrame { directs: vec![d], ray_count: 123, ..Default::default() };
        let title = acoustic_title(&frame, &AcousticView::default(), "capturing");
        assert!(title.contains("123 rays traced") && title.contains("outside: reflections skipped"), "{title}");
    }
}
