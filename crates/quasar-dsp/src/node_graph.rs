use crate::audio_buffer::{AudioBuffer, MAX_AUDIO_CHANNELS};
use crate::crossfader::EqualPowerCrossfader;
use quasar_core::param_exchange::SpatialCoefficients;

pub trait AudioNode: Send {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &quasar_core::param_exchange::SpatialCoefficients);
    fn reset(&mut self);
    fn input_channels(&self) -> u16;
    fn output_channels(&self) -> u16;
}

/// Indexed read access to per-source [`SpatialCoefficients`], so a caller can lend its
/// own storage (e.g. a crossfader bank) to [`AudioNodeGraph::process_with_params`] without
/// cloning the coefficients into a temporary `Vec` on the audio thread (#79).
pub trait ParamSource {
    /// Number of sources that have parameters.
    fn param_count(&self) -> usize;
    /// Parameters of source `i`, or `None` if out of range.
    fn params_at(&self, i: usize) -> Option<&SpatialCoefficients>;
}

impl ParamSource for [SpatialCoefficients] {
    fn param_count(&self) -> usize {
        self.len()
    }
    fn params_at(&self, i: usize) -> Option<&SpatialCoefficients> {
        self.get(i)
    }
}

impl ParamSource for [EqualPowerCrossfader] {
    fn param_count(&self) -> usize {
        self.len()
    }
    fn params_at(&self, i: usize) -> Option<&SpatialCoefficients> {
        self.get(i).map(|c| c.current_coefficients())
    }
}

/// A connection between two nodes in the graph.
#[derive(Clone, Debug)]
pub struct AudioConnection {
    pub from_node: usize,
    pub from_channel: u16,
    pub to_node: usize,
    pub to_channel: u16,
    pub gain: f32,
    /// Which source's parameters to use when processing the destination node.
    pub source_id: usize,
}

pub struct AudioNodeGraph {
    nodes: Vec<Box<dyn AudioNode>>,
    connections: Vec<AudioConnection>,
    scratch: Vec<AudioBuffer>,
    /// Per connection: staging buffer for the gain-scaled routed channel (preallocated at
    /// config time so `process` never allocates; a ~32 KB `AudioBuffer` must not live on the stack).
    temps: Vec<AudioBuffer>,
    /// Per node: `true` if it has an outgoing connection (intermediate, not mixed to the output).
    /// Maintained at config time.
    has_outgoing: Vec<bool>,
}

impl AudioNodeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            scratch: Vec::new(),
            temps: Vec::new(),
            has_outgoing: Vec::new(),
        }
    }

    /// Recompute the preallocated routing scratch (config time only; allocates).
    fn rebuild_routing_scratch(&mut self) {
        self.has_outgoing.clear();
        self.has_outgoing.resize(self.nodes.len(), false);
        for conn in &self.connections {
            if let Some(f) = self.has_outgoing.get_mut(conn.from_node) {
                *f = true;
            }
        }
        self.temps.clear();
        for conn in &self.connections {
            let (ch, n) = self
                .scratch
                .get(conn.from_node)
                .map(|s| (s.channels(), s.samples()))
                .unwrap_or((2, 256));
            self.temps.push(AudioBuffer::new(ch, n));
        }
    }

    /// Add a node to the graph. Returns the node index.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        let out_ch = self.nodes[idx].output_channels();
        if out_ch > 0 {
            self.scratch.push(AudioBuffer::new(out_ch, 256));
        } else {
            self.scratch.push(AudioBuffer::new(2, 256));
        }
        self.rebuild_routing_scratch();
        idx
    }

    /// Connect two nodes with gain and an optional source_id for params.
    pub fn connect(&mut self, from: usize, from_ch: u16, to: usize, to_ch: u16, gain: f32) {
        self.connect_with_source(from, from_ch, to, to_ch, gain, 0);
    }

    /// Connect two nodes, specifying which source's params to use.
    pub fn connect_with_source(
        &mut self,
        from: usize,
        from_ch: u16,
        to: usize,
        to_ch: u16,
        gain: f32,
        source_id: usize,
    ) {
        self.connections.push(AudioConnection {
            from_node: from,
            from_channel: from_ch,
            to_node: to,
            to_channel: to_ch,
            gain,
            source_id,
        });
        self.rebuild_routing_scratch();
    }

    /// Connect with unity gain.
    pub fn connect_direct(&mut self, from: usize, from_ch: u16, to: usize, to_ch: u16) {
        self.connect(from, from_ch, to, to_ch, 1.0);
    }

    /// Remove all connections from a node.
    pub fn disconnect_node(&mut self, node: usize) {
        self.connections.retain(|c| c.from_node != node && c.to_node != node);
        self.rebuild_routing_scratch();
    }

    /// Process the entire graph.
    ///
    /// `inputs`: one `AudioBuffer` per source being rendered.
    /// `params`: one `SpatialCoefficients` per source.
    /// `output`: the final mixed output buffer.
    ///
    /// Never allocates (all scratch is preallocated by `add_node` / `connect*`).
    pub fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        params: &[SpatialCoefficients],
        output: &mut AudioBuffer,
    ) {
        self.process_with_params(inputs, params, output);
    }

    /// Like [`process`](Self::process) but reads the per-source parameters through a
    /// [`ParamSource`] (e.g. a `[EqualPowerCrossfader]`) instead of a slice of owned values.
    /// Sources/connections whose parameters are missing are skipped. Never allocates.
    pub fn process_with_params<P: ParamSource + ?Sized>(
        &mut self,
        inputs: &[&AudioBuffer],
        params: &P,
        output: &mut AudioBuffer,
    ) {
        output.clear();

        let num_sources = inputs.len().min(self.nodes.len()).min(params.param_count());

        // Phase 1: process each source node with its input and params
        for src_idx in 0..num_sources {
            let Some(param) = params.params_at(src_idx) else { continue };
            let node = &mut *self.nodes[src_idx];
            let scratch = &mut self.scratch[src_idx];
            node.process(inputs[src_idx], scratch, param);
        }

        // Phase 2: route connections using per-source params
        for conn_idx in 0..self.connections.len() {
            let conn = &self.connections[conn_idx];
            if conn.from_node >= self.nodes.len() || conn.to_node >= self.nodes.len() {
                continue;
            }
            let from_idx = conn.from_node;
            let to_idx = conn.to_node;
            let from_ch = conn.from_channel as usize;
            let to_ch = conn.to_channel as usize;
            let gain = conn.gain;

            // Use the connection's source_id to pick the right params (fall back to source 0).
            let Some(conn_params) = params.params_at(conn.source_id).or_else(|| params.params_at(0)) else {
                continue;
            };

            let src_scratch = &self.scratch[from_idx];
            let Some(temp) = self.temps.get_mut(conn_idx) else { continue };
            temp.clear();
            if from_ch < src_scratch.channels() as usize && to_ch < MAX_AUDIO_CHANNELS {
                let src_ch_data = src_scratch.channel(from_ch as u16);
                let dst_ch = temp.channel_mut(to_ch as u16);
                let len = dst_ch.len().min(src_ch_data.len());
                for i in 0..len {
                    dst_ch[i] = src_ch_data[i] * gain;
                }
            }

            let node = &mut *self.nodes[to_idx];
            let dst_scratch = &mut self.scratch[to_idx];
            node.process(temp, dst_scratch, conn_params);
        }

        // Phase 3: sum only leaf nodes (nodes with no outgoing connections)
        // into output. Intermediate buffers (occlusion, reverb) are not output.
        for ch in 0..output.channels() as usize {
            let out_ch = output.channel_mut(ch as u16);
            for src_idx in 0..self.scratch.len() {
                if self.has_outgoing.get(src_idx).copied().unwrap_or(false) {
                    continue; // skip intermediate nodes
                }
                let sc = &self.scratch[src_idx];
                if ch < sc.channels() as usize {
                    let src_slice = sc.channel(ch as u16);
                    let len = out_ch.len().min(src_slice.len());
                    for i in 0..len {
                        out_ch[i] += src_slice[i];
                    }
                }
            }
        }
    }

    /// Reset all nodes.
    pub fn reset_all(&mut self) {
        for node in self.nodes.iter_mut() {
            node.reset();
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

impl Default for AudioNodeGraph {
    fn default() -> Self {
        Self::new()
    }
}
