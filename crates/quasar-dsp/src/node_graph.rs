use crate::audio_buffer::AudioBuffer;

pub trait AudioNode: Send {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &quasar_core::param_exchange::SpatialCoefficients);
    fn reset(&mut self);
    fn input_channels(&self) -> u16;
    fn output_channels(&self) -> u16;
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
}

impl AudioNodeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            scratch: Vec::new(),
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
    }

    /// Connect with unity gain.
    pub fn connect_direct(&mut self, from: usize, from_ch: u16, to: usize, to_ch: u16) {
        self.connect(from, from_ch, to, to_ch, 1.0);
    }

    /// Remove all connections from a node.
    pub fn disconnect_node(&mut self, node: usize) {
        self.connections.retain(|c| c.from_node != node && c.to_node != node);
    }

    /// Process the entire graph.
    ///
    /// `inputs`: one `AudioBuffer` per source being rendered.
    /// `params`: one `SpatialCoefficients` per source.
    /// `output`: the final mixed output buffer.
    pub fn process(
        &mut self,
        inputs: &[&AudioBuffer],
        params: &[quasar_core::param_exchange::SpatialCoefficients],
        output: &mut AudioBuffer,
    ) {
        output.clear();

        let num_sources = inputs.len().min(self.nodes.len());

        // Phase 1: process each source node with its input and params
        for src_idx in 0..num_sources {
            let input = inputs[src_idx];
            let param = &params[src_idx];
            let node = &mut *self.nodes[src_idx];
            let scratch = &mut self.scratch[src_idx];
            node.process(input, scratch, param);
        }

        // Phase 2: route connections using per-source params
        let num_connections = self.connections.len();
        if num_connections > 0 {
            for conn_idx in 0..num_connections {
                let conn = &self.connections[conn_idx];
                if conn.from_node < self.nodes.len() && conn.to_node < self.nodes.len() {
                    let from_idx = conn.from_node;
                    let to_idx = conn.to_node;
                    let from_ch = conn.from_channel as usize;
                    let to_ch = conn.to_channel as usize;
                    let gain = conn.gain;

                    // Use the connection's source_id to pick the right params
                    let conn_params = if conn.source_id < params.len() {
                        &params[conn.source_id]
                    } else {
                        &params[0]
                    };

                    let src_scratch = &self.scratch[from_idx];

                    let mut temp = AudioBuffer::new(src_scratch.channels(), src_scratch.samples());
                    if from_ch < src_scratch.channels() as usize {
                        let src_ch_data = src_scratch.channel(from_ch as u16);
                        let dst_ch = temp.channel_mut(to_ch as u16);
                        let len = dst_ch.len().min(src_ch_data.len());
                        for i in 0..len {
                            dst_ch[i] = src_ch_data[i] * gain;
                        }
                    }

                    let node = &mut *self.nodes[to_idx];
                    let dst_scratch = &mut self.scratch[to_idx];
                    node.process(&temp, dst_scratch, conn_params);
                }
            }
        }

        // Phase 3: sum all scratch buffers into output
        for ch in 0..output.channels() as usize {
            let out_ch = output.channel_mut(ch as u16);
            for src_idx in 0..self.scratch.len() {
                let sc = &self.scratch[src_idx];
                let src_chs = sc.channels() as usize;
                if ch < src_chs {
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
