#[cfg(feature = "cpu-simd")]
pub mod cpu_simd;
#[cfg(feature = "wgpu-compute")]
pub mod wgpu_compute;
pub mod hw_stub;

#[cfg(feature = "cpu-simd")]
pub use cpu_simd::CpuSimdComputeBackend;
#[cfg(feature = "wgpu-compute")]
pub use wgpu_compute::WgpuComputeBackend;
pub use hw_stub::HardwareAcceleratorStub;
