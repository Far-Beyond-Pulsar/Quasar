struct MaterialInstance {
    model_id: u32,
    param_offset: u32,
    param_size: u32,
    _pad: u32,
}

struct AcousticResponse {
    absorption: array<f32, 8>,
    scattering: array<f32, 8>,
    transmission: array<f32, 8>,
}

fn evaluate_material(
    instance: MaterialInstance,
    params: ptr<storage, array<u8>>,
    freq_centres: ptr<function, array<f32, 8>>,
) -> AcousticResponse {
    var result: AcousticResponse;

    switch (instance.model_id) {
        // TABULAR_MODEL_ID = 1: direct lookup
        case 1u: {
            let offset = instance.param_offset;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let idx = offset + i * 4u;
                result.absorption[i]  = bitcast<f32>((*params)[idx + 0u]);
                result.scattering[i]  = bitcast<f32>((*params)[idx + 32u]);
                result.transmission[i] = bitcast<f32>((*params)[idx + 64u]);
            }
        }

        // DELANY_BAZLEY_MODEL_ID = 2: porous absorber
        case 2u: {
            let offset = instance.param_offset;
            let flow_resistivity = bitcast<f32>((*params)[offset + 0u]);
            let thickness_m      = bitcast<f32>((*params)[offset + 4u]);

            for (var i = 0u; i < 8u; i = i + 1u) {
                let freq = (*freq_centres)[i];
                let rho = 1.204;
                let c0 = 343.0;
                let z0 = rho * c0;
                let e = rho * freq / flow_resistivity;

                // Characteristic impedance (complex)
                let zc_real = z0 * (1.0 + 0.0571 * pow(e, -0.754));
                let zc_imag = -z0 * 0.087 * pow(e, -0.732);

                // Propagation constant (complex)
                let omega_over_c0 = (2.0 * 3.14159265 * freq) / c0;
                let k_real = omega_over_c0 * (1.0 + 0.0978 * pow(e, -0.700));
                let k_imag = -omega_over_c0 * 0.189 * pow(e, -0.595);

                // cot(k * d) — complex
                let kd_real = k_real * thickness_m;
                let kd_imag = k_imag * thickness_m;
                let cos_kd_real = cos(kd_real) * cosh(kd_imag);
                let cos_kd_imag = -sin(kd_real) * sinh(kd_imag);
                let sin_kd_real = sin(kd_real) * cosh(kd_imag);
                let sin_kd_imag = cos(kd_real) * sinh(kd_imag);
                let denom = sin_kd_real * sin_kd_real + sin_kd_imag * sin_kd_imag;
                var cot_real = 0.0;
                var cot_imag = 0.0;
                if (denom > 1e-12) {
                    cot_real = (cos_kd_real * sin_kd_real + cos_kd_imag * sin_kd_imag) / denom;
                    cot_imag = (cos_kd_imag * sin_kd_real - cos_kd_real * sin_kd_imag) / denom;
                }

                // Surface impedance Zs = -j * Zc * cot(kd)
                let a = zc_real * cot_real - zc_imag * cot_imag;
                let b = zc_real * cot_imag + zc_imag * cot_real;
                let zs_real = b;
                let zs_imag = -a;

                // Reflection coefficient
                let rn_real = zs_real - z0;
                let rn_imag = zs_imag;
                let rd_real = zs_real + z0;
                let rd_imag = zs_imag;
                let rd_mag2 = rd_real * rd_real + rd_imag * rd_imag;
                var r_real = 0.0;
                var r_imag = 0.0;
                if (rd_mag2 > 1e-12) {
                    r_real = (rn_real * rd_real + rn_imag * rd_imag) / rd_mag2;
                    r_imag = (rn_imag * rd_real - rn_real * rd_imag) / rd_mag2;
                }
                let r_mag2 = r_real * r_real + r_imag * r_imag;
                result.absorption[i] = clamp(1.0 - r_mag2, 0.0, 1.0);
                result.scattering[i] = 0.0;
                result.transmission[i] = 0.0;
            }
        }

        // RESONANT_PANEL_MODEL_ID = 3: membrane absorber
        case 3u: {
            let offset = instance.param_offset;
            let panel_mass_kgm2 = bitcast<f32>((*params)[offset + 0u]);
            let cavity_depth_m   = bitcast<f32>((*params)[offset + 4u]);

            let rho = 1.204;
            let c0 = 343.0;
            let f0 = (c0 / (2.0 * 3.14159265)) * sqrt(rho / (panel_mass_kgm2 * cavity_depth_m));
            let q = 2.0 + 4.0 * min(panel_mass_kgm2 / 5.0, 1.0) + 4.0 * min(cavity_depth_m / 0.2, 1.0);

            for (var i = 0u; i < 8u; i = i + 1u) {
                let freq = (*freq_centres)[i];
                let eta = freq / f0;
                result.absorption[i] = 0.95 / (1.0 + q * q * (eta - 1.0 / eta) * (eta - 1.0 / eta));
                result.scattering[i] = 0.0;
                result.transmission[i] = 0.0;
            }
        }

        // Unknown model: fully reflective
        default: {
            for (var i = 0u; i < 8u; i = i + 1u) {
                result.absorption[i]  = 0.0;
                result.scattering[i]  = 0.0;
                result.transmission[i] = 0.0;
            }
        }
    }

    return result;
}