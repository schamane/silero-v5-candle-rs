//! `JitV4Backbone`: ConvNeXt-Blöcke mit selbst geschriebener depthwise Conv.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};
use rayon::prelude::*;

pub(crate) struct Backbone {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
}

struct Block {
    /// Die sieben Gewichtsspalten der depthwise Conv, je `[1, 1, C]`.
    dw_taps: Vec<Tensor>,
    /// Bias der depthwise Conv als `[1, 1, C]`.
    dw_bias: Tensor,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

impl Backbone {
    pub(crate) fn load(vb: VarBuilder) -> Result<Self> {
        let embed = candle_nn::conv1d(
            192,
            512,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("embed"),
        )?;
        let norm = candle_nn::layer_norm(512, 1e-6, vb.pp("norm"))?;
        let mut blocks = Vec::with_capacity(8);
        for index in 0..8 {
            let block = vb.pp(format!("convnext.{index}"));
            let dw_weight = block.get((512, 1, 7), "dwconv.weight")?;
            blocks.push(Block {
                dw_taps: depthwise_taps(&dw_weight)?,
                dw_bias: block.get(512, "dwconv.bias")?.reshape((1, 1, 512))?,
                norm: candle_nn::layer_norm(512, 1e-6, block.pp("norm"))?,
                pw1: candle_nn::linear(512, 1536, block.pp("pwconv1"))?,
                pw2: candle_nn::linear(1536, 512, block.pp("pwconv2"))?,
                gamma: block.get(512, "gamma")?,
            });
        }
        let final_norm = candle_nn::layer_norm(512, 1e-6, vb.pp("final_layer_norm"))?;
        Ok(Self {
            embed,
            norm,
            blocks,
            final_norm,
        })
    }

    /// Mel `[1, 192, T]` -> Features `[1, T, 512]`.
    pub(crate) fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(mel)?;
        // Ab hier bleibt alles im NTC-Layout `[1, T, C]`.
        let mut x = self.norm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.final_norm.forward(&x)
    }
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = depthwise(&self.dw_taps, &self.dw_bias, x)?;
        let y = self.norm.forward(&y)?;
        let y = gelu(&self.pw1.forward(&y)?)?;
        let y = self.pw2.forward(&y)?;
        let y = y.broadcast_mul(&self.gamma)?;
        x.add(&y)
    }
}

/// GELU (exakte Erf-Form). Auf der CPU eine parallele Näherung von `erf`
/// (Abweichung < 1e-6 gegenüber `gelu_erf`); andere Geräte nutzen `gelu_erf`.
fn gelu(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() || x.dtype() != DType::F32 {
        return x.gelu_erf();
    }
    let shape = x.shape().clone();
    let mut data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    data.par_chunks_mut(GELU_CHUNK)
        .for_each(|chunk| chunk.iter_mut().for_each(|v| *v = gelu_scalar(*v)));
    Tensor::from_vec(data, shape, x.device())
}

/// Chunk-Größe der parallelen GELU-Schleife (Elemente pro Rayon-Task).
const GELU_CHUNK: usize = 4096;

/// `0.5 * x * (1 + erf(x / sqrt(2)))` mit der Näherung 7.1.26 nach Abramowitz/Stegun.
fn gelu_scalar(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_7,
        -1.453_152_1,
        1.061_405_4,
    ];
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + P * z.abs());
    let poly = ((((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0]) * t;
    let erf_abs = 1.0 - poly * (-z * z).exp();
    let erf = erf_abs.copysign(z);
    0.5 * x * (1.0 + erf)
}

/// Zerlegt das Gewicht `[C, 1, 7]` einmalig in sieben Spalten `[1, 1, C]`.
fn depthwise_taps(weight: &Tensor) -> Result<Vec<Tensor>> {
    let (channels, _, kernel) = weight.dims3()?;
    (0..kernel)
        .map(|tap| weight.narrow(2, tap, 1)?.reshape((1, 1, channels)))
        .collect()
}

/// candle zerlegt `groups = 512` in 512 einzelne Convs; hier sieben
/// verschobene Slices mal Gewichtsspalte. Die Akkumulation startet mit dem
/// ersten Produkt statt mit Nullen, das Ergebnis ist dasselbe.
fn depthwise(taps: &[Tensor], bias: &Tensor, x: &Tensor) -> Result<Tensor> {
    let (_, len, channels) = x.dims3()?;
    let zeros = Tensor::zeros((1, 3, channels), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&zeros, x, &zeros], 1)?;
    let (first, rest) = taps.split_first().ok_or_else(|| {
        candle_core::Error::Msg("depthwise: keine Gewichtsspalten geladen".into())
    })?;
    let mut out = padded.narrow(1, 0, len)?.broadcast_mul(first)?;
    for (offset, w) in rest.iter().enumerate() {
        let slice = padded.narrow(1, offset + 1, len)?;
        out = out.add(&slice.broadcast_mul(w)?)?;
    }
    out.broadcast_add(bias)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::gelu;

    #[test]
    fn gelu_matches_gelu_erf_on_cpu() {
        let values: Vec<f32> = (0..20_000u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 8) % 160_001) as f32 / 10_000.0 - 8.0)
            .collect();
        let x = Tensor::from_vec(values, (1, 100, 200), &Device::Cpu).expect("Tensor");
        let fast = gelu(&x).expect("gelu");
        let exact = x.gelu_erf().expect("gelu_erf");
        assert_eq!(fast.dims(), exact.dims(), "Form weicht ab");
        let diff = (fast - exact)
            .expect("sub")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("skalar");
        assert!(diff < 1e-6, "max. Abweichung {diff} >= 1e-6");
    }
}
