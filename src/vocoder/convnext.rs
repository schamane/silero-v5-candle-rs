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
    pointwise: Pointwise,
}

/// Die beiden Pointwise-Layer samt Layer-Scale `gamma`.
enum Pointwise {
    /// CPU: Linear, Linear, dann `broadcast_mul(gamma)`.
    Plain {
        pw1: Linear,
        pw2: Linear,
        gamma: Tensor,
    },
    /// Metal: candles Broadcast-Kernel ist dort rund fünfmal langsamer als ein
    /// zusammenhängender. Der Bias steckt deshalb als letzte Gewichtszeile im GEMM
    /// (Ones-Spalte am Eingang), `gamma` ist in `pw2` eingerechnet.
    Fused {
        /// `[512 + 1, 1536]`.
        w1: Tensor,
        /// `[1536 + 1, 512]`, mit `gamma` skaliert.
        w2: Tensor,
    },
}

impl Pointwise {
    fn load(block: &VarBuilder) -> Result<Self> {
        let pw1 = candle_nn::linear(512, 1536, block.pp("pwconv1"))?;
        let pw2 = candle_nn::linear(1536, 512, block.pp("pwconv2"))?;
        let gamma = block.get(512, "gamma")?;
        if !block.device().is_metal() {
            return Ok(Self::Plain { pw1, pw2, gamma });
        }
        let (Some(b1), Some(b2)) = (pw1.bias(), pw2.bias()) else {
            return Err(candle_core::Error::Msg("pwconv: Bias fehlt".into()));
        };
        let w1 = Tensor::cat(
            &[&pw1.weight().t()?.contiguous()?, &b1.reshape((1, 1536))?],
            0,
        )?;
        let w2 = Tensor::cat(
            &[
                &pw2.weight()
                    .broadcast_mul(&gamma.reshape((512, 1))?)?
                    .t()?
                    .contiguous()?,
                &b2.mul(&gamma)?.reshape((1, 512))?,
            ],
            0,
        )?;
        Ok(Self::Fused { w1, w2 })
    }

    /// `ones`: `[batch * T, 1]`; nur der Metal-Pfad braucht sie.
    fn forward(&self, y: &Tensor, ones: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain { pw1, pw2, gamma } => {
                let y = gelu(&pw1.forward(y)?)?;
                pw2.forward(&y)?.broadcast_mul(gamma)
            }
            Self::Fused { w1, w2 } => {
                let (batch, len, channels) = y.dims3()?;
                let rows = batch * len;
                let y = Tensor::cat(&[&y.reshape((rows, channels))?, ones], 1)?.matmul(w1)?;
                let y = gelu(&y)?;
                Tensor::cat(&[&y, ones], 1)?
                    .matmul(w2)?
                    .reshape((batch, len, channels))
            }
        }
    }
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
            blocks.push(Block::load(&vb.pp(format!("convnext.{index}")))?);
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
        let (batch, len, _) = x.dims3()?;
        let ones = Tensor::ones((batch * len, 1), x.dtype(), x.device())?;
        for block in &self.blocks {
            x = block.forward(&x, &ones)?;
        }
        self.final_norm.forward(&x)
    }
}

impl Block {
    fn load(block: &VarBuilder) -> Result<Self> {
        let dw_weight = block.get((512, 1, 7), "dwconv.weight")?;
        Ok(Self {
            dw_taps: depthwise_taps(&dw_weight)?,
            dw_bias: block.get(512, "dwconv.bias")?.reshape((1, 1, 512))?,
            norm: candle_nn::layer_norm(512, 1e-6, block.pp("norm"))?,
            pointwise: Pointwise::load(block)?,
        })
    }

    fn forward(&self, x: &Tensor, ones: &Tensor) -> Result<Tensor> {
        let y = depthwise(&self.dw_taps, &self.dw_bias, x)?;
        let y = self.norm.forward(&y)?;
        let y = self.pointwise.forward(&y, ones)?;
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use std::collections::HashMap;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use candle_core::DType;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use candle_nn::VarBuilder;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::Block;
    use super::gelu;

    /// Zufällige Blockgewichte; `gamma` absichtlich weit von 1 entfernt.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn block_tensors() -> HashMap<String, Tensor> {
        let shapes: [(&str, Vec<usize>); 9] = [
            ("dwconv.weight", vec![512, 1, 7]),
            ("dwconv.bias", vec![512]),
            ("norm.weight", vec![512]),
            ("norm.bias", vec![512]),
            ("pwconv1.weight", vec![1536, 512]),
            ("pwconv1.bias", vec![1536]),
            ("pwconv2.weight", vec![512, 1536]),
            ("pwconv2.bias", vec![512]),
            ("gamma", vec![512]),
        ];
        shapes
            .into_iter()
            .map(|(name, shape)| {
                let scale = if name == "gamma" { 0.3 } else { 0.1 };
                let tensor = Tensor::randn(0f32, scale, shape, &Device::Cpu).expect("randn");
                (name.to_owned(), tensor)
            })
            .collect()
    }

    /// Der Metal-Block (Bias im GEMM, `gamma` eingerechnet) stimmt mit dem
    /// ungefusten CPU-Block überein. Ohne Metal-Gerät wird übersprungen.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn fused_metal_block_matches_unfused_reference() {
        let Ok(metal) = Device::new_metal(0) else {
            eprintln!("SKIP: kein Metal-Gerät");
            return;
        };
        let tensors = block_tensors();
        let load = |device: &Device| {
            let moved = tensors
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.to_device(device).expect("to_device")))
                .collect();
            let vb = VarBuilder::from_tensors(moved, DType::F32, device);
            Block::load(&vb).expect("Block")
        };
        let (cpu_block, metal_block) = (load(&Device::Cpu), load(&metal));
        let len = 200;
        let input = Tensor::randn(0f32, 1f32, (1, len, 512), &Device::Cpu).expect("randn");
        let ones = |device: &Device| Tensor::ones((len, 1), DType::F32, device).expect("ones");
        let want = cpu_block
            .forward(&input, &ones(&Device::Cpu))
            .expect("CPU-Block");
        let got = metal_block
            .forward(&input.to_device(&metal).expect("to_device"), &ones(&metal))
            .expect("Metal-Block")
            .to_device(&Device::Cpu)
            .expect("Rücklesen");
        let want = want
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
        let got = got
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = want
            .iter()
            .zip(&got)
            .fold(0f32, |m, (w, g)| m.max((w - g).abs()));
        assert!(
            diff / scale < 1e-4,
            "relative Abweichung {} >= 1e-4",
            diff / scale
        );
    }

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
