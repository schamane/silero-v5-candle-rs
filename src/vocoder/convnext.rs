//! `JitV4Backbone`: ConvNeXt-Blöcke mit selbst geschriebener depthwise Conv.

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

pub struct Backbone {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
}

struct Block {
    /// Die sieben Gewichtsspalten der depthwise Conv, je `[1, C, 1]`.
    dw_taps: Vec<Tensor>,
    /// Bias der depthwise Conv als `[1, C, 1]`.
    dw_bias: Tensor,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

impl Backbone {
    pub fn load(vb: VarBuilder) -> Result<Self> {
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
                dw_bias: block.get(512, "dwconv.bias")?.reshape((1, 512, 1))?,
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
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(mel)?;
        let x = self.norm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        let mut x = x.transpose(1, 2)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.final_norm.forward(&x.transpose(1, 2)?.contiguous()?)
    }
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = depthwise(&self.dw_taps, &self.dw_bias, x)?;
        let y = self.norm.forward(&y.transpose(1, 2)?.contiguous()?)?;
        let y = self.pw1.forward(&y)?.gelu_erf()?;
        let y = self.pw2.forward(&y)?;
        let y = y.broadcast_mul(&self.gamma)?;
        x.add(&y.transpose(1, 2)?)
    }
}

/// Zerlegt das Gewicht `[C, 1, 7]` einmalig in sieben Spalten `[1, C, 1]`.
fn depthwise_taps(weight: &Tensor) -> Result<Vec<Tensor>> {
    let (channels, _, kernel) = weight.dims3()?;
    (0..kernel)
        .map(|tap| weight.narrow(2, tap, 1)?.reshape((1, channels, 1)))
        .collect()
}

/// candle zerlegt `groups = 512` in 512 einzelne Convs; hier sieben
/// verschobene Slices mal Gewichtsspalte. Die Akkumulation startet mit dem
/// ersten Produkt statt mit Nullen, das Ergebnis ist dasselbe.
fn depthwise(taps: &[Tensor], bias: &Tensor, x: &Tensor) -> Result<Tensor> {
    let (_, channels, len) = x.dims3()?;
    let zeros = Tensor::zeros((1, channels, 3), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&zeros, x, &zeros], 2)?;
    let (first, rest) = taps.split_first().ok_or_else(|| {
        candle_core::Error::Msg("depthwise: keine Gewichtsspalten geladen".into())
    })?;
    let mut out = padded.narrow(2, 0, len)?.broadcast_mul(first)?;
    for (offset, w) in rest.iter().enumerate() {
        let slice = padded.narrow(2, offset + 1, len)?;
        out = out.add(&slice.broadcast_mul(w)?)?;
    }
    out.broadcast_add(bias)
}
