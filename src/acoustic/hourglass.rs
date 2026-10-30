//! HourGlass-Decoder: `shorten_layers[0]` wird berechnet und verworfen, Rust
//! lädt und rechnet nur `shorten_layers[1]`.

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use super::transformer::{FftBlock, PositionalEncoding};

pub struct HourGlass {
    pos_encoder: PositionalEncoding,
    pre: FftBlock,
    shorten: FftBlock,
    post: FftBlock,
    upsample: Linear,
    norm: LayerNorm,
}

impl HourGlass {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let pos_encoder = PositionalEncoding::load(vb.pp("pos_encoder"), 128)?;
        let pre = FftBlock::load(vb.pp("pre_vanilla_layers.0"), 128, 512, 2, 1e-5)?;
        let shorten = FftBlock::load(vb.pp("shorten_layers.1"), 128, 512, 2, 1e-5)?;
        let post = FftBlock::load(vb.pp("post_vanilla_layers.0"), 128, 512, 2, 1e-5)?;
        let upsample = candle_nn::linear(128, 384, vb.pp("upsample.proj"))?;
        let norm = candle_nn::layer_norm(128, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            pos_encoder,
            pre,
            shorten,
            post,
            upsample,
            norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original = x.dim(1)?;
        let x = self.pos_encoder.forward(x)?;
        let x = self.pre.forward(&x)?;
        let padded = original.div_ceil(3) * 3;
        let x = if padded > original {
            let zeros = Tensor::zeros((1, padded - original, 128), x.dtype(), x.device())?;
            Tensor::cat(&[&x, &zeros], 1)?
        } else {
            x
        };
        let residual = x.clone();
        let down = x.reshape((1, padded / 3, 3, 128))?.mean(2)?;
        let down = self.shorten.forward(&down)?;
        let up = self.upsample.forward(&down)?;
        let up = up.reshape((1, padded, 128))?;
        let z = up.add(&residual)?;
        let z = z.narrow(1, 0, original)?;
        let z = self.post.forward(&z)?;
        self.norm.forward(&z)
    }
}
