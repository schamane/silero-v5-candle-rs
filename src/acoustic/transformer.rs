//! FastPitch-Transformatoren: PositionalEncoding, Aufmerksamkeit, FFTBlock.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Module, VarBuilder, layer_norm};

use crate::config::MAX_POSITIONS;

pub struct PositionalEncoding {
    scale: Tensor,
    pe: Tensor,
    max_positions: usize,
}

impl PositionalEncoding {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let scale = vb.get(1, "scale")?;
        let pe = vb.get((MAX_POSITIONS, 1, dim), "pe")?.squeeze(1)?;
        Ok(Self {
            scale,
            pe,
            max_positions: MAX_POSITIONS,
        })
    }

    /// `x + scale * pe[:T]`; `x` ist `[1, T, dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        if t > self.max_positions {
            candle_core::bail!("Sequenz zu lang: {t} > {MAX_POSITIONS} Positionen");
        }
        let pe = self.pe.narrow(0, 0, t)?.unsqueeze(0)?;
        let pe = pe.broadcast_mul(&self.scale)?;
        x.add(&pe)
    }
}

pub struct MultiHeadAttention {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn load(vb: VarBuilder, dim: usize, heads: usize) -> Result<Self> {
        let weight = vb.get((3 * dim, dim), "in_proj_weight")?;
        let bias = vb.get(3 * dim, "in_proj_bias")?;
        let q_w = weight.narrow(0, 0, dim)?.contiguous()?;
        let k_w = weight.narrow(0, dim, dim)?.contiguous()?;
        let v_w = weight.narrow(0, 2 * dim, dim)?.contiguous()?;
        let q_b = bias.narrow(0, 0, dim)?.contiguous()?;
        let k_b = bias.narrow(0, dim, dim)?.contiguous()?;
        let v_b = bias.narrow(0, 2 * dim, dim)?.contiguous()?;
        let out_w = vb.get((dim, dim), "out_proj.weight")?;
        let out_b = vb.get(dim, "out_proj.bias")?;
        Ok(Self {
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            out_w,
            out_b,
            heads,
            head_dim: dim / heads,
        })
    }

    fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&weight.t()?)?.broadcast_add(bias)
    }

    /// `x` ist `[1, T, dim]`, die Padding-Maske ist bei Batch 1 leer.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, len, dim) = x.dims3()?;
        let q = Self::linear(x, &self.q_w, &self.q_b)?;
        let k = Self::linear(x, &self.k_w, &self.k_b)?;
        let v = Self::linear(x, &self.v_w, &self.v_b)?;
        let split = |t: &Tensor| -> Result<Tensor> {
            t.reshape((batch, len, self.heads, self.head_dim))?
                .permute((0, 2, 1, 3))
        };
        let q = split(&q)?;
        let k = split(&k)?;
        let v = split(&v)?;
        let scale = (self.head_dim as f64).sqrt();
        let q = (q / scale)?;
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = scores.matmul(&v)?;
        let context = context
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, len, dim))?;
        Self::linear(&context, &self.out_w, &self.out_b)
    }
}

pub struct FftBlock {
    attention: MultiHeadAttention,
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl FftBlock {
    pub fn load(vb: VarBuilder, dim: usize, ffn: usize, heads: usize, eps: f64) -> Result<Self> {
        let attention = MultiHeadAttention::load(vb.pp("self_attn"), dim, heads)?;
        let conv1 = candle_nn::conv1d(
            dim,
            ffn,
            9,
            Conv1dConfig {
                padding: 4,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(ffn, dim, 1, Conv1dConfig::default(), vb.pp("conv2"))?;
        let norm1 = layer_norm(dim, eps, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, eps, vb.pp("norm2"))?;
        Ok(Self {
            attention,
            conv1,
            conv2,
            norm1,
            norm2,
        })
    }

    /// Nur der normale Pfad: `pitch_pred` ist in allen Blöcken `False`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(x)?;
        let y = self.norm1.forward(&x.add(&attention)?)?;
        let conv = y.transpose(1, 2)?;
        let conv = self.conv1.forward(&conv)?.relu()?;
        let conv = self.conv2.forward(&conv)?;
        let conv = conv.transpose(1, 2)?;
        self.norm2.forward(&y.add(&conv)?)
    }
}

pub struct ForwardTransformer {
    pos_encoder: PositionalEncoding,
    layers: Vec<FftBlock>,
    norm: LayerNorm,
}

impl ForwardTransformer {
    pub fn load(
        vb: VarBuilder,
        dim: usize,
        ffn: usize,
        heads: usize,
        layers: usize,
        eps: f64,
    ) -> Result<Self> {
        let pos_encoder = PositionalEncoding::load(vb.pp("pos_encoder"), dim)?;
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            blocks.push(FftBlock::load(
                vb.pp(format!("layers.{index}")),
                dim,
                ffn,
                heads,
                eps,
            )?);
        }
        let norm = layer_norm(dim, eps, vb.pp("norm"))?;
        Ok(Self {
            pos_encoder,
            layers: blocks,
            norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.pos_encoder.forward(x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

pub fn embedding_table(vb: VarBuilder, rows: usize, dim: usize) -> Result<Tensor> {
    vb.get((rows, dim), "weight")
}

pub fn embedding(weight: &Tensor, ids: &[u32], device: &Device) -> Result<Tensor> {
    let ids = Tensor::from_slice(ids, ids.len(), device)?;
    weight.embedding(&ids)
}
