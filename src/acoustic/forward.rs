//! `JitMultiForward` (Encoder, LengthRegulator, HourGlass, `lin`) und die
//! Dauer-/Pitch-Nachbearbeitung aus `MultiTTSModel.forward`.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use super::hourglass::HourGlass;
use super::predictors::Predictors;
use super::transformer::{ForwardTransformer, embedding};
use crate::config::MAX_POSITIONS;

pub struct Tacotron {
    embedding: Tensor,
    speaker_embedding: Tensor,
    encoder: ForwardTransformer,
    pitch_proj: Conv1d,
    decoder: HourGlass,
    lin: Linear,
    device: Device,
}

impl Tacotron {
    pub fn load(
        vb: VarBuilder,
        device: &Device,
        symbol_count: usize,
        speaker_count: usize,
    ) -> Result<Self> {
        let embedding = super::transformer::embedding_table(vb.pp("embedding"), symbol_count, 128)?;
        let speaker_embedding =
            super::transformer::embedding_table(vb.pp("speaker_embedding"), speaker_count, 128)?;
        let encoder = ForwardTransformer::load(vb.pp("encoder"), 128, 512, 2, 4, 1e-5)?;
        let pitch_proj = candle_nn::conv1d(
            1,
            128,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("pitch_proj"),
        )?;
        let decoder = HourGlass::load(vb.pp("decoder"))?;
        let lin = candle_nn::linear(128, 192, vb.pp("lin"))?;
        Ok(Self {
            embedding,
            speaker_embedding,
            encoder,
            pitch_proj,
            decoder,
            lin,
            device: device.clone(),
        })
    }

    pub fn encoder_output(&self, ids: &[u32]) -> Result<Tensor> {
        let embedded = embedding(&self.embedding, ids, &self.device)?.unsqueeze(0)?;
        self.encoder.forward(&embedded)
    }

    fn speaker(&self, speaker_id: u32) -> Result<Tensor> {
        self.speaker_embedding
            .index_select(&Tensor::from_vec(vec![speaker_id], 1, &self.device)?, 0)?
            .reshape((1, 1, 128))
    }

    pub fn forward(
        &self,
        ids: &[u32],
        speaker_id: u32,
        pitch: &Tensor,
        dur: &[u32],
    ) -> Result<TacotronOutput> {
        let encoder = self.encoder_output(ids)?;
        let cond = encoder.broadcast_add(&self.speaker(speaker_id)?)?;
        let pitch = self.pitch_proj.forward(pitch)?.transpose(1, 2)?;
        let cond = cond.add(&pitch)?;
        let expanded = length_regulate(&cond, dur)?;
        let decoder = self.decoder.forward(&expanded)?;
        let mel = self.lin.forward(&decoder)?.transpose(1, 2)?;
        Ok(TacotronOutput {
            encoder,
            cond,
            expanded,
            decoder,
            mel,
        })
    }
}

pub struct TacotronOutput {
    pub encoder: Tensor,
    pub cond: Tensor,
    pub expanded: Tensor,
    pub decoder: Tensor,
    pub mel: Tensor,
}

pub fn length_regulate(x: &Tensor, dur: &[u32]) -> Result<Tensor> {
    let total: usize = dur.iter().map(|d| *d as usize).sum();
    let mut indices = Vec::with_capacity(total);
    for (index, count) in dur.iter().enumerate() {
        for _ in 0..*count {
            indices.push(index as u32);
        }
    }
    let indices = Tensor::from_vec(indices, total, x.device())?;
    x.index_select(&indices, 1)
}

/// `exp(pred) - 1`, negativ auf 0, `round`, `/ durs_rate`, `round`, dann die
/// vier festen Korrekturen. `torch.round` rundet Halbwerte gerade.
pub fn postprocess_durations(raw: &Tensor, durs_rate: f32) -> Result<Vec<u32>> {
    let rows = raw.to_vec2::<f32>()?;
    let values = &rows[0];
    let mut dur: Vec<f32> = values
        .iter()
        .map(|value| ((value.exp() - 1.0).max(0.0)).round_ties_even())
        .map(|value| (value / durs_rate).round_ties_even())
        .collect();
    let len = dur.len();
    if len < 3 {
        candle_core::bail!("Sequenz zu kurz für die Dauer-Korrekturen: {len} < 3 Symbole");
    }
    if dur[0] > 5.0 {
        dur[0] = 5.0;
    }
    if dur[len - 1] > 7.0 {
        dur[len - 1] = 7.0;
    }
    dur[len - 2] = 13.0;
    if dur[len - 3] > 13.0 {
        dur[len - 3] = 13.0;
    }
    Ok(dur.into_iter().map(|value| value as u32).collect())
}

/// Bei Koeffizient 1.0 bleibt von `update_pitch_coef` nur die Nullung kleiner
/// Werte; die Satzzeichen-Nullung des Originals ist toter Code.
///
/// Werte mit `|x| < 0.001` werden zu `0.0`. Alle anderen bleiben unverändert, auch `NaN`,
/// weil der Vergleich dafür falsch ist.
///
/// # Errors
///
/// Gibt einen Fehler zurück, wenn ein Tensor-Operator fehlschlägt.
pub fn update_pitch(pitch: &Tensor) -> Result<Tensor> {
    let small = pitch.abs()?.lt(0.001f32)?;
    small.where_cond(&pitch.zeros_like()?, pitch)
}

pub struct Acoustic {
    tacotron: Tacotron,
    pub predictors: Predictors,
    pub device: Device,
}

pub struct AcousticOutput {
    pub dur: Vec<u32>,
    pub dur_raw: Tensor,
    pub pitch_raw: Tensor,
    pub pitch: Tensor,
    pub encoder: Tensor,
    pub cond: Tensor,
    pub expanded: Tensor,
    pub decoder: Tensor,
    pub mel: Tensor,
}

impl Acoustic {
    pub fn load(
        vb: VarBuilder,
        device: &Device,
        symbol_count: usize,
        speaker_count: usize,
    ) -> Result<Self> {
        let tacotron = Tacotron::load(vb.pp("tacotron"), device, symbol_count, speaker_count)?;
        let predictors = Predictors::load(vb, device, symbol_count, speaker_count)?;
        Ok(Self {
            tacotron,
            predictors,
            device: device.clone(),
        })
    }

    pub fn forward(&self, ids: &[u32], speaker_id: u32) -> Result<AcousticOutput> {
        if ids.len() > MAX_POSITIONS {
            candle_core::bail!("Sequenz zu lang: {} Symbole > {MAX_POSITIONS}", ids.len());
        }
        let dur_raw = self.predictors.durations(ids, speaker_id)?;
        let dur = postprocess_durations(&dur_raw, 1.0)?;
        let frames: usize = dur.iter().map(|d| *d as usize).sum();
        if frames > MAX_POSITIONS {
            candle_core::bail!("Audio zu lang: {frames} Frames > {MAX_POSITIONS}");
        }
        let pitch_raw = self.predictors.pitch(ids, speaker_id)?;
        let pitch = update_pitch(&pitch_raw)?;
        let out = self.tacotron.forward(ids, speaker_id, &pitch, &dur)?;
        Ok(AcousticOutput {
            dur,
            dur_raw,
            pitch_raw,
            pitch,
            encoder: out.encoder,
            cond: out.cond,
            expanded: out.expanded,
            decoder: out.decoder,
            mel: out.mel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_tokens_are_postprocessed() -> Result<()> {
        let raw = Tensor::from_vec(vec![7f32.ln(), 2f32.ln(), 30f32.ln()], (1, 3), &Device::Cpu)?;
        let dur = postprocess_durations(&raw, 1.0)?;
        assert_eq!(
            dur,
            vec![5, 13, 7],
            "exp(raw) - 1 = [6, 1, 29] before the corrections"
        );
        Ok(())
    }
}
