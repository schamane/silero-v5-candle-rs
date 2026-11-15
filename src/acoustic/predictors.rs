//! `JitDurPredictor` und `JitOldPitchPredictor` (nur der genutzte Teil).

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::transformer::{ForwardTransformer, embedding, speaker_vector};

pub(crate) struct SeriesPredictor {
    embedding: Tensor,
    speaker: Tensor,
    transformer: ForwardTransformer,
    lin: Linear,
    device: Device,
}

impl SeriesPredictor {
    pub(crate) fn load(
        vb: VarBuilder,
        device: &Device,
        symbol_count: usize,
        speaker_count: usize,
    ) -> Result<Self> {
        let embedding = super::transformer::embedding_table(vb.pp("embedding"), symbol_count, 64)?;
        let speaker =
            super::transformer::embedding_table(vb.pp("speaker_embedding"), speaker_count, 64)?;
        let transformer = ForwardTransformer::load(vb.pp("transformer"), 64, 64, 2, 4, 1e-5)?;
        let lin = candle_nn::linear(64, 1, vb.pp("lin"))?;
        Ok(Self {
            embedding,
            speaker,
            transformer,
            lin,
            device: device.clone(),
        })
    }

    /// `text` sind die Symbol-IDs; Rückgabe `[1, T, 1]`.
    pub(crate) fn forward(&self, ids: &[u32], speaker_id: u32) -> Result<Tensor> {
        let embedded = embedding(&self.embedding, ids, &self.device)?;
        let embedded = embedded.unsqueeze(0)?;
        let speaker = speaker_vector(&self.speaker, speaker_id, 64, &self.device)?;
        let x = embedded.broadcast_add(&speaker)?;
        let x = self.transformer.forward(&x)?;
        self.lin.forward(&x)
    }
}

pub(crate) struct Predictors {
    pub duration: SeriesPredictor,
    pub pitch: SeriesPredictor,
}

impl Predictors {
    pub(crate) fn load(
        vb: VarBuilder,
        device: &Device,
        symbol_count: usize,
        speaker_count: usize,
    ) -> Result<Self> {
        let duration = SeriesPredictor::load(
            vb.pp("dur_predictor.dur_pred"),
            device,
            symbol_count,
            speaker_count,
        )?;
        let pitch = SeriesPredictor::load(
            vb.pp("pitch_predictor.pitch_pred"),
            device,
            symbol_count,
            speaker_count,
        )?;
        Ok(Self { duration, pitch })
    }

    /// `[1, T]` rohe Log-Dauern.
    pub(crate) fn durations(&self, ids: &[u32], speaker_id: u32) -> Result<Tensor> {
        self.duration.forward(ids, speaker_id)?.squeeze(2)
    }

    /// `[1, 1, T]` rohe Pitch-Werte (transponiert wie `JitOldPitchPredictor`).
    pub(crate) fn pitch(&self, ids: &[u32], speaker_id: u32) -> Result<Tensor> {
        self.pitch.forward(ids, speaker_id)?.transpose(1, 2)
    }
}
