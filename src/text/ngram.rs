//! Gemeinsamer Zahlenkern der n-Gramm-Akzentoren: n-Gramm-Tabelle,
//! Mean-EmbeddingBag, lineare Schichten und Softmax.
//!
//! Beide Akzentoren (`v5_ru` und ukrainisch) benutzen dieselbe fastText-artige
//! Einbettung und dieselbe MLP-Form. Ein gemeinsamer Kern hält die
//! Gleitkomma-Reihenfolge an einer Stelle, damit die Python-Parität nicht
//! zwischen zwei Kopien auseinanderläuft.

use std::collections::HashMap;
use std::path::Path;

use candle_core::Tensor;

use crate::error::io_at;
use crate::{Error, Result};

/// Dichte lineare Schicht in Zeilenreihenfolge (`weight[out][in]`).
#[derive(Debug)]
pub(crate) struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    in_dim: usize,
}

impl Linear {
    /// Lädt eine Schicht mit erwarteter Form.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Model`] zurück, wenn Gewicht oder Bias fehlen oder nicht
    /// `out * in_dim` beziehungsweise `out` Werte haben.
    pub(crate) fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        out: usize,
        in_dim: usize,
    ) -> Result<Self> {
        let w = tensor_vec(tensors, &format!("{prefix}.weight"))?;
        let b = tensor_vec(tensors, &format!("{prefix}.bias"))?;
        if w.len() != out * in_dim {
            return Err(Error::Model(format!(
                "{prefix}.weight: {} Werte statt {out}x{in_dim}",
                w.len()
            )));
        }
        if b.len() != out {
            return Err(Error::Model(format!(
                "{prefix}.bias: {} Werte statt {out}",
                b.len()
            )));
        }
        Ok(Self { w, b, out, in_dim })
    }

    /// Lädt eine Schicht und leitet `out`/`in_dim` aus `weight` ab.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Model`] zurück, wenn `weight` fehlt, nicht zweidimensional ist
    /// oder Bias beziehungsweise Länge nicht zur Form passen.
    pub(crate) fn load_auto(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let tensor = tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| Error::Model(format!("{prefix}.weight fehlt")))?;
        let (out, in_dim) = tensor
            .dims2()
            .map_err(|error| Error::Model(format!("{prefix}.weight: {error}")))?;
        Self::load(tensors, prefix, out, in_dim)
    }

    /// Anzahl der Ausgabewerte.
    pub(crate) fn out(&self) -> usize {
        self.out
    }

    /// Anzahl der Eingabewerte.
    pub(crate) fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// `y = W x + b`.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.in_dim,
            "Linear: Eingabe {} statt {}",
            x.len(),
            self.in_dim
        );
        let mut y = vec![0.0; self.out];
        for (o, slot) in y.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut sum = self.b[o];
            for (w, x) in row.iter().zip(x) {
                sum += w * x;
            }
            *slot = sum;
        }
        y
    }
}

/// Liest einen Tensor als flache `f32`-Folge.
///
/// # Errors
///
/// Gibt [`Error::Model`] zurück, wenn der Tensor fehlt oder sich nicht als `f32`
/// lesen lässt.
pub(crate) fn tensor_vec(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Model(format!("{name} fehlt in den Akzentor-Gewichten")))?;
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| Error::Model(format!("{name}: {error}")))
}

/// MLP mit ReLU zwischen den Schichten, aber ohne Aktivierung am Ausgang.
pub(crate) fn run_mlp(layers: &[Linear], input: &[f32]) -> Vec<f32> {
    let mut x = input.to_vec();
    for (index, layer) in layers.iter().enumerate() {
        x = layer.forward(&x);
        if index + 1 < layers.len() {
            for value in &mut x {
                *value = value.max(0.0);
            }
        }
    }
    x
}

/// Softmax und Argmax; bei Gleichstand gewinnt der erste Index wie bei `torch.argmax`.
pub(crate) fn softmax_argmax(logits: &[f32]) -> (Vec<f32>, usize) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for value in &mut probs {
        *value /= sum;
    }
    let mut argmax = 0;
    for (index, value) in probs.iter().enumerate() {
        if *value > probs[argmax] {
            argmax = index;
        }
    }
    (probs, argmax)
}

/// fastText-artige n-Gramm-Tabelle plus Mean-EmbeddingBag.
#[derive(Debug)]
pub(crate) struct NgramEmbedding {
    ngrams: HashMap<String, u32>,
    weights: Vec<f32>,
    dim: usize,
    unk: u32,
}

impl NgramEmbedding {
    /// Lädt `ngrams.txt` und `{prefix}.weight` aus `tensors`.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn `ngrams.txt` nicht lesbar ist, und
    /// [`Error::Model`], wenn die Tabelle nicht `count` Einträge hat, ein n-Gramm
    /// doppelt vorkommt oder das Embedding nicht `count x dim` groß ist.
    pub(crate) fn load(
        ngrams_path: &Path,
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        count: usize,
        dim: usize,
    ) -> Result<Self> {
        let weights = tensor_vec(tensors, &format!("{prefix}.weight"))?;
        if weights.len() != count * dim {
            return Err(Error::Model(format!(
                "{prefix}.weight: {} Werte statt {count}x{dim}",
                weights.len()
            )));
        }
        let text = std::fs::read_to_string(ngrams_path).map_err(io_at(ngrams_path))?;
        let mut ngrams = HashMap::with_capacity(count);
        for (id, gram) in text.lines().enumerate() {
            if ngrams.insert(gram.to_string(), id as u32).is_some() {
                return Err(Error::Model(format!(
                    "ngrams.txt: doppeltes n-Gramm {gram:?} in Zeile {}",
                    id + 1
                )));
            }
        }
        if ngrams.len() != count {
            return Err(Error::Model(format!(
                "ngrams.txt: {} statt {count} Einträge",
                ngrams.len()
            )));
        }
        let unk = *ngrams
            .get("UNK")
            .ok_or_else(|| Error::Model("ngrams.txt: UNK fehlt".into()))?;
        Ok(Self {
            ngrams,
            weights,
            dim,
            unk,
        })
    }

    /// `word_ngrams(word, 1, len+3)`, dann `embedding_bag(mean)`. `UNK` wird nur
    /// verwendet, wenn kein n-Gramm im Vokabular liegt.
    pub(crate) fn word_vector(&self, word: &[char]) -> Vec<f32> {
        let max_len = word.len() + 3;
        let mut wrapped = Vec::with_capacity(word.len() + 2);
        wrapped.push('<');
        wrapped.extend_from_slice(word);
        wrapped.push('>');
        let mut ids = Vec::new();
        // Ein Puffer für alle n-Gramme. `clear` behält die Kapazität.
        let mut gram = String::new();
        for n in 1..=max_len {
            if n > wrapped.len() {
                continue;
            }
            for start in 0..=wrapped.len() - n {
                gram.clear();
                gram.extend(&wrapped[start..start + n]);
                if let Some(id) = self.ngrams.get(gram.as_str()) {
                    ids.push(*id as usize);
                }
            }
        }
        if word.is_empty()
            && let Some(id) = self.ngrams.get("")
        {
            ids.push(*id as usize);
        }
        if ids.is_empty() {
            ids.push(self.unk as usize);
        }
        let mut mean = vec![0.0f32; self.dim];
        for id in &ids {
            let row = &self.weights[id * self.dim..(id + 1) * self.dim];
            for (slot, value) in mean.iter_mut().zip(row) {
                *slot += value;
            }
        }
        let count = ids.len() as f32;
        for value in &mut mean {
            *value /= count;
        }
        mean
    }
}
