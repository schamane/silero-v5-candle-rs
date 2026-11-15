//! `JitISTFTHead` plus inverse STFT mit `padding="same"`.

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner};

use crate::Error;

/// Der ISTFT-Frame schreibt `n_fft` Samples und multipliziert sie mit dem
/// Fenster der Länge `win_length`; beide müssen deshalb übereinstimmen.
///
/// `pad = (win - hop) / 2` wird später als `y[pad..len - pad]` abgeschnitten.
/// Bei `pad == 0` weichen die Schnitte ab: Python `y[:, 0:-0]` ist leer, der Rust-Schnitt
/// `y[0..len]` enthält dagegen alles. Deshalb ist `pad == 0` ein Fehler.
fn validate_fft(n_fft: usize, hop: usize, win: usize) -> crate::Result<()> {
    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        return Err(Error::model_msg(format!(
            "n_fft muss gerade und > 0 sein: {n_fft}"
        )));
    }
    if win == 0 || hop == 0 || hop > win {
        return Err(Error::model_msg(format!(
            "hop_length/win_length unbrauchbar: {hop}/{win}"
        )));
    }
    if n_fft != win {
        return Err(Error::model_msg(format!(
            "n_fft {n_fft} != win_length {win}: der ISTFT-Frame schreibt n_fft Samples"
        )));
    }
    if (win - hop) / 2 == 0 {
        return Err(Error::model_msg(format!(
            "pad (win - hop) / 2 ist 0 bei hop_length {hop} / win_length {win}"
        )));
    }
    Ok(())
}

/// Spektrum in Frame-Major-Layout: jeweils `[1, T, 1201]`, zusammenhängend im Speicher.
pub(crate) struct Spectrum {
    pub mag: Tensor,
    pub cos: Tensor,
    pub sin: Tensor,
}

pub(crate) struct Head {
    out: Linear,
    /// Fenster mal `1 / n_fft`; spart die Skalierung pro Sample.
    window_scaled: Vec<f32>,
    /// Fenster zum Quadrat; die Überlappungshülle summiert diese Werte.
    window_squared: Vec<f32>,
    irfft: Arc<dyn ComplexToReal<f32>>,
    n_fft: usize,
    hop: usize,
    win: usize,
    pad: usize,
}

impl Head {
    /// Lädt den Ausgabe-Layer und plant die inverse FFT einmalig.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn die FFT-Geometrie unbrauchbar ist, die
    /// Fensterlänge nicht `win_length` entspricht oder der Layer nicht geladen
    /// werden kann.
    pub(crate) fn load(
        vb: VarBuilder,
        window: Vec<f32>,
        n_fft: usize,
        hop: usize,
        win: usize,
    ) -> crate::Result<Self> {
        validate_fft(n_fft, hop, win)?;
        if window.len() != win {
            return Err(Error::model_msg(format!(
                "ISTFT-Fenster hat {} statt win_length {win} Werte",
                window.len()
            )));
        }
        let bins = n_fft / 2 + 1;
        let out = candle_nn::linear(512, 2 * bins, vb.pp("out"))
            .map_err(|error| Error::model("out", error))?;
        let irfft = RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft);
        let scale = 1.0 / n_fft as f32;
        let window_scaled = window.iter().map(|value| value * scale).collect();
        let window_squared = window.iter().map(|value| value * value).collect();
        Ok(Self {
            out,
            window_scaled,
            window_squared,
            irfft,
            n_fft,
            hop,
            win,
            pad: (win - hop) / 2,
        })
    }

    /// Features `[1, T, 512]` -> Betrag, Kosinus und Sinus, jeweils `[1, T, 1201]`.
    ///
    /// Frame-major und zusammenhängend: `exp`/`cos`/`sin` laufen so auf dem schnellen
    /// Pfad von candle, und die ISTFT liest die Bins eines Frames am Stück.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Spectrum> {
        let x = self.out.forward(x)?;
        let bins = self.n_fft / 2 + 1;
        let mag = x.narrow(2, 0, bins)?.contiguous()?;
        let phase = x.narrow(2, bins, bins)?.contiguous()?;
        let mag = mag.exp()?.minimum(100f32)?;
        let cos = phase.cos()?;
        let sin = phase.sin()?;
        Ok(Spectrum { mag, cos, sin })
    }

    /// `ISTFT(mag * (cos + i sin))`, überlappt und durch die Fensterhülle
    /// geteilt.
    pub(crate) fn istft(&self, spectrum: &Spectrum) -> Result<Vec<f32>> {
        let (_, frames, bins) = spectrum.mag.dims3()?;
        let expected_bins = self.n_fft / 2 + 1;
        if bins != expected_bins {
            return Err(candle_core::Error::Msg(format!(
                "ISTFT: {bins} Bins statt {expected_bins}"
            )));
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        // Frame-major (`frame * bins + bin`): die Bin-Schleife liest zusammenhängend.
        let mag = flat(&spectrum.mag)?;
        let cos = flat(&spectrum.cos)?;
        let sin = flat(&spectrum.sin)?;

        let mut input = self.irfft.make_input_vec();
        let mut output = self.irfft.make_output_vec();
        let mut scratch = self.irfft.make_scratch_vec();

        let out_len = (frames - 1) * self.hop + self.win;
        let mut audio = vec![0.0f32; out_len];
        for frame in 0..frames {
            let base = frame * bins;
            for bin in 0..bins {
                let magnitude = mag[base + bin];
                input[bin] =
                    Complex32::new(magnitude * cos[base + bin], magnitude * sin[base + bin]);
            }
            // `irfft` ignoriert DC- und Nyquist-Imaginärteil; explizit nullen
            // macht die Absicht sichtbar.
            input[0].im = 0.0;
            input[bins - 1].im = 0.0;
            self.irfft
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .map_err(|error| candle_core::Error::Msg(format!("irfft: {error}")))?;
            let offset = frame * self.hop;
            for ((acc, value), weight) in audio[offset..offset + self.win]
                .iter_mut()
                .zip(&output)
                .zip(&self.window_scaled)
            {
                *acc += value * weight;
            }
        }

        // Die Hülle hängt nur von der Frame-Zahl ab, nicht von den Bins: einmal pro Aufruf.
        let mut envelope = vec![0.0f32; out_len];
        for frame in 0..frames {
            let offset = frame * self.hop;
            for (slot, value) in envelope[offset..offset + self.win]
                .iter_mut()
                .zip(&self.window_squared)
            {
                *slot += value;
            }
        }
        let start = self.pad;
        let end = out_len - self.pad;
        // Prüfung vor der Schleife: sie wäre sonst ein Verzweigungs- und
        // Bounds-Check pro Sample und würde die Divisionsschleife bremsen.
        if let Some(offset) = envelope[start..end]
            .iter()
            .position(|divisor| *divisor <= 1e-11)
        {
            return Err(candle_core::Error::Msg(format!(
                "ISTFT-Fensterhülle bei Sample {} ist null",
                start + offset
            )));
        }
        let mut samples = Vec::with_capacity(end - start);
        for (value, divisor) in audio[start..end].iter().zip(&envelope[start..end]) {
            samples.push(value / divisor);
        }
        Ok(samples)
    }
}

/// Liest `[1, frames, bins]` als flachen Vektor in Reihenfolge `frame * bins + bin`.
fn flat(spectrum: &Tensor) -> Result<Vec<f32>> {
    spectrum.flatten_all()?.to_vec1::<f32>()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use super::{Head, validate_fft};

    #[test]
    fn accepts_the_model_geometry() {
        assert!(validate_fft(2400, 600, 2400).is_ok());
    }

    #[test]
    fn rejects_inconsistent_fft_geometry() {
        assert!(validate_fft(2400, 600, 1200).is_err());
        assert!(validate_fft(2401, 600, 2401).is_err());
        assert!(validate_fft(2400, 2401, 2400).is_err());
        assert!(validate_fft(0, 0, 0).is_err());
    }

    #[test]
    fn rejects_zero_padding() {
        assert!(validate_fft(2400, 2400, 2400).is_err());
        assert!(validate_fft(2400, 2399, 2400).is_err());
    }

    #[test]
    fn rejects_window_length_mismatch() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let Err(error) = Head::load(vb, vec![0.0; 100], 2400, 600, 2400) else {
            panic!("Fensterlänge 100 bei win_length 2400 wurde akzeptiert");
        };
        let message = error.to_string();
        assert!(
            message.contains("100") && message.contains("2400"),
            "Meldung ohne beide Längen: {message}"
        );
    }
}
