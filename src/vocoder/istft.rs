//! `JitISTFTHead` plus inverse STFT mit `padding="same"`.

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use realfft::{ComplexToReal, RealFftPlanner};
use rustfft::num_complex::Complex32;

use crate::Error;

/// Der ISTFT-Frame schreibt `n_fft` Samples und multipliziert sie mit dem
/// Fenster der Länge `win_length`; beide müssen deshalb übereinstimmen.
///
/// `pad = (win - hop) / 2` wird später als `y[pad..len - pad]` abgeschnitten.
/// Bei `pad == 0` wäre der Schnitt leer, in Python `y[:, 0:-0]` dagegen
/// ergäbe der Schnitt bei Rust alles. Deshalb ist `pad == 0` ein Fehler.
fn validate_fft(n_fft: usize, hop: usize, win: usize) -> crate::Result<()> {
    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        return Err(Error::Model(format!(
            "n_fft muss gerade und > 0 sein: {n_fft}"
        )));
    }
    if win == 0 || hop == 0 || hop > win {
        return Err(Error::Model(format!(
            "hop_length/win_length unbrauchbar: {hop}/{win}"
        )));
    }
    if n_fft != win {
        return Err(Error::Model(format!(
            "n_fft {n_fft} != win_length {win}: der ISTFT-Frame schreibt n_fft Samples"
        )));
    }
    if (win - hop) / 2 == 0 {
        return Err(Error::Model(format!(
            "pad (win - hop) / 2 ist 0 bei hop_length {hop} / win_length {win}"
        )));
    }
    Ok(())
}

pub struct Head {
    out: Linear,
    window: Vec<f32>,
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
    pub fn load(
        vb: VarBuilder,
        window: Vec<f32>,
        n_fft: usize,
        hop: usize,
        win: usize,
    ) -> crate::Result<Self> {
        validate_fft(n_fft, hop, win)?;
        if window.len() != win {
            return Err(Error::Model(format!(
                "ISTFT-Fenster hat {} statt win_length {win} Werte",
                window.len()
            )));
        }
        let bins = n_fft / 2 + 1;
        let out = candle_nn::linear(512, 2 * bins, vb.pp("out"))
            .map_err(|error| Error::Model(format!("out: {error}")))?;
        let irfft = RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft);
        Ok(Self {
            out,
            window,
            irfft,
            n_fft,
            hop,
            win,
            pad: (win - hop) / 2,
        })
    }

    /// Features `[1, T, 512]` -> `(mag, cos, sin)`, jeweils `[1, 1201, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.out.forward(x)?.transpose(1, 2)?;
        let bins = self.n_fft / 2 + 1;
        let mag = x.narrow(1, 0, bins)?;
        let phase = x.narrow(1, bins, bins)?;
        let mag = mag.exp()?.clamp(f32::MIN, 100.0)?;
        let cos = phase.cos()?;
        let sin = phase.sin()?;
        Ok((mag, cos, sin))
    }

    /// `ISTFT(mag * (cos + i sin))`, überlappt und durch die Fensterhülle
    /// geteilt.
    pub fn istft(&self, mag: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Vec<f32>> {
        let (_, bins, frames) = mag.dims3()?;
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
        let mag = frame_major(mag)?;
        let cos = frame_major(cos)?;
        let sin = frame_major(sin)?;

        let mut input = self.irfft.make_input_vec();
        let mut output = self.irfft.make_output_vec();
        let mut scratch = self.irfft.make_scratch_vec();

        let out_len = (frames - 1) * self.hop + self.win;
        let mut audio = vec![0.0f32; out_len];
        let scale = 1.0 / self.n_fft as f32;
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
                .zip(&self.window)
            {
                *acc += value * weight * scale;
            }
        }

        // Die Hülle hängt nur von der Frame-Zahl ab, nicht von den Bins: einmal pro Aufruf.
        let mut envelope = vec![0.0f32; out_len];
        for frame in 0..frames {
            let offset = frame * self.hop;
            for (index, value) in self.window.iter().enumerate() {
                envelope[offset + index] += value * value;
            }
        }
        let start = self.pad;
        let end = out_len - self.pad;
        let mut samples = Vec::with_capacity(end - start);
        for index in start..end {
            let divisor = envelope[index];
            if divisor <= 1e-11 {
                return Err(candle_core::Error::Msg(format!(
                    "ISTFT-Fensterhülle bei Sample {index} ist null"
                )));
            }
            samples.push(audio[index] / divisor);
        }
        Ok(samples)
    }
}

/// Liest `[1, bins, frames]` als flachen Vektor in Reihenfolge `frame * bins + bin`.
fn frame_major(spectrum: &Tensor) -> Result<Vec<f32>> {
    spectrum
        .transpose(1, 2)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()
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
