//! `JitISTFTHead` plus inverse STFT mit `padding="same"`.

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use realfft::{ComplexToReal, RealFftPlanner};
use rustfft::num_complex::Complex32;

/// Der ISTFT-Frame schreibt `n_fft` Samples und multipliziert sie mit dem
/// Fenster der Länge `win_length`; beide müssen deshalb übereinstimmen.
fn validate_fft(n_fft: usize, hop: usize, win: usize) -> Result<()> {
    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        candle_core::bail!("n_fft muss gerade und > 0 sein: {n_fft}");
    }
    if win == 0 || hop == 0 || hop > win {
        candle_core::bail!("hop_length/win_length unbrauchbar: {hop}/{win}");
    }
    if n_fft != win {
        candle_core::bail!(
            "n_fft {n_fft} != win_length {win}: der ISTFT-Frame schreibt n_fft Samples"
        );
    }
    Ok(())
}

pub struct Head {
    out: Linear,
    window: Vec<f32>,
    n_fft: usize,
    hop: usize,
    win: usize,
    pad: usize,
}

impl Head {
    pub fn load(
        vb: VarBuilder,
        window: Vec<f32>,
        n_fft: usize,
        hop: usize,
        win: usize,
    ) -> Result<Self> {
        validate_fft(n_fft, hop, win)?;
        let bins = n_fft / 2 + 1;
        let out = candle_nn::linear(512, 2 * bins, vb.pp("out"))?;
        Ok(Self {
            out,
            window,
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
        let mag = mag.contiguous()?.to_vec3::<f32>()?;
        let cos = cos.contiguous()?.to_vec3::<f32>()?;
        let sin = sin.contiguous()?.to_vec3::<f32>()?;
        let mag = &mag[0];
        let cos = &cos[0];
        let sin = &sin[0];

        let mut planner = RealFftPlanner::<f32>::new();
        let irfft: Arc<dyn ComplexToReal<f32>> = planner.plan_fft_inverse(self.n_fft);
        let mut input = irfft.make_input_vec();
        let mut output = irfft.make_output_vec();

        let out_len = (frames - 1) * self.hop + self.win;
        let mut audio = vec![0.0f32; out_len];
        let scale = 1.0 / self.n_fft as f32;
        for frame in 0..frames {
            for bin in 0..bins {
                let real = mag[bin][frame] * cos[bin][frame];
                let imag = mag[bin][frame] * sin[bin][frame];
                input[bin] = Complex32::new(real, imag);
            }
            // `irfft` ignoriert DC- und Nyquist-Imaginärteil; explizit nullen
            // macht die Absicht sichtbar.
            input[0].im = 0.0;
            input[bins - 1].im = 0.0;
            irfft
                .process(&mut input, &mut output)
                .map_err(|error| candle_core::Error::Msg(format!("irfft: {error}")))?;
            let offset = frame * self.hop;
            for (index, value) in output.iter().enumerate() {
                audio[offset + index] += value * self.window[index] * scale;
            }
        }

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

#[cfg(test)]
mod tests {
    use super::validate_fft;

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
}
