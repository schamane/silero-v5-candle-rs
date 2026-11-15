//! `JitISTFTHead` plus inverse STFT mit `padding="same"`.

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use rayon::prelude::*;
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

/// Frames je Rayon-Block der parallelen ISTFT; groß genug, dass IFFT-Arbeit den
/// Task-Overhead überwiegt, klein genug für gute Lastverteilung.
const FRAME_BLOCK: usize = 32;

/// Obergrenze der Magnitude (`min(exp(x), 100)`).
const MAG_MAX: f32 = 100.0;

/// Spektrum in Frame-Major-Layout: jeweils `[1, T, 1201]`, zusammenhängend im Speicher.
pub(crate) struct Spectrum {
    pub mag: Tensor,
    pub cos: Tensor,
    pub sin: Tensor,
}

/// Auf den Host zurückgelesenes Spektrum: je Frame `mag`, `cos`, `sin` hintereinander.
pub(crate) struct SpectrumData {
    data: Vec<f32>,
    frames: usize,
}

pub(crate) struct Head {
    out: Linear,
    /// Nur auf Metal: Gewicht `[512 + 1, 2 * bins]` mit dem Bias als letzter Zeile.
    /// Candles Broadcast-Kernel ist auf Metal rund fünfmal langsamer als ein
    /// zusammenhängender; der Bias läuft deshalb als Teil des GEMM (Ones-Spalte).
    out_biased: Option<Tensor>,
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
        let out_biased = if vb.device().is_metal() {
            let bias = out
                .bias()
                .ok_or_else(|| Error::model_msg("out: Bias fehlt"))?
                .reshape((1, 2 * bins))
                .map_err(|error| Error::model("out", error))?;
            let weight = out
                .weight()
                .t()
                .and_then(|weight| weight.contiguous())
                .and_then(|weight| Tensor::cat(&[&weight, &bias], 0))
                .map_err(|error| Error::model("out", error))?;
            Some(weight)
        } else {
            None
        };
        let irfft = RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft);
        let scale = 1.0 / n_fft as f32;
        let window_scaled = window.iter().map(|value| value * scale).collect();
        let window_squared = window.iter().map(|value| value * value).collect();
        Ok(Self {
            out,
            out_biased,
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
        let bins = self.n_fft / 2 + 1;
        let (x, clamp) = match &self.out_biased {
            Some(weight) => {
                let (batch, frames, channels) = x.dims3()?;
                let ones = Tensor::ones((batch * frames, 1), x.dtype(), x.device())?;
                let x = Tensor::cat(&[&x.reshape((batch * frames, channels))?, &ones], 1)?
                    .matmul(weight)?
                    .reshape((batch, frames, 2 * bins))?;
                // Die Begrenzung der Magnitude übernimmt die ISTFT-Schleife auf der CPU.
                (x, false)
            }
            None => (self.out.forward(x)?, true),
        };
        let mag = x.narrow(2, 0, bins)?.contiguous()?;
        let phase = x.narrow(2, bins, bins)?.contiguous()?;
        let mag = if clamp {
            mag.exp()?.minimum(MAG_MAX)?
        } else {
            mag.exp()?
        };
        let cos = phase.cos()?;
        let sin = phase.sin()?;
        Ok(Spectrum { mag, cos, sin })
    }

    /// Liest das Spektrum vom Gerät zurück. Das ist der letzte Schritt, der das
    /// Gerät berührt; alles danach ist reine CPU-Arbeit.
    pub(crate) fn readback(&self, spectrum: &Spectrum) -> Result<SpectrumData> {
        let (_, frames, bins) = spectrum.mag.dims3()?;
        let expected_bins = self.n_fft / 2 + 1;
        if bins != expected_bins {
            return Err(candle_core::Error::Msg(format!(
                "ISTFT: {bins} Bins statt {expected_bins}"
            )));
        }
        if frames == 0 {
            return Ok(SpectrumData {
                data: Vec::new(),
                frames,
            });
        }
        // Ein gemeinsamer Readback statt drei: `to_vec` committet und wartet auf den
        // eigenen Command-Buffer; auf Metal kostet jeder Wait ca. 1-2 ms. Die drei
        // Tensoren sind frame-major, deshalb liegen nach dem Cat die Kanäle
        // `mag`, `cos`, `sin` je Frame hintereinander.
        let joined = Tensor::cat(&[&spectrum.mag, &spectrum.cos, &spectrum.sin], 2)?;
        let (_, joined_frames, joined_bins) = joined.dims3()?;
        let stride = 3 * bins;
        if joined_frames != frames || joined_bins != stride {
            return Err(candle_core::Error::Msg(format!(
                "ISTFT: gepacktes Spektrum {:?} statt [{frames}, {stride}]",
                joined.dims()
            )));
        }
        let data = flat(&joined)?;
        Ok(SpectrumData { data, frames })
    }

    /// `ISTFT(mag * (cos + i sin))`: überlappt die zurückgelesenen Frames und teilt
    /// durch die Fensterhülle; läuft nur auf der CPU.
    pub(crate) fn overlap_add(&self, spectrum: &SpectrumData) -> Result<Vec<f32>> {
        let SpectrumData { data, frames } = spectrum;
        let frames = *frames;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let out_len = (frames - 1) * self.hop + self.win;
        let mut audio = vec![0.0f32; out_len];

        // Frames blockweise parallel in einen Zwischenpuffer, die Überlappung danach
        // sequentiell in Frame-Reihenfolge: je Sample dieselbe Additionsreihenfolge wie
        // ein rein sequentieller Lauf, das Ergebnis ist also bitgleich. Rayon nutzt den
        // Pool des aufrufenden Threads (Chunk-Pool oder globaler Pool), es entsteht
        // kein eigener Pool.
        let win = self.win;
        let mut frame_buf = vec![0.0f32; frames * win];
        frame_buf
            .par_chunks_mut(FRAME_BLOCK * win)
            .enumerate()
            .try_for_each(|(block, chunk)| -> Result<()> {
                let mut input = self.irfft.make_input_vec();
                let mut output = self.irfft.make_output_vec();
                let mut scratch = self.irfft.make_scratch_vec();
                for (index, dst) in chunk.chunks_mut(win).enumerate() {
                    self.frame_into(
                        data,
                        block * FRAME_BLOCK + index,
                        &mut input,
                        &mut output,
                        &mut scratch,
                    )?;
                    for ((slot, value), weight) in
                        dst.iter_mut().zip(&output).zip(&self.window_scaled)
                    {
                        *slot = value * weight;
                    }
                }
                Ok(())
            })?;
        for (frame, src) in frame_buf.chunks(win).enumerate() {
            let offset = frame * self.hop;
            for (acc, value) in audio[offset..offset + win].iter_mut().zip(src) {
                *acc += value;
            }
        }
        self.finish(audio, frames)
    }

    /// Baut das komplexe Halbspektrum eines Frames und führt die inverse FFT aus;
    /// das Ergebnis steht danach in `output` (ohne Fenster und Skalierung).
    ///
    /// Die Magnitude wird hier auf 100 begrenzt. Auf der CPU ist das ein No-op (der
    /// Tensor ist schon begrenzt), auf Metal ersetzt es das `minimum` im Graphen.
    fn frame_into(
        &self,
        data: &[f32],
        frame: usize,
        input: &mut [Complex32],
        output: &mut [f32],
        scratch: &mut [Complex32],
    ) -> Result<()> {
        let bins = self.n_fft / 2 + 1;
        let base = frame * 3 * bins;
        for bin in 0..bins {
            let magnitude = data[base + bin].min(MAG_MAX);
            input[bin] = Complex32::new(
                magnitude * data[base + bins + bin],
                magnitude * data[base + 2 * bins + bin],
            );
        }
        // `irfft` ignoriert DC- und Nyquist-Imaginärteil; explizit nullen
        // macht die Absicht sichtbar.
        input[0].im = 0.0;
        input[bins - 1].im = 0.0;
        self.irfft
            .process_with_scratch(input, output, scratch)
            .map_err(|error| candle_core::Error::Msg(format!("irfft: {error}")))
    }

    /// Hüllen-Division und Randschnitt nach der Überlappung.
    fn finish(&self, audio: Vec<f32>, frames: usize) -> Result<Vec<f32>> {
        let out_len = audio.len();
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

/// Liest `[1, frames, 3 * bins]` als flachen Vektor in Reihenfolge
/// `frame * 3 * bins + channel * bins + bin`.
fn flat(spectrum: &Tensor) -> Result<Vec<f32>> {
    spectrum.flatten_all()?.to_vec1::<f32>()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use realfft::num_complex::Complex32;

    use super::{Head, SpectrumData, validate_fft};

    /// Der frühere rein sequentielle Algorithmus als Referenz für die parallele ISTFT.
    fn overlap_add_sequential(head: &Head, spectrum: &SpectrumData) -> Vec<f32> {
        let SpectrumData { data, frames } = spectrum;
        let bins = head.n_fft / 2 + 1;
        let stride = 3 * bins;
        let mut input = head.irfft.make_input_vec();
        let mut output = head.irfft.make_output_vec();
        let mut scratch = head.irfft.make_scratch_vec();
        let out_len = (frames - 1) * head.hop + head.win;
        let mut audio = vec![0.0f32; out_len];
        for frame in 0..*frames {
            let base = frame * stride;
            for bin in 0..bins {
                let magnitude = data[base + bin];
                input[bin] = Complex32::new(
                    magnitude * data[base + bins + bin],
                    magnitude * data[base + 2 * bins + bin],
                );
            }
            input[0].im = 0.0;
            input[bins - 1].im = 0.0;
            head.irfft
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .expect("irfft");
            let offset = frame * head.hop;
            for ((acc, value), weight) in audio[offset..offset + head.win]
                .iter_mut()
                .zip(&output)
                .zip(&head.window_scaled)
            {
                *acc += value * weight;
            }
        }
        let mut envelope = vec![0.0f32; out_len];
        for frame in 0..*frames {
            let offset = frame * head.hop;
            for (slot, value) in envelope[offset..offset + head.win]
                .iter_mut()
                .zip(&head.window_squared)
            {
                *slot += value;
            }
        }
        audio[head.pad..out_len - head.pad]
            .iter()
            .zip(&envelope[head.pad..out_len - head.pad])
            .map(|(value, divisor)| value / divisor)
            .collect()
    }

    #[test]
    fn parallel_overlap_add_is_bit_identical_to_sequential() {
        let (n_fft, hop) = (2400usize, 600usize);
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n_fft as f32).cos())
            .collect();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let head = Head::load(vb, window, n_fft, hop, n_fft).expect("Head");
        let bins = n_fft / 2 + 1;
        // 77 Frames: mehrere volle Blöcke plus ein unvollständiger Rest.
        for frames in [1usize, 5, 32, 33, 77] {
            let mut state = 0x9E37_79B9u32;
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32
            };
            let mut data = Vec::with_capacity(frames * 3 * bins);
            for _ in 0..frames {
                data.extend((0..bins).map(|_| next() * 4.0));
                data.extend((0..bins).map(|_| next() * 2.0 - 1.0));
                data.extend((0..bins).map(|_| next() * 2.0 - 1.0));
            }
            let spectrum = SpectrumData { data, frames };
            let want = overlap_add_sequential(&head, &spectrum);
            let got = head.overlap_add(&spectrum).expect("overlap_add");
            assert_eq!(got.len(), want.len(), "Länge bei {frames} Frames");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "nicht bitgleich bei {frames} Frames"
            );
        }
    }

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
