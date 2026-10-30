//! `silero`: WAV aus einem Chunk, zum Anhören und für Messungen.

use std::path::{Path, PathBuf};
use std::time::Instant;

use silero_v5_ru_candle::Silero;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut dir = PathBuf::from("models/silero-v5-ru");
    let mut voice = "baya".to_string();
    let mut out: Option<PathBuf> = None;
    let mut text = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => dir = PathBuf::from(args.next().ok_or("--model braucht einen Pfad")?),
            "--voice" => voice = args.next().ok_or("--voice braucht einen Namen")?,
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().ok_or("--out braucht einen Pfad")?,
                ))
            }
            other => text.push(other.to_string()),
        }
    }
    let text = text.join(" ");
    if text.trim().is_empty() {
        return Err("Text fehlt".into());
    }
    let start = Instant::now();
    let model = Silero::load(&dir)?;
    let sample_rate = model.sample_rate();
    let load = start.elapsed();
    let synth = Instant::now();
    let samples = model.synthesize(&text, &voice)?;
    let synth = synth.elapsed();
    let duration = samples.len() as f64 / sample_rate as f64;
    eprintln!(
        "{duration:.2}s Audio, synth {:.2}s (RTF {:.3}), load {:.2}s",
        synth.as_secs_f64(),
        synth.as_secs_f64() / duration,
        load.as_secs_f64()
    );
    if let Some(path) = out {
        write_wav(&path, &samples, sample_rate)?;
        eprintln!("{}", path.display());
    }
    Ok(())
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let mut data = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = u32::try_from(samples.len() * 2)
        .map_err(|_| "Audio zu groß für eine WAV-Datei".to_string())?;
    let riff_len = data_len
        .checked_add(36)
        .ok_or_else(|| "Audio zu groß für eine WAV-Datei".to_string())?;
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&riff_len.to_le_bytes());
    data.extend_from_slice(b"WAVEfmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&sample_rate.to_le_bytes());
    data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        data.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, data).map_err(|e| format!("{}: {e}", path.display()))
}
