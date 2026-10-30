//! `silero`: WAV aus einem Chunk, zum Anhören und für Messungen.

use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use silero_v5_candle::{Error, Result, Silero};

const DEFAULT_MODEL: &str = "models/silero-v5-ru";
const DEFAULT_VOICE: &str = "baya";
const USAGE: &str = "\
usage: silero --out DATEI [--model VERZEICHNIS] [--voice NAME] [--] TEXT...

Synthetisiert TEXT zu einer WAV-Datei (16 Bit, 48 kHz, mono).

Optionen:
  --out DATEI           WAV-Datei, die geschrieben wird (Pflicht)
  --model VERZEICHNIS   Modellverzeichnis (Standard: models/silero-v5-ru)
  --voice NAME          Stimme, etwa baya (Standard: baya)
  --help                diese Hilfe anzeigen
  --                    beendet die Optionen; danach darf TEXT mit -- beginnen
";

/// Was der Aufruf tun soll.
enum Command {
    Help,
    Synthesize(Request),
}

struct Request {
    dir: PathBuf,
    voice: String,
    out: PathBuf,
    text: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            let mut cause = std::error::Error::source(&error);
            while let Some(source) = cause {
                eprintln!("  caused by: {source}");
                cause = source.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let request = match parse_args(std::env::args().skip(1))? {
        Command::Help => {
            print!("{USAGE}");
            return Ok(());
        }
        Command::Synthesize(request) => request,
    };
    let start = Instant::now();
    let model = Silero::load(&request.dir)?;
    let sample_rate = model.sample_rate();
    let load = start.elapsed();
    let synth = Instant::now();
    let samples = model.synthesize_text(&request.text, &request.voice)?;
    let synth = synth.elapsed();
    let duration = samples.len() as f64 / sample_rate as f64;
    eprintln!(
        "{duration:.2}s Audio, synth {:.2}s (RTF {:.3}), load {:.2}s",
        synth.as_secs_f64(),
        synth.as_secs_f64() / duration,
        load.as_secs_f64()
    );
    write_wav(&request.out, &samples, sample_rate)?;
    eprintln!("{}", request.out.display());
    Ok(())
}

/// Liest die Argumente. `--` beendet die Optionen, danach ist alles Text.
fn parse_args<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut dir = PathBuf::from(DEFAULT_MODEL);
    let mut voice = DEFAULT_VOICE.to_string();
    let mut out = None;
    let mut words = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => return Ok(Command::Help),
            "--" => {
                words.extend(args.by_ref());
                break;
            }
            "--model" => dir = PathBuf::from(value(&mut args, "--model", "einen Pfad")?),
            "--voice" => voice = value(&mut args, "--voice", "einen Namen")?,
            "--out" => out = Some(PathBuf::from(value(&mut args, "--out", "einen Pfad")?)),
            flag if flag.starts_with("--") => {
                return Err(Error::Input(format!("unbekannte Option {flag}")));
            }
            other => words.push(other.to_string()),
        }
    }
    let out = out.ok_or_else(|| Error::Input("--out fehlt".into()))?;
    let text = words.join(" ");
    if text.trim().is_empty() {
        return Err(Error::Input("text fehlt".into()));
    }
    Ok(Command::Synthesize(Request {
        dir,
        voice,
        out,
        text,
    }))
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str, what: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::Input(format!("{flag} braucht {what}")))
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let io_error = |source: io::Error| Error::Io {
        path: Some(path.to_path_buf()),
        source,
    };
    let too_large = || io_error(io::Error::other("audio zu groß für eine WAV-Datei"));
    let mut data = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = u32::try_from(samples.len() * 2).map_err(|_| too_large())?;
    let riff_len = data_len.checked_add(36).ok_or_else(too_large)?;
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
    std::fs::write(path, data).map_err(io_error)
}
