//! `silero`: WAV aus beliebig langem Text (`synthesize_text`), zum Anhören und für Messungen.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use silero_v5_candle::{DeviceChoice, Error, Result, Silero, write_wav};

const DEFAULT_MODEL: &str = "models/silero-v5-ru";
const DEFAULT_VOICE: &str = "baya";
const USAGE: &str = "\
usage: silero --out DATEI [--model VERZEICHNIS] [--voice NAME] [--device GERÄT] [--] TEXT...

Synthetisiert TEXT zu einer WAV-Datei (16 Bit, 48 kHz, mono).

Optionen:
  --out DATEI           WAV-Datei, die geschrieben wird (Pflicht)
  --model VERZEICHNIS   Modellverzeichnis (Standard: models/silero-v5-ru)
  --voice NAME          Stimme, etwa baya (Standard: baya)
  --device GERÄT        auto, cpu oder metal (Standard: $SILERO_DEVICE, sonst auto)
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
    device: DeviceChoice,
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
    let model = Silero::load_with_choice(&request.dir, request.device)?;
    let sample_rate = model.sample_rate();
    let load = start.elapsed();
    if model.device().is_metal() && load.as_secs_f64() > 1.0 {
        eprintln!("Metal-Shader werden beim ersten Start einmalig kompiliert (kann ~10 s dauern)");
    }
    let synth = Instant::now();
    let samples = model.synthesize_text(&request.text, &request.voice)?;
    let synth = synth.elapsed();
    let duration = samples.len() as f64 / sample_rate as f64;
    let device = if model.device().is_metal() {
        "metal"
    } else {
        "cpu"
    };
    eprintln!(
        "{duration:.2}s Audio, synth {:.2}s (RTF {:.3}), load {:.2}s, Gerät {device}",
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
    let mut device = None;
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
            "--device" => device = Some(value(&mut args, "--device", "ein Gerät")?.parse()?),
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
    let device = match device {
        Some(device) => device,
        None => env_device()?,
    };
    Ok(Command::Synthesize(Request {
        dir,
        voice,
        out,
        text,
        device,
    }))
}

/// `SILERO_DEVICE` als Rückfall, wenn `--device` fehlt.
fn env_device() -> Result<DeviceChoice> {
    match std::env::var("SILERO_DEVICE") {
        Ok(value) => value.parse(),
        Err(_) => Ok(DeviceChoice::Auto),
    }
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str, what: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::Input(format!("{flag} braucht {what}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(args: &[&str]) -> Request {
        let args = args.iter().map(|arg| (*arg).to_owned());
        match parse_args(args).expect("Argumente werden gelesen") {
            Command::Synthesize(request) => request,
            Command::Help => panic!("Hilfe statt Synthese"),
        }
    }

    #[test]
    fn parses_device_choices() {
        assert_eq!(
            request(&["--out", "out.wav", "--device", "cpu", "Текст"]).device,
            DeviceChoice::Cpu
        );
        assert_eq!(
            request(&["--out", "out.wav", "--device", "metal", "Текст"]).device,
            DeviceChoice::Metal
        );
        assert_eq!(
            request(&["--out", "out.wav", "Текст"]).device,
            DeviceChoice::Auto
        );
    }

    #[test]
    fn rejects_unknown_device() {
        assert!(matches!(
            parse_args(
                ["--out", "out.wav", "--device", "tpu", "Текст"]
                    .map(str::to_owned)
                    .into_iter()
            ),
            Err(Error::Input(_))
        ));
    }
}
