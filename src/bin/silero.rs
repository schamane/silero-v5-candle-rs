//! `silero`: WAV aus beliebig langem Text (`synthesize_text`), zum Anhören und für Messungen.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::Device;
use silero_v5_candle::{Error, Result, Silero, default_device, write_wav};

const DEFAULT_MODEL: &str = "models/silero-v5-ru";
const DEFAULT_VOICE: &str = "baya";
const USAGE: &str = "\
usage: silero --out DATEI [--model VERZEICHNIS] [--voice NAME] [--device GERÄT] [--] TEXT...

Synthetisiert TEXT zu einer WAV-Datei (16 Bit, 48 kHz, mono).

Optionen:
  --out DATEI           WAV-Datei, die geschrieben wird (Pflicht)
  --model VERZEICHNIS   Modellverzeichnis (Standard: models/silero-v5-ru)
  --voice NAME          Stimme, etwa baya (Standard: baya)
  --device GERÄT        auto, cpu oder metal (Standard: auto)
  --help                diese Hilfe anzeigen
  --                    beendet die Optionen; danach darf TEXT mit -- beginnen
";

/// Gerätewahl der CLI. `Auto` entspricht [`Silero::load`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DeviceChoice {
    #[default]
    Auto,
    Cpu,
    Metal,
}

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
    let model = match request.device {
        DeviceChoice::Auto => Silero::load(&request.dir)?,
        DeviceChoice::Cpu => Silero::load_with_device(&request.dir, &Device::Cpu)?,
        DeviceChoice::Metal => {
            let device = default_device();
            if !device.is_metal() {
                return Err(Error::Input(
                    "Metal ist auf diesem Gerät nicht verfügbar".into(),
                ));
            }
            Silero::load_with_device(&request.dir, &device)?
        }
    };
    let sample_rate = model.sample_rate();
    let load = start.elapsed();
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
    let mut device = DeviceChoice::Auto;
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
            "--device" => device = parse_device(value(&mut args, "--device", "ein Gerät")?)?,
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
        device,
    }))
}

fn parse_device(value: String) -> Result<DeviceChoice> {
    match value.as_str() {
        "auto" => Ok(DeviceChoice::Auto),
        "cpu" => Ok(DeviceChoice::Cpu),
        "metal" => Ok(DeviceChoice::Metal),
        other => Err(Error::Input(format!(
            "unbekanntes Gerät {other}; erwartet auto, cpu oder metal"
        ))),
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
