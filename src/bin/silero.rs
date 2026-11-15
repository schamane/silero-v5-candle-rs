//! `silero`: WAV aus beliebig langem Text (`synthesize_text`), zum Anhören und für Messungen.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use silero_v5_candle::{Error, Result, Silero, write_wav};

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
