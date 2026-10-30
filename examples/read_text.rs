//! Liest eine Textdatei mit dem passenden Silero-v5-Modell vor.
//!
//! ```sh
//! cargo run --release --example read_text -- \
//!     --file examples/lina.txt --lang uk --out lina.wav
//! ```
//!
//! `--lang ru` nutzt `models/silero-v5-ru` (Akzentor setzt die Betonung selbst),
//! `--lang uk` nutzt `models/silero-v5-cis-base-nostress` (ukrainischer Akzentor
//! setzt die Betonung). Ohne `--model` wechselt das Beispiel automatisch zu
//! `v5_cis_ext`, wenn die gewählte Stimme nur dort existiert (etwa `ukr_mykyta`).
//! Bringt das Ext-Modell den Akzentor mit, wird er für `ukr_*`-Stimmen ebenfalls
//! automatisch angewendet; sonst muss der Text `+`-Marken enthalten.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use silero_v5_candle::{Error, Result, Silero};

/// Modell mit den `v5_cis_ext`-Stimmen; wird bei Bedarf automatisch geladen.
const EXT_MODEL: &str = "models/silero-v5-cis-ext";

const USAGE: &str = "\
usage: read_text --file DATEI [--lang ru|uk] [--voice NAME] [--out DATEI] [--model VERZEICHNIS] [--no-play]

Liest DATEI (UTF-8) und schreibt/ spielt eine 48-kHz-Mono-WAV-Datei.

Optionen:
  --file, -f DATEI      Textdatei (Pflicht)
  --lang, -l SPRACHE    ru oder uk (Standard: uk)
  --voice NAME          Sprecher; Standard: baya (ru), ukr_igor (uk).
                        Ext-Stimmen wie ukr_mykyta laden v5_cis_ext automatisch
  --out, -o DATEI       WAV-Datei (Standard: <Eingabe>.wav)
  --model VERZEICHNIS   Modellverzeichnis; Standard: models/silero-v5-ru bzw.
                        models/silero-v5-cis-base-nostress
  --no-play             nur schreiben, nicht abspielen
  --help, -h            diese Hilfe
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Ru,
    Uk,
}

impl Language {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "ru" | "russian" | "russisch" => Ok(Self::Ru),
            "uk" | "ua" | "ukrainian" | "ukrainisch" => Ok(Self::Uk),
            other => Err(Error::Input(format!(
                "unbekannte Sprache {other:?}; erwartet ru oder uk"
            ))),
        }
    }

    fn model_dir(self) -> &'static str {
        match self {
            Self::Ru => "models/silero-v5-ru",
            Self::Uk => "models/silero-v5-cis-base-nostress",
        }
    }

    fn default_voice(self) -> &'static str {
        match self {
            Self::Ru => "baya",
            Self::Uk => "ukr_igor",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ru => "ru",
            Self::Uk => "uk",
        }
    }
}

#[derive(Debug)]
struct Request {
    file: PathBuf,
    language: Language,
    voice: String,
    model: PathBuf,
    /// `true`, wenn `--model` gesetzt wurde; dann kein automatischer Wechsel.
    model_explicit: bool,
    out: PathBuf,
    play: bool,
}

#[derive(Debug)]
enum CommandLine {
    Help,
    Read(Request),
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fehler: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Lädt das Modell und wechselt bei einer `v5_cis_ext`-Stimme automatisch dorthin,
/// sofern `--model` nicht gesetzt wurde.
fn load_model(request: &Request) -> Result<(Silero, PathBuf)> {
    let mut path = request.model.clone();
    let mut model = Silero::load(&path)?;
    if !request.model_explicit
        && request.language == Language::Uk
        && request.voice.starts_with("ukr_")
        && !has_voice(&model, &request.voice)
    {
        let ext_path = PathBuf::from(EXT_MODEL);
        let ext = Silero::load(&ext_path).map_err(|error| {
            Error::Input(format!(
                "unbekannte Stimme {:?}; {} ist nicht verfügbar: {error}",
                request.voice,
                ext_path.display()
            ))
        })?;
        if has_voice(&ext, &request.voice) {
            model = ext;
            path = ext_path;
        }
    }
    if !has_voice(&model, &request.voice) {
        let mut available: Vec<&str> = model
            .voices()
            .into_iter()
            .filter(|name| request.language != Language::Uk || name.starts_with("ukr_"))
            .collect();
        if available.is_empty() {
            available = model.voices();
        }
        let hint = if request.language == Language::Uk {
            format!("; ohne --model wird {EXT_MODEL} automatisch geladen")
        } else {
            String::new()
        };
        return Err(Error::Input(format!(
            "unbekannte Stimme {:?}; verfügbar: {}{hint}",
            request.voice,
            available.join(", ")
        )));
    }
    Ok((model, path))
}

fn has_voice(model: &Silero, voice: &str) -> bool {
    model.voices().contains(&voice)
}

fn run() -> Result<()> {
    let command = parse_args(std::env::args().skip(1))?;
    let request = match command {
        CommandLine::Help => {
            print!("{USAGE}");
            return Ok(());
        }
        CommandLine::Read(request) => request,
    };

    let text = std::fs::read_to_string(&request.file).map_err(|error| Error::Io {
        path: Some(request.file.clone()),
        source: error,
    })?;
    if text.trim().is_empty() {
        return Err(Error::Input(format!("{} ist leer", request.file.display())));
    }

    let (model, model_path) = load_model(&request)?;
    if !model.has_accentor_for(&request.voice) && !text.contains('+') {
        eprintln!(
            "hinweis: {} hat keinen Akzentor; '+vor dem betonten Vokal' oder\n\
             'uv run scripts/stress_uk.py' verbessert die Aussprache",
            model_path.display()
        );
    }

    let started = std::time::Instant::now();
    let samples = model.synthesize_text(&text, &request.voice)?;
    let seconds = samples.len() as f64 / f64::from(model.sample_rate());
    write_wav(&request.out, &samples, model.sample_rate())?;
    eprintln!(
        "{}: {seconds:.2} s Audio in {:.2} s (Sprache {}, Stimme {})",
        request.out.display(),
        started.elapsed().as_secs_f64(),
        request.language.name(),
        request.voice
    );

    if request.play {
        play(&request.out);
    }
    Ok(())
}

fn parse_args<I>(args: I) -> Result<CommandLine>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut file = None;
    let mut language = Language::Uk;
    let mut voice = None;
    let mut model = None;
    let mut out = None;
    let mut play = true;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(CommandLine::Help),
            "--file" | "-f" => file = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--lang" | "-l" => language = Language::parse(&value(&mut args, &arg)?)?,
            "--voice" => voice = Some(value(&mut args, &arg)?),
            "--model" => model = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--out" | "-o" => out = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--no-play" => play = false,
            other => {
                return Err(Error::Input(format!(
                    "unbekannte Option {other:?}; --help zeigt die Hilfe"
                )));
            }
        }
    }

    let file = file.ok_or_else(|| Error::Input("--file fehlt; --help zeigt die Hilfe".into()))?;
    let out = out.unwrap_or_else(|| file.with_extension("wav"));
    let model_explicit = model.is_some();
    Ok(CommandLine::Read(Request {
        file,
        language,
        voice: voice.unwrap_or_else(|| language.default_voice().to_owned()),
        model: model.unwrap_or_else(|| PathBuf::from(language.model_dir())),
        model_explicit,
        out,
        play,
    }))
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::Input(format!("{flag} braucht einen Wert")))
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(Error::from)?;
    }
    let file = std::fs::File::create(path).map_err(|error| Error::Io {
        path: Some(path.to_path_buf()),
        source: error,
    })?;
    let mut writer = io::BufWriter::new(file);
    let data_len = (samples.len() as u32) * 2;
    let byte_rate = sample_rate * 2;
    write(&mut writer, b"RIFF")?;
    write(&mut writer, &(36 + data_len).to_le_bytes())?;
    write(&mut writer, b"WAVEfmt ")?;
    write(&mut writer, &16u32.to_le_bytes())?;
    write(&mut writer, &1u16.to_le_bytes())?; // PCM
    write(&mut writer, &1u16.to_le_bytes())?; // mono
    write(&mut writer, &sample_rate.to_le_bytes())?;
    write(&mut writer, &byte_rate.to_le_bytes())?;
    write(&mut writer, &2u16.to_le_bytes())?; // block align
    write(&mut writer, &16u16.to_le_bytes())?; // bits per sample
    write(&mut writer, b"data")?;
    write(&mut writer, &data_len.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer
            .write_all(&value.to_le_bytes())
            .map_err(Error::from)?;
    }
    writer.flush().map_err(Error::from)?;
    Ok(())
}

fn write(writer: &mut impl io::Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes).map_err(Error::from)
}

fn play(path: &Path) {
    let players: &[(&str, &[&str])] = &[
        ("afplay", &[]),
        ("aplay", &["-q"]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "error"]),
    ];
    for (player, flags) in players {
        let status = Command::new(player).args(*flags).arg(path).status();
        if let Ok(status) = status {
            if !status.success() {
                eprintln!("{player} endete mit {status}");
            }
            return;
        }
    }
    eprintln!("kein Audioplayer gefunden (afplay/aplay/ffplay)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ukrainian_defaults() {
        let CommandLine::Read(request) =
            parse_args(["--file".to_owned(), "in.txt".to_owned()]).expect("Argumente")
        else {
            panic!("Hilfe statt Anfrage");
        };
        assert_eq!(request.language, Language::Uk);
        assert_eq!(request.voice, "ukr_igor");
        assert_eq!(
            request.model,
            PathBuf::from("models/silero-v5-cis-base-nostress")
        );
        assert!(!request.model_explicit);
        assert_eq!(request.out, PathBuf::from("in.wav"));
    }

    #[test]
    fn explicit_model_disables_fallback() {
        let CommandLine::Read(request) = parse_args([
            "--file".to_owned(),
            "in.txt".to_owned(),
            "--model".to_owned(),
            EXT_MODEL.to_owned(),
        ])
        .expect("Argumente") else {
            panic!("Hilfe statt Anfrage");
        };
        assert!(request.model_explicit);
        assert_eq!(request.model, PathBuf::from(EXT_MODEL));
    }

    #[test]
    fn rejects_unknown_language() {
        let error = parse_args([
            "--file".to_owned(),
            "in.txt".to_owned(),
            "--lang".to_owned(),
            "de".to_owned(),
        ])
        .expect_err("de ist kein Modell");
        assert!(error.to_string().contains("de"), "{error}");
    }
}
