//! Messprogramm für die Dokumentation in `docs/messungen.html`.
//!
//! Aufruf: `cargo bench --bench sbench -- run <cpu|metal> short,long,corpus[,chunks]`
//! (weitere Modi: `load`, `rss`, `dump`, `cold`). `RUNS` setzt die Zahl der gemessenen
//! Läufe, `MODEL` den Modellordner. Ohne Argumente tut das Programm nichts, damit
//! `cargo test --all-targets` es gefahrlos mitbaut und startet.

use silero_v5_candle::{DeviceChoice, Silero};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn Error>>;

const SHORT: &str = "Привет, как у тебя дела сегодня?";
const VOICE: &str = "baya";
const USAGE: &str = "usage: sbench run <cpu|metal> <short,long,corpus[,chunks]> | load <dev> | rss <dev> | cold <dev> | dump <dev> <kind> <file>";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn corpus() -> Res<Vec<String>> {
    let raw = std::fs::read_to_string(root().join("tests/corpus.txt"))?;
    Ok(raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect())
}

fn long_text() -> Res<String> {
    let mut long = String::new();
    for l in corpus()? {
        let cand = format!("{} {},", long, l.trim_end_matches(['.', '!', '?']));
        if cand.chars().count() > 310 {
            break;
        }
        long = cand;
    }
    Ok(long.trim().to_string())
}

fn text(kind: &str) -> Res<String> {
    Ok(match kind {
        "short" => SHORT.to_string(),
        "long" => long_text()?,
        _ => corpus()?.into_iter().take(20).collect::<Vec<_>>().join(" "),
    })
}

fn model(dev: &str) -> Res<Silero> {
    let dir =
        std::env::var_os("MODEL").map_or_else(|| root().join("models/silero-v5-ru"), PathBuf::from);
    Ok(Silero::load_with_choice(
        &dir,
        dev.parse::<DeviceChoice>()?,
    )?)
}

fn synth(m: &Silero, kind: &str, t: &str) -> Res<Vec<f32>> {
    Ok(match kind {
        "chunks" => {
            let mut all = Vec::new();
            for chunk in m.synthesize_chunks(t, VOICE) {
                all.extend(chunk?);
            }
            all
        }
        "corpus" => m.synthesize_text(t, VOICE)?,
        _ => m.synthesize(t, VOICE)?,
    })
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn run() -> Res<()> {
    let a: Vec<String> = std::env::args().collect();
    // `cargo test --all-targets` startet das Programm ohne Modus.
    let (Some(mode), Some(dev)) = (a.get(1), a.get(2)) else {
        return Ok(());
    };
    let n: usize = std::env::var("RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);
    match mode.as_str() {
        "run" => {
            let m = model(dev)?;
            for kind in a.get(3).ok_or(USAGE)?.split(',') {
                let t = text(if kind == "chunks" { "corpus" } else { kind })?;
                synth(&m, kind, &t)?;
                synth(&m, kind, &t)?;
                let mut v = vec![];
                let mut len = 0;
                for _ in 0..n {
                    let s = Instant::now();
                    len = std::hint::black_box(synth(&m, kind, &t)?).len();
                    v.push(ms(s));
                }
                println!(
                    "{{\"dev\":\"{dev}\",\"kind\":\"{kind}\",\"samples\":{len},\"runs_ms\":{v:?}}}"
                );
            }
        }
        "load" => {
            drop(model(dev)?);
            let mut v = vec![];
            for _ in 0..n.max(9) {
                let s = Instant::now();
                std::hint::black_box(model(dev)?);
                v.push(ms(s));
            }
            println!("{{\"dev\":\"{dev}\",\"kind\":\"load\",\"runs_ms\":{v:?}}}");
        }
        "rss" => {
            let m = model(dev)?;
            std::hint::black_box(synth(&m, "corpus", &text("corpus")?)?);
        }
        "dump" => {
            let m = model(dev)?;
            let kind = a.get(3).ok_or(USAGE)?;
            let audio = synth(&m, kind, &text(kind)?)?;
            let bytes: Vec<u8> = audio.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(a.get(4).ok_or(USAGE)?, bytes)?;
        }
        "cold" => {
            let s = Instant::now();
            let m = model(dev)?;
            let load = ms(s);
            let s = Instant::now();
            synth(&m, "short", SHORT)?;
            println!(
                "{{\"dev\":\"{dev}\",\"kind\":\"cold\",\"load_ms\":{load:.1},\"first_synth_ms\":{:.1}}}",
                ms(s)
            );
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}

fn main() -> Res<()> {
    run()
}
