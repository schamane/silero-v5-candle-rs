//! Prints the accented line for every case in a stress suite JSON.
//!
//! `cargo run --release --example accent_suite -- models/silero-v5-ru \
//!  tests/data/stress_hard60.json baya`

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use serde_json::Value;
use silero_v5_candle::Silero;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: accent_suite <model-dir> <suite.json> [voice]");
        return Err("wrong number of arguments".into());
    }
    let voice = args.get(3).map_or("baya", String::as_str);
    let suite: Vec<Value> = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    let model = Silero::load(Path::new(&args[1]))?;
    for case in &suite {
        let text = case["text"]
            .as_str()
            .ok_or_else(|| format!("case {:?} has no text", case["id"]))?;
        println!("{}", model.accent_text(text, voice)?);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
