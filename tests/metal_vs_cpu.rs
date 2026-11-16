//! Metal und CPU müssen dieselbe Stimme bis auf Rundung liefern: kurzer und langer Text.
//!
//! Braucht das RU-Modell unter `models/silero-v5-ru/` und ein Metal-Gerät; sonst
//! wird der Test übersprungen.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::io::Write;
use std::path::Path;

use candle_core::Device;
use silero_v5_candle::{Silero, default_device};

const VOICE: &str = "baya";
const SHORT: &str = "Привет, как дела?";
/// Zeilen 30 und 6 aus `tests/corpus.txt` (zusammen ~300 Zeichen).
const CORPUS: &str = include_str!("corpus.txt");
/// Gemessen: 1e-5 (kurz) und 2.6e-4 (lang); die Grenze lässt Spielraum für andere GPUs.
const TOLERANCE: f32 = 1e-3;

fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

#[test]
fn metal_audio_matches_cpu() {
    if cfg!(debug_assertions) {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber der Test braucht --release"
        );
        skip("Test braucht --release (candle im Debug-Build zu langsam)");
        return;
    }
    let metal = default_device();
    if !metal.is_metal() {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber kein Metal-Gerät"
        );
        skip("kein Metal-Gerät; Test übersprungen");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-ru");
    if !dir.is_dir() {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
            dir.display()
        );
        skip(&format!("{} fehlt; Test übersprungen", dir.display()));
        return;
    }
    let cpu = Silero::load_with_device(&dir, &Device::Cpu).expect("CPU-Modell lädt");
    let gpu = Silero::load_with_device(&dir, &metal).expect("Metal-Modell lädt");
    let lines: Vec<&str> = CORPUS.lines().collect();
    let long = format!("{} {}", lines[29], lines[5]);
    for (name, text) in [("kurz", SHORT), ("lang", long.as_str())] {
        let want = cpu.synthesize(text, VOICE).expect("CPU-Synthese läuft");
        let got = gpu.synthesize(text, VOICE).expect("Metal-Synthese läuft");
        assert_eq!(got.len(), want.len(), "{name}: Länge");
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Metal vs. CPU {name}: max {max:.3e}");
        assert!(
            max <= TOLERANCE,
            "{name}: max. Abweichung {max} > {TOLERANCE}"
        );
    }
}
