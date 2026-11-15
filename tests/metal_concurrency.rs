//! Gleichzeitiges `synthesize` auf einem gemeinsamen Metal-Modell darf nicht abstürzen
//! und muss bitgleiche Ergebnisse liefern (prozessweite Metal-Sperre, siehe `lib.rs`).
//!
//! Braucht das RU-Modell unter `models/silero-v5-ru/` und ein Metal-Gerät; sonst
//! wird der Test übersprungen.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::io::Write;
use std::path::Path;

use silero_v5_candle::{Silero, default_device};

const VOICE: &str = "baya";
const TEXT: &str = "Привет, как дела?";
const THREADS: usize = 8;
const ROUNDS: usize = 5;

fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

#[test]
fn concurrent_synthesize_on_metal_is_bit_identical() {
    if cfg!(debug_assertions) {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber der Test braucht --release"
        );
        skip("Test braucht --release (candle im Debug-Build zu langsam)");
        return;
    }
    let device = default_device();
    if !device.is_metal() {
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
    let model = Silero::load_with_device(&dir, &device).expect("Modell lädt");
    let reference = model.synthesize(TEXT, VOICE).expect("Einzelaufruf läuft");
    assert!(!reference.is_empty(), "Referenz-Audio ist leer");

    for round in 0..ROUNDS {
        let outputs: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| scope.spawn(|| model.synthesize(TEXT, VOICE).expect("Thread läuft")))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("Thread panikt nicht"))
                .collect()
        });
        for (thread, audio) in outputs.iter().enumerate() {
            assert!(
                audio.len() == reference.len()
                    && audio
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "Runde {round}, Thread {thread}: Audio weicht vom Einzelaufruf ab"
            );
        }
    }
}
