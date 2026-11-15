//! `synthesize_text` (parallel auf der CPU) muss bitgleich zu den nacheinander
//! berechneten Chunks von `synthesize_chunks` sein.
//!
//! Braucht das RU-Modell unter `models/silero-v5-ru/` (nicht eingecheckt); ohne
//! Verzeichnis wird der Test übersprungen.

use std::io::Write;
use std::path::Path;

use candle_core::Device;
use silero_v5_candle::{Error, Silero};

const VOICE: &str = "baya";

fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn load_cpu_model() -> Option<Silero> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-ru");
    if !dir.is_dir() {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
            dir.display()
        );
        skip(&format!("{} fehlt; Test übersprungen", dir.display()));
        return None;
    }
    Some(Silero::load_with_device(&dir, &Device::Cpu).expect("Modell lädt"))
}

#[test]
fn parallel_text_equals_sequential_chunks() {
    if cfg!(debug_assertions) {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber der Test braucht --release"
        );
        skip("Test braucht --release (candle im Debug-Build zu langsam)");
        return;
    }
    let Some(model) = load_cpu_model() else {
        return;
    };
    let text = "Привет, как дела? Сегодня хорошая погода, и мы гуляем по парку. \
                Замок стоит на холме, а рядом течёт река. Я купил муку и испёк хлеб.";

    let parts: Vec<Vec<f32>> = model
        .synthesize_chunks(text, VOICE)
        .collect::<Result<_, _>>()
        .expect("Chunks synthetisieren");
    assert!(parts.len() > 1, "Text muss mehrere Chunks ergeben");
    let sequential: Vec<f32> = parts.concat();

    let parallel = model.synthesize_text(text, VOICE).expect("parallel");
    assert_eq!(parallel.len(), sequential.len(), "Länge");
    let differing = parallel
        .iter()
        .zip(&sequential)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differing, 0, "{differing} Samples weichen bitweise ab");

    // Unbekannte Stimme scheitert vor jeder Rechenarbeit.
    assert!(matches!(
        model.synthesize_text(text, "gibt_es_nicht"),
        Err(Error::Input(_))
    ));
    assert!(matches!(
        model.synthesize_chunks(text, "gibt_es_nicht").next(),
        Some(Err(Error::Input(_)))
    ));
}
