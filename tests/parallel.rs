//! `synthesize_text` (parallel auf CPU und Metal) muss bitgleich zu den nacheinander
//! berechneten Chunks von `synthesize_chunks` sein.
//!
//! Braucht das RU-Modell unter `models/silero-v5-ru/` (nicht eingecheckt); ohne
//! Verzeichnis wird der Test übersprungen.

use std::io::Write;
use std::path::Path;

use candle_core::Device;
use silero_v5_candle::{Error, Silero, default_device};

const VOICE: &str = "baya";

fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn skip_debug() -> bool {
    if cfg!(debug_assertions) {
        assert!(
            !require_golden(),
            "SILERO_REQUIRE_GOLDEN gesetzt, aber der Test braucht --release"
        );
        skip("Test braucht --release (candle im Debug-Build zu langsam)");
        true
    } else {
        false
    }
}

fn load_model(device: &Device) -> Option<Silero> {
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
    Some(Silero::load_with_device(&dir, device).expect("Modell lädt"))
}

fn check_parallel(model: &Silero, label: &str) {
    let text = "Привет, как дела? Сегодня хорошая погода, и мы гуляем по парку. \
                Замок стоит на холме, а рядом течёт река. Я купил муку и испёк хлеб.";

    let parts: Vec<Vec<f32>> = model
        .synthesize_chunks(text, VOICE)
        .collect::<Result<_, _>>()
        .expect("Chunks synthetisieren");
    assert!(parts.len() > 1, "{label}: Text muss mehrere Chunks ergeben");
    let sequential: Vec<f32> = parts.concat();

    let parallel = model.synthesize_text(text, VOICE).expect("parallel");
    assert_eq!(parallel.len(), sequential.len(), "{label}: Länge");
    let differing = parallel
        .iter()
        .zip(&sequential)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{label}: {differing} Samples weichen bitweise ab"
    );

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

#[test]
fn parallel_text_equals_sequential_chunks_on_cpu() {
    if skip_debug() {
        return;
    }
    let Some(model) = load_model(&Device::Cpu) else {
        return;
    };
    check_parallel(&model, "CPU");
}

#[test]
fn parallel_text_equals_sequential_chunks_on_default_device() {
    if skip_debug() {
        return;
    }
    let device = default_device();
    let label = if device.is_metal() {
        "Metal"
    } else {
        "CPU (default)"
    };
    let Some(model) = load_model(&device) else {
        return;
    };
    check_parallel(&model, label);
}
