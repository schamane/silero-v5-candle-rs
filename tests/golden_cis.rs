//! Golden für `v5_cis_ext` (ukrainisch): ein Satz, eine Stimme.
//!
//! Der Export liegt unter `models/silero-v5-cis-ext/` und ist wie alle
//! Modellgewichte nicht eingecheckt. Ohne Verzeichnis wird der Test übersprungen.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use silero_v5_candle::Silero;

const TEXT: &str = "+Я з р+аннього дит+инства д+уже любл+ю сл+ухати цік+аві к+азки.";
const VOICE: &str = "ukr_mykyta";

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-cis-ext")
}

fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn golden(path: &Path) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(path, &Device::Cpu).expect("Golden lädt")
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .expect("Tensor lässt sich flachen")
        .to_vec1::<f32>()
        .expect("Tensor ist als f32 lesbar")
}

fn integers(tensor: &Tensor) -> Vec<u32> {
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_dtype(candle_core::DType::U32))
        .expect("Tensor lässt sich als u32 lesen")
        .to_vec1::<u32>()
        .expect("Tensor ist als u32 lesbar")
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "{name}: Shape {:?} statt {:?}",
        actual.dims(),
        expected.dims()
    );
    let actual = values(actual);
    let expected = values(expected);
    assert_eq!(actual.len(), expected.len(), "{name}: Länge");
    let mut max = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let diff = (a - e).abs();
        max = max.max(diff);
        assert!(
            diff <= tolerance,
            "{name}: Abweichung {diff} bei {index} ({a} statt {e}) > {tolerance}"
        );
    }
    eprintln!("{name}: max {max:.3e}");
}

#[test]
fn ukrainian_golden_matches_python() {
    if cfg!(debug_assertions) {
        if require_golden() {
            panic!("SILERO_REQUIRE_GOLDEN gesetzt, aber Golden-Tests brauchen --release");
        }
        skip("Golden-Tests brauchen --release");
        return;
    }
    let dir = model_dir();
    let golden_path = dir.join("golden/uk-0.safetensors");
    if !golden_path.is_file() {
        if require_golden() {
            panic!(
                "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
                golden_path.display()
            );
        }
        skip(&format!("{} fehlt", golden_path.display()));
        return;
    }
    let model = Silero::load(&dir).expect("CIS-Modell lädt");
    if model.has_accentor() {
        assert!(model.has_accentor_for(VOICE));
        assert!(!model.has_accentor_for("kaz_abai"));
        let plain = model
            .debug_stages("Привіт, як справи?", VOICE)
            .expect("Akzentor-Synthese läuft");
        assert!(
            plain.accented.contains(&'+'),
            "Ext-Modell mit Akzentor muss unbetonten Text markieren"
        );
    }
    let stages = model.debug_stages(TEXT, VOICE).expect("Synthese läuft");
    let expected = golden(&golden_path);

    let sequence = integers(&expected["sequence"]);
    assert_eq!(stages.ids.len(), sequence.len(), "sequence");
    for (index, (actual, wanted)) in stages.ids.iter().zip(&sequence).enumerate() {
        assert_eq!(actual, wanted, "sequence[{index}]");
    }
    let dur = integers(&expected["dur"]);
    assert_eq!(stages.output.dur, dur, "dur");

    compare(
        "dur_raw",
        &stages.output.dur_raw,
        &expected["dur_raw"],
        1e-4,
    );
    compare(
        "pitch_raw",
        &stages.output.pitch_raw,
        &expected["pitch_raw"],
        1e-4,
    );
    compare("pitch", &stages.output.pitch, &expected["pitch"], 1e-5);
    compare(
        "encoder",
        &stages.output.encoder,
        &expected["encoder"],
        1e-4,
    );
    compare("cond", &stages.output.cond, &expected["cond"], 1e-4);
    compare(
        "expanded",
        &stages.output.expanded,
        &expected["expanded"],
        1e-4,
    );
    compare(
        "decoder",
        &stages.output.decoder,
        &expected["decoder"],
        1e-3,
    );
    compare("mel", &stages.output.mel, &expected["mel"], 1e-3);
    compare("backbone", &stages.features, &expected["backbone"], 3e-4);
    compare("mag", &stages.mag, &expected["mag"], 5e-3);
    compare("cos", &stages.cos, &expected["cos"], 3e-3);
    compare("sin", &stages.sin, &expected["sin"], 3e-3);

    let wanted_audio = values(&expected["audio"]);
    assert_eq!(stages.audio.len(), wanted_audio.len(), "audio: Länge");
    let mut max = 0.0f32;
    for (index, (actual, wanted)) in stages.audio.iter().zip(&wanted_audio).enumerate() {
        let diff = (actual - wanted).abs();
        max = max.max(diff);
        assert!(
            diff <= 5e-3,
            "audio: Abweichung {diff} bei {index} ({actual} statt {wanted})"
        );
    }
    eprintln!("audio: max {max:.3e}");
}
