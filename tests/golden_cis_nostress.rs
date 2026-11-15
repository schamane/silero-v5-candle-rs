//! Golden für `v5_cis_base_nostress` (ukrainisch, automatischer Akzentor).
//!
//! Der Python-Export akzentuiert zuerst den vorbereiteten Text mit
//! `silero-stress` und rechnet dann die 48-kHz-Pipeline. Genau diese Reihenfolge
//! prüft der Test. Ohne Modellverzeichnis wird er übersprungen.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use silero_v5_candle::Silero;

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-cis-base-nostress")
}

fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn load_model() -> Option<Silero> {
    let dir = model_dir();
    let golden = dir.join("golden/text.jsonl");
    if !golden.is_file() {
        if require_golden() {
            panic!(
                "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
                golden.display()
            );
        }
        skip(&format!(
            "{} fehlt; Nostress-Golden übersprungen",
            golden.display()
        ));
        return None;
    }
    Some(Silero::load(&dir).expect("Nostress-Modell lädt"))
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

/// Absolute Toleranz plus relativer Anteil; für den ConvNeXt-Ausgang.
///
/// Bereits 1e-5 Unterschied im Mel verstärken sich im Backbone messbar (maximal
/// ~5e-3); derselbe Rust-Backbone auf Pythons Mel weicht dagegen nur 1e-4 ab.
/// Die hörbaren Stufen `mag`/`cos`/`sin` und das Audio bleiben eng geprüft.
fn compare_scaled(name: &str, actual: &Tensor, expected: &Tensor, rtol: f32) {
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
    let mut worst = 0;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 5e-3 + rtol * e.abs();
        let diff = (a - e).abs();
        if diff > max {
            max = diff;
            worst = index;
        }
        assert!(
            diff <= tolerance,
            "{name}: Abweichung {diff} bei {index} ({a} statt {e}) > {tolerance}"
        );
    }
    eprintln!("{name}: max {max:.3e} bei {worst}");
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
    let mut worst = 0;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let diff = (a - e).abs();
        if diff > max {
            max = diff;
            worst = index;
        }
    }
    assert!(
        max <= tolerance,
        "{name}: Abweichung {max} bei {worst} ({} statt {}) > {tolerance}",
        actual[worst],
        expected[worst]
    );
    eprintln!("{name}: max {max:.3e}");
}

#[test]
fn ukrainian_nostress_golden_matches_python() {
    if cfg!(debug_assertions) {
        if require_golden() {
            panic!("SILERO_REQUIRE_GOLDEN gesetzt, aber Golden-Tests brauchen --release");
        }
        skip("Golden-Tests brauchen --release");
        return;
    }
    let Some(model) = load_model() else { return };
    assert!(model.has_accentor(), "Nostress-Modell braucht Akzentor");
    assert!(model.has_accentor_for("ukr_igor"));
    assert!(!model.has_accentor_for("kaz_zhadyra"));
    let dir = model_dir();
    let text =
        std::fs::read_to_string(dir.join("golden/text.jsonl")).expect("text.jsonl ist lesbar");
    let entries: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("text.jsonl Zeile ist gültiges JSON"))
        .collect();
    for entry in &entries {
        let id = entry["id"].as_u64().expect("Feld id ist eine Zahl");
        let raw = entry["text"].as_str().expect("Feld text ist ein String");
        let voice = entry["voice"].as_str().expect("Feld voice ist ein String");
        let prepared = entry["prepared"]
            .as_str()
            .expect("Feld prepared ist ein String");
        let accented = entry["accented"]
            .as_str()
            .expect("Feld accented ist ein String");
        let stages = model.debug_stages(raw, voice).expect("Pipeline");
        assert_eq!(
            stages.prepared.sentence.iter().collect::<String>(),
            prepared,
            "Satz {id}: prepared"
        );
        assert_eq!(
            stages.accented.iter().collect::<String>(),
            accented,
            "Satz {id}: accented"
        );
        if id == 1 {
            assert!(
                accented.contains("к+азки"),
                "Satz 1: vorhandene Betonung muss erhalten bleiben"
            );
        }

        let expected = golden(&dir.join(format!("golden/uk-{id}.safetensors")));
        let sequence = integers(&expected["sequence"]);
        assert_eq!(stages.ids, sequence, "Satz {id}: sequence");
        let dur = integers(&expected["dur"]);
        assert_eq!(stages.output.dur, dur, "Satz {id}: dur");

        compare(
            &format!("Satz {id} dur_raw"),
            &stages.output.dur_raw,
            &expected["dur_raw"],
            1e-4,
        );
        compare(
            &format!("Satz {id} pitch_raw"),
            &stages.output.pitch_raw,
            &expected["pitch_raw"],
            1e-4,
        );
        compare(
            &format!("Satz {id} pitch"),
            &stages.output.pitch,
            &expected["pitch"],
            1e-5,
        );
        compare(
            &format!("Satz {id} encoder"),
            &stages.output.encoder,
            &expected["encoder"],
            1e-4,
        );
        compare(
            &format!("Satz {id} cond"),
            &stages.output.cond,
            &expected["cond"],
            1e-4,
        );
        compare(
            &format!("Satz {id} expanded"),
            &stages.output.expanded,
            &expected["expanded"],
            1e-4,
        );
        compare(
            &format!("Satz {id} decoder"),
            &stages.output.decoder,
            &expected["decoder"],
            1e-3,
        );
        compare(
            &format!("Satz {id} mel"),
            &stages.output.mel,
            &expected["mel"],
            1e-3,
        );
        compare_scaled(
            &format!("Satz {id} backbone"),
            &stages.features,
            &expected["backbone"],
            1e-3,
        );
        compare_scaled(
            &format!("Satz {id} mag"),
            &stages.mag,
            &expected["mag"],
            5e-3,
        );
        // Die Phase streut bei kleinen Magnituden deutlich; das Audio bleibt
        // trotzdem bei 7e-4. Der ISTFT gewichtet mit `mag`.
        compare(
            &format!("Satz {id} cos"),
            &stages.cos,
            &expected["cos"],
            2.5e-2,
        );
        compare(
            &format!("Satz {id} sin"),
            &stages.sin,
            &expected["sin"],
            2.5e-2,
        );

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
        eprintln!("Satz {id}: audio max {max:.3e}");
    }
}
