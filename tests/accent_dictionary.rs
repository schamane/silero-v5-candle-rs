//! `accent_text` uses the optional `stress_dict.tsv` without synthesizing audio.
//!
//! The test is skipped unless the RU model and the generated dictionary are present.

use std::io::Write;
use std::path::Path;

use candle_core::Device;
use silero_v5_candle::Silero;

fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

#[test]
fn accent_text_uses_dictionary_and_keeps_explicit_marks() {
    if cfg!(debug_assertions) {
        skip("Test braucht --release (Modelllast)");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-ru");
    if !dir.is_dir() {
        skip(&format!("{} fehlt", dir.display()));
        return;
    }
    if !dir.join("stress_dict.tsv").is_file() {
        skip("stress_dict.tsv fehlt; scripts/export_stress_dict.py erzeugt sie");
        return;
    }
    let model = Silero::load_with_device(&dir, &Device::Cpu).expect("Modell lädt");

    let accented = model
        .accent_text("Книга лежит на столе.", "baya")
        .expect("Betonung");
    assert!(
        accented.contains("стол+е"),
        "Wörterbuch nicht angewandt: {accented:?}"
    );

    let explicit = model
        .accent_text("Прив+ет, как дела?", "baya")
        .expect("Betonung");
    assert!(
        explicit.contains("прив+ет"),
        "explizite Marke verloren: {explicit:?}"
    );
}
