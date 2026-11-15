//! Fehlertyp der Bibliothek.

use std::path::{Path, PathBuf};

/// Fehler beim Laden eines Modells oder bei der Synthese.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Eine Datei lässt sich nicht lesen oder schreiben. `path` fehlt, wenn der Ort unbekannt ist.
    #[error("{}", io_message(.path.as_deref()))]
    Io {
        /// Betroffene Datei, falls bekannt.
        path: Option<PathBuf>,
        /// Ursache aus dem Dateisystem.
        source: std::io::Error,
    },
    /// Das Modellverzeichnis ist unvollständig oder ungültig: fehlende oder falsch geformte
    /// Datei, falsche Konfiguration.
    #[error("{context}")]
    Model {
        /// Was geprüft wurde, meist Dateiname oder Teilmodell.
        context: String,
        /// Ursache, falls die Meldung nicht schon alles sagt. Sie steht nur in der
        /// `source()`-Kette, nicht in der Anzeige.
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Die Eingabe ist unbrauchbar: leerer Text, unbekannte Stimme, unbekanntes Symbol.
    #[error("{0}")]
    Input(String),
    /// Ein Schritt der Pipeline ist fehlgeschlagen. `stage` nennt den Schritt.
    #[error("{stage} fehlgeschlagen")]
    Pipeline {
        /// Name des Schritts, etwa `akustik` oder `istft`.
        stage: &'static str,
        /// Ursache aus `candle`.
        source: candle_core::Error,
    },
}

/// Ergebnis mit dem Fehler [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Meldungstext des [`Error::Io`]-Fehlers, mit Pfad, wenn er bekannt ist.
fn io_message(path: Option<&Path>) -> String {
    match path {
        Some(path) => format!(
            "datei {}: lesen oder schreiben fehlgeschlagen",
            path.display()
        ),
        None => "ein-/ausgabe fehlgeschlagen".to_owned(),
    }
}

impl Error {
    /// [`Error::Model`] mit Kontext und der zugrunde liegenden Ursache.
    pub(crate) fn model(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Model {
            context: context.into(),
            source: Some(Box::new(source)),
        }
    }

    /// [`Error::Model`] ohne weitere Ursache.
    pub(crate) fn model_msg(context: impl Into<String>) -> Self {
        Self::Model {
            context: context.into(),
            source: None,
        }
    }
}

/// Wandelt einen `candle`-Fehler in [`Error::Pipeline`] der genannten Stufe um.
pub(crate) fn stage(name: &'static str) -> impl FnOnce(candle_core::Error) -> Error {
    move |source| Error::Pipeline {
        stage: name,
        source,
    }
}

/// Wandelt einen E/A-Fehler in [`Error::Io`] mit dem betroffenen Pfad um.
pub(crate) fn io_at(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.into();
    move |source| Error::Io {
        path: Some(path),
        source,
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<Error>();
};
