//! WAV-Ausgabe: 16 Bit PCM, mono.

use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::Result;
use crate::error::io_at;

/// Größe von `WAVEfmt `-Block und `data`-Kopf ohne `RIFF`-Kopf, die zu `data_len` addiert wird.
const RIFF_OVERHEAD: u32 = 36;

/// Schreibt `samples` (Gleitkomma in `-1.0..=1.0`, außerhalb wird begrenzt) als
/// 16-Bit-PCM-WAV nach `path`; fehlende Elternverzeichnisse werden angelegt.
///
/// # Errors
///
/// Gibt [`Error::Io`](crate::Error::Io) mit dem Pfad zurück, wenn das Schreiben scheitert
/// oder das Audio für den 32-Bit-Größenzähler eines WAV-Containers zu groß ist.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(io_at(parent))?;
    }
    let too_large = || io::Error::other("audio zu groß für eine WAV-Datei");
    let sizes = samples
        .len()
        .checked_mul(2)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .and_then(|data_len| Some((data_len, data_len.checked_add(RIFF_OVERHEAD)?)))
        .zip(sample_rate.checked_mul(2));
    let ((data_len, riff_len), byte_rate) = sizes.ok_or_else(too_large).map_err(io_at(path))?;
    let file = std::fs::File::create(path).map_err(io_at(path))?;
    let mut writer = BufWriter::new(file);
    write_pcm16(
        &mut writer,
        samples,
        sample_rate,
        (data_len, riff_len),
        byte_rate,
    )
    .and_then(|()| writer.flush())
    .map_err(io_at(path))
}

fn write_pcm16(
    writer: &mut impl Write,
    samples: &[f32],
    sample_rate: u32,
    (data_len, riff_len): (u32, u32),
    byte_rate: u32,
) -> io::Result<()> {
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVEfmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&1u16.to_le_bytes())?; // mono
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?; // Block-Align
    writer.write_all(&16u16.to_le_bytes())?; // Bits je Sample
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
