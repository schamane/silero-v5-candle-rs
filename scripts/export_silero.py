#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "scipy", "safetensors", "soundfile"]
# ///
"""Export Silero v5_ru weights, dictionaries and golden tensors for silero-v5-candle-rs.

Reads the packaged TorchScript model once, unpacks the quantized BERT embeddings
exactly once (via one warm apply_tts), writes f32 safetensors plus the text
frontend dictionaries, and records golden tensors for every stage of the
pipeline. Run with `uv run scripts/export_silero.py`; the output directory is
gitignored (CC BY-NC-SA 4.0, derived weights).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path.home() / ".cache/torch/hub/snakers4_silero-models_master/src/silero/model/v5_ru.pt"
EXPECTED_SHA256 = "7ba04d42340fe0398042eed2e0d12d62e23096d626b1b9feff4dcb1309197ab4"
OUT = ROOT / "models/silero-v5-ru"
CORPUS = ROOT / "tests/corpus.txt"
VOICES = ["baya", "kseniya", "xenia", "aidar", "eugene"]
# Sentences with full intermediate tensors (only baya); the rest get mel/audio.
FULL_GOLDEN = 6
# Voices beyond baya are exported for the first N sentences (mel/audio only).
MULTI_VOICE_GOLDEN = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Mirrors `silero_v5_candle::text::normalize::{split_chunks, has_cyrillic}`.
CYRILLIC = re.compile("[а-яё]")  # after lower(), as `has_cyrillic`
CHUNK_LIMIT = 400


def _split_sentences(text: str) -> list[str]:
    """`SENTENCE_SPLIT = (?<=[.!?…])\\s+|\\n+`, keeping the punctuation."""
    chars = list(text)
    pieces: list[str] = []
    start = 0
    i = 0
    while i < len(chars):
        if chars[i] in ".!?…":
            j = i + 1
            while j < len(chars) and chars[j].isspace():
                j += 1
            if j > i + 1:
                pieces.append("".join(chars[start : i + 1]))
                start = i = j
                continue
            i += 1
        elif chars[i] == "\n":
            j = i
            while j < len(chars) and chars[j] == "\n":
                j += 1
            pieces.append("".join(chars[start:i]))
            start = i = j
        else:
            i += 1
    if start < len(chars):
        pieces.append("".join(chars[start:]))
    return pieces


def _last_substring(chars: list[str], needle: str, limit: int) -> int:
    """Last start index of `needle` that ends at or before `limit`; 0 if none."""
    found = 0
    if not needle or len(chars) < len(needle):
        return found
    wanted = list(needle)
    for start in range(len(chars) - len(needle) + 1):
        if start + len(needle) > limit:
            break
        if chars[start : start + len(needle)] == wanted:
            found = start
    return found


def split_chunks(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    limit = max(3, limit)
    chunks: list[str] = []
    for sentence in _split_sentences(text):
        sentence = list(sentence.strip())
        while len(sentence) > limit:
            cut = max(
                _last_substring(sentence, ", ", limit),
                _last_substring(sentence, "; ", limit),
                _last_substring(sentence, " — ", limit),
            )
            if cut < limit // 3:
                cut = _last_substring(sentence, " ", limit)
            if cut < limit // 3:
                cut = limit
            head = "".join(sentence[:cut]).strip()
            if head:
                chunks.append(head)
            sentence = sentence[cut:]
            while sentence and sentence[0] in " ,;":
                sentence.pop(0)
        tail = "".join(sentence).strip()
        if tail:
            chunks.append(tail)
    return chunks


def contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def state_dict(module) -> dict[str, torch.Tensor]:
    return {name: contiguous(value) for name, value in module.state_dict().items()}


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: contiguous(value) for name, value in tensors.items()}, str(path))


def postprocess_durations(raw: torch.Tensor) -> torch.Tensor:
    """The gt_durs is None path of MultiTTSModel.forward."""
    dur = torch.exp(raw) - 1
    dur = torch.where(dur < 0, torch.zeros_like(dur), dur)
    dur = torch.round(dur)
    dur = torch.round(dur / 1.0)
    if bool(dur[0, 0] > 5):
        dur[0, 0] = 5
    if bool(dur[0, -1] > 7):
        dur[0, -1] = 7
    dur[0, -2] = 13
    if bool(dur[0, -3] > 13):
        dur[0, -3] = 13
    return dur


def homo_logits(homosolver, sentence: str):
    """Replicates HomoSolver.__call__ up to the model logits."""
    import torch as t

    tagged = homosolver._find_and_tag_homos(sentence)
    starts, ends, sents, words = [], [], [], []
    for start, end, word, _, raw_mark in tagged:
        if raw_mark is None:
            continue
        ids = t.tensor(homosolver.tokenizer(raw_mark))
        homo_start = t.where(ids == homosolver.tokenizer.homo_start_id)[0][0]
        homo_end = t.where(ids == homosolver.tokenizer.homo_end_id)[0][0]
        starts.append(homo_start)
        ends.append(homo_end)
        sents.append(ids)
        words.append(word)
    if not sents:
        return None, None, []
    padded = t.nn.utils.rnn.pad_sequence(
        sents, batch_first=True, padding_value=homosolver.tokenizer.pad_token_id
    )
    with t.no_grad():
        logits = homosolver.model(
            padded, t.LongTensor(starts), t.LongTensor(ends)
        )
    return logits, t.round(t.sigmoid(logits)), words


def run_pipeline(part, tsm, sentence: str, voice: str):
    """Computes every intermediate exactly as the TorchScript forward does."""
    speaker_id = part.speaker_to_ids[0][voice]
    speaker_ids = torch.tensor([speaker_id])
    sequence, _, _, _ = part.merge_batch_model([sentence], [None], [1.0], [1.0])
    seq_len = sequence.shape[1]
    mask = torch.zeros(1, seq_len, dtype=torch.bool)

    raw_dur = tsm.dur_predictor(sequence, speaker_ids, mask, 1.0, None)
    raw_pitch = tsm.pitch_predictor(sequence, speaker_ids, mask, 1.0)
    pitch_coefs = torch.ones(1, seq_len)
    pitch = tsm.update_pitch_coef(raw_pitch, pitch_coefs, speaker_ids)
    dur = postprocess_durations(raw_dur)

    tacotron = tsm.tacotron
    embedded = tacotron.embedding(sequence)
    speaker = tacotron.speaker_embedding(speaker_ids).unsqueeze(1).expand(-1, seq_len, -1)
    encoder = tacotron.encoder(embedded, mask, None)
    cond = encoder + speaker
    cond = cond + tacotron.pitch_proj(pitch).transpose(1, 2)
    expanded = tacotron.len_reg(cond, dur)
    decoder = tacotron.decoder(expanded, None, None)
    mel = tacotron.lin(decoder).transpose(1, 2)

    backbone = tsm.vocoder.backbone(mel)
    spec = tsm.vocoder.head(backbone)
    audio = part.wrapped_jit_v(spec, 48000)[0]

    reference = part.apply_tts(
        text=sentence, speaker=voice, sample_rate=48000, put_accent=True, put_yo=True
    )
    if audio.shape != reference.shape or not torch.allclose(audio, reference, atol=0, rtol=0):
        diff = (audio - reference).abs().max().item() if audio.shape == reference.shape else float("nan")
        raise SystemExit(
            f"manual pipeline differs from apply_tts for '{sentence}' ({voice}): "
            f"diff={diff} manual={tuple(audio.shape)} reference={tuple(reference.shape)} dur_sum={dur.sum().item()}"
        )

    return {
        "sequence": sequence,
        "mask": mask,
        "dur": dur,
        "dur_raw": raw_dur,
        "pitch": pitch,
        "pitch_raw": raw_pitch,
        "encoder": encoder,
        "cond": cond,
        "expanded": expanded,
        "decoder": decoder,
        "mel": mel,
        "backbone": backbone,
        "mag": spec[0],
        "cos": spec[1],
        "sin": spec[2],
        "audio": audio,
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    digest = sha256(PACKAGE)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"{PACKAGE}: sha256 {digest} != {EXPECTED_SHA256}")

    print(f"loading {PACKAGE}", file=sys.stderr)
    importer = PackageImporter(str(PACKAGE))
    model = importer.load_pickle("tts_models", "model")
    part = model.packages[0]
    tsm = part.models[0]
    accentor_model = part.accentor.accentor.model
    homosolver = part.accentor.homosolver

    # Warm run: this is the only place the int8 BERT embeddings are unpacked.
    warm = "Привет, как дела?"
    warm_audio = model.apply_tts(text=warm, speaker="baya", sample_rate=48000)
    if not part.q_model_unpacked:
        raise SystemExit("warm apply_tts did not unpack the quantized BERT embeddings")
    again = model.apply_tts(text=warm, speaker="baya", sample_rate=48000)
    if not torch.equal(warm_audio, again):
        raise SystemExit("apply_tts is not deterministic across runs")

    OUT.mkdir(parents=True, exist_ok=True)
    tts_state = state_dict(tsm)
    accentor_state = state_dict(accentor_model)
    homo_state = state_dict(homosolver.model)
    window = contiguous(part.wrapped_jit_v.istft.window)
    write_safetensors(OUT / "acoustic.safetensors", tts_state)
    write_safetensors(OUT / "accentor.safetensors", accentor_state)
    write_safetensors(OUT / "homosolver.safetensors", homo_state)
    write_safetensors(OUT / "istft_window.safetensors", {"window": window})

    # Gate: every tensor can be loaded back strictly and audio stays bit-identical.
    tsm.load_state_dict({k: v for k, v in tts_state.items()}, strict=True)
    accentor_model.load_state_dict(accentor_state, strict=True)
    homosolver.model.load_state_dict(homo_state, strict=True)
    reloaded = model.apply_tts(text=warm, speaker="baya", sample_rate=48000)
    if not torch.equal(warm_audio, reloaded):
        raise SystemExit("audio changed after strict state_dict reload")

    symbols = list(part.symbols)
    config = {
        "symbols": "".join(symbols),
        "symbol_to_id": part.symbol_to_id,
        "speakers": part.speaker_to_ids[0],
        "sample_rate": 48000,
        "n_fft": int(part.wrapped_jit_v.istft.n_fft),
        "hop_length": int(part.wrapped_jit_v.istft.hop_length),
        "win_length": int(part.wrapped_jit_v.istft.win_length),
        "max_positions": 5000,
        "sil_value": float(part.models[0].sil_value),
        "mean_std_coef": [float(x) for x in part.models[0].mean_std_coef],
        "punkt_symbols": [int(x) for x in part.models[0].punkt_symbols],
        "point_id": int(part.models[0].point_id),
        "comma_ids": [int(x) for x in part.models[0].comma_ids],
        "bert": {
            "vocab_size": 83830,
            "hidden_size": 312,
            "num_hidden_layers": 3,
            "num_attention_heads": 12,
            "intermediate_size": 600,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 2048,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "use_cache": False,
            "classifier_dropout": None,
            "model_type": None,
        },
        "tokenizer": {
            "never_split": sorted(homosolver.tokenizer.never_split),
            "tokenize_chinese_chars": True,
            "model_max_length": 2048,
            "unk_token_id": homosolver.tokenizer.unk_token_id,
            "sep_token_id": homosolver.tokenizer.sep_token_id,
            "pad_token_id": homosolver.tokenizer.pad_token_id,
            "cls_token_id": homosolver.tokenizer.cls_token_id,
            "mask_token_id": homosolver.tokenizer.mask_token_id,
            "homo_start_id": homosolver.tokenizer.homo_start_id,
            "homo_end_id": homosolver.tokenizer.homo_end_id,
        },
    }
    if config["symbols"] != "_~|!+,-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё–… ":
        raise SystemExit(f"unexpected symbols: {config['symbols']!r}")
    if len(config["speakers"]) != 5 or config["speakers"]["baya"] != 1:
        raise SystemExit("unexpected speaker ids")
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    ngrams = accentor_model.embedding.ngram_dict
    ordered = sorted(ngrams, key=ngrams.__getitem__)
    if [ngrams[g] for g in ordered] != list(range(len(ordered))):
        raise SystemExit("ngram ids are not contiguous")
    (OUT / "ngrams.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")

    exceptions = part.accentor.accentor.exceptions
    lines = [f"{word}\t{stress}\t{yo}" for word, (stress, yo) in exceptions.items()]
    (OUT / "exceptions.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "homodict.json").write_text(
        json.dumps(homosolver.homodict, ensure_ascii=False, indent=0, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    vocab = sorted(homosolver.tokenizer.vocab, key=homosolver.tokenizer.vocab.__getitem__)
    if [homosolver.tokenizer.vocab[t] for t in vocab] != list(range(len(vocab))):
        raise SystemExit("BERT vocab ids are not contiguous")
    (OUT / "bert_vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    corpus = [line.rstrip("\n") for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = []
    for index, line in enumerate(corpus):
        prepared, clean, has_text = part.prepare_text_input(line)
        entry = {
            "id": index,
            "text": line,
            "prepared": prepared,
            "clean": clean,
            "has_text": bool(has_text),
        }
        if not has_text:
            texts.append(entry)
            continue
        homo = homosolver(prepared, put_stress=True, put_yo=True, stress_single_vowel=True)
        accented = part.accentor.accentor(homo, put_stress=True, put_yo=True, stress_single_vowel=True)
        no_homo = part.accentor.accentor(
            prepared,
            put_stress=True,
            put_yo=True,
            stress_single_vowel=True,
            skip_stress_words=homosolver.homodict,
            skip_yo_words=homosolver.yohomodict,
        )
        logits, preds, words = homo_logits(homosolver, prepared)
        sequence, _, _, _ = part.merge_batch_model([prepared], [None], [1.0], [1.0])
        expected = [part.symbol_to_id[c] for c in part.sos_token + accented + part.eos_token]
        if sequence[0].tolist() != expected:
            raise SystemExit(f"sentence {index}: merge_batch_model sequence differs from accentor output")
        entry.update(
            {
                "homo": homo,
                "accented": accented,
                "accented_no_homo": no_homo,
                "sequence": sequence[0].tolist(),
                "homo_words": words,
                "homo_preds": preds.flatten().int().tolist() if preds is not None else [],
            }
        )
        texts.append(entry)
        if logits is not None:
            write_safetensors(OUT / f"golden/homo-{index}.safetensors", {"logits": logits})

        voices = VOICES if index < MULTI_VOICE_GOLDEN else ["baya"]
        for voice in voices:
            state = run_pipeline(part, tsm, prepared, voice)
            keys = ["dur", "dur_raw", "pitch", "pitch_raw", "mel", "audio"]
            if voice == "baya":
                keys += ["backbone"]
            tensors = {key: state[key] for key in keys}
            if index < FULL_GOLDEN and voice == "baya":
                tensors.update(
                    {key: state[key] for key in ["encoder", "cond", "expanded", "decoder", "mag", "cos", "sin"]}
                )
            write_safetensors(OUT / f"golden/{index}-{voice}.safetensors", tensors)
        print(f"golden {index}: {' '.join(entry['homo_words']) or '-'}", file=sys.stderr)

    with (OUT / "golden/text.jsonl").open("w", encoding="utf-8") as handle:
        for entry in texts:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    chunks = []
    for index, line in enumerate(corpus):
        split = split_chunks(line)
        kept = [chunk for chunk in split if CYRILLIC.search(chunk.lower())]
        chunks.append({"id": index, "chunks": split, "kept": kept})
    (OUT / "golden/chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({total / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
