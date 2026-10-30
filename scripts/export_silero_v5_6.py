#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "scipy", "safetensors", "soundfile"]
# ///
"""Export Silero v5_6_ru weights, dictionaries and golden tensors.

Reads the packaged TorchScript model once, unpacks the quantized BERT embeddings
exactly once (via one warm apply_tts), writes f32 safetensors plus the text
frontend dictionaries, and records golden tensors for every stage of the
pipeline. Run with `uv run scripts/export_silero_v5_6.py /path/to/v5_6_ru.pt`;
the output directory is gitignored (CC BY-NC-SA 4.0, derived weights).

The v5_6 package differs from v5_ru: the vocoder (with its ISTFT window) lives
inside the acoustic model, the duration and pitch predictors carry type, focus
and emphasis embeddings, and the forward pass is conditioned on per-sentence
utterance types. `golden/*.safetensors` therefore also store `type_ids`.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models/silero-v5_6-ru"
CORPUS = ROOT / "tests/corpus.txt"
VOICES = ["baya", "kseniya", "xenia", "aidar", "eugene"]
EXPECTED_SHA256 = "04cc9b1d08ce38bcaa9d02c488d80d3e6806d17f7eaa7c95982f36a3cbf50b2e"
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


def contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def state_dict(module) -> dict[str, torch.Tensor]:
    return {name: contiguous(value) for name, value in module.state_dict().items()}


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: contiguous(value) for name, value in tensors.items()}, str(path))


def postprocess_durations(raw: torch.Tensor) -> torch.Tensor:
    """`exp(pred) - 1`, clamp, round, `cap_sos(5)` and `fx_durs` without emphasis."""
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


def type_ids_for(part, sequence: torch.Tensor, text: str, classify_text, classify_sentence) -> torch.Tensor:
    """`apply_tts` for plain text: sentence types over the token sequence, no emphasis."""
    spans, sent_texts = part.token_sent_spans(sequence[0])
    types_list = classify_text(text).split("|")
    if len(types_list) != len(spans):
        types_list = [classify_sentence(t.replace("+", "")) for t in sent_texts]
    return part.build_type_ids_tokens(sequence[0], types_list, set())


def run_pipeline(part, tsm, sentence: str, voice: str, classify_text, classify_sentence):
    """Computes every intermediate exactly as the TorchScript forward does."""
    speaker_id = part.speaker_to_ids[0][voice]
    speaker_ids = torch.tensor([speaker_id])
    sequence, _, _, pitch_coefs = part.merge_batch_model([sentence], [None], [1.0], [1.0])
    seq_len = sequence.shape[1]
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    type_ids = type_ids_for(part, sequence, sentence, classify_text, classify_sentence)
    focus = torch.zeros(1, seq_len, dtype=torch.long)

    raw_dur = tsm.dur_predictor(
        sequence, speaker_ids, mask, 1.0, None, type_ids=type_ids, focus_mask=focus
    )
    dur = postprocess_durations(raw_dur)
    raw_pitch = tsm.pitch_predictor(
        sequence, speaker_ids, mask, type_ids=type_ids, focus_mask=focus
    )
    coef = torch.unsqueeze(pitch_coefs[0], 0)
    pitch = tsm.update_pitch_coef(raw_pitch, coef, speaker_ids)

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
    packed = tsm.vocoder.head.out(backbone).transpose(1, 2)
    mag_raw, phase = torch.chunk(packed, 2, 1)
    mag = torch.clamp(torch.exp(mag_raw), None, 100.0)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    audio = tsm.vocoder(mel, 48000).reshape(-1)

    reference = part.apply_tts(text=sentence, speaker=voice, sample_rate=48000)
    if audio.shape != reference.shape or not torch.equal(audio, reference):
        diff = (audio - reference).abs().max().item() if audio.shape == reference.shape else float("nan")
        raise SystemExit(
            f"manual pipeline differs from apply_tts for '{sentence}' ({voice}): "
            f"diff={diff} manual={tuple(audio.shape)} reference={tuple(reference.shape)}"
        )

    return {
        "sequence": sequence,
        "mask": mask,
        "type_ids": type_ids,
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
        "mag": mag,
        "cos": cos,
        "sin": sin,
        "audio": audio,
    }


def homo_logits(homosolver, sentence: str):
    """Replicates HomoSolver.__call__ up to the model logits."""
    tagged = homosolver._find_and_tag_homos(sentence)
    starts, ends, sents, words = [], [], [], []
    for start, end, word, _, raw_mark, _ in tagged:
        if raw_mark is None:
            continue
        ids = torch.tensor(homosolver.tokenizer(raw_mark))
        homo_start = torch.where(ids == homosolver.tokenizer.homo_start_id)[0][0]
        homo_end = torch.where(ids == homosolver.tokenizer.homo_end_id)[0][0]
        starts.append(homo_start)
        ends.append(homo_end)
        sents.append(ids)
        words.append(word)
    if not sents:
        return None, None, []
    padded = torch.nn.utils.rnn.pad_sequence(
        sents, batch_first=True, padding_value=homosolver.tokenizer.pad_token_id
    )
    with torch.no_grad():
        logits = homosolver.model(padded, torch.LongTensor(starts), torch.LongTensor(ends))
    return logits, torch.round(torch.sigmoid(logits)), words


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_silero_v5_6.py /path/to/v5_6_ru.pt")
    package = Path(sys.argv[1])
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    digest = sha256(package)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"{package}: sha256 {digest} != {EXPECTED_SHA256}")

    print(f"loading {package}", file=sys.stderr)
    importer = PackageImporter(str(package))
    model = importer.load_pickle("tts_models", "model")
    part = model.packages[0]
    tsm = part.models[0]
    accentor_model = part.accentor.accentor.model
    homosolver = part.accentor.homosolver
    globals_ = type(part).apply_tts.__globals__
    classify_text = globals_["classify_text"]
    classify_sentence = globals_["classify_sentence"]

    # Warm run: this is the only place the int8 BERT embeddings are unpacked.
    warm = "привет, как дела?"
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
    window = contiguous(tsm.vocoder.head.istft.window)
    write_safetensors(OUT / "acoustic.safetensors", tts_state)
    write_safetensors(OUT / "accentor.safetensors", accentor_state)
    write_safetensors(OUT / "homosolver.safetensors", homo_state)
    write_safetensors(OUT / "istft_window.safetensors", {"window": window})

    # Gate: every tensor can be loaded back strictly and audio stays bit-identical.
    tsm.load_state_dict(tts_state, strict=True)
    accentor_model.load_state_dict(accentor_state, strict=True)
    homosolver.model.load_state_dict(homo_state, strict=True)
    reloaded = model.apply_tts(text=warm, speaker="baya", sample_rate=48000)
    if not torch.equal(warm_audio, reloaded):
        raise SystemExit("audio changed after strict state_dict reload")

    symbols = list(part.symbols)
    istft = tsm.vocoder.head.istft
    config = {
        "symbols": "".join(symbols),
        "symbol_to_id": part.symbol_to_id,
        "speakers": part.speaker_to_ids[0],
        "sample_rate": 48000,
        "n_fft": int(istft.n_fft),
        "hop_length": int(istft.hop_length),
        "win_length": int(istft.win_length),
        "max_positions": 5000,
        "sil_value": float(tsm.sil_value),
        "mean_std_coef": [float(x) for x in tsm.mean_std_coef],
        "punkt_symbols": [int(x) for x in tsm.punkt_symbols],
        "point_id": int(tsm.point_id),
        "comma_ids": [int(x) for x in tsm.comma_ids],
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
        prepared, clean, has_text = part.prepare_text_input(line, None)
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
                "type_ids": type_ids_for(part, sequence, prepared, classify_text, classify_sentence)[0].tolist(),
                "homo_words": words,
                "homo_preds": preds.flatten().int().tolist() if preds is not None else [],
            }
        )
        texts.append(entry)
        if logits is not None:
            write_safetensors(OUT / f"golden/homo-{index}.safetensors", {"logits": logits})

        voices = VOICES if index < MULTI_VOICE_GOLDEN else ["baya"]
        for voice in voices:
            state = run_pipeline(part, tsm, prepared, voice, classify_text, classify_sentence)
            keys = ["type_ids", "dur", "dur_raw", "pitch", "pitch_raw", "mel", "audio"]
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

    # Chunking and Cyrillic filter come from the v5_ru exporter (same contract as export_silero.py).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_silero

    chunks = []
    for index, line in enumerate(corpus):
        split = export_silero.split_chunks(line)
        kept = [chunk for chunk in split if export_silero.CYRILLIC.search(chunk.lower())]
        chunks.append({"id": index, "chunks": split, "kept": kept})
    (OUT / "golden/chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({total / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
