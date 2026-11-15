#!/usr/bin/env bash
# Erzeugt die WAV-Dateien für listening/index.html.
# Die WAVs sind wegen `*.wav` nicht im Git-Index.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p listening/audio

cargo build --release --bin silero
cargo build --release --no-default-features --features accelerate \
    --target-dir target/accel-release --bin silero

texts=(
    "Привет, как дела?"
    "Замок стоит на холме, а рядом течёт река."
    "Я купил муку и испёк хлеб."
    "Все пришли вовремя, но всё равно опоздали."
    "Сегодня хорошая погода, и мы гуляем по парку."
)

for index in "${!texts[@]}"; do
    id=$(printf 't%02d' "$((index + 1))")
    text=${texts[$index]}
    printf '== %s: %s\n' "$id" "$text"
    target/release/silero --device metal --model models/silero-v5-ru --voice baya \
        --out "listening/audio/$id-metal.wav" "$text"
    target/release/silero --device cpu --model models/silero-v5-ru --voice baya \
        --out "listening/audio/$id-cpu.wav" "$text"
    target/accel-release/release/silero --device cpu --model models/silero-v5-ru --voice baya \
        --out "listening/audio/$id-accel.wav" "$text"
done

printf '\nFertig. Öffne listening/index.html im Browser.\n'
