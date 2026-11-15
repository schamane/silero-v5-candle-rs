# Wiederholbare Prüfungen. `make lint-rust` ist der Torwächter, den die
# README-Befehle verwenden.
.PHONY: fmt lint-rust test check

fmt:
	cargo fmt --all

lint-rust:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-targets

check: lint-rust test
