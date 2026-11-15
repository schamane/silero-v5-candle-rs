# Wiederholbare Prüfungen. `make lint-rust` ist der Torwächter, den die
# README-Befehle verwenden.
.PHONY: fmt lint-rust test test-release test-release-cpu test-release-accelerate check

fmt:
	cargo fmt --all

lint-rust:
	cargo clippy --workspace --all-targets --features test-util -- -D warnings
	cargo clippy --workspace --all-targets --no-default-features --features test-util -- -D warnings
	cargo clippy --workspace --all-targets --no-default-features --features accelerate,test-util -- -D warnings

test:
	cargo test --workspace --all-targets

test-release:
	cargo test --release --workspace --all-targets

test-release-cpu:
	cargo test --release --workspace --all-targets --no-default-features

test-release-accelerate:
	cargo test --release --workspace --all-targets --no-default-features --features accelerate

check: lint-rust test-release test-release-cpu test-release-accelerate
