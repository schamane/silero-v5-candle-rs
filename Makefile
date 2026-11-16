# Wiederholbare Prüfungen. `make lint-rust` ist der Torwächter, den die
# README-Befehle verwenden.
.PHONY: fmt lint-rust check-docs check-linux test test-debug test-release test-release-cpu test-release-accelerate check

fmt:
	cargo fmt --all

lint-rust:
	cargo clippy --locked --workspace --all-targets --features test-util -- -D warnings
	cargo clippy --locked --workspace --all-targets --no-default-features --features test-util -- -D warnings
	cargo clippy --locked --workspace --all-targets --no-default-features --features accelerate,test-util -- -D warnings

# Lib-Cross-Check gegen Linux, nur wenn das Ziel installiert ist.
check-linux:
	@if rustup target list --installed | rg -qx x86_64-unknown-linux-gnu; then \
		cargo check --lib --locked --target x86_64-unknown-linux-gnu; \
	else \
		echo "SKIP: Ziel x86_64-unknown-linux-gnu nicht installiert"; \
	fi

# Goldens brauchen --release (im Debug-Build übersprungen); daher ist das der Standard.
test:
	cargo test --workspace --all-targets --release

# Schneller Debug-Lauf; Golden-Tests werden dabei übersprungen.
test-debug:
	cargo test --workspace --all-targets

test-release:
	cargo test --release --workspace --all-targets

test-release-cpu:
	cargo test --release --workspace --all-targets --no-default-features

test-release-accelerate:
	cargo test --release --workspace --all-targets --no-default-features --features accelerate

check-docs:
	./scripts/check_docs.sh

check: lint-rust check-docs check-linux test-release test-release-cpu test-release-accelerate
