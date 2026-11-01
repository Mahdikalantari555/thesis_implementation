# ssl_anomaly_rs (Gate G3)

Rust acceleration for `ssl_anomaly`. **Not compiled in v1** (gate G3).

## Build (in WSL or Windows with Rust toolchain)

```bash
cd rust
cargo build --release
maturin develop  # requires maturin
```

## Parity
`tests/parity/test_backends_parity.py` compares Rust vs NumPy outputs (rtol=1e-6).
No Rust code runs before gate G3 passes profiling.