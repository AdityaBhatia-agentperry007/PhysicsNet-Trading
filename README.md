# PhysicsNet-Trading

> physics-informed deep learning for Bitcoin price prediction · built with [Moses Lua](https://github.com/moseslua)

## overview

sub-millisecond Bitcoin price prediction pipeline modelling order book as a **stochastic non-Newtonian fluid**. deployed across 5 venues simultaneously.

not a transformer. not a basic LSTM. a 3-stage hybrid that treats market microstructure like a physics simulation.

## architecture

```
RNN → GNN → TFT
 |      |      |
 |      |      └── multi-horizon quantile forecasts (1s-15min)
 |      └── 250+ node heterogeneous graph (Binance/Coinbase/Kraken/Deribit/CME)
 └── SPDE Navier-Stokes fluid dynamics + Ensemble Kalman Filter
```

## what makes this different

- SPDE Navier-Stokes fluid dynamics + deep learning — velocity, pressure, viscosity fields via Ensemble Kalman Filter
- heterogeneous graph: 250+ nodes capturing cross-venue arbitrage and derivatives lead-lag
- Cond-GRU temporal encoder with semimartingale-consistent regularisation
- TFT multi-horizon quantile forecasts with regime-aware blending via Hurst exponent
- critical path: ~100μs p99 (Rust EnKF → ONNX RNN → PyTorch Geometric GNN → TensorRT TFT)

## results

- deployed live across 5 venues
- per-tick reactivity on live exchange WebSocket
- regime-aware: H < 0.5 (mean-reversion), H > 0.5 (trending)

```
stack: Python · Rust · PyTorch · TensorRT · ONNX · TorchScript
       Kafka · TimescaleDB · FastAPI · Docker
```

## collaborators
- [Moses Lua Jian Hao](https://github.com/moseslua) — lead researcher, architecture design
- Aditya Bhatia — pipeline integration, signal validation, deployment infrastructure

