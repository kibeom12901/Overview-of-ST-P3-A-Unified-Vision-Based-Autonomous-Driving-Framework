### Prediction Combination:

- **Equation**: 
  \[
  \hat{x}_{t+1} = \mathcal{G}(x_t, \eta_t) \oplus \mathcal{G}(x_{0:t})
  \]
  - Here, \(\mathcal{G}\) represents the GRU process, and \(\oplus\) denotes the combination of these predictions.

- **Future State Predictions**: This combined prediction serves as the base for future state predictions (up to \(H\) horizons).
