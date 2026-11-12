"""Stateful causal streaming model with a fixed K-frame lookahead (research_20261027).

Unlike the windowed MobilePoser cascade (bidirectional LSTMs over a [t-80, t+fut] window), this model carries one
recurrent state over the whole sequence (unbounded causal history) and emits the pose of frame t when the IMU of
frame t+K arrives (K=5 at 60 fps = 83 ms).  Inputs are expressed relative to the pelvis sensor (PNP-style).
"""
