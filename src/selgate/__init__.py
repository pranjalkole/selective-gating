"""selgate — Illumination-Aware Selective Gating for Low-Light Image Enhancement.

x_out = g ⊙ F_theta(x) + (1 - g) ⊙ x

where F_theta is Retinexformer and g is a cheap, interpretable pixel-wise gate
driven by a hand-crafted illumination estimate L(x).
"""

__version__ = "0.1.0"
