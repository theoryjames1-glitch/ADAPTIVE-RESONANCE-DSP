# 🔷 Adaptive Resonance Optimization (ARO)

> **Learning law for optimization**:  
> ARO = **Gradient descent + Adaptive Resonance dynamics**.  
> It extends optimizers (SGD/AdamW) with **resonant attractors** and **vigilance-gated memory**, yielding stable but plastic learning.

---

## 📖 Overview

Traditional optimizers (SGD, Adam, AdamW) treat learning as **filtered gradient descent**:
- Parameters `θ` updated by gradients `g`.
- Momentum and variance tracking = **1st-order IIR filters**.
- Great for smoothing noise, but **forgetful**: no structural memory, risk of catastrophic forgetting.

**Adaptive Resonance Theory (ART)**, from Grossberg, defines learning laws as **differential equations**:
- Prototypes `w_j` evolve toward matched hidden states.
- Hidden states `h` resonate with prototypes.
- Vigilance `ρ` gates whether to update or allocate new memory.
- Balances **stability** (don’t overwrite) and **plasticity** (learn new).

**ARO unifies these two views**:  
👉 Error-driven gradient descent **plus** resonance-driven prototype dynamics.

---

## ⚙️ System Model

State:
```

x\_t = \[ θ\_t, h\_t, α\_t, μ\_t, σ\_t, W\_t ]

```
- `θ_t` – plant parameters (DSP knobs, NN weights, etc.)
- `h_t` – optimizer buffers (e.g. momentum)
- `α_t, μ_t` – adaptive hyperparameters (LR, momentum)
- `σ_t` – dither strength
- `W_t = { w_j }` – ART prototypes (memory traces)

Feedback:
```

y\_t = \[ ℓ\_t, r\_t ]

```
- `ℓ_t` – loss  
- `r_t` – optional reward  

---

## 🔹 Learning Laws

### 1. Error-driven path (AdamW-style)
```

m\_t = μ m\_{t-1} + (1-μ) g\_t      # momentum = 1st-order IIR
u\_t = α · precondition(m\_t)      # control action
θ\_{t+1} = θ\_t - u\_t + σ D(θ)η\_t  # update + shaped dither

```

### 2. Resonance-driven path (ART-style)
```

h*\_{t+1} = h*\_t + λ (w\_j - h*\_t)              # hidden state pulled to prototype
w\_j ← w\_j + η ((h* ∧ w\_j) - w\_j)              # prototype update law
M\_j = |h\* ∧ w\_j| / |h\*| ≥ ρ  ?  update : new  # vigilance gate

```

### 3. Coupling (two choices)
- **Regularizer**:
```

L\_total = L\_task + λ\_ART ||h\* - w\_j||^2

```
- **Feedback correction**:
```

θ\_{t+1} = θ\_t - u\_t + β (h*\_{t+1} - h*\_t)

````

---

## 🎛 Application to DSP Plants

- **Plant**: differentiable DSP chain (oscillators, filters, envelopes, effects).  
- **Embedding `h*`**: learned features (STFT/mel, small encoder).  
- **Prototypes `w_j`**: represent stable timbral/feature regimes.  
- **Vigilance `ρ`**: spawns new prototypes for novel sounds.  
- **Coupling**: resonance loss aligns features with prototypes; error loss drives task learning.  

---

## 🛠 Design Rules

- Clamp gains: `α ∈ [α_min, α_max]`, `μ ∈ [0,1)`, `σ ≪ 1`.  
- Dampen LR when loss variance is high.  
- Use small ART weight (`λ_ART ~ 1e-4`) in regression.  
- Prototype LR `η ~ 1e-3 … 1e-2`.  
- Vigilance `ρ ∈ [0.7, 0.9]` → controls memory granularity.  

---

## 📊 Comparison

| Aspect              | AdamW                             | ARO (AdamW + ART)                    |
|---------------------|-----------------------------------|--------------------------------------|
| Gradient smoothing  | EMA of g, EMA of g²               | Same                                 |
| Step-size scaling   | Variance-normalized               | Same (inherits AdamW)                |
| Weight decay        | Explicit, decoupled               | Can add, not core                    |
| Memory              | Buffers only (m,v)                | Explicit prototypes `w_j`            |
| Nonlinearity        | Smooth (all filters)              | Vigilance = switching nonlinearity   |
| Stability           | By normalization/damping          | By prototypes as attractors          |
| Plasticity          | LR scheduling only                | Vigilance-spawned new prototypes     |

---

## ✅ Why It Matters

- **Stability–Plasticity balance**: prototypes anchor learned states, vigilance handles novelty.  
- **Continual learning**: no catastrophic forgetting; optimizer itself preserves structure.  
- **Control-theoretic grounding**: behaves like a 2-pole system (momentum + resonance).  
- **General**: wraps around *any plant* (DSP chain, NN, controller).  

---

## 🚀 Quick Start (PyTorch)

```python
model = MyDSPPlant()
opt = AdamWAdaptiveResonance(
  model.parameters(),
  lr=1e-3, betas=(0.9,0.999), weight_decay=1e-2,
  resonance_weight=1e-3, vigilance=0.8, proto_lr=5e-3
)

for x, y in data:
  y_pred, h = model(x)
  loss = mse(y_pred, y) + opt.resonance_weight * opt.resonance_loss(h)
  loss.backward()
  opt.step(h_star=h)
  opt.zero_grad()
````

---

## 📖 Citation-style Thesis

> **Adaptive Resonance Optimization (ARO)** augments gradient descent with resonance dynamics, turning the optimizer into a **2nd-order resonant control system**. Prototypes serve as explicit attractors in representation space, and vigilance regulates stability–plasticity. This reframes optimization as not just error minimization, but as a **dynamical system with memory**.

---

```

