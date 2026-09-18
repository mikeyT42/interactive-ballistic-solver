# ══════════════════════════════════════════════════════════════════════════════
#  Greek Letter Names
# ══════════════════════════════════════════════════════════════════════════════
# Δ : Uppercase Delta
# ψ : Lowercase Psi
# ρ : Lowercase Rho
# ϵ : Lowercase Epsilon
# θ : Lowercase Theta

# ══════════════════════════════════════════════════════════════════════════════
#  Physical Constants
# ══════════════════════════════════════════════════════════════════════════════
const g   = 9.81                       # gravitational acceleration     [m/s²]
const M   = 0.216                      # ball mass                        [kg]
const R   = 0.075                      # ball radius                      [m]
const A   = π * R^2                    # cross-sectional area             [m²]
const ρ   = 1.199                      # air density (20°C,101 kPa,50%RH)[kg/m³]
const C_D = 0.47                       # drag coefficient
const C_L = 0.031                      # lift (Magnus) coefficient

# ══════════════════════════════════════════════════════════════════════════════
#  Solver Tuning
# ══════════════════════════════════════════════════════════════════════════════
const N_SECANT = 5                     # max secant iterations
const Δt₀     = 0.01                   # integration time-step            [s]
const t_max   = 5.0                    # simulation time ceiling          [s]
const v_max   = 35.0                   # flywheel speed ceiling           [m/s]
const ε_h     = 0.02                   # height-error tolerance           [m]
