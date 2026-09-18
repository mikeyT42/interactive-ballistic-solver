include("constants.jl")

function secant_integration(d_floor::Float16, Δz::Float16, cosθ::Float32,
        tanθ::Float32)
    # 2 ── Vacuum initial guess ──
    num = g * d_floor^2
    den = 2cosθ^2 * (d_floor * tanθ - Δz)
    den = den ≤ 0.0 ? 0.001 : den             # guard NaN
    v_w = √(num / den)
    m̂   = v_w * cosθ                          # vacuum guess (m-hat)
    m̂   = clamp(m̂, 0.1, v_max)                # keep seed in the valid domain

    # 3 ── Secant iteration on m ──
    m₀ = m̂
    m₁ = m̂ + 0.5
    h₀ = h_at_m(m₀, Vx, Vy, cosφ, sinφ, tanθ, d_floor)
    h₁ = h_at_m(m₁, Vx, Vy, cosφ, sinφ, tanθ, d_floor)

    ms = [m₀, m₁]                             # archive for viz

    converged = false
    for _ in 1:N_SECANT
        # ── Flat-slope guard ──
        # A near-zero secant slope only means convergence if h₁ is ALSO
        # within tolerance of the target — a flat region far from Δz is
        # not a solution.
        if abs(h₁ - h₀) < 1e-4
            converged = abs(h₁ - Δz) < ε_h; break
        end
        ε₁ = h₁ - Δz
        ε₀ = h₀ - Δz
        m_new = m₁ - ε₁ * (m₁ - m₀) / (ε₁ - ε₀)
        # ── Clamp to a physically meaningful range ──
        # Without this, the secant step can go negative (ball travels
        # backward → simulate's stall guard loops pointlessly) or blow up
        # far past v_max.
        m_new = clamp(m_new, 0.1, v_max)

        push!(ms, m_new)

        m₀, h₀ = m₁, h₁
        m₁      = m_new
        h₁      = h_at_m(m₁, Vx, Vy, cosφ, sinφ, tanθ, d_floor)

        if abs(h₁ - Δz) < 0.01
            converged = true; break
        end
    end

    return converged, ms
end
