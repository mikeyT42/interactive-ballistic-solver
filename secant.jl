include("rk4.jl")

# vˣ = 2 dᶠ = 1.48

# ══════════════════════════════════════════════════════════════════════════════
#  Secant Method Root Finder
# ══════════════════════════════════════════════════════════════════════════════

function secant_root_find(dᶠ, Δz, cosθ, tanθ, cosϕ, sinϕ, vˣ, vʸ)
    # This number controls how long it will take to search the solutioin space.
    # Basically to say, this variable controls it's max root searches.
    N_SECANT = 10

    # 1 ── Vacuum initial guess ──
    num = g * dᶠ^2
    den = 2cosθ^2 * (dᶠ * tanθ - Δz)
    den = den ≤ 0.0 ? 0.001 : den             # guard NaN
    v̂ = √(num / den)
    m̂   = v̂ * cosθ                            # vacuum guess (m-hat)
    m̂   = clamp(m̂, 0.1, v̄)                    # keep seed in the valid domain

    # 2 ── Secant iteration on m ──
    m₀ = m̂
    m₁ = m̂ + 0.5
    h₀ = h_at_m(m₀, vˣ, vʸ, cosϕ, sinϕ, tanθ, dᶠ)
    h₁ = h_at_m(m₁, vˣ, vʸ, cosϕ, sinϕ, tanθ, dᶠ)

    mₛ = [m₀, m₁]                             # archive for viz

    converged = false
    hₙ₋₁ = h₀
    hₙ = h₁
    mₙ = m₁
    mₙ₋₁ = m₀
    for _ in 1:N_SECANT
        # ── Flat-slope guard ──
        # A near-zero secant slope only means convergence if h₁ is ALSO
        # within tolerance of the target — a flat region far from Δz is
        # not a solution.
        if abs(hₙ - hₙ₋₁) < 1e-4
            converged = abs(hₙ - Δz) < εᶻ; break
        end
        εₙ = hₙ - Δz
        εₙ₋₁ = hₙ₋₁ - Δz
        mₙ₊₁ = mₙ - εₙ * (mₙ - mₙ₋₁) / (εₙ - εₙ₋₁)
        # ── Clamp to a physically meaningful range ──
        # Without this, the secant step can go negative (ball travels
        # backward → simulate's stall guard loops pointlessly) or blow up
        # far past v̄.
        mₙ₊₁ = clamp(mₙ₊₁, 0.1, v̄)

        push!(mₛ, mₙ₊₁)

        mₙ₋₁, hₙ₋₁  = mₙ, hₙ
        mₙ          = mₙ₊₁
        hₙ          = h_at_m(mₙ, vˣ, vʸ, cosϕ, sinϕ, tanθ, dᶠ)

        if abs(hₙ - Δz) < 0.01
            converged = true; break
        end
    end

    return converged, mₛ, mₙ, hₙ, m̂
end

# -------
function h_at_m(m, vˣ, vʸ, cosϕ, sinϕ, tanθ, d)
    vₛˣ  = m * cosϕ - vˣ                # shooter velocity x  (field)
    vₛʸ  = m * sinϕ - vʸ                # shooter velocity y  (field)
    vₕ = hypot(vₛˣ, vₛʸ)                # horizontal speed, shooter frame
    vᶻ = vₕ * tanθ                      # vertical from fixed hood
    return rk4_simulate(m, vᶻ, d)
end
