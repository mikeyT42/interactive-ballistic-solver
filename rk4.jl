# ══════════════════════════════════════════════════════════════════════════════
#  RK4 Trajectory Integrator
# ══════════════════════════════════════════════════════════════════════════════
"""
    rk4_simulate(vₓ₀, vz₀, d; trace=false)

Integrate the 2-D ballistic arc with drag + Magnus until radial
distance reaches `d`.

Returns final height `z`.  With `trace=true` returns `(z, points)`.
"""
function rk4_simulate(vₓ₀, vz₀, d; trace=false)
    x,  z  = 0.0, 0.0
    vₓ, vz = Float64(vₓ₀), Float64(vz₀)
    Δt     = Δt₀
    t      = 0.0

    pts = trace ? Tuple{Float64,Float64}[(0.0, 0.0)] : nothing

    while x < d && t < t̄
        # ── Stall guard ──
        # If drag (+ Magnus on ascent) has bled vₓ to zero or negative,
        # the ball can never reach d — without this guard the trim step
        # below divides by zero/negative and the loop can misbehave.
        vₓ <= 0.0 && break

        # Trim the last step so x lands on d
        if x + vₓ * Δt > d
            Δt = (d - x) / vₓ
        end

        # ── RK4 stages ────────────────────────────
        kₓ₁ = aₓ(vₓ, vz)
        kz₁ = az(vₓ, vz)

        kₓ₂ = aₓ(vₓ + kₓ₁*Δt/2, vz + kz₁*Δt/2)
        kz₂ = az(vₓ + kₓ₁*Δt/2, vz + kz₁*Δt/2)

        kₓ₃ = aₓ(vₓ + kₓ₂*Δt/2, vz + kz₂*Δt/2)
        kz₃ = az(vₓ + kₓ₂*Δt/2, vz + kz₂*Δt/2)

        kₓ₄ = aₓ(vₓ + kₓ₃*Δt,   vz + kz₃*Δt  )
        kz₄ = az(vₓ + kₓ₃*Δt,   vz + kz₃*Δt  )

        # Update state with the weighted averages
        x  += vₓ * Δt + ((Δt^2)/6) * (kₓ₁ + kₓ₂ + kₓ₃)
        z  += vz * Δt + ((Δt^2)/6) * (kz₁ + kz₂ + kz₃)

        vₓ += Δt/6 * (kₓ₁ + 2kₓ₂ + 2kₓ₃ + kₓ₄)
        vz += Δt/6 * (kz₁ + 2kz₂ + 2kz₃ + kz₄)

        t  += Δt

        !isnothing(pts) && push!(pts, (x, z))
    end

    return trace ? (z, pts) : z
end

# ══════════════════════════════════════════════════════════════════════════════
#  Aerodynamic Acceleration  (2-D radial–vertical plane)
#
#    vₓ : horizontal (radial) velocity     [m/s]
#    vz : vertical velocity  (+up)         [m/s]
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
"Horizontal (radial) acceleration: drag + Magnus cross-term."
function aₓ(vₓ, vz)
    v = hypot(vₓ, vz)
    v == 0.0 && return 0.0
    Fd = -0.5 * ρ * A * C_D * v * vₓ
    Fl = -0.5 * ρ * A * C_L * v * vz

    return (Fd + Fl) / M
end

# ──────────────────────────────────────────────────────────────────────────────
"Vertical acceleration: gravity + drag + Magnus lift."
function az(vₓ, vz)
    v = hypot(vₓ, vz)
    v == 0.0 && return -g
    Fg = -M * g
    Fd = -0.5 * ρ * A * C_D * v * vz
    Fl =  0.5 * ρ * A * C_L * v * vₓ
    return (Fg + Fd + Fl) / M
end
