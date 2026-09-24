# ══════════════════════════════════════════════════════════════════════════════
#  Ballistic Solver
# ══════════════════════════════════════════════════════════════════════════════

"""
    calculate(dᶠ, Δz, ϕᵣ°, ψ°, vˣ, vʸ, θ°)

Full ballistic solve with moving-reference-frame compensation.

| symbol  | meaning                                |
|---------|----------------------------------------|
| dᶠ      | radial floor distance to target  [m]   |
| Δz      | target vertical offset (height)  [m]   |
| ϕᵣ°     | target azimuth, **robot** frame  [°]   |
| ψ°      | robot heading (field, Pigeon 2)  [°]   |
| vˣ, vʸ  | field-centric robot velocity     [m/s] |
| θ°      | fixed hood launch angle          [°]   |

Returns `NamedTuple`:
  flywheel, turret_yaw, error, valid, trajectory, estimates, m_guess

`turret_yaw` is lead-compensated: it points along the muzzle velocity
(ground-frame target speed minus robot velocity), robot-relative, in
[−180, 180]. When the robot is stationary it reduces to the plain target azimuth
ϕᵣ.
"""
function calculate(dᶠ, Δz, ϕᵣ°, ψ°, vˣ, vʸ, θ°)
    # ── Early bail-out ──
    if dᶠ < 0.1
        return (flywheel   = 0.0,  turret_yaw = 0.0,
                error      = 999.0, valid      = false,
                trajectory = Tuple{Float64,Float64}[],
                estimates  = Vector{Tuple{Float64,Float64}}[],
                m_guess    = 0.0)
    end

    # 1 ── Angle conversion  (robot → field) ──
    ϕᵣ         = deg2rad(ϕᵣ°)
    ψ          = deg2rad(ψ°)
    
    ϕ          = ϕᵣ + ψ     # field-centric azimuth
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)

    θ    = deg2rad(θ°)
    cosθ = cos(θ)
    tanθ = tan(θ)

    # 2 ── Simulate ──
    converged, mₛ, mᶠ, zᶠ, m̂ = secant_root_find(dᶠ, Δz, cosθ, tanθ, cosϕ, sinϕ,
                                               vˣ, vʸ)

    # 4 ── Final flywheel speed ──
    vₛˣ  = mᶠ * cosϕ - vˣ
    vₛʸ  = mᶠ * sinϕ - vʸ
    vₛᶻ  = hypot(vₛˣ, vₛʸ)
    flywheel = cosθ > 1e-3 ? vₛᶻ / cosθ : 0.0

    # 5 ── Turret yaw (shot leading: aim along the muzzle velocity) ──
    yaw = turret_yaw(vₛˣ, vₛʸ, ψ°)

    # 6 ── Validity (Stage-1 math/geometry) ──
    sim_err = abs(zᶠ - Δz)
    valid   = converged && sim_err ≤ εᶻ && flywheel > 0 && flywheel ≤ v̄

    # 7 ── Trajectories for plot ──
    estimates, final_pts = plot_trajectories(mₛ, cosϕ, sinϕ, tanθ, vˣ, vʸ, vₛᶻ,
                                             dᶠ, mᶠ)

    return (flywheel   = flywheel,
            turret_yaw = yaw,
            error      = sim_err,
            valid      = valid,
            trajectory = final_pts,
            estimates  = estimates,
            m_guess    = m̂)
end

# ──────────────────────────────────────────────────────────────────────────────
function turret_yaw(vₛˣ, vₛʸ, ψ°)
    # The muzzle must fire along (vₛˣ, vₛʸ) so that after adding the robot's
    # own velocity, the ball's ground-frame path points at the target.
    # atan gives a FIELD-frame angle; subtract heading for the turret PID.
    fieldYaw° = rad2deg(atan(vₛʸ, vₛˣ))
    yaw = fieldYaw° - ψ°
    yaw = mod(yaw + 180.0, 360.0) - 180.0     # → [−180, 180]
end

