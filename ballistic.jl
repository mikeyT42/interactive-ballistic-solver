using GLMakie


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

# ══════════════════════════════════════════════════════════════════════════════
#  Main Ballistic Solver
#  (1:1 with Java  VelocityAngleSolver.calculate)
# ══════════════════════════════════════════════════════════════════════════════

"""
    calculate(d_floor, Δz, φᵣ_deg, ψ_deg, Vx, Vy, θ_deg)

Full ballistic solve with moving-reference-frame compensation.

| symbol  | meaning                                |
|---------|----------------------------------------|
| d_floor | radial floor distance to target  [m]   |
| Δz      | target vertical offset (height)  [m]   |
| φᵣ_deg  | target azimuth, **robot** frame  [°]   |
| ψ_deg   | robot heading (field, Pigeon 2)  [°]   |
| Vx, Vy  | field-centric robot velocity     [m/s] |
| θ_deg   | fixed hood launch angle          [°]   |

Returns `NamedTuple`:
  flywheel, turret_yaw, error, valid, trajectory, estimates, m_guess
"""
function calculate(d_floor, Δz, φᵣ_deg, ψ_deg, Vx, Vy, θ_deg)

    # ── Early bail-out ──
    if d_floor < 0.1
        return (flywheel   = 0.0,  turret_yaw = 0.0,
                error      = 999.0, valid      = false,
                trajectory = Tuple{Float64,Float64}[],
                estimates  = Vector{Tuple{Float64,Float64}}[],
                m_guess    = 0.0)
    end

    # 1 ── Angle conversion  (robot → field) ──
    φᵣ         = deg2rad(φᵣ_deg)
    ψ          = deg2rad(ψ_deg)
    φ          = φᵣ + ψ                       # field-centric azimuth
    cosφ, sinφ = cos(φ), sin(φ)

    θ    = deg2rad(θ_deg)
    cosθ = cos(θ)
    tanθ = tan(θ)

    # 2 ── Vacuum initial guess ──
    num = g * d_floor^2
    den = 2cosθ^2 * (d_floor * tanθ - Δz)
    den = den ≤ 0.0 ? 0.001 : den             # guard NaN
    v_w = √(num / den)
    m̂   = v_w * cosθ                          # vacuum guess (m-hat)

    # 3 ── Secant iteration on m ──
    m₀ = m̂
    m₁ = m̂ + 0.5
    h₀ = h_at_m(m₀, Vx, Vy, cosφ, sinφ, tanθ, d_floor)
    h₁ = h_at_m(m₁, Vx, Vy, cosφ, sinφ, tanθ, d_floor)

    ms = [m₀, m₁]                             # archive for viz

    converged = false
    for _ in 1:N_SECANT
        if abs(h₁ - h₀) < 1e-4
            converged = true; break
        end
        ε₁ = h₁ - Δz
        ε₀ = h₀ - Δz
        m_new = m₁ - ε₁ * (m₁ - m₀) / (ε₁ - ε₀)
        push!(ms, m_new)

        m₀, h₀ = m₁, h₁
        m₁      = m_new
        h₁      = h_at_m(m₁, Vx, Vy, cosφ, sinφ, tanθ, d_floor)

        if abs(h₁ - Δz) < 0.01
            converged = true; break
        end
    end

    # 4 ── Final flywheel speed ──
    m_f = m₁
    sx  = m_f * cosφ - Vx
    sy  = m_f * sinφ - Vy
    v_h = hypot(sx, sy)
    flywheel = cosθ > 1e-3 ? v_h / cosθ : 0.0

    # 5 ── Turret yaw  (shot leading removed — pure direction) ──
    yaw = rad2deg(φ) - ψ_deg
    yaw = mod(yaw + 180.0, 360.0) - 180.0     # → [−180, 180]

    # 6 ── Validity  (Stage-1 math/geometry) ──
    sim_err = abs(h₁ - Δz)
    valid   = converged && sim_err ≤ ε_h && flywheel > 0 && flywheel ≤ v_max

    # 7 ── Trajectories for plot ──
    estimates = Vector{Vector{Tuple{Float64,Float64}}}()
    for m in ms[1:end-1]
        m > 0 || continue
        ex  = m * cosφ - Vx
        ey  = m * sinφ - Vy
        evz = hypot(ex, ey) * tanθ
        _, pts = simulate(m, evz, d_floor; trace=true)
        push!(estimates, pts)
    end

    _, final_pts = simulate(m_f, v_h * tanθ, d_floor; trace=true)

    return (flywheel   = flywheel,
            turret_yaw = yaw,
            error      = sim_err,
            valid      = valid,
            trajectory = final_pts,
            estimates  = estimates,
            m_guess    = m̂)
end

# ══════════════════════════════════════════════════════════════════════════════
#  Aerodynamic Acceleration  (2-D radial–vertical plane)
#
#    vₓ : horizontal (radial) velocity     [m/s]
#    vz : vertical velocity  (+up)         [m/s]
# ══════════════════════════════════════════════════════════════════════════════

"Horizontal (radial) acceleration: drag + Magnus cross-term."
function aₓ(vₓ, vz)
    v = hypot(vₓ, vz)
    v == 0.0 && return 0.0
    Fd = -0.5 * ρ * A * C_D * v * vₓ
    Fl = -0.5 * ρ * A * C_L * v * vz

    return (Fd + Fl) / M
end

"Vertical acceleration: gravity + drag + Magnus lift."
function az(vₓ, vz)
    v = hypot(vₓ, vz)
    v == 0.0 && return -g
    Fg = -M * g
    Fd = -0.5 * ρ * A * C_D * v * vz
    Fl =  0.5 * ρ * A * C_L * v * vₓ
    return (Fg + Fd + Fl) / M
end

# ══════════════════════════════════════════════════════════════════════════════
#  RK4 Trajectory Integrator
# ══════════════════════════════════════════════════════════════════════════════

"""
    simulate(vₓ₀, vz₀, d; trace=false)

Integrate the 2-D ballistic arc with drag + Magnus until radial
distance reaches `d`.

Returns final height `z`.  With `trace=true` returns `(z, points)`.
"""
function simulate(vₓ₀, vz₀, d; trace=false)
    x,  z  = 0.0, 0.0
    vₓ, vz = Float64(vₓ₀), Float64(vz₀)
    Δt     = Δt₀
    t      = 0.0

    pts = trace ? Tuple{Float64,Float64}[(0.0, 0.0)] : nothing

    while x < d && t < t_max
        # Trim the last step so x lands on d
        if x + vₓ * Δt > d
            Δt = (d - x) / vₓ
        end

        # ── RK4 stages ────────────────────────────
        kₓ₁ = aₓ(vₓ,             vz            )
        kz₁ = az(vₓ,             vz            )

        kₓ₂ = aₓ(vₓ + kₓ₁*Δt/2, vz + kz₁*Δt/2)
        kz₂ = az(vₓ + kₓ₁*Δt/2, vz + kz₁*Δt/2)

        kₓ₃ = aₓ(vₓ + kₓ₂*Δt/2, vz + kz₂*Δt/2)
        kz₃ = az(vₓ + kₓ₂*Δt/2, vz + kz₂*Δt/2)

        kₓ₄ = aₓ(vₓ + kₓ₃*Δt,   vz + kz₃*Δt  )
        kz₄ = az(vₓ + kₓ₃*Δt,   vz + kz₃*Δt  )

        # Weighted average
        āₓ = (kₓ₁ + 2kₓ₂ + 2kₓ₃ + kₓ₄) / 6
        āz = (kz₁ + 2kz₂ + 2kz₃ + kz₄) / 6

        # Update state
        x  += vₓ * Δt + 0.5 * āₓ * Δt^2
        z  += vz * Δt + 0.5 * āz * Δt^2
        vₓ += āₓ * Δt
        vz += āz * Δt
        t  += Δt

        !isnothing(pts) && push!(pts, (x, z))
    end

    return trace ? (z, pts) : z
end

# ══════════════════════════════════════════════════════════════════════════════
#  Fixed-Hood Height Helper  (called inside secant loop)
# ══════════════════════════════════════════════════════════════════════════════

"""
Height at target distance `d` for world horizontal speed `m`,
subtracting robot velocity and deriving vz from the fixed hood angle.
"""
function h_at_m(m, Vx, Vy, cosφ, sinφ, tanθ, d)
    sx  = m * cosφ - Vx                # shooter velocity x  (field)
    sy  = m * sinφ - Vy                # shooter velocity y  (field)
    v_h = hypot(sx, sy)                # horizontal speed, shooter frame
    v_z = v_h * tanθ                   # vertical from fixed hood
    return simulate(m, v_z, d)
end

# ══════════════════════════════════════════════════════════════════════════════
#  Interactive GLMakie Visualisation  (3-D field view)
# ══════════════════════════════════════════════════════════════════════════════
#
#  World layout:
#    • Target sits fixed at the field-frame origin, height Δz.
#    • Robot position is placed d_floor away from the target, back along the
#      field-centric azimuth φ = φᵣ + ψ (so "azimuth 0, heading 0" places the
#      robot on the −X side of the target, aiming in +X).
#    • Trajectory points returned by `calculate` are (radial, height) pairs in
#      the shooter's own aiming plane — they get rotated into field X/Y by the
#      same cosφ, sinφ used inside `calculate` itself.
#    • Two arrows drawn from the robot marker show its field-centric Vx and Vy
#      velocity components; both grow/shrink with magnitude and (for Vx) swing
#      direction if the sign flips.
# ══════════════════════════════════════════════════════════════════════════════

function interactive_solver()
    fig = Figure(size = (1100, 820))

    ax = Axis3(fig[1, 1];
        title  = "Ballistic Trajectory  (3-D Field View)",
        xlabel = "Field X  [m]",
        ylabel = "Field Y  [m]",
        zlabel = "Height  [m]")
    xlims!(ax, -6, 6)
    ylims!(ax, -6, 6)
    zlims!(ax, 0, 8)

    sg = SliderGrid(fig[2, 1],
        (label = "d_floor  [m]",
         range = 0.5:0.01:3.0,       startvalue = 1.0),
        (label = "Δz  (height)  [m]",
         range = 0.0:0.05:2.0,       startvalue = 1.83),
        (label = "φ_robot  (azimuth)  [°]",
         range = -180.0:1.0:180.0,   startvalue = 0.0),
        (label = "ψ  (heading)  [°]",
         range = -180.0:1.0:180.0,   startvalue = 0.0),
        (label = "Vₓ  robot (field)  [m/s]",
         range = -3.0:0.1:3.0,       startvalue = 0.0),
        (label = "Vy  robot (field)  [m/s]",
         range = -3.0:0.1:3.0,       startvalue = 0.0),
        (label = "θ  (launch angle)  [°]",
         range = 60.0:1.0:80.0,     startvalue = 80.0),
    )

    info = Observable("—")
    Label(fig[3, 1], info; tellwidth = false)

    sl = sg.sliders
    palette = [:red, :orange, :gold, :green, :cyan, :purple]

    # Scale factor turning m/s of robot velocity into a visible arrow length
    arrow_scale = 0.6

    onany(sl[1].value, sl[2].value, sl[3].value,
          sl[4].value, sl[5].value, sl[6].value, sl[7].value
    ) do d, dz, φr_deg, ψ_deg, vx, vy, θ

        empty!(ax)

        sol = calculate(d, dz, φr_deg, ψ_deg, vx, vy, θ)

        # Field-centric azimuth and robot position (target fixed at origin)
        φ          = deg2rad(φr_deg + ψ_deg)
        cosφ, sinφ = cos(φ), sin(φ)
        robot_x    = -d * cosφ
        robot_y    = -d * sinφ

        "Rotate a shooter-frame (radial, height) trace into field X/Y/Z."
        rotate_trace(pts) = (
            [robot_x + p[1] * cosφ for p in pts],
            [robot_y + p[1] * sinφ for p in pts],
            [p[2] for p in pts]
        )

        # Secant-estimate arcs  (dashed)
        for (i, pts) in enumerate(sol.estimates)
            isempty(pts) && continue
            xs, ys, zs = rotate_trace(pts)
            lines!(ax, xs, ys, zs;
                   color     = palette[mod1(i, length(palette))],
                   linewidth = 1.2,
                   alpha     = 0.6,
                   linestyle = :dash)
        end

        # Converged trajectory  (solid blue)
        if !isempty(sol.trajectory)
            xs, ys, zs = rotate_trace(sol.trajectory)
            lines!(ax, xs, ys, zs; color = :blue, linewidth = 4)
        end

        # Target marker
        scatter!(ax, [0.0], [0.0], [dz]; color = :red, markersize = 20)

        # Robot marker
        scatter!(ax, [robot_x], [robot_y], [0.0]; color = :black,
                 markersize = 18)

        # ── Robot velocity-vector arrows (Vx, Vy) ──
        # Vx drawn along field X, Vy drawn along field Y. Each arrow starts
        # just outside the robot marker (offset along its own direction) so
        # it doesn't overlap/obstruct the dot, and grows/shrinks/flips with
        # the sliders since it's recomputed from vx, vy every callback.
        stand_off = 0.12   # gap between marker edge and arrow start [m]

        vx_dir = Vec3f(sign(vx) == 0 ? 1.0 : sign(vx), 0.0, 0.0)
        vy_dir = Vec3f(0.0, sign(vy) == 0 ? 1.0 : sign(vy), 0.0)

        arrow_origins = [Point3f(robot_x, robot_y, 0.05) + stand_off * vx_dir,
                          Point3f(robot_x, robot_y, 0.05) + stand_off * vy_dir]
        arrow_dirs    = [Vec3f(vx * arrow_scale, 0.0, 0.0),
                          Vec3f(0.0, vy * arrow_scale, 0.0)]

        arrows3d!(ax, arrow_origins, arrow_dirs;
            color      = [:orange, :purple],
            shaftradius = 0.05,
            tipradius   = 0.1,
            tiplength   = 0.2)

        # ── Heading (ψ) and Azimuth (φ) direction indicators ──
        # These show *direction only*, not magnitude, so both arrows are
        # drawn at a fixed length regardless of slider values. Raised in Z
        # above the Vx/Vy arrows so all four don't visually collide.
        angle_len   = 0.8   # fixed arrow length for direction indicators [m]
        angle_z     = 0.35  # height above ground for these arrows [m]

        ψ_rad = deg2rad(ψ_deg)
        ψ_dir = Vec3f(cos(ψ_rad) * angle_len, sin(ψ_rad) * angle_len, 0.0)
        φ_dir = Vec3f(cosφ * angle_len, sinφ * angle_len, 0.0)

        angle_origins = [Point3f(robot_x, robot_y, angle_z),
                          Point3f(robot_x, robot_y, angle_z)]
        angle_dirs    = [ψ_dir, φ_dir]

        arrows3d!(ax, angle_origins, angle_dirs;
            color       = [:dodgerblue, :hotpink],
            shaftradius = 0.05,
            tipradius   = 0.1,
            tiplength   = 0.2)

        # Info bar
        tag = sol.valid ? "✓ VALID" : "✗ INVALID"
        info[] =
            "Flywheel: $(round(sol.flywheel; digits=2)) m/s  │  " *
            "Turret Yaw: $(round(sol.turret_yaw; digits=2))°  │  " *
            "Sim Error: $(round(sol.error; digits=3)) m  │  $tag\n" *
            "m̂ (vacuum): $(round(sol.m_guess; digits=2))  │  " *
            "φ_field: $(round(φr_deg + ψ_deg; digits=1))°  │  " *
            "Robot V = ($(round(vx; digits=2)), $(round(vy; digits=2))) m/s\n" *
            "Arrows — orange: Vₓ  │  purple: Vy  │  " *
            "dodgerblue: heading ψ  │  hotpink: azimuth φ (→ target)"
    end

    notify(sl[1].value)          # trigger first render
    display(fig)
end

# ── Launch ──
interactive_solver()
