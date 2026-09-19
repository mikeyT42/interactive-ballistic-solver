using GLMakie

include("constants.jl")
include("ShotResult.jl")
include("secant.jl")

# ══════════════════════════════════════════════════════════════════════════════
#  Greek Letter Names
# ══════════════════════════════════════════════════════════════════════════════
# Δ : Uppercase Delta
# ψ : Lowercase Psi
# ρ : Lowercase Rho
# ϵ : Lowercase Epsilon
# θ : Lowercase Theta

# ══════════════════════════════════════════════════════════════════════════════
#  Main Ballistic Solver
#  (1:1 with Java  VelocityAngleSolver.calculate)
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
    
    ϕ          = ϕᵣ + ψ                       # field-centric azimuth
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)

    θ    = deg2rad(θ°)
    cosθ = cos(θ)
    tanθ = tan(θ)

    converged, mₛ, m₁, h₁, m̂ = secant_root_find(dᶠ, Δz, cosθ, tanθ, cosϕ, sinϕ,
                                               vˣ, vʸ)

    # 4 ── Final flywheel speed ──
    mᶠ   = m₁
    vₛˣ  = mᶠ * cosϕ - vˣ
    vₛʸ  = mᶠ * sinϕ - vʸ
    vʰ   = hypot(vₛˣ, vₛʸ)
    flywheel = cosθ > 1e-3 ? vʰ / cosθ : 0.0

    # 5 ── Turret yaw  (shot leading removed — pure direction) ──
    yaw = rad2deg(ϕ) - ψ°
    yaw = mod(yaw + 180.0, 360.0) - 180.0     # → [−180, 180]

    # 6 ── Validity  (Stage-1 math/geometry) ──
    sim_err = abs(h₁ - Δz)
    valid   = converged && sim_err ≤ εᶻ && flywheel > 0 && flywheel ≤ v̄

    # 7 ── Trajectories for plot ──
    estimates = Vector{Vector{Tuple{Float64,Float64}}}()
    for m in mₛ[1:end-1]
        ex  = m * cosϕ - vˣ
        ey  = m * sinϕ - vʸ
        evz = hypot(ex, ey) * tanθ
        _, pts = simulate(m, evz, dᶠ; trace=true)
        push!(estimates, pts)
    end

    _, final_pts = simulate(mᶠ, vʰ * tanθ, dᶠ; trace=true)

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
#  Interactive GLMakie Visualisation  (3-D field view)
# ══════════════════════════════════════════════════════════════════════════════

function interactive_solver()
    set_theme!(theme_dark())
    fig = Figure(size = (1100, 820), backgroundcolor = :grey10)

    ax = Axis3(fig[1, 1];
        title  = "Ballistic Trajectory  (3-D Field View)",
        xlabel = "Field X  [m]",
        ylabel = "Field Y  [m]",
        zlabel = "Height  [m]",
        backgroundcolor = :grey10,
        xspinecolor_1 = :grey50, xspinecolor_2 = :grey50, xspinecolor_3 = :grey50,
        yspinecolor_1 = :grey50, yspinecolor_2 = :grey50, yspinecolor_3 = :grey50,
        zspinecolor_1 = :grey50, zspinecolor_2 = :grey50, zspinecolor_3 = :grey50,
        xgridcolor = (:white, 0.12), ygridcolor = (:white, 0.12),
        zgridcolor = (:white, 0.12))
    xlims!(ax, -6, 6)
    ylims!(ax, -6, 6)
    zlims!(ax, 0, 8)

    sg = SliderGrid(fig[2, 1],
        (label = "dᶠ [m]",
         range = 0.5:0.01:3.0,       startvalue = 1.0),
        (label = "Δz  (height)  [m]",
         range = 0.0:0.05:2.0,       startvalue = 1.83),
        (label = "φ_robot  (azimuth)  [°]",
         range = -180.0:1.0:180.0,   startvalue = 0.0),
        (label = "ψ  (heading)  [°]",
         range = -180.0:1.0:180.0,   startvalue = 0.0),
        (label = "vˣ  robot (field)  [m/s]",
         range = -3.0:0.1:3.0,       startvalue = 0.0),
        (label = "vʸ  robot (field)  [m/s]",
         range = -3.0:0.1:3.0,       startvalue = 0.0),
        (label = "θ  (launch angle)  [°]",
         range = 60.0:1.0:80.0,     startvalue = 80.0),
    )

    # ── Slider colors ──
    # color_inactive : the unfilled track
    # color_active   : the filled portion of the track + the handle while dragging
    # color_active_dimmed : the handle while idle
    for s in sg.sliders
        s.color_inactive[]      = RGBf(0.22, 0.22, 0.26)
        s.color_active[]        = RGBf(0.00, 0.75, 1.00)   # deep sky blue (matches trajectory)
        s.color_active_dimmed[] = RGBf(0.00, 0.55, 0.80)
    end

    info = Observable("—")
    Label(fig[3, 1], info; tellwidth = false)

    sl = sg.sliders
    palette = [:tomato, :orange, :gold, :springgreen, :cyan, :mediumorchid1]

    onany(sl[1].value, sl[2].value, sl[3].value,
          sl[4].value, sl[5].value, sl[6].value, sl[7].value
    ) do d, dz, φr_deg, ψ°, vx, vy, θ

        empty!(ax)

        sol = calculate(d, dz, φr_deg, ψ°, vx, vy, θ)

        # Field-centric azimuth and robot position (target fixed at origin)
        φ          = deg2rad(φr_deg + ψ°)
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
            lines!(ax, xs, ys, zs; color = :deepskyblue, linewidth = 4)
        end

        # Target marker
        scatter!(ax, [0.0], [0.0], [dz]; color = :red, markersize = 20)

        # Robot marker
        scatter!(ax, [robot_x], [robot_y], [0.0]; color = :white,
                 markersize = 18)

        # ── Robot velocity-vector arrows (vˣ, vʸ) ──
        # vˣ drawn along field X, vʸ drawn along field Y. Each arrow starts
        # just outside the robot marker (offset along its own direction) so
        # it doesn't overlap/obstruct the dot, and grows/shrinks/flips with
        # the sliders since it's recomputed from vx, vy every callback.
        stand_off = 0.12   # gap between marker edge and arrow start [m]

        vx_dir = Vec3f(sign(vx) == 0 ? 1.0 : sign(vx), 0.0, 0.0)
        vy_dir = Vec3f(0.0, sign(vy) == 0 ? 1.0 : sign(vy), 0.0)

        # Scale factor turning m/s of robot velocity into a visible arrow length
        arrow_scale = 0.9

        arrow_origins = [Point3f(robot_x, robot_y, 0.05) + stand_off * vx_dir,
                          Point3f(robot_x, robot_y, 0.05) + stand_off * vy_dir]
        arrow_dirs    = [Vec3f(vx * arrow_scale, 0.0, 0.0),
                          Vec3f(0.0, vy * arrow_scale, 0.0)]

        arrows3d!(ax, arrow_origins, arrow_dirs;
            color      = [:orange, :mediumorchid1],
            shaftradius = 0.05,
            tipradius   = 0.1,
            tiplength   = 0.2)

        # ── Heading (ψ) and Azimuth (φ) direction indicators ──
        # These show *direction only*, not magnitude, so both arrows are
        # drawn at a fixed length regardless of slider values. Raised in Z
        # above the vˣ/vʸ arrows so all four don't visually collide.
        angle_len   = 0.8   # fixed arrow length for direction indicators [m]
        angle_z     = 0.35  # height above ground for these arrows [m]

        ψ_rad = deg2rad(ψ°)
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
            "φ_field: $(round(φr_deg + ψ°; digits=1))°  │  " *
            "Robot V = ($(round(vx; digits=2)), $(round(vy; digits=2))) m/s\n" *
            "Arrows — orange: vˣ  │  violet: vʸ  │  " *
            "dodgerblue: heading ψ  │  hotpink: azimuth φ (→ target)"
    end

    notify(sl[1].value)          # trigger first render
    display(fig)
end

# ── Launch ──
interactive_solver()
