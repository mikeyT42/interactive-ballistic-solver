using GLMakie

include("constants.jl")
include("ShotResult.jl")
include("secant.jl")

# ══════════════════════════════════════════════════════════════════════════════
#  Greek Letter Names
# ══════════════════════════════════════════════════════════════════════════════
# Δ : Uppercase Delta
# ψ : Lowercase Psi
# ϕ : Lowercase Phi
# ρ : Lowercase Rho
# ϵ : Lowercase Epsilon
# θ : Lowercase Theta

# ══════════════════════════════════════════════════════════════════════════════
#  Main Ballistic Solver
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

    converged, mₛ, mᶠ, zᶠ, m̂ = secant_root_find(dᶠ, Δz, cosθ, tanθ, cosϕ, sinϕ,
                                               vˣ, vʸ)

    # 4 ── Final flywheel speed ──
    vₛˣ  = mᶠ * cosϕ - vˣ
    vₛʸ  = mᶠ * sinϕ - vʸ
    vₛᶻ  = hypot(vₛˣ, vₛʸ)
    flywheel = cosθ > 1e-3 ? vₛᶻ / cosθ : 0.0

    # 5 ── Turret yaw  (shot leading: aim along the muzzle velocity) ──
    yaw = turret_yaw(vₛˣ, vₛʸ, ψ°)

    # 6 ── Validity  (Stage-1 math/geometry) ──
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

function turret_yaw(vₛˣ, vₛʸ, ψ°)
    # The muzzle must fire along (vₛˣ, vₛʸ) so that after adding the robot's
    # own velocity, the ball's ground-frame path points at the target.
    # atan gives a FIELD-frame angle; subtract heading for the turret PID.
    fieldYaw° = rad2deg(atan(vₛʸ, vₛˣ))
    yaw = fieldYaw° - ψ°
    yaw = mod(yaw + 180.0, 360.0) - 180.0     # → [−180, 180]
end

# ══════════════════════════════════════════════════════════════════════════════
#  Interactive GLMakie Visualisation  (3-D field view)
# ══════════════════════════════════════════════════════════════════════════════

function plot_trajectories(mₛ, cosϕ, sinϕ, tanθ, vˣ, vʸ, vᶻ, dᶠ, mᶠ)
    estimates = Vector{Vector{Tuple{Float64,Float64}}}()
    for m in mₛ[1:end-1]
        evˣ  = m * cosϕ - vˣ
        evʸ  = m * sinϕ - vʸ
        evᶻ = hypot(evˣ, evʸ) * tanθ
        _, pts = rk4_simulate(m, evᶻ, dᶠ; trace=true)
        push!(estimates, pts)
    end

    _, final_pts = rk4_simulate(mᶠ, vᶻ * tanθ, dᶠ; trace=true)

    return estimates, final_pts
end

function interactive_solver()
    set_theme!(theme_dark())
    fig = Figure(size = (1100, 820), backgroundcolor = :grey10)

    ax = Axis3(fig[1, 1];
        title  = "Ballistic Trajectory  (3-D Field View)",
        xlabel = "Field X  [m]",
        ylabel = "Field Y  [m]",
        zlabel = "Height  [m]",
        backgroundcolor = :grey10,
        xspinecolor_1 = :grey50, xspinecolor_2 = :grey50,
            xspinecolor_3 = :grey50,
        yspinecolor_1 = :grey50, yspinecolor_2 = :grey50,
            yspinecolor_3 = :grey50,
        zspinecolor_1 = :grey50, zspinecolor_2 = :grey50,
            zspinecolor_3 = :grey50,
        xgridcolor = (:white, 0.12), ygridcolor = (:white, 0.12),
        zgridcolor = (:white, 0.12))
    xlims!(ax, -4, 4)
    ylims!(ax, -4, 4)
    zlims!(ax, 0, 4)

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
    # color_active : the filled portion of the track + the handle while dragging
    # color_active_dimmed : the handle while idle
    for s in sg.sliders
        s.color_inactive[]      = RGBf(0.22, 0.22, 0.26)
        s.color_active[]        = RGBf(0.00, 0.75, 1.00)
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

        # ── Heading (ψ), Azimuth (φ) and Turret aim direction indicators ──
        # These show *direction only*, not magnitude, so all arrows are
        # drawn at a fixed length regardless of slider values. Raised in Z
        # above the vˣ/vʸ arrows so they don't visually collide.
        angle_len   = 0.8   # fixed arrow length for direction indicators [m]
        angle_z     = 0.35  # height above ground for heading/azimuth arrows [m]
        turret_z    = 0.55  # turret arrow sits higher so it stays visible
                            # (not hidden behind the azimuth arrow) at zero lead

        ψ_rad = deg2rad(ψ°)
        ψ_dir = Vec3f(cos(ψ_rad) * angle_len, sin(ψ_rad) * angle_len, 0.0)
        φ_dir = Vec3f(cosφ * angle_len, sinφ * angle_len, 0.0)

        # sol.turret_yaw is ROBOT-relative; arrows live in the FIELD frame,
        # so add the heading back to get where the turret actually points.
        turret_field = deg2rad(sol.turret_yaw + ψ°)
        turret_dir   = Vec3f(cos(turret_field) * angle_len,
                             sin(turret_field) * angle_len, 0.0)

        angle_origins = [Point3f(robot_x, robot_y, angle_z),
                          Point3f(robot_x, robot_y, angle_z),
                          Point3f(robot_x, robot_y, turret_z)]
        angle_dirs    = [ψ_dir, φ_dir, turret_dir]

        arrows3d!(ax, angle_origins, angle_dirs;
            color       = [:dodgerblue, :hotpink, :limegreen],
            shaftradius = 0.05,
            tipradius   = 0.1,
            tiplength   = 0.2)

        # Info bar
        tag = sol.valid ? "✓ VALID" : "✗ INVALID"
        lead = mod(sol.turret_yaw - φr_deg + 180.0, 360.0) - 180.0
        info[] =
            "Flywheel: $(round(sol.flywheel; digits=2)) m/s  │  " *
            "Turret Yaw: $(round(sol.turret_yaw; digits=2))°  │  " *
            "Lead: $(round(lead; digits=2))°  │  " *
            "Sim Error: $(round(sol.error; digits=3)) m  │  $tag\n" *
            "m̂ (vacuum): $(round(sol.m_guess; digits=2))  │  " *
            "φ_field: $(round(φr_deg + ψ°; digits=1))°  │  " *
            "Robot V = ($(round(vx; digits=2)), $(round(vy; digits=2))) m/s\n" *
            "Arrows — orange: vˣ  │  violet: vʸ  │  " *
            "dodgerblue: heading ψ  │  hotpink: azimuth φ (→ target)  │  " *
            "limegreen: turret aim"
    end

    notify(sl[1].value)          # trigger first render
    display(fig)
end

# ── Launch ──
interactive_solver()
