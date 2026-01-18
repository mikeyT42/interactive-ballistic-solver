# ==============================================================================
# PHYSICAL CONSTANTS (Tune these to your specific ball/environment)
# ==============================================================================
const G = 9.81
const MASS = 0.216
const RADIUS = 0.075
const AREA = π * RADIUS * RADIUS
const AIR_DENSITY = 1.225
const DRAG_COEFF = 0.47
const LIFT_COEFF = 0.15

# ==============================================================================
# SOLVER SETTINGS
# ==============================================================================
const MAX_SECANT_ITERS = 5
const TIME_STEP = 0.01
const MAX_SIM_TIME = 5.0

# ==============================================================================
# PUBLIC API
# ==============================================================================
"""
Main solver function - 1:1 match to original Java logic.
Returns: launch angle (deg), launch velocity (m/s), final trajectory points
(Vector of (x,y)), list of estimate trajectories, entry angle, and initial v 
guess.
"""
function calculate(target_dist_x::Float64, target_height_y::Float64,
    α_shape_scalar::Float64, is_blocked_mode::Bool)

    θ_rad = 0.0
    entry_deg = NaN

    # --- STEP 1: Determine the fixed Launch Angle (Theta θ) ---
    if is_blocked_mode
        θ_deg = 80.0
        θ_rad = deg2rad(θ_deg)
    else
        min_entry = -45.0
        max_entry = -75.0
        entry_deg = min_entry + (α_shape_scalar * (max_entry - min_entry))
        entry_rad = deg2rad(entry_deg)
        term1 = (2 * target_height_y) / target_dist_x
        term2 = tan(entry_rad)
        θ_rad = atan(term1 - term2)
    end

    # --- STEP 2: Estimate Initial Velocity (Vacuum Guess) ---
    cos_θ = cos(θ_rad)
    tan_θ = tan(θ_rad)
    numerator = G * target_dist_x^2
    denominator = 2 * cos_θ * cos_θ *
                  ((target_dist_x * tan_θ) - target_height_y)

    v_guess = NaN
    if denominator <= 0
        return 45.0, 0.0, Vector{Tuple{Float64,Float64}}(),
        Vector{Vector{Tuple{Float64,Float64}}}(), entry_deg, v_guess
    end

    v_guess = √(numerator / denominator)

    # --- STEP 3: Refine Velocity using Secant Method ---
    v₀ = v_guess
    v₁ = v_guess + 0.5
    y₀ = simulate_shot_height(v₀, θ_rad, target_dist_x)
    y₁ = simulate_shot_height(v₁, θ_rad, target_dist_x)

    vs = [v₀, v₁]  # Collect all velocities used for plotting estimates

    for _ in 1:MAX_SECANT_ITERS
        if abs(y₁ - y₀) < 0.0001
            break
        end
        error₁ = y₁ - target_height_y
        error₀ = y₀ - target_height_y
        v_new = v₁ - error₁ * (v₁ - v₀) / (error₁ - error₀)
        push!(vs, v_new)
        v₀ = v₁
        y₀ = y₁
        v₁ = v_new
        y₁ = simulate_shot_height(v₁, θ_rad, target_dist_x)
        if abs(y₁ - target_height_y) < 0.01
            break
        end
    end

    # --- STEP 4: Get Trajectories for Plotting ---
    estimates_2d = Vector{Vector{Tuple{Float64,Float64}}}()
    for v in vs[1:end-1]  # All but final
        if v > 0
            _, points = simulate_shot_height(v, θ_rad, target_dist_x;
                collect_points=true)
            push!(estimates_2d, points)
        end
    end

    _, final_points = simulate_shot_height(v₁, θ_rad, target_dist_x;
        collect_points=true)

    return rad2deg(θ_rad), v₁, final_points, estimates_2d, entry_deg,
    v_guess
end

# ==============================================================================
# PRIVATE HELPER - exact match to Java
# ==============================================================================
function simulate_shot_height(v₀::Float64, θ::Float64, target_x::Float64;
    collect_points::Bool=false)

    x = 0.0
    y = 0.0
    vx = v₀ * cos(θ)
    vy = v₀ * sin(θ)

    Δt = TIME_STEP
    time = 0.0

    points = collect_points ? Vector{Tuple{Float64,Float64}}() : nothing
    if collect_points
        push!(points, (x, y))
    end

    while x < target_x && time < MAX_SIM_TIME
        predicted_x = x + vx * Δt
        if predicted_x > target_x
            remaining_dist = target_x - x
            Δt = remaining_dist / vx
        end

        # RK4 k1
        ax₁ = get_acc_x(vx, vy)
        ay₁ = get_acc_y(vx, vy)
        # k2
        vx₂ = vx + ax₁ * (Δt * 0.5)
        vy₂ = vy + ay₁ * (Δt * 0.5)
        ax₂ = get_acc_x(vx₂, vy₂)
        ay₂ = get_acc_y(vx₂, vy₂)
        # k3
        vx₃ = vx + ax₂ * (Δt * 0.5)
        vy₃ = vy + ay₂ * (Δt * 0.5)
        ax₃ = get_acc_x(vx₃, vy₃)
        ay₃ = get_acc_y(vx₃, vy₃)
        # k4
        vx₄ = vx + ax₃ * Δt
        vy₄ = vy + ay₃ * Δt
        ax₄ = get_acc_x(vx₄, vy₄)
        ay₄ = get_acc_y(vx₄, vy₄)

        ax_avg = (ax₁ + 2 * ax₂ + 2 * ax₃ + ax₄) / 6.0
        ay_avg = (ay₁ + 2 * ay₂ + 2 * ay₃ + ay₄) / 6.0

        x += vx * Δt + 0.5 * ax_avg * Δt * Δt
        y += vy * Δt + 0.5 * ay_avg * Δt * Δt
        vx += ax_avg * Δt
        vy += ay_avg * Δt
        time += Δt

        if collect_points
            push!(points, (x, y))
        end
    end

    if collect_points
        return y, points
    else
        return y
    end
end

function get_acc_x(vx::Float64, vy::Float64)::Float64
    v = √(vx * vx + vy * vy)
    if v == 0
        return 0.0
    end
    f_drag_mag = 0.5 * AIR_DENSITY * AREA * DRAG_COEFF * v^2
    f_drag_x = -f_drag_mag * (vx / v)
    f_lift_mag = 0.5 * AIR_DENSITY * AREA * LIFT_COEFF * v^2
    f_lift_x = -f_lift_mag * (vy / v)
    return (f_drag_x + f_lift_x) / MASS
end

function get_acc_y(vx::Float64, vy::Float64)::Float64
    v = √(vx * vx + vy * vy)
    if v == 0
        return -G
    end
    f_grav_y = -MASS * G
    f_drag_mag = 0.5 * AIR_DENSITY * AREA * DRAG_COEFF * v^2
    f_drag_y = -f_drag_mag * (vy / v)
    f_lift_mag = 0.5 * AIR_DENSITY * AREA * LIFT_COEFF * v^2
    f_lift_y = f_lift_mag * (vx / v)
    return (f_grav_y + f_drag_y + f_lift_y) / MASS
end

# ==============================================================================
# 2D INTERACTIVE PLOTTING
# ==============================================================================
using GLMakie

function interactive_ballistic_solver_2d()
    fig = Figure(size=(900, 600))

    ax = Axis(fig[1, 1],
        title="Ballistic Trajectory (2D)",
        xlabel="Horizontal Distance (m)",
        ylabel="Height (m)",
        xgridvisible=true,
        ygridvisible=true
    )

    # Fixed limits - adjust if needed
    xlims!(ax, 0, 8)
    ylims!(ax, 0, 10)

    # Sliders
    sg = SliderGrid(fig[2, 1],
        (label="Target Distance X (m)", range=0:0.1:6.14, startvalue=3.0),
        (label="Target Height Y (m)", range=0:0.1:1.83, startvalue=1.83),
        (label="Shape Scalar (0-1)", range=0:0.01:1, startvalue=0.5),
    )

    toggles = Toggle(fig, active=false)
    fig[2, 2] = hgrid!(toggles, Label(fig, "Blocked Mode"))

    # Text display for results
    result_text = Observable("--")
    Label(fig[3, 1], result_text, tellwidth=false)

    # Observables for inputs
    dist_x = sg.sliders[1].value
    height_y = sg.sliders[2].value
    shape_scalar = sg.sliders[3].value
    blocked_mode = toggles.active

    # Color cycle for estimates
    estimate_colors = [:red, :orange, :gold, :green, :blue, :purple]

    onany(dist_x, height_y, shape_scalar, blocked_mode) do dx, hy, ss, bm
        empty!(ax)

        angle, vel, final_points, estimates, entry_angle, v_guess =
            calculate(dx, hy, ss, bm)

        # Plot estimate trajectories (convergence steps)
        for (i, pts) in enumerate(estimates)
            if !isempty(pts)
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                col = estimate_colors[mod1(i, length(estimate_colors))]
                lines!(ax, xs, ys, color=col, linewidth=1.2, alpha=0.6,
                    linestyle=:dash)
            end
        end

        # Plot final solved trajectory
        if !isempty(final_points)
            xs = [p[1] for p in final_points]
            ys = [p[2] for p in final_points]
            lines!(ax, xs, ys, color=:blue, linewidth=4)
        end

        # Target point
        scatter!(ax, [Point2f(dx, hy)], color=:red, markersize=20)

        # Update result text
        result_text[] =
            "Launch Angle: $(round(angle, digits=2)) °  " *
            "Entry Angle: $(round(entry_angle, digits=2)) °  " *
            "Velocity: $(round(vel, digits=2)) m/s  " *
            "Velocity Guess: $(round(v_guess, digits=2)) m/s " *
            "Percent Error $(round(abs((vel-v_guess)/v_guess) * 100)) %"
    end

    # Initial plot
    notify(dist_x)
    display(GLMakie.Screen(), fig)
end

# ==============================================================================
# 3D INTERACTIVE PLOTTING
# ==============================================================================
function interactive_ballistic_solver_3d()
    # ---------------------------------------------------------
    # Fixed target in world space
    # ---------------------------------------------------------
    TARGET_X = 6.14
    TARGET_Y = 0.0
    TARGET_Z = 1.83

    fig = Figure(size=(900, 600))

    ax = Axis3(fig[1, 1],
        title="Ballistic Trajectory (3D)",
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)"
    )

    xlims!(ax, -1, 7)
    ylims!(ax, -4, 4)
    zlims!(ax, 0, 10)

    # ---------------------------------------------------------
    # Controls (3D window only)
    # ---------------------------------------------------------
    sg = SliderGrid(fig[2, 1],
        (label="Shooter Distance From Target (m)", range=1.0:0.05:7.0,
            startvalue=6.14),
        (label="Shooter Lateral Y (m)", range=-4.0:0.1:4.0,
            startvalue=0.0),
        (label="Shape Scalar (0–1)", range=0:0.01:1,
            startvalue=0.5),
    )

    blocked_toggle = Toggle(fig, active=false)
    fig[2, 2] = hgrid!(Label(fig, "Blocked Mode"), blocked_toggle)

    result_text = Observable("—")
    Label(fig[3, 1:2], result_text, tellwidth=false)

    # ---------------------------------------------------------
    # Observables
    # ---------------------------------------------------------
    shooter_dist = sg.sliders[1].value
    shooter_lat = sg.sliders[2].value
    shape_scalar = sg.sliders[3].value
    blocked = blocked_toggle.active

    estimate_colors = [:red, :orange, :gold, :green, :blue, :purple]

    # ---------------------------------------------------------
    # Update loop (ONLY responds to 3D sliders)
    # ---------------------------------------------------------
    onany(shooter_dist, shooter_lat, shape_scalar, blocked) do d, lat, ss, bm
        empty!(ax)

        # Shooter world position
        shooter_x = TARGET_X - d
        shooter_y = lat
        shooter_z = 0.0

        dx = d
        dy = TARGET_Z

        # Run solver locally
        angle, vel, final_pts, estimates, entry_angle, v_guess =
            calculate(dx, dy, ss, bm)

        # Plot target
        scatter!(ax, [TARGET_X], [TARGET_Y], [TARGET_Z],
            color=:red, markersize=25)

        # Plot shooter
        scatter!(ax, [shooter_x], [shooter_y], [shooter_z],
            color=:black, markersize=15)

        # Estimate trajectories
        for (i, pts) in enumerate(estimates)
            if isempty(pts)
                continue
            end
            xs = Float64[]
            ys = Float64[]
            zs = Float64[]
            for (x2d, y2d) in pts
                push!(xs, shooter_x + x2d)
                push!(ys, shooter_y * (1 - x2d / dx))
                push!(zs, y2d)
            end
            col = estimate_colors[mod1(i, length(estimate_colors))]
            lines!(ax, xs, ys, zs, linewidth=1.2,
                linestyle=:dash, color=col, alpha=0.6)
        end

        # Final trajectory
        if !isempty(final_pts)
            xs = Float64[]
            ys = Float64[]
            zs = Float64[]
            for (x2d, y2d) in final_pts
                push!(xs, shooter_x + x2d)
                push!(ys, shooter_y * (1 - x2d / dx))
                push!(zs, y2d)
            end
            lines!(ax, xs, ys, zs, linewidth=4, color=:blue)
        end

        result_text[] =
            "Shooter @ ($(round(shooter_x; digits=2)), " *
            "$(round(shooter_y; digits=2)), 0)  " *
            "Launch Angle: $(round(angle; digits=2))°  " *
            "Entry Angle: $(round(entry_angle))  " *
            "Velocity: $(round(vel; digits=2)) m/s  " *
            "Velocity Guess: $(round(v_guess; digits=2)) m/s  " *
            "Percent Error $(round(abs((vel-v_guess)/v_guess) * 100)) %"
    end

    notify(shooter_dist)
    display(GLMakie.Screen(), fig)
end

@async interactive_ballistic_solver_2d()
@async interactive_ballistic_solver_3d()
