using LinearAlgebra

# =========================================================================
# PHYSICAL CONSTANTS (Tune these to your specific ball/environment)
# =========================================================================
const G = 9.81
const MASS = 0.27
const RADIUS = 0.12
const AREA = π * RADIUS * RADIUS
const AIR_DENSITY = 1.225
const DRAG_COEFF = 0.47
const LIFT_COEFF = 0.15

# =========================================================================
# SOLVER SETTINGS
# =========================================================================
const MAX_SECANT_ITERS = 5
const TIME_STEP = 0.01
const MAX_SIM_TIME = 5.0

# =========================================================================
# PUBLIC API
# =========================================================================
"""
Main solver function - 1:1 match to original Java logic.
Returns: launch angle (deg), launch velocity (m/s), final trajectory points (Vector of (x,y)), list of estimate trajectories, entry angle, and initial v guess.
"""
function calculate(target_dist_x::Float64, target_height_y::Float64, shape_scalar::Float64, is_blocked_mode::Bool)
    theta_rad = 0.0
    entry_deg = NaN

    # --- STEP 1: Determine the fixed Launch Angle (Theta) ---
    if is_blocked_mode
        min_launch = 50.0
        max_launch = 80.0
        theta_deg = min_launch + (shape_scalar * (max_launch - min_launch))
        theta_rad = deg2rad(80)
    else
        min_entry = -35.0
        max_entry = -65.0
        entry_deg = min_entry + (shape_scalar * (max_entry - min_entry))
        entry_rad = deg2rad(entry_deg)
        term1 = (2 * target_height_y) / target_dist_x
        term2 = tan(entry_rad)
        theta_rad = atan(term1 - term2)
    end

    # --- STEP 2: Estimate Initial Velocity (Vacuum Guess) ---
    cos_theta = cos(theta_rad)
    tan_theta = tan(theta_rad)
    numerator = G * target_dist_x * target_dist_x
    denominator = 2 * cos_theta * cos_theta * ((target_dist_x * tan_theta) - target_height_y)

    v_guess = NaN
    if denominator <= 0
        return 45.0, 0.0, Vector{Tuple{Float64,Float64}}(), Vector{Vector{Tuple{Float64,Float64}}}(), entry_deg, v_guess
    end

    v_guess = sqrt(numerator / denominator)

    # --- STEP 3: Refine Velocity using Secant Method ---
    v0 = v_guess
    v1 = v_guess + 0.5
    y0 = simulate_shot_height(v0, theta_rad, target_dist_x)
    y1 = simulate_shot_height(v1, theta_rad, target_dist_x)

    vs = [v0, v1]  # Collect all velocities used for plotting estimates

    for i in 1:MAX_SECANT_ITERS
        if abs(y1 - y0) < 0.0001
            break
        end
        error1 = y1 - target_height_y
        error0 = y0 - target_height_y
        v_new = v1 - error1 * (v1 - v0) / (error1 - error0)
        push!(vs, v_new)
        v0 = v1
        y0 = y1
        v1 = v_new
        y1 = simulate_shot_height(v1, theta_rad, target_dist_x)
        if abs(y1 - target_height_y) < 0.01
            break
        end
    end

    # --- STEP 4: Get Trajectories for Plotting ---
    estimates_2d = Vector{Vector{Tuple{Float64,Float64}}}()
    for v in vs[1:end-1]  # All but final
        if v > 0
            _, points = simulate_shot_height(v, theta_rad, target_dist_x; collect_points=true)
            push!(estimates_2d, points)
        end
    end

    _, final_points = simulate_shot_height(v1, theta_rad, target_dist_x; collect_points=true)

    return rad2deg(theta_rad), v1, final_points, estimates_2d, entry_deg, v_guess
end

# =========================================================================
# PRIVATE HELPER - exact match to Java
# =========================================================================
function simulate_shot_height(v0::Float64, theta::Float64, target_x::Float64; collect_points::Bool=false)
    x = 0.0
    y = 0.0
    vx = v0 * cos(theta)
    vy = v0 * sin(theta)

    dt = TIME_STEP
    time = 0.0

    points = collect_points ? Vector{Tuple{Float64,Float64}}() : nothing
    if collect_points
        push!(points, (x, y))
    end

    while x < target_x && time < MAX_SIM_TIME
        predicted_x = x + vx * dt
        if predicted_x > target_x
            remaining_dist = target_x - x
            dt = remaining_dist / vx
        end

        # RK4 k1
        ax1 = get_acc_x(vx, vy)
        ay1 = get_acc_y(vx, vy)
        # k2
        vx2 = vx + ax1 * (dt * 0.5)
        vy2 = vy + ay1 * (dt * 0.5)
        ax2 = get_acc_x(vx2, vy2)
        ay2 = get_acc_y(vx2, vy2)
        # k3
        vx3 = vx + ax2 * (dt * 0.5)
        vy3 = vy + ay2 * (dt * 0.5)
        ax3 = get_acc_x(vx3, vy3)
        ay3 = get_acc_y(vx3, vy3)
        # k4
        vx4 = vx + ax3 * dt
        vy4 = vy + ay3 * dt
        ax4 = get_acc_x(vx4, vy4)
        ay4 = get_acc_y(vx4, vy4)

        ax_avg = (ax1 + 2*ax2 + 2*ax3 + ax4) / 6.0
        ay_avg = (ay1 + 2*ay2 + 2*ay3 + ay4) / 6.0

        x += vx * dt + 0.5 * ax_avg * dt * dt
        y += vy * dt + 0.5 * ay_avg * dt * dt
        vx += ax_avg * dt
        vy += ay_avg * dt
        time += dt

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
    v = sqrt(vx*vx + vy*vy)
    if v == 0
        return 0.0
    end
    f_drag_mag = 0.5 * AIR_DENSITY * AREA * DRAG_COEFF * v * v
    f_drag_x = -f_drag_mag * (vx / v)
    f_lift_mag = 0.5 * AIR_DENSITY * AREA * LIFT_COEFF * v * v
    f_lift_x = -f_lift_mag * (vy / v)
    return (f_drag_x + f_lift_x) / MASS
end

function get_acc_y(vx::Float64, vy::Float64)::Float64
    v = sqrt(vx*vx + vy*vy)
    if v == 0
        return -G
    end
    f_grav_y = -MASS * G
    f_drag_mag = 0.5 * AIR_DENSITY * AREA * DRAG_COEFF * v * v
    f_drag_y = -f_drag_mag * (vy / v)
    f_lift_mag = 0.5 * AIR_DENSITY * AREA * LIFT_COEFF * v * v
    f_lift_y = f_lift_mag * (vx / v)
    return (f_grav_y + f_drag_y + f_lift_y) / MASS
end

# =========================================================================
# 2D INTERACTIVE PLOTTING
# =========================================================================
using GLMakie

function interactive_ballistic_solver()
    fig = Figure(size=(900, 600))

    ax = Axis(fig[1,1],
        title = "Ballistic Trajectory (2D)",
        xlabel = "Horizontal Distance (m)",
        ylabel = "Height (m)",
        xgridvisible = true,
        ygridvisible = true
    )

    # Fixed limits - adjust if needed
    xlims!(ax, 0, 8)
    ylims!(ax, 0, 10)

    # Sliders
    sg = SliderGrid(fig[2,1],
        (label="Target Distance X (m)", range=0:0.1:6.14, startvalue=3.0),
        (label="Target Height Y (m)",    range=0:0.1:1.83, startvalue=1.83),
        (label="Shape Scalar (0-1)",     range=0:0.01:1, startvalue=0.5),
    )

    toggles = Toggle(fig, active=false)
    fig[2,2] = hgrid!(toggles, Label(fig, "Blocked Mode"))

    # Text display for results
    result_text = Observable("Launch Angle: -- °   Velocity: -- m/s   Entry Angle: -- °")
    Label(fig[3,1], result_text, tellwidth=false)

    # Observables for inputs
    dist_x = sg.sliders[1].value
    height_y = sg.sliders[2].value
    shape_scalar = sg.sliders[3].value
    blocked_mode = toggles.active

    # Color cycle for estimates
    estimate_colors = [:red, :orange, :gold, :green, :blue, :purple]

    onany(dist_x, height_y, shape_scalar, blocked_mode) do dx, hy, ss, bm
        empty!(ax)

        angle, vel, final_points, estimates, entry_angle, v_guess = calculate(dx, hy, ss, bm)

        # Plot estimate trajectories (convergence steps)
        for (i, pts) in enumerate(estimates)
            if !isempty(pts)
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                col = estimate_colors[mod1(i, length(estimate_colors))]
                lines!(ax, xs, ys, color=col, linewidth=1.2, alpha=0.6, linestyle=:dash)
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
        result_text[] = "Launch Angle: $(round(angle, digits=2)) °  Entry Angle: $(round(entry_angle, digits=2)) °  Velocity: $(round(vel, digits=2)) m/s  Velocity Guess: $(round(v_guess, digits=2)) m/s Percent Error $(round(abs((vel-v_guess)/v_guess) * 100)) %"
    end

    # Initial plot
    notify(dist_x)

    display(fig)
end

# To run:
# interactive_ballistic_solver()
