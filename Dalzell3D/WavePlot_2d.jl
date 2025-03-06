module WavePlot

# Import packages at the module level
using JLD2
using Plots
using Statistics
using FFTW
using Printf
using TOML
using DelimitedFiles
using LinearAlgebra

# Export public functions
export generate_plots, load_results

# Load the saved results
function load_results(filename="wave_batch_results_2d.jld2")
    @load filename batch
    return batch
end

# Generate Gaussian spectrum (for plotting)
function generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    return Hs^2/16/k_w/sqrt(2*pi) .* exp.(-(kvec_rs .- k_p).^2 ./ 2 / (k_w)^2)
end

# JONSWAP spectrum (for plotting)
function generate_jonswap_spectrum(option, kvec_rs, Hs, k_p, k_w, gamma=3.3, sigma_a=0.07, sigma_b=0.09)
    # Initialize spectrum vector
    Sk_vec = zeros(length(kvec_rs))

    # Convert to omega based on water depth
    g = 9.81
    if option[:option] == "deepwater"
        omega_p = sqrt(g * k_p)
        omega_rs = sqrt.(g .* kvec_rs)
    elseif option[:option] == "finitewater"
        omega_p = sqrt(g * k_p * tanh(k_p * option[:h]))
        omega_rs = sqrt.(g .* kvec_rs .* tanh.(kvec_rs .* option[:h]))
    end
    f_p = omega_p / 2 / pi
    f_rs = omega_rs ./ 2 ./ pi
    # JONSWAP parameters
    alpha = Hs^2 * k_p^2 / 16  # Scaling factor to match significant wave height
    
    # Calculate spectrum for each wave number
    for i in 1:length(kvec_rs)
        k = kvec_rs[i]
        f = f_rs[i]
        # Frequency ratio term
        x = f / f_p
        
        # PM spectrum
        pm_term = alpha * f^(-5) * exp(-1.25 * (x^(-4)))
        
        # Peak enhancement factor
        sigma = f <= f_p ? sigma_a : sigma_b
        r = exp(-(f - f_p)^2 / (2 * sigma^2 * f_p^2))
        
        # JONSWAP spectrum
        Sk_vec[i] = pm_term * gamma^r
    end
    
    # Normalize to ensure correct significant wave height
    scaling_factor = (Hs^2/16) / sum(Sk_vec * (kvec_rs[2] - kvec_rs[1]))
    return Sk_vec .* scaling_factor
end

# Try to load custom spectrum from file
function load_custom_spectrum(filename, kvec_rs, Hs)
    # Read custom spectrum from file
    # Format expected: CSV with two columns (wave number, spectral density)
    data = nothing
    try
        data = readdlm(filename, ',', Float64)
    catch e
        @warn "Error reading custom spectrum file: $e"
        return generate_gaussian_spectrum(kvec_rs, Hs, kvec_rs[div(length(kvec_rs),2)], 0.1*mean(kvec_rs))
    end
    
    # Extract wave numbers and spectral densities
    k_values = data[:, 1]
    S_values = data[:, 2]
    
    # Interpolate to match the required wave numbers
    Sk_vec = zeros(length(kvec_rs))
    for i in 1:length(kvec_rs)
        # Find closest indices in original data
        k = kvec_rs[i]
        if k < minimum(k_values) || k > maximum(k_values)
            Sk_vec[i] = 0.0  # Outside the range
        else
            # Linear interpolation
            idx_above = findfirst(x -> x >= k, k_values)
            if idx_above == 1
                Sk_vec[i] = S_values[1]
            else
                idx_below = idx_above - 1
                k_below = k_values[idx_below]
                k_above = k_values[idx_above]
                S_below = S_values[idx_below]
                S_above = S_values[idx_above]
                # Interpolate
                Sk_vec[i] = S_below + (S_above - S_below) * (k - k_below) / (k_above - k_below)
            end
        end
    end
    
    # Normalize to ensure correct significant wave height
    scaling_factor = (Hs^2/16) / sum(Sk_vec * (kvec_rs[2] - kvec_rs[1]))
    return Sk_vec .* scaling_factor
end

# Helper function to generate spectrum information string
function generate_spectrum_info(params)
    spectrum_type = get(params, :spectrum_type, "gaussian")
    simulation_type = get(params, :simulation_type, "probe")
    
    # Base information about wave parameters
    base_info = "ϵ=$(params[:eps_0]), f_p=$(round(1/params[:T_p], digits=3)), " *
                "h=$(round(params[:h], digits=2)), k_p=$(params[:k_p])"
                
    # Add directional information for 2D simulations
    if get(params, :is_directional, false)
        base_info *= ", θw=$(round(params[:tht_w], digits=3))"
    end
                
    # Add k-range information
    k_range_info = ""
    if haskey(params, :actual_kmin) && haskey(params, :actual_kmax)
        k_range_info = "\nk-range: [$(round(params[:actual_kmin], digits=3)), $(round(params[:actual_kmax], digits=3))]"
    end
    
    # Add simulation type info
    sim_info = "\nSimulation: $(simulation_type)"
    if get(params, :is_directional, false)
        sim_info *= ", Directional"
    else
        sim_info *= ", Unidirectional"
    end
    
    # Spectrum-specific information
    if spectrum_type == "gaussian"
        k_w_rel = params[:k_w] / params[:k_p]
        return "$base_info\nSpectrum: Gaussian, k_w/k_p=$(round(k_w_rel, digits=3))$k_range_info$sim_info"
    elseif spectrum_type == "jonswap"
        k_w_rel = params[:k_w] / params[:k_p]
        gamma = get(params, :gamma, 3.3)
        sigma_a = get(params, :sigma_a, 0.07)
        sigma_b = get(params, :sigma_b, 0.09)
        return "$base_info\nSpectrum: JONSWAP, γ=$gamma, σa=$sigma_a, σb=$sigma_b, " *
               "k_w/k_p=$(round(k_w_rel, digits=3))$k_range_info$sim_info"
    elseif spectrum_type == "custom"
        spectrum_file = get(params, :spectrum_file, "custom_spectrum.csv")
        return "$base_info\nSpectrum: Custom ($spectrum_file)$k_range_info$sim_info"
    else
        return "$base_info\nSpectrum: Unknown$k_range_info$sim_info"
    end
end

# Function to plot the spectrum
function plot_spectrum(params, output_dir, plot_size)
    # Create a water depth option for spectrum generation
    option = Dict(:option => params[:option], :h => params[:h])
    
    spectrum_type = get(params, :spectrum_type, "gaussian")
    k_p = params[:k_p]
    k_w = params[:k_w]
    Hs = params[:Hs]
    
    # Use actual k-range if available, otherwise calculate
    if haskey(params, :actual_kmin) && haskey(params, :actual_kmax)
        kmin = params[:actual_kmin]
        kmax = params[:actual_kmax]
    else
        kmin_factor = get(params, :kmin_factor, 4.0)
        kmax_factor = get(params, :kmax_factor, 4.0)
        custom_kmin = get(params, :custom_kmin, -1.0)
        custom_kmax = get(params, :custom_kmax, -1.0)
        
        if custom_kmin > 0
            kmin = custom_kmin
        else
            kmin = k_p - kmin_factor * k_w
            if kmin < 0
                kmin = 0.1 * k_p
            end
        end
        
        if custom_kmax > 0
            kmax = custom_kmax
        else
            kmax = k_p + kmax_factor * k_w
        end
    end
    
    # Create wave number range for plotting with more resolution
    k_values = range(kmin, kmax, length=500)
    k_values_arr = collect(k_values)
    
    # Generate spectrum based on type
    if spectrum_type == "gaussian"
        S_values = generate_gaussian_spectrum(k_values_arr, Hs, k_p, k_w)
        title_text = "Gaussian Spectrum"
    elseif spectrum_type == "jonswap"
        gamma = get(params, :gamma, 3.3)
        sigma_a = get(params, :sigma_a, 0.07)
        sigma_b = get(params, :sigma_b, 0.09)
        S_values = generate_jonswap_spectrum(option, k_values_arr, Hs, k_p, k_w, gamma, sigma_a, sigma_b)
        title_text = "JONSWAP Spectrum (γ=$gamma)"
    elseif spectrum_type == "custom"
        # For custom spectrum, we would need to read from file and interpolate
        spectrum_file = get(params, :spectrum_file, "custom_spectrum.csv")
        if isfile(spectrum_file)
            S_values = load_custom_spectrum(spectrum_file, k_values_arr, Hs)
            title_text = "Custom Spectrum ($spectrum_file)"
        else
            S_values = generate_gaussian_spectrum(k_values_arr, Hs, k_p, k_w)
            title_text = "Custom Spectrum (file not found, showing Gaussian approximation)"
        end
    else
        S_values = generate_gaussian_spectrum(k_values_arr, Hs, k_p, k_w)
        title_text = "Unknown Spectrum Type"
    end
    
    # Create spectrum plot
    spectrum_info = generate_spectrum_info(params)
    
    p = plot(
        k_values_arr, 
        S_values,
        xlabel="Wave Number (k)", 
        ylabel="Spectral Density S(k)",
        title="$title_text\n$spectrum_info",
        size=plot_size,
        dpi=300,
        legend=false
    )
    
    # Add vertical line at peak wave number
    vline!(p, [k_p], linestyle=:dash, color=:red, label="Peak wave number")
    
    # Mark the actual range used in simulation
    if haskey(params, :actual_kmin) && haskey(params, :actual_kmax)
        vline!(p, [params[:actual_kmin]], linestyle=:dot, color=:blue, label="kmin")
        vline!(p, [params[:actual_kmax]], linestyle=:dot, color=:blue, label="kmax")
    end
    
    display(p)
    savefig(p, joinpath(output_dir, "wave_spectrum.png"))
    
    # Also generate a log-scale version to better see the tails
    p_log = plot(
        k_values_arr, 
        S_values,
        xlabel="Wave Number (k)", 
        ylabel="Spectral Density S(k)",
        title="$title_text (Log Scale)\n$spectrum_info",
        size=plot_size,
        dpi=300,
        legend=false,
        yscale=:log10,
        ylim=(maximum(S_values)*1e-6, maximum(S_values)*2)
    )
    
    # Add vertical line at peak wave number
    vline!(p_log, [k_p], linestyle=:dash, color=:red, label="Peak wave number")
    
    # Mark the actual range used in simulation
    if haskey(params, :actual_kmin) && haskey(params, :actual_kmax)
        vline!(p_log, [params[:actual_kmin]], linestyle=:dot, color=:blue, label="kmin")
        vline!(p_log, [params[:actual_kmax]], linestyle=:dot, color=:blue, label="kmax")
    end
    
    display(p_log)
    savefig(p_log, joinpath(output_dir, "wave_spectrum_logscale.png"))
    
    # If we have directional simulation, plot directional spreading
    if get(params, :is_directional, false)
        # Create direction range
        tht_w = params[:tht_w]
        tht_values = range(-4*tht_w, 4*tht_w, length=200)
        tht_values_arr = collect(tht_values)
        
        # Generate directional spreading function
        D_values = exp.(-((tht_values_arr .- 0).^2) ./ (2 * tht_w^2))
        
        # Normalize
        D_values = D_values / sum(D_values * (tht_values_arr[2] - tht_values_arr[1]))
        
        # Create directional spreading plot
        p_dir = plot(
            tht_values_arr, 
            D_values,
            xlabel="Direction (radians)", 
            ylabel="Directional Distribution D(θ)",
            title="Directional Spreading\n$spectrum_info",
            size=plot_size,
            dpi=300,
            legend=false
        )
        
        display(p_dir)
        savefig(p_dir, joinpath(output_dir, "directional_spreading.png"))
    end
end

# Plot time series data from probe simulation
function plot_probe_time_series(batch, params, output_dir, plot_size)
    # Get the number of simulations
    N_test = params[:N_test]
    
    # Generate spectrum type description for plot titles
    spectrum_info = generate_spectrum_info(params)
    
    # Create time vector
    data_length = length(batch[:zeta_total][1])
    time_vector = collect(range(0, params[:Nperiod], length=data_length))
    
    # Time series plot of first simulation
    p1 = plot(
        title="Time Series (First Simulation)\n$spectrum_info",
        size=plot_size,
        dpi=300
    )
    
    plot!(p1, time_vector, 
          batch[:zeta_1][1], 
          linestyle=:dash, color=:red, 
          label="Linear (zeta1)")
    
    plot!(p1, time_vector, 
          batch[:zeta_total][1], 
          linestyle=:solid, color=:blue, 
          label="Total (zeta1 + zeta20 + zeta22)", 
          xlabel="Time (T_p)", ylabel="Elevation")
    
    display(p1)
    savefig(p1, joinpath(output_dir, "time_series_plot.png"))
    
    # Create a focused view of a portion of the time series
    focus_start = min(250, length(time_vector) ÷ 8)
    focus_end = min(focus_start + 50, length(time_vector))
    
    p1_zoom = plot(
        title="Time Series (Zoomed)\n$spectrum_info",
        size=plot_size,
        dpi=300
    )
    
    plot!(p1_zoom, time_vector[focus_start:focus_end], 
          batch[:zeta_1][1][focus_start:focus_end], 
          linestyle=:dash, color=:red, 
          label="Linear (zeta1)")
    
    plot!(p1_zoom, time_vector[focus_start:focus_end], 
          batch[:zeta_total][1][focus_start:focus_end], 
          linestyle=:solid, color=:blue, 
          label="Total (zeta1 + zeta20 + zeta22)", 
          xlabel="Time (T_p)", ylabel="Elevation")
    
    display(p1_zoom)
    savefig(p1_zoom, joinpath(output_dir, "time_series_zoomed_plot.png"))
    
    # Plot second-order components
    p_components = plot(
        title="Wave Components (Zoomed)\n$spectrum_info",
        size=plot_size,
        dpi=300,
        layout=(3,1)
    )
    
    plot!(p_components[1], time_vector[focus_start:focus_end], 
          batch[:zeta_1][1][focus_start:focus_end], 
          linestyle=:solid, color=:blue, 
          label="Linear (zeta1)", 
          xlabel="", ylabel="zeta1")
    
    plot!(p_components[2], time_vector[focus_start:focus_end], 
          batch[:zeta_20][1][focus_start:focus_end], 
          linestyle=:solid, color=:red, 
          label="Difference (zeta20)", 
          xlabel="", ylabel="zeta20")
    
    plot!(p_components[3], time_vector[focus_start:focus_end], 
          batch[:zeta_22][1][focus_start:focus_end], 
          linestyle=:solid, color=:green, 
          label="Sum (zeta22)", 
          xlabel="Time (T_p)", ylabel="zeta22")
    
    display(p_components)
    savefig(p_components, joinpath(output_dir, "wave_components_plot.png"))
    
    # FFT Analysis
    Nt = data_length
    df_fft = 1 / (params[:Nperiod] * params[:T_p])
    f_fft = df_fft .* vcat(0:(floor(Int, Nt/2) - 1), (-ceil(Int, Nt/2)):-1)  # for fft
    f_p = 1/params[:T_p]
    
    p2 = plot(
        fftshift(f_fft), 
        fftshift(abs.(fft(batch[:zeta_total][1]))),
        xlabel="Frequency (Hz)", 
        ylabel="Amplitude",
        xlim=(0, 10 * f_p),
        title="FFT Analysis\n$spectrum_info",
        size=plot_size,
        dpi=300
    )
    
    # Add vertical line at peak frequency
    vline!(p2, [f_p], linestyle=:dash, color=:red, label="Peak frequency")
    
    display(p2)
    savefig(p2, joinpath(output_dir, "fft_plot.png"))
    
    # Batch stats - Exceedance probability
    p3 = plot(
        xlabel="Threshold (η/Hs)", 
        ylabel="Exceedance Probability", 
        title="Separate Time Series\n$spectrum_info", 
        yscale=:log10, 
        ylim=(1e-6, 1), 
        xlim=(0, 2),
        grid=true, 
        legend=:bottomleft,
        size=plot_size,
        dpi=300
    )
    
    for k in 1:N_test
        plot!(p3, batch[:thresholds][k], batch[:exceed_prob][k], 
              color=:black, label=k == 1 ? "Simulations" : "")
    end
    
    # Add theoretical curves
    plot!(p3, batch[:thresholds][1], exp.(-8 .* batch[:thresholds][1].^2), 
          color=:red, linewidth=2, label="Rayleigh")
    
    plot!(p3, batch[:thresholds][1], 
          exp.(-8 ./ params[:Hs]^2 ./ params[:k_p]^2 .* 
               (sqrt.(1 .+ 2 .* params[:k_p] .* batch[:thresholds][1] .* params[:Hs]) .- 1).^2), 
          color=:blue, linewidth=2, label="Tayfun")
    
    display(p3)
    savefig(p3, joinpath(output_dir, "exceedance_prob_plot.png"))
    
    # Kurtosis and skewness convergence
    kurt_conv = zeros(N_test)
    skew_conv = zeros(N_test)
    kurt_conv_std = zeros(N_test)
    skew_conv_std = zeros(N_test)
    
    for k in 1:N_test
        kurt_conv[k] = mean(batch[:kurtosis][1:k])
        skew_conv[k] = mean(batch[:skewness][1:k])
        
        kurt_conv_std[k] = std(batch[:kurtosis][1:k])
        skew_conv_std[k] = std(batch[:skewness][1:k])
    end
    
    p4 = plot(
        1:N_test, 
        kurt_conv, 
        yerror=kurt_conv_std, 
        marker=:x, 
        line=:solid,
        xlabel="Number of simulations", 
        ylabel="Kurtosis",
        title="Kurtosis Convergence\n$spectrum_info",
        size=plot_size,
        dpi=300
    )
    
    display(p4)
    savefig(p4, joinpath(output_dir, "kurtosis_convergence.png"))
    
    p5 = plot(
        1:N_test, 
        skew_conv, 
        yerror=skew_conv_std, 
        marker=:o, 
        line=:solid,
        xlabel="Number of simulations", 
        ylabel="Skewness",
        title="Skewness Convergence\n$spectrum_info",
        size=plot_size,
        dpi=300
    )
    
    display(p5)
    savefig(p5, joinpath(output_dir, "skewness_convergence.png"))
    
    # Batch average stats
    exceed_prob_avg = zeros(size(batch[:exceed_prob][1]))
    
    for k in 1:N_test
        exceed_prob_avg .+= batch[:exceed_prob][k]
    end
    exceed_prob_avg ./= N_test
    
    p6 = plot(
        batch[:thresholds][1], 
        exceed_prob_avg, 
        yscale=:log10, 
        ylim=(1e-6, 1), 
        xlim=(0, 2),
        xlabel="Threshold (η/Hs)", 
        ylabel="Exceedance probability",
        title="Averaged across $N_test runs\n$spectrum_info",
        grid=true, 
        legend=:bottomleft,
        size=plot_size,
        dpi=300
    )
    
    plot!(p6, batch[:thresholds][1], exp.(-8 .* batch[:thresholds][1].^2), 
          color=:red, linewidth=2, label="Rayleigh")
    
    plot!(p6, batch[:thresholds][1], 
          exp.(-8 ./ params[:Hs]^2 ./ params[:k_p]^2 .* 
               (sqrt.(1 .+ 2 .* params[:k_p] .* batch[:thresholds][1] .* params[:Hs]) .- 1).^2), 
          color=:blue, linewidth=2, label="Tayfun")
    
    display(p6)
    savefig(p6, joinpath(output_dir, "avg_exceedance_prob.png"))
    
    # Additional plot: Simulation statistics summary
    p7 = plot(
        xlabel="Simulation", 
        ylabel="Value",
        title="Simulation Statistics Summary\n$spectrum_info",
        size=plot_size,
        dpi=300,
        legend=:outertopright
    )
    
    # Add horizontal line for theoretical Hs
    hline!(p7, [params[:Hs]], color=:black, linestyle=:dash, label="Target Hs")
    
    # Plot Hs values
    plot!(p7, 1:N_test, [batch[:Hs]], color=:blue, marker=:circle, label="Hs (time domain)")
    plot!(p7, 1:N_test, [batch[:Hs_m0]], color=:red, marker=:star, label="Hs (4σ)")
    
    display(p7)
    savefig(p7, joinpath(output_dir, "simulation_stats_summary.png"))
    
    # Histogram of kurtosis and skewness
    p8 = plot(
        title="Distribution of Statistics\n$spectrum_info",
        size=plot_size,
        dpi=300,
        layout=(2,1)
    )
    
    histogram!(p8[1], batch[:kurtosis], bins=15, 
               xlabel="", ylabel="Count", title="Kurtosis",
               label="Kurtosis", color=:blue)
    vline!(p8[1], [3.0], color=:red, linestyle=:dash, label="Gaussian (3.0)")
    vline!(p8[1], [mean(batch[:kurtosis])], color=:black, linestyle=:solid, 
           label="Mean ($(round(mean(batch[:kurtosis]), digits=3)))")
    
    histogram!(p8[2], batch[:skewness], bins=15, 
               xlabel="Value", ylabel="Count", title="Skewness",
               label="Skewness", color=:green)
    vline!(p8[2], [0.0], color=:red, linestyle=:dash, label="Gaussian (0.0)")
    vline!(p8[2], [mean(batch[:skewness])], color=:black, linestyle=:solid, 
           label="Mean ($(round(mean(batch[:skewness]), digits=3)))")
    
    display(p8)
    savefig(p8, joinpath(output_dir, "statistics_distribution.png"))
end

# Plot grid simulation results
function plot_grid_results(batch, params, output_dir, plot_size, plot_config)
    # Generate spectrum type description for plot titles
    spectrum_info = generate_spectrum_info(params)
    
    # Get grid parameters
    x = params[:x]
    y = params[:y]
    
    # Get contour levels
    contour_levels = 15
    if haskey(plot_config, "contour_levels")
        contour_levels = plot_config["contour_levels"]
    end
    
    # Get whether to show 3D plots
    show_3d = get(plot_config, "show_3d", true)
    
    # Plot each time step for the first simulation
    for time_step in batch[:time_steps]
        # Get data for this time step
        grid_data = batch[:grid_results][1]["grid_data"][time_step]
        zeta_total = grid_data["zeta_total"]
        
        # Create contour plot
        p_contour = contourf(
            x, y, zeta_total,
            color=:turbo,
            levels=contour_levels,
            xlabel="x (m)",
            ylabel="y (m)",
            title="Wave Elevation at t = $(time_step) · dt\n$spectrum_info",
            size=plot_size,
            dpi=300
        )
        
        display(p_contour)
        savefig(p_contour, joinpath(output_dir, "wave_contour_t$(time_step).png"))
        
        # Create 3D surface plot if requested
        if show_3d
            p_surface = surface(
                x, y, zeta_total,
                color=:turbo,
                xlabel="x (m)",
                ylabel="y (m)",
                zlabel="Elevation",
                title="Wave Elevation at t = $(time_step) · dt\n$spectrum_info",
                size=plot_size,
                dpi=300
            )
            
            display(p_surface)
            savefig(p_surface, joinpath(output_dir, "wave_surface_t$(time_step).png"))
        end
        
        # If component fields are available, plot them too
        if haskey(grid_data, "zeta_1")
            # Linear component
            p_linear = contourf(
                x, y, grid_data["zeta_1"],
                color=:turbo,
                levels=contour_levels,
                xlabel="x (m)",
                ylabel="y (m)",
                title="Linear Wave Component (zeta1) at t = $(time_step) · dt\n$spectrum_info",
                size=plot_size,
                dpi=300
            )
            
            display(p_linear)
            savefig(p_linear, joinpath(output_dir, "linear_contour_t$(time_step).png"))
            
            # Difference frequency component
            p_diff = contourf(
                x, y, grid_data["zeta_20"],
                color=:turbo,
                levels=contour_levels,
                xlabel="x (m)",
                ylabel="y (m)",
                title="Difference Frequency Component (zeta20) at t = $(time_step) · dt\n$spectrum_info",
                size=plot_size,
                dpi=300
            )
            
            display(p_diff)
            savefig(p_diff, joinpath(output_dir, "diff_contour_t$(time_step).png"))
            
            # Sum frequency component
            p_sum = contourf(
                x, y, grid_data["zeta_22"],
                color=:turbo,
                levels=contour_levels,
                xlabel="x (m)",
                ylabel="y (m)",
                title="Sum Frequency Component (zeta22) at t = $(time_step) · dt\n$spectrum_info",
                size=plot_size,
                dpi=300
            )
            
            display(p_sum)
            savefig(p_sum, joinpath(output_dir, "sum_contour_t$(time_step).png"))
            
            # Combined plot of all components
            p_combined = plot(
                layout=(2,2),
                size=(plot_size[1]*1.5, plot_size[2]*1.5),
                dpi=300,
                title="Wave Components at t = $(time_step) · dt\n$spectrum_info"
            )
            
            # Total
            contourf!(p_combined[1], x, y, grid_data["zeta_total"],
                    color=:turbo, levels=contour_levels,
                    xlabel="x (m)", ylabel="y (m)", title="Total")
            
            # Linear
            contourf!(p_combined[2], x, y, grid_data["zeta_1"],
                    color=:turbo, levels=contour_levels,
                    xlabel="x (m)", ylabel="y (m)", title="Linear (zeta1)")
            
            # Difference
            contourf!(p_combined[3], x, y, grid_data["zeta_20"],
                    color=:turbo, levels=contour_levels,
                    xlabel="x (m)", ylabel="y (m)", title="Difference (zeta20)")
            
            # Sum
            contourf!(p_combined[4], x, y, grid_data["zeta_22"],
                    color=:turbo, levels=contour_levels,
                    xlabel="x (m)", ylabel="y (m)", title="Sum (zeta22)")
            
            display(p_combined)
            savefig(p_combined, joinpath(output_dir, "components_t$(time_step).png"))
        end
    end
    
    # If multiple simulations, create statistics across them
    if length(batch[:grid_results]) > 1
        # We'll calculate statistics for the first time step only
        time_step = batch[:time_steps][1]
        
        # Collect mean and standard deviation across simulations
        mean_elevation = zeros(size(batch[:grid_results][1]["grid_data"][time_step]["zeta_total"]))
        std_elevation = zeros(size(mean_elevation))
        
        # First pass - calculate mean
        for i in 1:length(batch[:grid_results])
            mean_elevation .+= batch[:grid_results][i]["grid_data"][time_step]["zeta_total"]
        end
        mean_elevation ./= length(batch[:grid_results])
        
        # Second pass - calculate standard deviation
        for i in 1:length(batch[:grid_results])
            std_elevation .+= (batch[:grid_results][i]["grid_data"][time_step]["zeta_total"] .- mean_elevation).^2
        end
        std_elevation = sqrt.(std_elevation ./ length(batch[:grid_results]))
        
        # Plot mean elevation
        p_mean = contourf(
            x, y, mean_elevation,
            color=:turbo,
            levels=contour_levels,
            xlabel="x (m)",
            ylabel="y (m)",
            title="Mean Wave Elevation at t = $(time_step) · dt\n$spectrum_info",
            size=plot_size,
            dpi=300
        )
        
        display(p_mean)
        savefig(p_mean, joinpath(output_dir, "mean_elevation_t$(time_step).png"))
        
        # Plot standard deviation
        p_std = contourf(
            x, y, std_elevation,
            color=:turbo,
            levels=contour_levels,
            xlabel="x (m)",
            ylabel="y (m)",
            title="Standard Deviation of Wave Elevation at t = $(time_step) · dt\n$spectrum_info",
            size=plot_size,
            dpi=300
        )
        
        display(p_std)
        savefig(p_std, joinpath(output_dir, "std_elevation_t$(time_step).png"))
    end
end

# Main plotting function
function generate_plots(batch, config_file=nothing)
    # Load config if provided
    plot_config = nothing
    output_config = nothing
    
    if !isnothing(config_file)
        full_config = TOML.parsefile(config_file)
        if haskey(full_config, "plot")
            plot_config = full_config["plot"]
        end
        if haskey(full_config, "output")
            output_config = full_config["output"]
        end
    end
    
    # Extract simulation parameters
    params = batch[:parameters]
    
    # Default output directory
    output_dir = "."
    if !isnothing(output_config) && haskey(output_config, "directory")
        output_dir = output_config["directory"]
    end
    
    # Create directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Set default plot size
    default_size = (800, 600)
    plot_size = default_size
    if !isnothing(plot_config) && haskey(plot_config, "size")
        plot_size = Tuple(plot_config["size"])
    end
    
    # Create wave spectrum plot
    plot_spectrum(params, output_dir, plot_size)
    
    # Generate plots based on simulation type
    simulation_type = get(params, :simulation_type, "probe")
    
    if simulation_type == "probe"
        # Plot time series data
        plot_probe_time_series(batch, params, output_dir, plot_size)
    else # grid simulation
        # Plot grid data
        plot_grid_results(batch, params, output_dir, plot_size, plot_config !== nothing ? plot_config : Dict())
    end
    
    println("All plots generated successfully in $(output_dir)!")
end

end # module
