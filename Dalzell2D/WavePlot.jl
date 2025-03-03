module WavePlot

# Import packages at the module level
using JLD2
using Plots
using Statistics
using FFTW
using Printf
using TOML
using DelimitedFiles  # For reading custom spectrum files

# Export public functions
export generate_plots, load_results

# Load the saved results
function load_results(filename="wave_batch_results.jld2")
    @load filename batch
    return batch
end

# Generate Gaussian spectrum (for plotting)
function generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    return Hs^2/16/k_w/sqrt(2*pi) .* exp.(-(kvec_rs .- k_p).^2 ./ 2 / (k_w)^2)
end

# JONSWAP spectrum (for plotting)
function generate_jonswap_spectrum(kvec_rs, Hs, k_p, k_w, gamma=3.3, sigma_a=0.07, sigma_b=0.09)
    # Initialize spectrum vector
    Sk_vec = zeros(length(kvec_rs))
    
    # JONSWAP parameters
    alpha = Hs^2 * k_p^2 / 16  # Scaling factor to match significant wave height
    
    # Calculate spectrum for each wave number
    for i in 1:length(kvec_rs)
        k = kvec_rs[i]
        
        # Frequency ratio term
        x = k / k_p
        
        # PM spectrum
        pm_term = alpha * k^(-3) * exp(-1.25 * (x^(-4)))
        
        # Peak enhancement factor
        sigma = k <= k_p ? sigma_a : sigma_b
        r = exp(-(x - 1)^2 / (2 * sigma^2))
        
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
# Helper function to generate spectrum information string
function generate_spectrum_info(params)
    spectrum_type = get(params, :spectrum_type, "gaussian")
    
    base_info = "ϵ=$(params[:eps_0]), f_p=$(round(1/params[:T_p], digits=3)), " *
                "h=$(round(params[:h], digits=2)), k_p=$(params[:k_p])"
                
    # Add k-range information
    k_range_info = ""
    if haskey(params, :actual_kmin) && haskey(params, :actual_kmax)
        k_range_info = "\nk-range: [$(round(params[:actual_kmin], digits=3)), $(round(params[:actual_kmax], digits=3))]"
    end
    
    if spectrum_type == "gaussian"
        k_w_rel = params[:k_w] / params[:k_p]
        return "$base_info\nSpectrum: Gaussian, k_w/k_p=$(round(k_w_rel, digits=3))$k_range_info"
    elseif spectrum_type == "jonswap"
        k_w_rel = params[:k_w] / params[:k_p]
        gamma = get(params, :gamma, 3.3)
        sigma_a = get(params, :sigma_a, 0.07)
        sigma_b = get(params, :sigma_b, 0.09)
        return "$base_info\nSpectrum: JONSWAP, γ=$gamma, σa=$sigma_a, σb=$sigma_b, " *
               "k_w/k_p=$(round(k_w_rel, digits=3))$k_range_info"
    elseif spectrum_type == "custom"
        spectrum_file = get(params, :spectrum_file, "custom_spectrum.csv")
        return "$base_info\nSpectrum: Custom ($spectrum_file)$k_range_info"
    else
        return "$base_info\nSpectrum: Unknown$k_range_info"
    end
end

# Function to plot the spectrum
function plot_spectrum(params, output_dir, plot_size)
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
        S_values = generate_jonswap_spectrum(k_values_arr, Hs, k_p, k_w, gamma, sigma_a, sigma_b)
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
end
# Main plotting function
function generate_plots(batch, config_file=nothing)
    # Load config if provided
    plot_config = nothing
    if !isnothing(config_file)
        plot_config = TOML.parsefile(config_file)
    end
    
    # Extract simulation parameters
    params = batch[:parameters]
    
    # Determine number of simulations
    N_test = params[:N_test]
    
    # Get the length of data for time calculations
    data_length = length(batch[:zeta_total][1])
    
    # Default output directory
    output_dir = "."
    if !isnothing(plot_config) && haskey(plot_config, "output") && haskey(plot_config["output"], "directory")
        output_dir = plot_config["output"]["directory"]
    end
    
    # Create directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Set default plot size
    default_size = (800, 600)
    plot_size = default_size
    if !isnothing(plot_config) && haskey(plot_config, "plot") && haskey(plot_config["plot"], "size")
        plot_size = Tuple(plot_config["plot"]["size"])
    end
    
    # Generate spectrum type description for plot titles
    spectrum_info = generate_spectrum_info(params)
    
    # Create wave spectrum plot
    plot_spectrum(params, output_dir, plot_size)
    
    # Example: Time series plot sample
    time_vector = collect(range(0, params[:Nperiod], length=data_length))
    
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
    
    # FFT
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
    
    println("All plots generated successfully in $(output_dir)!")
end

end # module
