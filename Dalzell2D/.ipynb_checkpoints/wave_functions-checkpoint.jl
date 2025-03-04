# Core wave analysis functions

function obtain_2dwaveSpectra_irregular(Hs, eps0, k_p, k_w, option, Nk, seed, spectrum_params)
    # Set random seed for reproducibility
    Random.seed!(seed)
    g = 9.81
    
    # Define wave number range based on parameters or custom values
    kmin_factor = get(spectrum_params, "kmin_factor", 4.0)
    kmax_factor = get(spectrum_params, "kmax_factor", 4.0)
    custom_kmin = get(spectrum_params, "custom_kmin", -1.0)
    custom_kmax = get(spectrum_params, "custom_kmax", -1.0)
    
    # Apply custom kmin if specified
    if custom_kmin > 0
        kmin = custom_kmin
    else
        kmin = k_p - kmin_factor * k_w
        if kmin < 0
            kmin = 0.1 * k_p
        end
    end
    
    # Apply custom kmax if specified
    if custom_kmax > 0
        kmax = custom_kmax
    else
        kmax = k_p + kmax_factor * k_w
    end
    
    kvec_range = range(kmin, kmax, length=Nk)
    kvec_rs = collect(kvec_range)  # Convert range to array
    
    # Generate spectrum based on type
    spectrum_type = get(spectrum_params, "spectrum_type", "gaussian")
    
    if spectrum_type == "gaussian"
        Sk_vec = generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    elseif spectrum_type == "jonswap"
        gamma = get(spectrum_params, "gamma", 3.3)
        sigma_a = get(spectrum_params, "sigma_a", 0.07)
        sigma_b = get(spectrum_params, "sigma_b", 0.09)
        Sk_vec = generate_jonswap_spectrum(option, kvec_rs, Hs, k_p, k_w, gamma, sigma_a, sigma_b)
    elseif spectrum_type == "custom"
        spectrum_file = get(spectrum_params, "spectrum_file", "")
        if isempty(spectrum_file) || !isfile(spectrum_file)
            @warn "Custom spectrum file not found, falling back to Gaussian spectrum"
            Sk_vec = generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
        else
            Sk_vec = load_custom_spectrum(spectrum_file, kvec_rs, Hs)
        end
    else
        @warn "Unknown spectrum type: $spectrum_type, falling back to Gaussian"
        Sk_vec = generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    end
    
    # Calculate frequencies based on water depth
    if option[:option] == "deepwater"
        omega_rs = sqrt.(g .* kvec_rs)
    elseif option[:option] == "finitewater"
        omega_rs = sqrt.(g .* kvec_rs .* tanh.(kvec_rs .* option[:h]))
    end
    
    # Implementing the Rayleigh random distribution
    dk = kvec_range[2] - kvec_range[1]
    amp_product = sqrt.(-2 .* log.(rand(Nk))) .* sqrt.(Sk_vec .* dk)
    
    rand_phase = 2*pi*rand(Nk)
    
    # Save wave number range in the return values
    return kvec_rs, omega_rs, amp_product, rand_phase, spectrum_type, (kmin, kmax)
end

# Gaussian spectrum
function generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    return Hs^2/16/k_w/sqrt(2*pi) .* exp.(-(kvec_rs .- k_p).^2 ./ 2 / (k_w)^2)
end

# JONSWAP spectrum
function generate_jonswap_spectrum(option, kvec_rs, Hs, k_p, k_w, gamma=3.3, sigma_a=0.07, sigma_b=0.09)
    # Initialize spectrum vector
    Sk_vec = zeros(length(kvec_rs))

    # Convert to omega based on water depth
    g = 9.81
    if option[:option] == "deepwater"
        omega_p = sqrt.(g * k_p)
        omega_rs = sqrt.(g .* kvec_rs)
    elseif option[:option] == "finitewater"
        omega_p = sqrt.(g * k_p * tanh(k_p * option[:h]))
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
        #x = k / k_p
        x = f / f_p
        
        # PM spectrum
        #pm_term = alpha * k^(-3) * exp(-1.25 * (x^(-4)))
        pm_term = alpha * f^(-5) * exp(-1.25 * (x^(-4)))
        
        # Peak enhancement factor
        #sigma = k <= k_p ? sigma_a : sigma_b
        sigma = f <= f_p ? sigma_a : sigma_b
        r = exp(-(f - f_p)^2 / (2 * sigma^2 * f_p^2))
        
        # JONSWAP spectrum
        Sk_vec[i] = pm_term * gamma^r
    end
    
    # Normalize to ensure correct significant wave height
    scaling_factor = (Hs^2/16) / sum(Sk_vec * (kvec_rs[2] - kvec_rs[1]))
    return Sk_vec .* scaling_factor
end

# Load custom spectrum from file
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
function Dalzell_2D_xyt_YanLi_probe(t, x, t0, option, kvec_rs, omega_rs, phase_rs, amp_rs)
    # Li & Li (pof, 2021) or Li (JFM, 2021) 
    g = 9.81
    h = option[:h]
    
    # Initialize values
    amp_rs = amp_rs ./ 2
    zeta22 = 0
    zeta20 = 0
    zeta1 = 0
    psi1 = kvec_rs .* x .- omega_rs .* (t - t0) .+ phase_rs
    
    for iik in 1:length(kvec_rs)
        # cos_tht = cos(tht_rs - tht_rs(iik))
        cos_tht = 1
        amp_12 = amp_rs .* amp_rs[iik]
        
        if option[:option] == "deepwater"
            dem_mean = (omega_rs .- omega_rs[iik]).^2 .- g .* abs.(kvec_rs .- kvec_rs[iik])
            dem_plus = (omega_rs .+ omega_rs[iik]).^2 .- g .* abs.(kvec_rs .+ kvec_rs[iik])
            coeff_phi20 = omega_rs .* omega_rs[iik] .* (omega_rs .- omega_rs[iik]) .* (1 .+ cos_tht) ./ dem_mean
            coeff_phi22 = -omega_rs .* omega_rs[iik] .* (omega_rs .+ omega_rs[iik]) .* (1 .- cos_tht) ./ dem_plus
            
            num_zeta20 = (omega_rs .- omega_rs[iik]).^2 .+ g .* abs.(kvec_rs .- kvec_rs[iik])
            num_zeta22 = (omega_rs .+ omega_rs[iik]).^2 .+ g .* abs.(kvec_rs .+ kvec_rs[iik])
            
            Bm_1 = (omega_rs.^2 .+ omega_rs[iik]^2) ./ g ./ 2
            coeff_zt20 = omega_rs .* omega_rs[iik] .* (1 .+ cos_tht) .* num_zeta20 ./ g ./ 2 ./ dem_mean .+ Bm_1
            
            coeff_zt22 = -omega_rs .* omega_rs[iik] .* (1 .- cos_tht) .* num_zeta22 ./ g ./ 2 ./ dem_plus .+ Bm_1
            
        elseif option[:option] == "finitewater"
            k2iik = kvec_rs[iik]
            
            th_1 = tanh.(kvec_rs .* h)
            th_2 = tanh(k2iik * h)
            sh_1 = sinh.(kvec_rs .* h)
            sh_2 = sinh(k2iik * h)
            omega2iik = omega_rs[iik]
            dem_20 = (omega_rs .- omega2iik).^2 .- g .* abs.(kvec_rs .- k2iik) .* tanh.((abs.(kvec_rs .- k2iik)) .* h)
            dem_22 = (omega_rs .+ omega2iik).^2 .- g .* abs.(kvec_rs .+ k2iik) .* tanh.((abs.(kvec_rs .+ k2iik)) .* h)
            
            coeff_zt22 = (omega_rs.^2 .+ omega2iik^2) ./ 2 ./ g .- omega_rs .* omega2iik ./ 2 ./ g .*
                (1 .- cos_tht ./ th_1 ./ th_2) .*
                ((omega_rs .+ omega2iik).^2 .+ g .* abs.(kvec_rs .+ k2iik) .* tanh.(abs.(kvec_rs .+ k2iik) .* h)) ./
                dem_22 .+ (omega_rs .+ omega2iik) ./ 2 ./ g ./ dem_22 .*
                (omega_rs.^3 ./ sh_1.^2 .+ omega2iik^3 ./ sh_2^2)
            
            coeff_zt20 = (omega_rs.^2 .+ omega2iik^2) ./ 2 ./ g .+ omega_rs .* omega2iik ./ 2 ./ g .*
                (1 .+ cos_tht ./ th_1 ./ th_2) .*
                ((omega_rs .- omega2iik).^2 .+ g .* abs.(kvec_rs .- k2iik) .* tanh.(abs.(kvec_rs .- k2iik) .* h)) ./
                dem_20 .+ (omega_rs .- omega2iik) ./ 2 ./ g ./ dem_20 .*
                (omega_rs.^3 ./ sh_1.^2 .- omega2iik^3 ./ sh_2^2)
        end
        
        # Handle NaN and Inf values
        coeff_zt20[isnan.(coeff_zt20)] .= 0
        coeff_zt20[isinf.(coeff_zt20)] .= 0
        
        coeff_zt22[isnan.(coeff_zt22)] .= 0
        coeff_zt22[isinf.(coeff_zt22)] .= 0
        
        psi2 = psi1[iik]
        cos_plus = cos.(psi1 .+ psi2)
        cos_minus = cos.(psi1 .- psi2)
        
        zeta22 = sum(amp_12 .* coeff_zt22 .* cos_plus) + zeta22
        zeta20 = sum(amp_12 .* coeff_zt20 .* cos_minus) + zeta20
        zeta1 = amp_rs[iik] * cos(psi2) + zeta1
    end
    
    zeta22 = 2 * zeta22
    zeta20 = 2 * zeta20
    zeta1 = 2 * zeta1
    
    return zeta20, zeta22, zeta1
end

function exceedance_probability(zeta, prescribed_Hs)
    # Find peaks (crests) in the signal
    crests = Float64[]
    for i in 2:(length(zeta)-1)
        if zeta[i] > 0 && zeta[i] > zeta[i-1] && zeta[i] > zeta[i+1]
            push!(crests, zeta[i])
        end
    end
    
    # Handle edge case of no peaks found
    if isempty(crests)
        @warn "No peaks found in the signal"
        return 0.0, 0.0, 0.0, 0.0, zeros(500), collect(range(0, 2, length=500))
    end
    
    # Calculate significant wave height (Hs)
    sorted_crests = sort(crests, rev=true)
    n_highest = max(1, round(Int, length(sorted_crests)/3))
    Hs = 2 * mean(sorted_crests[1:n_highest])
    
    # Calculate Hs from standard deviation (Hs_m0)
    Hs_m0 = 4 * std(zeta)
    
    # Calculate kurtosis
    μ = mean(zeta)
    σ = std(zeta)
    n = length(zeta)
    
    # Centralized moments
    m4 = sum((zeta .- μ).^4) / n
    m2 = σ^2
    kurt = m4 / (m2^2)
    
    # Calculate skewness
    m3 = sum((zeta .- μ).^3) / n
    skew = m3 / (m2^(3/2))
    
    # Calculate exceedance probability
    thresholds = collect(range(0, 2, length=500))
    exceed_prob = zeros(length(thresholds))
    
    for i in 1:length(thresholds)
        # Count how many crests exceed the threshold (normalized by prescribed_Hs)
        exceed_prob[i] = sum(crests ./ prescribed_Hs .> thresholds[i]) / length(crests)
    end
    
    return Hs, Hs_m0, kurt, skew, exceed_prob, thresholds
end

# Main processing function that will run on each worker
function batch_run_seeded(seed, input, endtimeT_p)
    # Random params
    kvec_rs, omega_rs, amp_rs, phase_rs, spectrum_type, k_range = obtain_2dwaveSpectra_irregular(
        input[:Hs], input[:eps_0], input[:k_p], input[:k_w], input[:option], 
        input[:Nk], seed, input[:spectrum_params]
    )
    
    # Initialize arrays
    zeta_1_single_probe = zeros(endtimeT_p * input[:Ntp])
    zeta_22_single_probe = zeros(endtimeT_p * input[:Ntp])
    zeta_20_single_probe = zeros(endtimeT_p * input[:Ntp])
    
    # Process each time step
    for kk in 1:(endtimeT_p * input[:Ntp])
        zeta_20_single_probe[kk], zeta_22_single_probe[kk], zeta_1_single_probe[kk] = 
            Dalzell_2D_xyt_YanLi_probe(
                input[:time][kk], 0, input[:t0], input[:option], kvec_rs, omega_rs, 
                phase_rs, amp_rs
            )
    end
    
    return zeta_22_single_probe, zeta_20_single_probe, zeta_1_single_probe, spectrum_type, k_range
end
