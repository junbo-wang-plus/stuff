# Core wave analysis functions with support for 2D simulations
using Statistics
using Random
using FFTW
using DelimitedFiles
using LinearAlgebra

# Functions for loading and processing 2D wave spectra are now integrated directly

function load_2d_spectrum_netcdf(filename)
    using NetCDF
    
    # Open the NetCDF file
    nc = NetCDF.open(filename)
    
    # Read the specific variables from your WAM file
    # You'll need to inspect your WAM file to find the correct variable names
    # Common ones might be:
    freq = NetCDF.read(nc, "frequency")  # or similar name
    dir = NetCDF.read(nc, "direction")   # or similar name
    spectrum = NetCDF.read(nc, "e_fth")  # or similar name for energy spectrum
    
    # Convert frequency to wavenumber if needed
    g = 9.81
    depth = 50.0  # your water depth in meters
    k_values = zeros(length(freq))
    
    for i in 1:length(freq)
        omega = 2π * freq[i]
        # Dispersion relation to convert frequency to wavenumber
        k_est = omega^2 / g  # deep water approximation
        for _ in 1:5  # iterative solution
            k_est = omega^2 / (g * tanh(k_est * depth))
        end
        k_values[i] = k_est
    end
    
    # Close the NetCDF file
    NetCDF.close(nc)
    
    return k_values, dir, spectrum
end


function load_2d_spectrum_csv(filename)
    # Read the entire file content
    data = nothing
    try
        data = readdlm(filename, ',', Float64, '\n', skipstart=0)
    catch e
        error("Error reading 2D spectrum file: $e")
    end

    # Detect format based on data structure
    if size(data, 2) == 3
        # Three-column format: k, θ, S(k,θ)
        k_values = sort(unique(data[:, 1]))
        theta_values = sort(unique(data[:, 2]))
        
        # Initialize the spectrum grid
        spectrum_grid = zeros(length(theta_values), length(k_values))
        
        # Fill the grid
        for i in 1:size(data, 1)
            k = data[i, 1]
            theta = data[i, 2]
            S = data[i, 3]
            
            k_idx = findfirst(x -> x ≈ k, k_values)
            theta_idx = findfirst(x -> x ≈ theta, theta_values)
            
            if !isnothing(k_idx) && !isnothing(theta_idx)
                spectrum_grid[theta_idx, k_idx] = S
            end
        end
        
        return k_values, theta_values, spectrum_grid, "three_column"
    else
        # Grid format with header
        # First row (excluding first cell) contains k values
        k_values = vec(data[1, 2:end])
        
        # First column (excluding first cell) contains θ values
        theta_values = vec(data[2:end, 1])
        
        # Rest of the grid contains S(k,θ) values
        spectrum_grid = data[2:end, 2:end]
        
        return k_values, theta_values, spectrum_grid, "grid_with_header"
    end
end

"""
    load_2d_spectrum_netcdf(filename)

Load a 2D wave spectrum from a NetCDF file (typically from WAM or WAVEWATCH III).

# Arguments
- `filename`: Path to the NetCDF file

# Returns
- `k_values`: Array of wave numbers
- `theta_values`: Array of wave directions
- `spectrum_grid`: 2D array of spectral energy densities
"""
function load_2d_spectrum_netcdf(filename)
    # This requires the NetCDF.jl package
    error("""
    NetCDF support requires the NetCDF.jl package. 
    
    To implement this functionality:
    1. Add 'using NetCDF' to your imports
    2. Install the package with: ]add NetCDF
    3. Implement the specific NetCDF reading logic for your WAM/WAVEWATCH format
    
    The exact implementation depends on the specific structure of your NetCDF files.
    """)
    
    # Example implementation (would need to be adapted to specific file format):
    #=
    using NetCDF
    
    # Open the NetCDF file
    nc = NetCDF.open(filename)
    
    # Read the dimensions and variables based on your NetCDF structure
    # This is highly dependent on the specific output format from WAM/WAVEWATCH
    k_values = NetCDF.read(nc, "wavenumber")  # or "frequency" depending on the file
    theta_values = NetCDF.read(nc, "direction")
    spectrum_grid = NetCDF.read(nc, "efth")  # The 2D spectrum variable name
    
    # Close the file
    NetCDF.close(nc)
    
    return k_values, theta_values, spectrum_grid
    =#
    
    # For now, return empty arrays as this is just a placeholder
    return Float64[], Float64[], Array{Float64}(undef, 0, 0)
end

"""
    process_2d_spectrum(k_values, theta_values, spectrum_grid, Hs, params)

Process a 2D spectrum and create the necessary wave components.

# Arguments
- `k_values`: Array of wave numbers
- `theta_values`: Array of wave directions
- `spectrum_grid`: 2D array of spectral energy densities
- `Hs`: Target significant wave height (used only if normalize=true)
- `params`: Additional parameters for spectrum processing

# Returns
- `kx_values`: X-component of wave numbers
- `ky_values`: Y-component of wave numbers
- `k_magnitudes`: Magnitude of wave numbers
- `theta_flat`: Flattened array of directions
- `amplitudes`: Wave amplitudes
"""
function process_2d_spectrum(k_values, theta_values, spectrum_grid, Hs, params)
    # Flatten the grid to create vectors of k, θ, and S(k,θ)
    n_k = length(k_values)
    n_theta = length(theta_values)
    
    # Create meshgrid-like arrays
    k_grid = repeat(reshape(k_values, 1, n_k), n_theta, 1)
    theta_grid = repeat(reshape(theta_values, n_theta, 1), 1, n_k)
    
    # Flatten the arrays
    k_flat = vec(k_grid)
    theta_flat = vec(theta_grid)
    S_flat = vec(spectrum_grid)
    
    # Check if normalization is requested
    normalize = get(params, "normalize_spectrum", false)
    
    # Process the spectrum values
    if normalize
        # Calculate the sum for normalization
        S_sum = sum(S_flat)
        
        if S_sum ≈ 0.0
            error("The spectrum has zero or near-zero total energy")
        end
        
        # Calculate the amplitudes
        # The target variance is Hs²/16
        target_variance = Hs^2 / 16
        scaling_factor = sqrt(target_variance / S_sum)
        
        # Convert to amplitudes with normalization
        amplitudes = S_flat .* scaling_factor
    else
        # Use the spectrum values directly - simply take the square root to convert
        # from energy density to amplitude (E = a²/2 => a = sqrt(2E))
        amplitudes = sqrt.(2.0 .* S_flat)
        
        # Log the resulting significant wave height
        approx_Hs = 4 * sqrt(sum(S_flat))
        @info "Using 2D spectrum without normalization. Approximate Hs = $approx_Hs"
    end
    
    # Calculate kx and ky components
    kx_values = k_flat .* cos.(theta_flat)
    ky_values = k_flat .* sin.(theta_flat)
    
    return kx_values, ky_values, k_flat, theta_flat, amplitudes
end

"""
    obtain_directional_waveSpectra_irregular(Hs, eps0, k_p, k_w, tht_w, option, Nk, Ntht, seed, spectrum_params)

Generate random wave spectra for 2D waves with directional spreading.

# Arguments
- `Hs`: Significant wave height
- `eps0`: Wave steepness parameter
- `k_p`: Peak wave number
- `k_w`: Wave number bandwidth
- `tht_w`: Directional spreading parameter
- `option`: Dictionary with water depth options
- `Nk`: Number of wave number components
- `Ntht`: Number of directional components
- `seed`: Random seed for reproducibility
- `spectrum_params`: Dictionary with spectrum parameters

# Returns
- `kx_rs`: X-component of wave number
- `ky_rs`: Y-component of wave number
- `kvec_rs`: Magnitude of wave number
- `tht_rs`: Wave direction angles
- `omega_rs`: Angular frequencies
- `amp_rs`: Wave amplitudes
- `phase_rs`: Wave phases
- `spectrum_type`: Type of spectrum used
- `k_range`: Range of wave numbers (kmin, kmax)
"""
function obtain_directional_waveSpectra_irregular(Hs, eps0, k_p, k_w, tht_w, option, Nk, Ntht, seed, spectrum_params)
    # Set random seed for reproducibility
    Random.seed!(seed)
    g = 9.81
    
    # Check if we're using a full 2D spectrum file
    if get(spectrum_params, "spectrum_type", "") == "spectrum2d"
        return obtain_2d_spectrum_from_file(Hs, option, seed, spectrum_params)
    end
    
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
    
    # Create wave number and direction vectors
    kvec = range(kmin, kmax, length=Nk) |> collect
    
    # Define theta range based on spreading parameter
    tht_min = -4 * tht_w
    if abs(tht_min) > π
        tht_min = -π + 0.01
    end
    tht_max = -tht_min
    tht_vec = range(tht_min, tht_max, length=Ntht) |> collect
    
    # Create 2D grid of wave numbers and directions
    kmat = repeat(kvec', Ntht, 1)
    tht_mat = repeat(tht_vec, 1, Nk)
    
    # Reshape to vectors
    kvec_rs = vec(kmat)
    tht_rs = vec(tht_mat)
    
    # Generate spectrum based on type
    spectrum_type = get(spectrum_params, "spectrum_type", "gaussian")
    
    # Check if normalization is requested
    normalize_spectrum = get(spectrum_params, "normalize_spectrum", true)
    
    # Wave number spectrum
    if spectrum_type == "gaussian"
        # Use different spreading for high-frequency components
        k_w2 = sqrt(1.0) * k_w
        Sk_vec = exp.(-((kvec_rs .- k_p).^2) ./ (2 * k_w^2))
        Sk_vec_HF = exp.(-((kvec_rs .- k_p).^2) ./ (2 * k_w2^2))
        Sk_vec[kvec_rs .> k_p] = Sk_vec_HF[kvec_rs .> k_p]
    elseif spectrum_type == "jonswap"
        gamma = get(spectrum_params, "gamma", 3.3)
        sigma_a = get(spectrum_params, "sigma_a", 0.07)
        sigma_b = get(spectrum_params, "sigma_b", 0.09)
        
        # Get base spectrum
        Sk_base = generate_jonswap_spectrum(option, kvec_rs, Hs, k_p, k_w, gamma, sigma_a, sigma_b)
        Sk_vec = Sk_base
    elseif spectrum_type == "custom"
        spectrum_file = get(spectrum_params, "spectrum_file", "")
        if isempty(spectrum_file) || !isfile(spectrum_file)
            @warn "Custom spectrum file not found, falling back to Gaussian spectrum"
            Sk_vec = exp.(-((kvec_rs .- k_p).^2) ./ (2 * k_w^2))
        else
            # For custom spectra, respect the normalize_spectrum setting
            Sk_vec = load_custom_spectrum(spectrum_file, kvec_rs, Hs, normalize=normalize_spectrum)
        end
    else
        @warn "Unknown spectrum type: $spectrum_type, falling back to Gaussian"
        Sk_vec = exp.(-((kvec_rs .- k_p).^2) ./ (2 * k_w^2))
    end
    
    # Directional spreading function
    D_vec = exp.(-((tht_rs .- 0).^2) ./ (2 * tht_w^2))
    
    # Calculate frequencies based on water depth
    if option[:option] == "deepwater"
        omega_rs = sqrt.(g .* kvec_rs)
    elseif option[:option] == "finitewater"
        omega_rs = sqrt.(g .* kvec_rs .* tanh.(kvec_rs .* option[:h]))
    end
    
    # Wave number components
    kx_rs = kvec_rs .* cos.(tht_rs)
    ky_rs = kvec_rs .* sin.(tht_rs)
    
    # Normalize the spectrum
    Sum_ktheta = sum(Sk_vec .* D_vec)
    amp_rs = Sk_vec .* D_vec ./ Sum_ktheta .* eps0 ./ k_p
    
    # Random phases
    phase_rs = 2π * rand(length(kvec_rs))
    
    return kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, amp_rs, phase_rs, spectrum_type, (kmin, kmax)
end

"""
    obtain_2d_spectrum_from_file(Hs, option, seed, spectrum_params)

Generate wave components from a 2D spectrum file (e.g., from WAM or WAVEWATCH III).

# Arguments
- `Hs`: Significant wave height (used for normalization if requested)
- `option`: Dictionary with water depth options
- `seed`: Random seed for reproducibility
- `spectrum_params`: Dictionary with spectrum parameters including the file path

# Returns
- `kx_rs`: X-component of wave number
- `ky_rs`: Y-component of wave number
- `kvec_rs`: Magnitude of wave number
- `tht_rs`: Wave direction angles
- `omega_rs`: Angular frequencies
- `amp_rs`: Wave amplitudes
- `phase_rs`: Wave phases
- `spectrum_type`: "spectrum2d"
- `k_range`: Range of wave numbers (kmin, kmax)
"""
function obtain_2d_spectrum_from_file(Hs, option, seed, spectrum_params)
    # Set random seed for reproducibility
    Random.seed!(seed)
    g = 9.81
    
    # Get the spectrum file path
    spectrum_file = get(spectrum_params, "spectrum2d_file", "")
    if isempty(spectrum_file) || !isfile(spectrum_file)
        error("2D spectrum file not found: $spectrum_file")
    end
    
    # Determine file format and load accordingly
    file_format = get(spectrum_params, "spectrum2d_format", "auto")
    
    # Load the spectrum based on format
    k_values = Float64[]
    theta_values = Float64[]
    spectrum_grid = Array{Float64}(undef, 0, 0)
    
    if file_format == "auto"
        # Try to guess format based on extension
        if endswith(spectrum_file, ".nc")
            file_format = "netcdf"
        else
            file_format = "csv"
        end
    end
    
    if file_format == "netcdf"
        # Load from NetCDF (requires NetCDF.jl package)
        k_values, theta_values, spectrum_grid = load_2d_spectrum_netcdf(spectrum_file)
    else
        # Default to CSV format
        k_values, theta_values, spectrum_grid, detected_format = load_2d_spectrum_csv(spectrum_file)
        @info "Loaded 2D spectrum from $spectrum_file (format: $detected_format)"
    end
    
    # Check if normalization is requested - for 2D spectra, default to false
    normalize = get(spectrum_params, "normalize_spectrum", false)
    
    # Add the normalization parameter to the spectrum_params
    spectrum_params_with_norm = copy(spectrum_params)
    spectrum_params_with_norm["normalize_spectrum"] = normalize
    
    # Process the 2D spectrum to obtain wave components
    kx_rs, ky_rs, kvec_rs, tht_rs, amp_rs = process_2d_spectrum(
        k_values, theta_values, spectrum_grid, Hs, spectrum_params_with_norm
    )
    
    # Calculate frequencies based on water depth
    if option[:option] == "deepwater"
        omega_rs = sqrt.(g .* kvec_rs)
    elseif option[:option] == "finitewater"
        omega_rs = sqrt.(g .* kvec_rs .* tanh.(kvec_rs .* option[:h]))
    end
    
    # Generate random phases
    phase_rs = 2π * rand(length(kvec_rs))
    
    # Determine the k range for reporting
    k_range = (minimum(k_values), maximum(k_values))
    
    return kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, amp_rs, phase_rs, "spectrum2d", k_range
end

"""
    generate_jonswap_spectrum(option, kvec_rs, Hs, k_p, k_w, gamma=3.3, sigma_a=0.07, sigma_b=0.09)

Generate a JONSWAP spectrum.

# Arguments
- `option`: Dictionary with water depth options
- `kvec_rs`: Wave numbers
- `Hs`: Significant wave height
- `k_p`: Peak wave number
- `k_w`: Wave number bandwidth
- `gamma`: Peak enhancement factor
- `sigma_a`: Left spectral width
- `sigma_b`: Right spectral width

# Returns
- Spectral density values
"""
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

"""
    load_custom_spectrum(filename, kvec_rs, Hs; normalize=true)

Load a custom spectrum from a file.

# Arguments
- `filename`: Path to the spectrum file
- `kvec_rs`: Wave numbers
- `Hs`: Significant wave height (used only if normalize=true)
- `normalize`: Whether to normalize the spectrum to match the specified Hs

# Returns
- Spectral density values
"""
function load_custom_spectrum(filename, kvec_rs, Hs; normalize=true)
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
    
    # Normalize the spectrum if requested
    if normalize
        # Normalize to ensure correct significant wave height
        dk = kvec_rs[2] - kvec_rs[1]
        scaling_factor = (Hs^2/16) / sum(Sk_vec * dk)
        return Sk_vec .* scaling_factor
    else
        # Use the spectrum values directly
        approx_Hs = 4 * sqrt(sum(Sk_vec * (kvec_rs[2] - kvec_rs[1])))
        @info "Using custom spectrum without normalization. Approximate Hs = $approx_Hs"
        return Sk_vec
    end
end

# Helper function for Gaussian spectrum
function generate_gaussian_spectrum(kvec_rs, Hs, k_p, k_w)
    Sk_vec = exp.(-((kvec_rs .- k_p).^2) ./ (2 * k_w^2))
    scaling_factor = (Hs^2/16) / sum(Sk_vec * (kvec_rs[2] - kvec_rs[1]))
    return Sk_vec .* scaling_factor
end

"""
    Dalzell_3D_xyt_YanLi_probe(t, x, y, t0, option, kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs)

Calculate second-order wave components for 2D waves with directional spreading at a single point.

# Arguments
- `t`: Time
- `x`: X-position
- `y`: Y-position
- `t0`: Initial time
- `option`: Dictionary with water depth options
- `kx_rs`: X-component of wave number
- `ky_rs`: Y-component of wave number
- `kvec_rs`: Magnitude of wave number
- `tht_rs`: Wave direction angles
- `omega_rs`: Angular frequencies
- `phase_rs`: Wave phases
- `amp_rs`: Wave amplitudes

# Returns
- `zeta20`: Second-order difference frequency component
- `zeta22`: Second-order sum frequency component
- `zeta1`: First-order component
"""
function Dalzell_3D_xyt_YanLi_probe(t, x, y, t0, option, kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs)
    # Li & Li (pof, 2021) or Li (JFM, 2021) 
    g = 9.81
    h = option[:h]
    
    # Initialize values
    amp_rs = amp_rs ./ 2
    zeta22 = 0.0
    zeta20 = 0.0
    zeta1 = 0.0
    
    # Calculate phase for all wave components at the given point
    psi1 = kx_rs .* x .+ ky_rs .* y .- omega_rs .* (t - t0) .+ phase_rs
    
    for iik in 1:length(kvec_rs)
        # Directional cosine term
        cos_tht = cos.(tht_rs .- tht_rs[iik])
        amp_12 = amp_rs .* amp_rs[iik]
        
        if option[:option] == "deepwater"
            dem_mean = (omega_rs .- omega_rs[iik]).^2 .- g .* abs.(kvec_rs .- kvec_rs[iik])
            dem_plus = (omega_rs .+ omega_rs[iik]).^2 .- g .* abs.(kvec_rs .+ kvec_rs[iik])
            
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

"""
    Dalzell_3D_xyt_YanLi_grid(t, xvec, yvec, t0, option, kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs)

Calculate second-order wave components for 2D waves over a grid.

# Arguments
- `t`: Time
- `xvec`: Vector of X-positions
- `yvec`: Vector of Y-positions
- `t0`: Initial time
- `option`: Dictionary with water depth options
- `kx_rs`: X-component of wave number
- `ky_rs`: Y-component of wave number
- `kvec_rs`: Magnitude of wave number
- `tht_rs`: Wave direction angles
- `omega_rs`: Angular frequencies
- `phase_rs`: Wave phases
- `amp_rs`: Wave amplitudes

# Returns
- `zeta20`: Second-order difference frequency component grid
- `zeta22`: Second-order sum frequency component grid
- `zeta1`: First-order component grid
"""
function Dalzell_3D_xyt_YanLi_grid(t, xvec, yvec, t0, option, kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs)
    # Li & Li (pof, 2021) or Li (JFM, 2021) 
    g = 9.81
    h = option[:h]
    
    # Initialize values
    amp_rs = amp_rs ./ 2
    zeta22 = zeros(length(yvec), length(xvec))
    zeta20 = zeros(length(yvec), length(xvec))
    zeta1 = zeros(length(yvec), length(xvec))
    
    for iik in 1:length(kvec_rs)
        # Directional cosine term for the current wave component
        cos_tht = cos.(tht_rs .- tht_rs[iik])
        amp_12 = amp_rs .* amp_rs[iik]
        
        if option[:option] == "deepwater"
            dem_mean = (omega_rs .- omega_rs[iik]).^2 .- g .* abs.(kvec_rs .- kvec_rs[iik])
            dem_plus = (omega_rs .+ omega_rs[iik]).^2 .- g .* abs.(kvec_rs .+ kvec_rs[iik])
            
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
        
        # Process each y-row
        for iiy in 1:length(yvec)
            y0 = yvec[iiy]
            # Calculate phase for all wave components at current x,y positions
            psi1 = kx_rs .* xvec' .+ ky_rs .* y0 .- omega_rs .* (t - t0) .+ phase_rs
            psi2 = psi1[iik, :]
            
            # Linear wave component
            zeta1_y = amp_rs .* cos.(psi1)
            
            # Calculate second-order wave components
            cos_plus = cos.(psi1 .+ psi2')
            cos_minus = cos.(psi1 .- psi2')
            
            zeta22[iiy, :] = vec(sum(amp_12 .* coeff_zt22 .* cos_plus, dims=1)) .+ zeta22[iiy, :]
            zeta20[iiy, :] = vec(sum(amp_12 .* coeff_zt20 .* cos_minus, dims=1)) .+ zeta20[iiy, :]
            zeta1[iiy, :] = vec(sum(zeta1_y, dims=1)) .+ zeta1[iiy, :]
        end
    end
    
    zeta22 = 2 * zeta22
    zeta20 = 2 * zeta20
    zeta1 = 2 * zeta1
    
    return zeta20, zeta22, zeta1
end

"""
    exceedance_probability(zeta, prescribed_Hs)

Calculate exceedance probability and wave statistics.

# Arguments
- `zeta`: Wave elevation time series
- `prescribed_Hs`: Target significant wave height

# Returns
- `Hs`: Calculated significant wave height
- `Hs_m0`: Significant wave height from variance
- `kurt`: Kurtosis
- `skew`: Skewness
- `exceed_prob`: Exceedance probability
- `thresholds`: Threshold values
"""
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

# Main processing function for 2D wave simulation at a probe location
function batch_run_seeded(seed, input, endtimeT_p)
    # Random params with directional spreading
    kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, amp_rs, phase_rs, spectrum_type, k_range = 
        obtain_directional_waveSpectra_irregular(
            input[:Hs], input[:eps_0], input[:k_p], input[:k_w], input[:tht_w], input[:option], 
            input[:Nk], input[:Ntht], seed, input[:spectrum_params]
        )
    
    # Initialize arrays for a single probe location
    zeta_1_single_probe = zeros(endtimeT_p * input[:Ntp])
    zeta_22_single_probe = zeros(endtimeT_p * input[:Ntp])
    zeta_20_single_probe = zeros(endtimeT_p * input[:Ntp])
    
    # Probe location
    x_probe = get(input, :x_probe, 0.0)
    y_probe = get(input, :y_probe, 0.0)
    
    # Process each time step
    for kk in 1:(endtimeT_p * input[:Ntp])
        zeta_20_single_probe[kk], zeta_22_single_probe[kk], zeta_1_single_probe[kk] = 
            Dalzell_3D_xyt_YanLi_probe(
                input[:time][kk], x_probe, y_probe, input[:t0], input[:option], 
                kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs
            )
    end
    
    return zeta_22_single_probe, zeta_20_single_probe, zeta_1_single_probe, spectrum_type, k_range
end

# Main processing function for 2D wave simulations over a grid (for a single time step)
function batch_run_grid(seed, input, time_step)
    # Random params with directional spreading
    kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, amp_rs, phase_rs, spectrum_type, k_range = 
        obtain_directional_waveSpectra_irregular(
            input[:Hs], input[:eps_0], input[:k_p], input[:k_w], input[:tht_w], input[:option], 
            input[:Nk], input[:Ntht], seed, input[:spectrum_params]
        )
    
    # Calculate wave field at the given time step
    t = input[:time][time_step]
    zeta20, zeta22, zeta1 = Dalzell_3D_xyt_YanLi_grid(
        t, input[:x], input[:y], input[:t0], input[:option], 
        kx_rs, ky_rs, kvec_rs, tht_rs, omega_rs, phase_rs, amp_rs
    )
    
    return zeta20, zeta22, zeta1, spectrum_type, k_range
end
