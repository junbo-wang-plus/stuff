#!/usr/bin/env julia

# Load required packages
using Distributed
using Statistics
using Random
using FFTW
using JLD2
using TOML
using Plots
using ArgParse

# Parse command line arguments 
function parse_commandline()
    s = ArgParseSettings(description="Run 2D wave simulation with directional spreading")
    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to configuration file"
            arg_type = String
            default = "wave_config_2d.toml"
        "--plot-only", "-p"
            help = "Only generate plots from existing results"
            action = :store_true
        "--results", "-r"
            help = "Path to results file"
            arg_type = String
            default = "wave_batch_results_2d.jld2"
        "--no-plots"
            help = "Skip generating plots"
            action = :store_true
    end
    return parse_args(s)
end

# Get arguments
args = parse_commandline()

# Load config first
config = TOML.parsefile(args["config"])
use_parallel = get(get(config, "simulation", Dict()), "use_parallel", false)

# Set up workers if needed
if use_parallel && nprocs() == 1 && !args["plot-only"]
    num_workers = get(get(config, "simulation", Dict()), "num_workers", 0)
    if num_workers <= 0
        num_workers = min(Sys.CPU_THREADS - 1, 16)
    end
    addprocs(num_workers)
    @info "Added $(nprocs()-1) worker processes"
end

# Include all functions on all processes
@everywhere begin
    using Statistics
    using Random
    using FFTW
    using LinearAlgebra
end

# Include wave_functions.jl on all workers
@everywhere include("wave_functions_2d.jl")

# Load functions for plotting
include("WavePlot_2d.jl")
using .WavePlot

# === Load Config Function ===
function load_config(config_file)
    config = TOML.parsefile(config_file)
    
    input = Dict{Symbol, Any}()
    
    # Copy all parameters from the config
    for (key, value) in config["parameters"]
        input[Symbol(key)] = value
    end
    
    # Extract spectrum parameters
    spectrum_params = Dict{String, Any}()
    spectrum_params["spectrum_type"] = get(config["parameters"], "spectrum_type", "gaussian")
    
    # Get normalize_spectrum parameter
    spectrum_params["normalize_spectrum"] = get(config["parameters"], "normalize_spectrum", true)
    
    # Handle 2D spectrum file if specified
    if spectrum_params["spectrum_type"] == "spectrum2d"
        spectrum_params["spectrum2d_file"] = get(config["parameters"], "spectrum2d_file", "")
        spectrum_params["spectrum2d_format"] = get(config["parameters"], "spectrum2d_format", "auto")
    end
    
    # Get spectrum-specific parameters
    if spectrum_params["spectrum_type"] == "jonswap"
        spectrum_params["gamma"] = get(config["parameters"], "gamma", 3.3)
        spectrum_params["sigma_a"] = get(config["parameters"], "sigma_a", 0.07)
        spectrum_params["sigma_b"] = get(config["parameters"], "sigma_b", 0.09)
    elseif spectrum_params["spectrum_type"] == "custom"
        spectrum_params["spectrum_file"] = get(config["parameters"], "spectrum_file", "custom_spectrum.csv")
    end
    
    # Get wave number range parameters
    spectrum_params["kmin_factor"] = get(config["parameters"], "kmin_factor", 4.0)
    spectrum_params["kmax_factor"] = get(config["parameters"], "kmax_factor", 4.0)
    spectrum_params["custom_kmin"] = get(config["parameters"], "custom_kmin", -1.0)
    spectrum_params["custom_kmax"] = get(config["parameters"], "custom_kmax", -1.0)
    
    # Store spectrum parameters
    input[:spectrum_params] = spectrum_params
    
    # Apply k_w_relative if set
    if haskey(input, Symbol("k_w_relative")) && input[:k_w_relative]
        input[:k_w] = input[:k_w] * input[:k_p]
    end
    
    # Calculate derived parameters
    input[:eps_0] = get(input, :k_p, 0.0) * input[:Hs] / 2  # steepness (only if k_p is defined)
    input[:h] = input[:kph] / get(input, :k_p, 1.0)  # Use k_p if defined, otherwise default to 1.0
    input[:g] = 9.81
    
    # Calculate wave parameters if k_p is available (not needed for 2D spectrum files)
    if haskey(input, :k_p)
        input[:omega_p] = sqrt(input[:g] * input[:k_p] * tanh(input[:kph]))
        input[:f_p] = input[:omega_p] / (2 * π)
        input[:T_p] = 1 / input[:f_p]
        input[:lambda_p] = 2 * π / input[:k_p]
        input[:cg_p] = input[:omega_p] / input[:k_p] / 2 * (1 + 2 * input[:kph] / sinh(2 * input[:kph]))
    else
        # For 2D spectrum files, require T_p to be specified directly
        if !haskey(input, :T_p)
            error("When using a 2D spectrum file without k_p, you must specify T_p directly")
        end
        input[:f_p] = 1 / input[:T_p]
        # Derive lambda_p from water depth and period if needed
        if !haskey(input, :lambda_p)
            # Estimate k_p from the dispersion relation
            k_est = 0.0
            if input[:option] == "deepwater"
                k_est = (2 * π * input[:f_p])^2 / input[:g]
            else
                # Iterate to find k_p from the dispersion relation
                omega_p = 2 * π * input[:f_p]
                k_est = omega_p^2 / input[:g]  # Initial deepwater estimate
                for _ in 1:10  # Few iterations should converge
                    k_est = omega_p^2 / (input[:g] * tanh(k_est * input[:h]))
                end
            end
            input[:lambda_p] = 2 * π / k_est
        end
    end
    
    input[:dt] = input[:T_p] / input[:Ntp]  # time step determined from peak period
    input[:time] = (0:(input[:Nperiod] * input[:Ntp] - 1)) .* input[:dt] .+ input[:t0]
    input[:Nt] = input[:Nperiod] * input[:Ntp]
    
    # X-direction spatial grid
    input[:Nx] = input[:ppp] * input[:Nxperiod] # points per period
    input[:Lx] = input[:Nxperiod] * input[:lambda_p] # spatial range for periods
    input[:dx] = input[:Lx] / input[:Nx]
    input[:x] = (0:(input[:Nx] - 1)) .* input[:dx] .- input[:Lx] / 2
    
    # Y-direction spatial grid
    nyperiod = haskey(input, :Nyperiod) ? input[:Nyperiod] : input[:Nxperiod]
    ppp_y = haskey(input, :ppp_y) ? input[:ppp_y] : input[:ppp]
    
    input[:Ny] = ppp_y * nyperiod # points per period
    input[:Ly] = nyperiod * input[:lambda_p] # spatial range for periods
    input[:dy] = input[:Ly] / input[:Ny]
    input[:y] = (0:(input[:Ny] - 1)) .* input[:dy] .- input[:Ly] / 2
    
    # We always use directional spreading in this simplified version
    input[:is_directional] = true
    
    # Ensure probe positions are included
    input[:x_probe] = get(config["parameters"], "x_probe", 0.0)
    input[:y_probe] = get(config["parameters"], "y_probe", 0.0)
    
    # Use a Dictionary for option
    input[:option] = Dict(:option => config["parameters"]["water_type"], :h => input[:h])
    
    return input, config["simulation"], config["output"]
end

# === Run Simulation Function ===
function run_simulation(config_file)
    # Load configuration
    input, sim_config, output_config = load_config(config_file)
    
    # Extract simulation parameters
    N_test = get(sim_config, "N_test", 100)
    seed_max = get(sim_config, "seed_max", 500)
    seed_values = shuffle(collect(1:seed_max))[1:N_test]
    endtimeT_p = input[:Nperiod]
    simulation_type = get(sim_config, "simulation_type", "probe")
    
    # For grid simulations, get the time steps to calculate
    grid_time_steps = []
    if simulation_type == "grid"
        grid_time_steps = get(sim_config, "grid_time_steps", [100, 200, 300])
    end
    
    println("Starting batch run with $N_test simulations...")
    println("Simulation type: $simulation_type")
    t_start = time()
    
    # Parallel batch run - using a much simpler approach that works with distributed
    results = Vector{Dict{String, Any}}(undef, N_test)
    
    # Define the task function for probe simulations
    function run_single_probe_sim(i)
        @info "Starting probe simulation $i with seed $(seed_values[i])"
        t_sim_start = time()
        
        # Run the simulation
        zeta_22, zeta_20, zeta_1, spectrum_type, k_range = batch_run_seeded(seed_values[i], input, endtimeT_p)
        zeta_total = zeta_22 + zeta_20 + zeta_1
        
        # Calculate statistics
        Hs, Hs_m0, kurtosis, skewness, exceed_prob, thresholds = 
            exceedance_probability(zeta_total, input[:Hs])
        
        t_sim_end = time()
        @info "Completed simulation $i in $(round(t_sim_end - t_sim_start, digits=2)) seconds"
        
        # Return a dictionary with the results
        return Dict(
            "index" => i,
            "zeta_22" => zeta_22,
            "zeta_20" => zeta_20,
            "zeta_1" => zeta_1,
            "zeta_total" => zeta_total,
            "Hs" => Hs,
            "Hs_m0" => Hs_m0,
            "kurtosis" => kurtosis,
            "skewness" => skewness,
            "exceed_prob" => exceed_prob,
            "thresholds" => thresholds,
            "seed" => seed_values[i],
            "spectrum_type" => spectrum_type,
            "k_range" => k_range
        )
    end
    
    # Define the task function for grid simulations
    function run_single_grid_sim(i)
        @info "Starting grid simulation $i with seed $(seed_values[i])"
        t_sim_start = time()
        
        # Results structure for grid simulation
        grid_results = Dict(
            "index" => i,
            "seed" => seed_values[i],
            "time_steps" => grid_time_steps,
            "grid_data" => Dict()
        )
        
        # Only save total field if specified
        save_total_only = get(sim_config, "save_total_only", false)
        
        # Process each time step
        for time_step in grid_time_steps
            @info "Processing time step $time_step for simulation $i"
            zeta_20, zeta_22, zeta_1, spectrum_type, k_range = 
                batch_run_grid(seed_values[i], input, time_step)
            
            # Calculate total elevation
            zeta_total = zeta_20 + zeta_22 + zeta_1
            
            # Store the results for this time step
            grid_results["grid_data"][time_step] = Dict(
                "zeta_total" => zeta_total
            )
            
            # Only keep component fields if requested
            if !save_total_only
                grid_results["grid_data"][time_step]["zeta_20"] = zeta_20
                grid_results["grid_data"][time_step]["zeta_22"] = zeta_22
                grid_results["grid_data"][time_step]["zeta_1"] = zeta_1
            end
            
            # Store spectrum info
            grid_results["spectrum_type"] = spectrum_type
            grid_results["k_range"] = k_range
        end
        
        t_sim_end = time()
        @info "Completed grid simulation $i in $(round(t_sim_end - t_sim_start, digits=2)) seconds"
        
        return grid_results
    end
    
    # Choose which simulation function to use
    run_sim_func = simulation_type == "grid" ? run_single_grid_sim : run_single_probe_sim
    
    # Run simulations in parallel (without @distributed)
    if nprocs() > 1
        # Use pmap for parallel execution
        results = pmap(run_sim_func, 1:N_test)
    else
        # Sequential execution
        for i in 1:N_test
            results[i] = run_sim_func(i)
        end
    end
    
    t_end = time()
    println("All simulations completed in $(round(t_end - t_start, digits=2)) seconds")
    
    # Sort results by index
    sort!(results, by=r -> r["index"])
    
    # Process results based on simulation type
    batch = Dict{Symbol, Any}()
    
    if simulation_type == "probe"
        # Extract batch information into arrays for analysis
        for field in ["zeta_22", "zeta_20", "zeta_1", "zeta_total", "Hs", "Hs_m0", "kurtosis", 
                     "skewness", "exceed_prob", "thresholds", "seed", "spectrum_type", "k_range"]
            batch[Symbol(field)] = [r[field] for r in results]
        end
    else # grid simulation
        # Store grid results
        batch[:grid_results] = results
        batch[:time_steps] = grid_time_steps
        batch[:seed] = [r["seed"] for r in results]
        batch[:spectrum_type] = results[1]["spectrum_type"]
        batch[:k_range] = results[1]["k_range"]
    end
    
    # Add simulation parameters to the batch
    batch[:parameters] = Dict(
        :Hs => input[:Hs],
        :h => input[:h],
        :T_p => input[:T_p],
        :Nperiod => input[:Nperiod],
        :option => input[:option][:option],
        :Nk => input[:Nk],
        :Ntht => input[:Ntht],
        :tht_w => input[:tht_w], 
        :N_test => N_test,
        :spectrum_type => input[:spectrum_params]["spectrum_type"],
        :simulation_type => simulation_type,
        :is_directional => true
    )
    
    # Add k_p if it exists
    if haskey(input, :k_p)
        batch[:parameters][:k_p] = input[:k_p]
        batch[:parameters][:k_w] = input[:k_w]
        batch[:parameters][:eps_0] = input[:eps_0]
    end
    
    # Add spatial grid information
    batch[:parameters][:x] = input[:x]
    batch[:parameters][:y] = input[:y]
    batch[:parameters][:x_probe] = input[:x_probe]
    batch[:parameters][:y_probe] = input[:y_probe]
    
    # Add wave number range parameters
    batch[:parameters][:kmin_factor] = get(input[:spectrum_params], "kmin_factor", 4.0)
    batch[:parameters][:kmax_factor] = get(input[:spectrum_params], "kmax_factor", 4.0) 
    batch[:parameters][:custom_kmin] = get(input[:spectrum_params], "custom_kmin", -1.0)
    batch[:parameters][:custom_kmax] = get(input[:spectrum_params], "custom_kmax", -1.0)
    batch[:parameters][:normalize_spectrum] = get(input[:spectrum_params], "normalize_spectrum", false)
    
    if length(batch[:k_range]) > 0
        batch[:parameters][:actual_kmin] = batch[:k_range][1][1]
        batch[:parameters][:actual_kmax] = batch[:k_range][1][2]
    end
    
    # Add additional spectrum parameters
    if input[:spectrum_params]["spectrum_type"] == "jonswap"
        batch[:parameters][:gamma] = input[:spectrum_params]["gamma"]
        batch[:parameters][:sigma_a] = input[:spectrum_params]["sigma_a"]
        batch[:parameters][:sigma_b] = input[:spectrum_params]["sigma_b"]
    elseif input[:spectrum_params]["spectrum_type"] == "custom"
        batch[:parameters][:spectrum_file] = input[:spectrum_params]["spectrum_file"]
    elseif input[:spectrum_params]["spectrum_type"] == "spectrum2d"
        batch[:parameters][:spectrum2d_file] = input[:spectrum_params]["spectrum2d_file"]
    end
    
    # Save results
    filename = get(output_config, "filename", "wave_batch_results_2d.jld2")
    @save filename batch
    println("Results saved to $filename")
    
    # Return for potential further processing
    return batch, input
end

# Main function
function main()
    if args["plot-only"]
        # Load existing results and generate plots
        println("Loading results from $(args["results"]) and generating plots...")
        batch = WavePlot.load_results(args["results"])
        WavePlot.generate_plots(batch, args["config"])
    else
        # Run the simulation
        println("Running 2D wave simulation using config file: $(args["config"])")
        
        # Run the simulation
        batch, input = run_simulation(args["config"])
        
        # Generate plots if not disabled
        if !args["no-plots"]
            println("Generating plots...")
            WavePlot.generate_plots(batch, args["config"])
        end
    end
    
    println("Done!")
end

# Run the main function
main()
