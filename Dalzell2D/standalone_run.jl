#!/usr/bin/env julia

# First, load required packages on main process
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
    s = ArgParseSettings(description="Run wave simulation")
    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to configuration file"
            arg_type = String
            default = "wave_config.toml"
        "--plot-only", "-p"
            help = "Only generate plots from existing results"
            action = :store_true
        "--results", "-r"
            help = "Path to results file"
            arg_type = String
            default = "wave_batch_results.jld2"
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
end

# Include wave_functions.jl on all workers
@everywhere include("wave_functions.jl")

# Load functions for plotting
include("WavePlot.jl")
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
    input[:eps_0] = input[:k_p] * input[:Hs] / 2  # steepness
    input[:h] = input[:kph] / input[:k_p]
    input[:g] = 9.81
    input[:omega_p] = sqrt(input[:g] * input[:k_p] * tanh(input[:kph]))
    input[:f_p] = input[:omega_p] / (2 * π)
    input[:T_p] = 1 / input[:f_p]
    input[:lambda_p] = 2 * π / input[:k_p]
    input[:cg_p] = input[:omega_p] / input[:k_p] / 2 * (1 + 2 * input[:kph] / sinh(2 * input[:kph]))
    
    input[:dt] = input[:T_p] / input[:Ntp]  # time step determined from peak period
    input[:time] = (0:(input[:Nperiod] * input[:Ntp] - 1)) .* input[:dt] .+ input[:t0]
    input[:Nt] = input[:Nperiod] * input[:Ntp]
    
    input[:Nx] = input[:ppp] * input[:Nxperiod] # points per period
    input[:Lx] = input[:Nxperiod] * input[:lambda_p] # spatial range for periods
    input[:dx] = input[:Lx] / input[:Nx]
    input[:x] = (0:(input[:Nx] - 1)) .* input[:dx] .- input[:Lx] / 2
    
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
    
    println("Starting batch run with $N_test simulations...")
    t_start = time()
    
    # Parallel batch run - using a much simpler approach that works with distributed
    results = Vector{Dict{String, Any}}(undef, N_test)
    
    # Define the task function here in main scope
    function run_single_sim(i)
        @info "Starting simulation $i with seed $(seed_values[i])"
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
    
    # Run simulations in parallel (without @distributed)
    if nprocs() > 1
        # Use pmap for parallel execution
        results = pmap(run_single_sim, 1:N_test)
    else
        # Sequential execution
        for i in 1:N_test
            results[i] = run_single_sim(i)
        end
    end
    
    t_end = time()
    println("All simulations completed in $(round(t_end - t_start, digits=2)) seconds")
    
    # Sort results by index
    sort!(results, by=r -> r["index"])
    
    # Extract batch information into arrays for analysis
    batch = Dict{Symbol, Any}()
    
    for field in ["zeta_22", "zeta_20", "zeta_1", "zeta_total", "Hs", "Hs_m0", "kurtosis", 
                 "skewness", "exceed_prob", "thresholds", "seed", "spectrum_type", "k_range"]
        batch[Symbol(field)] = [r[field] for r in results]
    end
    
    # Add simulation parameters to the batch
    batch[:parameters] = Dict(
        :k_p => input[:k_p],
        :k_w => input[:k_w],
        :Hs => input[:Hs],
        :eps_0 => input[:eps_0],
        :h => input[:h],
        :T_p => input[:T_p],
        :Nperiod => input[:Nperiod],
        :option => input[:option][:option],
        :Nk => input[:Nk],
        :N_test => N_test,
        :spectrum_type => input[:spectrum_params]["spectrum_type"]
    )
    
    # Add wave number range parameters
    batch[:parameters][:kmin_factor] = get(input[:spectrum_params], "kmin_factor", 4.0)
    batch[:parameters][:kmax_factor] = get(input[:spectrum_params], "kmax_factor", 4.0) 
    batch[:parameters][:custom_kmin] = get(input[:spectrum_params], "custom_kmin", -1.0)
    batch[:parameters][:custom_kmax] = get(input[:spectrum_params], "custom_kmax", -1.0)
    batch[:parameters][:actual_kmin] = batch[:k_range][1][1]
    batch[:parameters][:actual_kmax] = batch[:k_range][1][2]
    
    # Add additional spectrum parameters
    if input[:spectrum_params]["spectrum_type"] == "jonswap"
        batch[:parameters][:gamma] = input[:spectrum_params]["gamma"]
        batch[:parameters][:sigma_a] = input[:spectrum_params]["sigma_a"]
        batch[:parameters][:sigma_b] = input[:spectrum_params]["sigma_b"]
    elseif input[:spectrum_params]["spectrum_type"] == "custom"
        batch[:parameters][:spectrum_file] = input[:spectrum_params]["spectrum_file"]
    end
    
    # Save results
    filename = get(output_config, "filename", "wave_batch_results.jld2")
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
        batch = load_results(args["results"])
        generate_plots(batch, args["config"])
    else
        # Run the simulation
        println("Running wave simulation using config file: $(args["config"])")
        
        # Run the simulation
        batch, input = run_simulation(args["config"])
        
        # Generate plots if not disabled
        if !args["no-plots"]
            println("Generating plots...")
            generate_plots(batch, args["config"])
        end
    end
    
    println("Done!")
end

# Run the main function
main()
