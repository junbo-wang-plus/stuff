#!/usr/bin/env julia

using ArgParse

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings(description="Generate plots from existing simulation results")
    @add_arg_table! s begin
        "--results", "-r"
            help = "Path to results file"
            arg_type = String
            default = "wave_batch_results.jld2"
        "--config", "-c"
            help = "Path to configuration file"
            arg_type = String
            default = "wave_config.toml"
        "--output-dir", "-o"
            help = "Output directory for plots"
            arg_type = String
    end
    return parse_args(s)
end

# Include the WavePlot module
include("WavePlot.jl")
using .WavePlot
using TOML

function main()
    args = parse_commandline()
    
    # Load the results
    println("Loading results from $(args["results"])...")
    batch = load_results(args["results"])
    
    # Update output directory if specified
    if haskey(args, "output-dir") && !isnothing(args["output-dir"])
        # Modify config to use specified output directory
        config = TOML.parsefile(args["config"])
        if !haskey(config, "output")
            config["output"] = Dict()
        end
        config["output"]["directory"] = args["output-dir"]
        
        # Write temporary config file
        temp_config = "temp_config.toml"
        open(temp_config, "w") do io
            TOML.print(io, config)
        end
        
        # Use the temporary config file
        generate_plots(batch, temp_config)
        
        # Clean up
        rm(temp_config)
    else
        # Use the provided config file
        generate_plots(batch, args["config"])
    end
    
    println("Done!")
end

main()
