#!/usr/bin/env julia

using ArgParse
using JLD2
using Statistics
using TOML
using Printf

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings(
        description="Extract wave statistics from simulation results",
        prog="wave_stats.jl"
    )
    
    @add_arg_table! s begin
        "--results", "-r"
            help = "Path to results file (JLD2)"
            arg_type = String
            default = "wave_batch_results.jld2"
        "--config", "-c"
            help = "Path to configuration file (TOML)"
            arg_type = String
            default = "wave_config.toml"
        "--format", "-f"
            help = "Output format (text, csv, json)"
            arg_type = String
            default = "text"
        "--output", "-o"
            help = "Output file (if not specified, prints to stdout)"
            arg_type = String
    end
    
    return parse_args(s)
end

# Load results from JLD2 file
function load_results(filename)
    @info "Loading results from $filename"
    results = nothing
    try
        @load filename batch
        return batch
    catch e
        error("Failed to load results from $filename: $e")
    end
end

# Extract statistics from the results
function extract_statistics(batch)
    stats = Dict{String, Any}()
    
    # Get number of simulations
    N_test = length(batch[:zeta_total])
    
    # Extract parameters
    params = batch[:parameters]
    
    # Calculate average statistics
    stats["num_simulations"] = N_test
    
    # Calculate mean of each component and the total elevation for each simulation
    linear_means = [mean(batch[:zeta_1][i]) for i in 1:N_test]
    zeta20_means = [mean(batch[:zeta_20][i]) for i in 1:N_test]
    zeta22_means = [mean(batch[:zeta_22][i]) for i in 1:N_test]
    total_means = [mean(batch[:zeta_total][i]) for i in 1:N_test]
    
    stats["mean_linear"] = mean(linear_means)
    stats["mean_linear_std"] = std(linear_means)
    stats["mean_zeta20"] = mean(zeta20_means)
    stats["mean_zeta20_std"] = std(zeta20_means)
    stats["mean_zeta22"] = mean(zeta22_means) 
    stats["mean_zeta22_std"] = std(zeta22_means)
    stats["mean_elevation"] = mean(total_means)
    stats["mean_elevation_std"] = std(total_means)
    
    # Normal statistics
    stats["skewness"] = mean(batch[:skewness])
    stats["skewness_std"] = std(batch[:skewness])
    stats["kurtosis"] = mean(batch[:kurtosis])
    stats["kurtosis_std"] = std(batch[:kurtosis])
    stats["Hs_time"] = mean(batch[:Hs])
    stats["Hs_time_std"] = std(batch[:Hs])
    stats["Hs_m0"] = mean(batch[:Hs_m0])
    stats["Hs_m0_std"] = std(batch[:Hs_m0])
    
    # Include key parameters
    stats["k_p"] = params[:k_p]
    stats["Hs_target"] = params[:Hs]
    stats["eps_0"] = params[:eps_0]
    stats["water_depth"] = params[:h]
    stats["water_type"] = params[:option]
    stats["spectrum_type"] = params[:spectrum_type]
    
    return stats
end

# Format statistics as text
function format_as_text(stats)
    lines = [
        "Wave Simulation Statistics",
        "==========================",
        "",
        "Simulation Parameters:",
        "---------------------",
        @sprintf("Spectrum Type: %s", stats["spectrum_type"]),
        @sprintf("Water Type: %s", stats["water_type"]),
        @sprintf("Peak Wave Number (k_p): %.4f", stats["k_p"]),
        @sprintf("Target Significant Wave Height (Hs): %.4f m", stats["Hs_target"]),
        @sprintf("Wave Steepness (eps_0): %.4f", stats["eps_0"]),
        @sprintf("Water Depth: %.4f m", stats["water_depth"]),
        "",
        @sprintf("Statistics Summary (from %d simulations):", stats["num_simulations"]),
        "--------------------------------------",
        @sprintf("Wave Component Means:"),
        @sprintf("  Linear (zeta1):   %.8f ± %.8f m", stats["mean_linear"], stats["mean_linear_std"]),
        @sprintf("  Set-down (zeta20): %.8f ± %.8f m", stats["mean_zeta20"], stats["mean_zeta20_std"]),
        @sprintf("  Sum-freq (zeta22): %.8f ± %.8f m", stats["mean_zeta22"], stats["mean_zeta22_std"]),
        @sprintf("  Total Elevation:  %.8f ± %.8f m", stats["mean_elevation"], stats["mean_elevation_std"]),
        "",
        @sprintf("Skewness: %.4f ± %.4f", stats["skewness"], stats["skewness_std"]),
        @sprintf("Kurtosis: %.4f ± %.4f", stats["kurtosis"], stats["kurtosis_std"]),
        "",
        @sprintf("Significant Wave Height (time domain): %.4f ± %.4f m", stats["Hs_time"], stats["Hs_time_std"]),
        @sprintf("Significant Wave Height (4σ): %.4f ± %.4f m", stats["Hs_m0"], stats["Hs_m0_std"]),
    ]
    
    return join(lines, "\n")
end

# Format statistics as CSV
function format_as_csv(stats)
    header = "statistic,value,std_dev"
    
    lines = [
        "spectrum_type,$(stats["spectrum_type"]),",
        "water_type,$(stats["water_type"]),",
        "k_p,$(stats["k_p"]),",
        "Hs_target,$(stats["Hs_target"]),",
        "eps_0,$(stats["eps_0"]),",
        "water_depth,$(stats["water_depth"]),",
        "num_simulations,$(stats["num_simulations"]),",
        "mean_linear,$(stats["mean_linear"]),$(stats["mean_linear_std"])",
        "mean_zeta20,$(stats["mean_zeta20"]),$(stats["mean_zeta20_std"])",
        "mean_zeta22,$(stats["mean_zeta22"]),$(stats["mean_zeta22_std"])",
        "mean_elevation,$(stats["mean_elevation"]),$(stats["mean_elevation_std"])",
        "skewness,$(stats["skewness"]),$(stats["skewness_std"])",
        "kurtosis,$(stats["kurtosis"]),$(stats["kurtosis_std"])",
        "Hs_time,$(stats["Hs_time"]),$(stats["Hs_time_std"])",
        "Hs_m0,$(stats["Hs_m0"]),$(stats["Hs_m0_std"])"
    ]
    
    return header * "\n" * join(lines, "\n")
end

# Format statistics as JSON
function format_as_json(stats)
    # Simple JSON formatting without external dependencies
    lines = [
        "{",
        "  \"simulation_parameters\": {",
        "    \"spectrum_type\": \"$(stats["spectrum_type"])\",",
        "    \"water_type\": \"$(stats["water_type"])\",",
        "    \"k_p\": $(stats["k_p"]),",
        "    \"Hs_target\": $(stats["Hs_target"]),",
        "    \"eps_0\": $(stats["eps_0"]),",
        "    \"water_depth\": $(stats["water_depth"])",
        "  },",
        "  \"statistics\": {",
        "    \"num_simulations\": $(stats["num_simulations"]),",
        "    \"mean_linear\": {",
        "      \"mean\": $(stats["mean_linear"]),",
        "      \"std\": $(stats["mean_linear_std"])",
        "    },",
        "    \"mean_zeta20\": {",
        "      \"mean\": $(stats["mean_zeta20"]),",
        "      \"std\": $(stats["mean_zeta20_std"])",
        "    },",
        "    \"mean_zeta22\": {",
        "      \"mean\": $(stats["mean_zeta22"]),",
        "      \"std\": $(stats["mean_zeta22_std"])",
        "    },",
        "    \"mean_elevation\": {",
        "      \"mean\": $(stats["mean_elevation"]),",
        "      \"std\": $(stats["mean_elevation_std"])",
        "    },",
        "    \"skewness\": {",
        "      \"mean\": $(stats["skewness"]),",
        "      \"std\": $(stats["skewness_std"])",
        "    },",
        "    \"kurtosis\": {",
        "      \"mean\": $(stats["kurtosis"]),",
        "      \"std\": $(stats["kurtosis_std"])",
        "    },",
        "    \"Hs_time\": {",
        "      \"mean\": $(stats["Hs_time"]),",
        "      \"std\": $(stats["Hs_time_std"])",
        "    },",
        "    \"Hs_m0\": {",
        "      \"mean\": $(stats["Hs_m0"]),",
        "      \"std\": $(stats["Hs_m0_std"])",
        "    }",
        "  }",
        "}"
    ]
    
    return join(lines, "\n")
end

# Main function
function main()
    args = parse_commandline()
    
    # Load results
    batch = load_results(args["results"])
    
    # Extract statistics
    stats = extract_statistics(batch)
    
    # Format output based on selected format
    output_text = ""
    if args["format"] == "text"
        output_text = format_as_text(stats)
    elseif args["format"] == "csv"
        output_text = format_as_csv(stats)
    elseif args["format"] == "json"
        output_text = format_as_json(stats)
    else
        @warn "Unknown format: $(args["format"]). Using text format."
        output_text = format_as_text(stats)
    end
    
    # Output to file or stdout
    if isnothing(args["output"]) || isempty(args["output"])
        println(output_text)
    else
        open(args["output"], "w") do io
            write(io, output_text)
        end
        println("Statistics written to $(args["output"])")
    end
end

# Run the script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
