"""
Rejection method. This method handles conditional intensity functions.
"""

mutable struct RegirJumpAggregation{T,S, F1, F2, RNG} <:
               AbstractSSAJumpAggregator
    next_jump::Int                    # the next jump to execute
    prev_jump::Int                    # the previous jump that was executed
    next_jump_time::T                 # the time of the next jump
    end_time::T                       # the time to stop a simulation
    cur_rates::Vector{T}              # vector of current propensity values
    sum_rate::Nothing                 # not used
    ma_jumps::S                       # MassActionJumps
    rates::F1                         # vector of rate functions
    affects!::F2                      # vector of affect functions for VariableRateJumps
    save_positions::Tuple{Bool, Bool} # tuple for whether to save the jumps before and/or after event
    rng::RNG                          # random number generator
    dep_gr::Nothing                       # map from jumps to jumps depending on it but we don't need it here as long as we don't have other type of jump
    urates::F1                        # vector of rate upper bound functions
    prev_jump_time::Vector{T}         # the time of the last jump for each reaction
    rmax::Real                        # current rmax
end

function RegirJumpAggregation(nj::Int, pj::Int, njt::T, et::T, crs::Vector{T}, sr::Nothing,
    maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
    rng::RNG; u::U, dep_graph::Nothing, urates, pjt::Vector{T}, rmax::Real) where {T, S, F1, F2, RNG, U}

    num_jumps = get_num_majumps(maj) + length(urates)

    RegirJumpAggregation{T, S, F1, F2, RNG}(nj, pj, njt, et, crs, sr, maj, rs, affs!, sps, rng,
           dep_graph, urates, pjt, rmax)
end



# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::Regir, u, p, t, end_time, condintensity_jumps,
                    ma_jumps, save_positions, rng; kwargs...)
    AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
                                                   Tuple{typeof(u), typeof(p), typeof(t)}}

    ncirjs = length(condintensity_jumps)
    
    affects! = Vector{AffectWrapper}(undef, ncirjs)
    rates = Vector{RateWrapper}(undef, ncirjs)
    urates = Vector{RateWrapper}(undef, ncirjs)

    idx = 1
    
    if condintensity_jumps !== nothing
        for (i, cij) in enumerate(condintensity_jumps)
            affects![idx] = AffectWrapper(integ -> (cij.affect!(integ); nothing))
            urates[idx] = RateWrapper(cij.urate)
            idx += 1
            rates[i] = RateWrapper(cij.rate)
        end
    end

    num_jumps = ncirjs
    cur_rates = Vector{typeof(t)}(undef, num_jumps)
    sum_rate = nothing
    dep_graph = nothing
    rmax=1.0
    next_jump = 0
    previous_jump=0
    next_jump_time = typemax(t)
    pjt = zeros(typeof(t),num_jumps)
    RegirJumpAggregation(next_jump, previous_jump, next_jump_time, end_time, cur_rates, sum_rate,
                            ma_jumps, rates, affects!, save_positions, rng; u,
                            dep_graph, urates, pjt, rmax)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RegirJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RegirJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)
    # update current jump rates and times
    update_prevjumptimes!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RegirJumpAggregation, integrator, u, params, t)
    @unpack end_time = p
    update_rmax!(p,u,params,t)
    p.next_jump_time,  p.next_jump = next_time(p, u, params, t, end_time)    
    nothing
end


######################## SSA specific helper routines ########################
function update_prevjumptimes!(p::RegirJumpAggregation, u, params, t)
    @unpack next_jump, next_jump_time = p
    p.prev_jump_time[next_jump] = next_jump_time
    nothing
end

#= @inline function get_ma_urate(p::RegirJumpAggregation, i, u, params, t)
    return evalrxrate(u, i, p.ma_jumps)
end =#

@inline function get_urate(p::RegirJumpAggregation, uidx, u, params, t)
    @inbounds return p.urates[uidx](u, params, t)
end


@inline function get_rate(p::RegirJumpAggregation, lidx, u, params, t)
    @inbounds return p.rates[lidx](u, params, t)
end


function update_rmax!(p::RegirJumpAggregation, u, params, t)  
"""
Update rmax
"""
    @unpack prev_jump_time = p
    rmax=0;
    for i in eachindex(prev_jump_time)                   
        t_start_min = prev_jump_time[i];
        tau_max = t - t_start_min
        rmax_temp = get_urate(p, i, u, params, tau_max);
        if isnan(rmax_temp) || isinf(rmax_temp)
            println("Problem computed rmax is", rmax_temp, "for time %.2e", tau_max)
            rmax_temp = 5*p.rates[i]; #most distribution rarely pass 5*r0
        end
        if rmax_temp>rmax
            rmax=rmax_temp;
        end
    end
    p.rmax=rmax;
    nothing
end

function calculate_rmax!(p::RegirJumpAggregation, u, params, t)  
    """
    Calculate and retrun rmax
    """
        @unpack prev_jump_time = p
        rmax=0;
        for i in eachindex(prev_jump_time)                   
            t_start_min = prev_jump_time[i];
            tau_max = t - t_start_min
            rmax_temp = get_urate(p, i, u, params, tau_max);
            if isnan(rmax_temp) || isinf(rmax_temp)
                println("Problem computed rmax is", rmax_temp, "for time %.2e", tau_max)
                rmax_temp = 5*p.rates[i]; #most distribution rarely pass 5*r0
            end
            if rmax_temp>rmax
                rmax=rmax_temp;
            end
        end
        return rmax
    end

function next_time(p::RegirJumpAggregation{T}, u, params, t, tstop::T) where {T}
    @unpack rng, prev_jump_time = p
    num_jumps = length(p.urates)

    t_current=t
    rmax=calculate_rmax!(p, u, params, t_current)
    a0=num_jumps*rmax;
    s = a0 == zero(t) ? typemax(t) : randexp(rng) / a0
    _ttemp = t_current + s
    rejectReaction=true
    mu=0
    while rejectReaction
        # choose the reaction mu that will occurs
        mu = rand(1:num_jumps)
        rnum2 = rand(rng)
        tau_max=t_current-prev_jump_time[mu]
        #println((get_rate(p,mu,u,params,tau_max), mu, tau_max))
        if rnum2 <= get_rate(p,mu,u,params,tau_max) / rmax
            rejectReaction = false
        end
        t_current=_ttemp
        rmax=calculate_rmax!(p, u, params, t_current)
        a0=num_jumps*rmax;
        s = a0 == zero(t) ? typemax(t) : randexp(rng) / a0
        _ttemp = t_current + s
        if _ttemp>tstop
            t_current = tstop
            mu = 1 #attention this should be change for a empty reaction
            rejectReaction = false
        end
    end
    return t_current, mu
end

