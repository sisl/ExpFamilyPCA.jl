# TODO: add use eval path


function _symbolics_to_julia(
    symbolics_expression::Num,
    symbolics_variable::Num;
)
    # NOTE: make sure that derivatives are expanded with expand_derivatives before calling the helper
    fn = x->begin
        result = substitute(
            symbolics_expression,
            Dict(symbolics_variable => x)
        ).val
        return result
    end
    return fn
end

function _binary_search_monotone(
    f::Function,
    target::Real; 
    low=-1e10,
    high=1e10,
    tol=1e-10,
    maxiter=1e6
)
    iter = 0
    while high - low > tol && iter < maxiter
        mid = (low + high) / 2
        if f(mid) < target
            low = mid
        else
            high = mid
        end
        iter += 1
    end
    return (low + high) / 2
end
