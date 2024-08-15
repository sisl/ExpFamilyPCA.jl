# Technical Notes

## EPCA Constructors 

EPCA can be induced from (assumes no verification that the user inputed arguments are correct)[^1]:

- [x] $F$
- [x] $G$
- [ ] $G, g$  # check for speedups with inverting g
- [x] $G, F$
- [ ] $G, f$  # figure out math
- [x] $F, g$  # check for speedups with inverting g
- [x] $F, f$
- [ ] $F, Fg$
- [ ] $F, fg$
- [ ] $f, Fg$
- [ ] $f, fg$
- [x] Bregman, $g$
- [x] Bregman, $G$
- [x] $F, f, G$
- [x] $F, f, g$
- [ ] $F, f, Fg$
- [ ] $F, f, fg$
- [ ] $F, Fg, fg$
- [ ] $f, Fg, fg$
- [ ] $G, g, f$
- [ ] $G, g$
- [ ] $L(V, A), g$
- [ ] $F, f, Fg, fg, g$
- [ ] $F, Fg, fg, g$

[^1]: These are only the constructions I could come up with in 20 minutes, there may be even more constructions.

### Rationale
* $F$ induces $f$
* $f$ induces $g_inverse$ and $g$
* $G$ induces $g$
* $g$ induces $g_inverse$ and $f$
* $g$ induces $F$

## Implementation Journey
We want to have multiple constructors for `EPCA`, but the problem is that Julia *only dispatches on positional arguments*, not keyword arguments. Moreover, it only dispatches on the *type* of positional arguments not their value. This means that you can't dispatch on the value of a boolean flag `foo(flag=true)` overwrites `foo(flag=false)` because both are `Bool` instances, i.e., `typeof(true) === typeof(false)`. This problem is addressed with Julia's ["Value types"](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22). While 


### TODO
- [ ] Rewrite utils.jl functions with loops using [seperate kernel functions](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions) to speed up Julia compiler.
- [ ] Add value initialization