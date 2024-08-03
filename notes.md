# Technical Notes

## EPCA Constructors 

EPCA can be induced from (assumes no verification that the user inputed arguments are correct)[^1]:

1. $F$
2. $f$
3. $G$
4. $G, g$
5. $G, F$
6. $G, f$
7. $F, g$
8. $F, f$
9.  $F, Fg$
10. $F, fg$
11. $f, Fg$
12. $f, fg$
13. Bregman, $g$
14. Bregman, $G$
15. $F, f, Fg$
16. $F, f, fg$
17. $F, Fg, fg$
18. $f, Fg, fg$
19. $G, g$
20. $L(V, A)$
21. $F, f, Fg, fg, g$
22. $F, Fg, fg, g$

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