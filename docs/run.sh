# Run this script from the docs/ directory to build and view documentation locally

julia --project make.jl
cd ..
julia -e 'using LiveServer; serve(dir="docs/build")'