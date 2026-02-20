## BurgersEqFV.jl

### Learn how to code the 1D viscous Burgers' equation solver in Julia!

We are interested in solving the 1D viscous Burgers' equation

$\partial_t u + \partial_x(u^2/2)=\nu\partial_{xx}u$

using the finite volume (FV) method. Additionally, we will use Julia's automatic differentiation (AD) tools to find optimal numerical scheme to compute it.

### Run the notebook

The [notebook.jl](notebook.jl) contains the tutorial building up the solver. To run it, you need to [install Julia](https://github.com/JuliaLang/juliaup) and then [install Pluto.jl](https://plutojl.org/#install), the notebook tool to run the code interactively:

After installing Julia, open the Julia REPL (terminal), install Pluto, and run the server withsession and run a Pluto server
```julia
import Pkg; Pkg.add("Pluto")
import Pluto; Pluto.run()
```
In the browser that Pluto opens, enter the notebook URL

```
https://github.com/b-fg/BurgersEqFV.jl/blob/main/notebook.jl
```

Then click on "Run notebook code", and that's it! The notebook take about 5 minutes to download and install all the dependencies before running the code. Just grab a coffee and start reading it in the meantime ;)

#### Alternative: Clone this repo and run the local notebook
You can always just clone this repository, install Pluto following the same procedure as before, and enter the path to the local notebook.jl file.