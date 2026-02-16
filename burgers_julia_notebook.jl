### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c483b5f8-da98-43e2-9f61-f5db8e89e9cc
md"""
# Deriving optimal data-driven numerical schemes for implicit turbulence modelling in Julia
"""

# ╔═╡ 6200cb59-cbf4-448e-bd34-e2ab8e1582fe
md"
## 1. Introduction to Julia

Julia is a programming language with the following properties

- **Compiled** via LLVM (fast!)
- **Just in time** (JIT) compilation
- **Functional language**: ecnouraging function evaluation, composability, functions as objects
- **Multiple dispatch**: function calls based on argument types
- **Dynamically typed**: variables can change during runtime
- **Metaprogramming**: powerful macros to generate (and avoid repeating) code
- **Interoperability** with C, Python, R, and Fortran libraries
- **Easy syntax**, similar to Python or Matlab
- **Package manager** that works for real (unlike Python...)
- **Open source** with a growing ecosystem
"

# ╔═╡ 64b763b5-3e17-469f-b6c5-69c109529977
md"
### Julia basics
#### Applying functions to vectors

Let's take a look at basic Julia sintax!
We start by creating a vector of `N=10` elements with type `T=Int`
"

# ╔═╡ d5727eea-3021-4c8f-9e15-088305ee096d
begin
	T = Int
	N = 10
	a = zeros(T, N)
end

# ╔═╡ f4496866-a27b-4d7a-982a-f530bde2b043
begin
	using BenchmarkTools
	b = zeros(T, N)
	@btime $b .= map(i -> i * i, 1:$N) # this one allocates!
	@btime map!(i -> i * i, $b, 1:$N) # allocation free!
	all(b .== a)
end

# ╔═╡ 6e5ef939-ea32-482f-9a2b-ca2cfb8b67b9
md"And now we loop through it to operate element-wise"

# ╔═╡ 4cd1ea7f-331b-4b3c-a280-2041de98af2e
begin
	
	for i in 1:N # even better use `eachindex(a)` instead of `1:N`
		a[i] = i * i
	end
	a
end

# ╔═╡ 7ed11855-7609-4c4e-b101-a80c650f243d
md"We can also use broadcasting, array comprehension, or function mapping to do the same element-wise operation!

**Broadcasting** with dot `.` syntax
"

# ╔═╡ 3f291428-adbc-43a0-8255-728d5a9b5afc
collect(1:N) .* collect(1:N) # collect(1:N) = [1,2,3,...,N]

# ╔═╡ 8ea71ce4-2959-42b7-b24c-69edbfee2449
md"**Array comprehension** with `[ ]` syntax"

# ╔═╡ 9115472d-4f45-41fc-b83d-a495a7dcfdfb
[i * i for i in 1:N]

# ╔═╡ 6dba3536-aae9-43b5-94fd-985da239580a
md"**Function mapping** with `map` syntax: applies a function to a *collection* (something iterable)"

# ╔═╡ 9361b6d9-542a-4f18-9a40-3e0592772117
map(i -> i * i, 1:N)

# ╔═╡ 7a7874d3-d19e-4bd0-ab4c-bf78fad88981
md" However, all these functions above generate a new array on the RHS eventually, thus allocating memory. Instead, if we want to assign this result to an already allocated vector `b`, we can use **in-place** element-wise operations.

This can be achieved with `map!`. In Julia, functions which mutate arguments typically contain the character `!` as a warning."

# ╔═╡ 7aff6f20-d270-4544-88a7-1ef035bedf13
md"We can actually define the function, instead of using a lambda (in-line) function"

# ╔═╡ 97618fbb-9b11-467f-8a63-0c93f337f4a1
begin
	f(i) = i * i # the array `a` is taken from global scope, not passed into `f`
	map!(f, b, 1:N)
end

# ╔═╡ 1312e529-e56a-46e7-8ff5-130437dde938
md"And we can also apply the broadcast directly to the function itself"

# ╔═╡ 9087ebf6-a9f8-44c4-83ad-63252a4334e8
f.(T.(1:N))

# ╔═╡ ba716968-6c11-4c90-a09a-efbc94c37f6a
md"
#### Multiple dispatch and Structs

Below is an example of multiple dispatch. In short, a certain function is called depending on the arguments data type
"

# ╔═╡ a0df8e19-7de9-48ea-9abc-f0a7bc2adb83
begin
	g(x) = println("Calling the `g` function for `Any` type")
	g(x::Int) = println("Calling the `g` function for `Int` eletype")
	g(x::Float64) = println("Calling the `g` function for `Float64` element type")
	
	g(1)
	g(1.0)
	g("Hi")
	g(g) # functions are objects of type `Function`!
end

# ╔═╡ 3d5e45c5-7fd0-4e24-ba8a-b498f628a074
md"And we can degine our own custom type using `Structs`. For example"

# ╔═╡ f5e58ad0-6998-4eb3-ade7-b4d70b4374df
begin
	struct Triangle{T}
		b :: T # base
		h :: T # height
	end
	struct Circle{T}
		r :: T # radius
	end
	area(x::Triangle) = (x.b * x.h) / 2
	area(x::Circle) = π * x.r^2
	area(x) = "Type of x: $(typeof(x)) does not have a defined area"
end

# ╔═╡ 9b00dd03-996e-4bb3-96a1-e6c19190a352
begin
	triangle = Triangle(4,5) # used default constractor
	area(triangle)
end

# ╔═╡ f67ff160-bb45-49df-a282-800313f19fdc
begin
	circle = Circle(3.0)
	area(circle)
end

# ╔═╡ 8ef9fa67-5f5f-4616-9342-6cc484ae4755
area(ones(N))

# ╔═╡ 5ba22a58-e17e-4054-8c60-5990bb4151d1
md"#### Other stuff?"

# ╔═╡ 19a15fe3-63c6-405c-8d2e-5e25816608ab
md"
## 2. Simulating the viscous Burgers' equation
The [viscous Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation) is a 1D PDE of the form

$\partial_t u + \partial_x f = 0$

with convection and diffusion fluxes

$f = u^2/2 - \nu\partial_xu$

where ``\nu`` is the diffusion coefficient (aka. viscosity). Because of the nonlinear term and the interplay with viscous effects, this PDE naturally distributes energy across a wide range of scales until it dissipates at the smallest scales. This resembles how *turbulence* behaves in fluid flows. so it is a good simple PDE to study numerical methods and turbulence models. The wave-propagating behaviour of the nonlinear term also gives rise to discontinuities, mimicking shock-wave phenomena of compressible flows.
"

# ╔═╡ 44d5b669-03f9-43b7-8248-6b4cfc2f8985
md"### Finite volume method discretization

To solve this PDE, we will resort to the FVM. Our basic assumption is that the continuous solution ``u(x,t)`` can be discretize in space into multiple cells. The solution at each cell ``i`` is a volume-average (or line for 1D) value

$u_i(x,t)=\dfrac{1}{\delta x}\int^{x_{i+1/2}}_{x_{i-1/2}}u(x,t)\mathrm{d}x$

where ``\delta x=(x_{i+1/2}-x_{i-1/2})`` is the cell volume.

The piecewise-constant values create discontinuities of the solution at the cells interface.  
"

# ╔═╡ 2b9a3d6f-afa1-46d2-809a-c33c0f88a408
md"![](https://github.com/b-fg/b-fg.github.io/blob/88fb634271ab49500f0049ab0ade3779a50aa1c1/assets/images/notebooks/fv.svg?raw=true {width='5'})"

# ╔═╡ b51af950-d3a6-42cb-8e78-840407d10196
$(Resource(
    "data:image/png;base64,#=base64-encoded file here=#",
    MIME("image/png"),
    (:height => 240)
))

# ╔═╡ f9103597-b7ba-4d6b-a152-b54b6a8ae088
LocalResource("./fv.svg")

# ╔═╡ b022fe41-000e-440a-9ef9-28697fdb6d72
md"
The FVM takes advantage of the Gauss theorem, where the integral form of the PDE can be transformed from the divergence of the flux into a summation of the flux across the faces of a cell

$\dfrac{1}{\delta x}\int_{V}\partial_x f \mathrm{d}V = \dfrac{1}{\delta x}\int_{S} f\cdot\hat{n}\mathrm{d}S$

Thus, we can express our PDE in semi-discrete form as

$\dfrac{\mathrm{d}u_i}{\mathrm{d}t}=-\dfrac{f_{i+1/2}-f_{i-1/2}}{\delta x}$
"

# ╔═╡ 8f324c5d-5a5d-404e-92de-9c4a0aa79397
md"#### Numerical flux

**Convective flux**

Our next job is to define the intercell flux, aka. numerical flux, and a time-intergrator method. First, we focus on the convective flux $f_c=u^2/2$. Instead of using the piece-wise constant values at the $_{i+1/2}$ face, $u_i$ and $u_{i+1}$, we will reconstruct a higher-order solution using the [$k$-scheme by Van Leer](https://en.wikipedia.org/wiki/MUSCL_scheme#Piecewise_parabolic_reconstruction), a high-order generalization of the [MUSCL scheme](https://en.wikipedia.org/wiki/MUSCL_scheme), both for the left side of the face, $u^L_{i+1/2}$ and the right, $u^R_{i+1/2}$ (which is written symmetrically)

$u^L_{i+1/2} = u_i + \dfrac{1}{4}\left[(1-k)(u_i-u_{i-1}) + (1+k)(u_{i+1}-u_i)\right]$
$u^R_{i+1/2} = u_{i+1} + \dfrac{1}{4}\left[(1-k)(u_{i+1}-u_{i+2}) + (1+k)(u_{i}-u_{i+1})\right]$

The $k$-scheme provides a tunable parameter, $k$, which can result in the following reconsuctrions:

-  $$k=-1$$: Upwind 2nd-order
-  $$k=0$$: Fromm
-  $$k=1/3$$: ? 3rd-order
-  $$k=1/2$$: Quick 3rd-order
-  $$k=1$$: Central 2nd-order

Note that the scheme goes from being fully upwind (-1) to fully central (1). Values of $k<-1$ and $k>1$ are allowed, and they bias the reconstruction even more to the upwind and downwind directions, respectively.

For the moment, let's define the function!
"


# ╔═╡ 51b551b9-a97d-4a4f-96ca-bc7edfe65814
kscheme(um, ui, up, k) = ui + 1 / 4 * ((1 - k) * (ui - um) + (1 + k) * (up - ui))

# ╔═╡ 02e1a59c-c747-4bf0-93a8-074ea7f88f9b
md"Next, after computing $u_L$ and $u_R$, we need to define how the intercell flux will be computed. Note that this numerical flux is unique to each face! That is $f_{i+1/2} = -f_{(i+1)-1/2}$.

We choose the [Rusanov flux](link) to solve the two-point value problem at the face.

$f_{i+1/2}(u_L,u_R) = \dfrac{1}{2}(f(u_L) + f(u_R)) - \dfrac{1}{2}\max{(|u_L|, |u_R|)}(u_R-u_L)$

Note that the numerical flux can also be computed using exact or approximate [Riemann solvers](https://en.wikipedia.org/wiki/Riemann_solver). 

Let's define the function for computing the Rusanov flux
"

# ╔═╡ e4327a21-904b-4bab-9179-442dea00dda5
begin
	flux(u) = u * u / 2
	function f_rusanov(uL, uR)
		1 / 2 * (flux(uL) + flux(uR)) - 1 / 2 * max(abs(uL), abs(uR)) * (uR - uL)
	end
end

# ╔═╡ 57633811-38aa-4b57-9c5d-2573c4aef2e6
md"
**Viscous flux**

The computation of viscous (aka. diffusive, aka. dissipative...) fluxes is more straightforward. Importantly, this flux contains the gradient $\partial_x u$. In the current grid configuration, gradient of cell-centered quantities, such as $u_i$, live in the cell faces, and that is great because we do not need to solve a two-point value problem, differently from the convective flux. Then, the divergence of the gradient, living in the face, already provides the unique viscous flux at that face.

Let's break it down. First, we compute the discrete gradient using central differences, for example at the $i+1/2$ face

$\nu\partial_x u|_{i+1/2} \approx \nu\dfrac{u_{i+1} - u_i}{\delta x}$

And then, using the divergence theorem again, we get the dissipative flux across the cell boundaries 

$\partial_x (\nu \partial_x u) \approx \dfrac{\nu\partial_x u|_{i+1/2} - \nu\partial_x u|_{i-1/2}}{\delta x} = \dfrac{\nu\dfrac{u_{i+1} - u_i}{\delta x} - \nu\dfrac{u_{i} - u_{i-1}}{\delta x}}{\delta x}=\nu\dfrac{u_{i+1}-2u_i+u_{i-1}}{\delta x^2}$

Let's write the dissipative flux function
"

# ╔═╡ fc23f17b-753a-40bd-80aa-cbf61a930717
f_diss(um, ui, up, ν, dx) = ν * (up - 2ui + um) / (dx^2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[compat]
BenchmarkTools = "~1.6.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "10fbcacea77e5f06204aa57f49837bbf31a9edbf"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "9297459be9e338e546f5c4bedb59b3b5674da7f1"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─c483b5f8-da98-43e2-9f61-f5db8e89e9cc
# ╟─6200cb59-cbf4-448e-bd34-e2ab8e1582fe
# ╟─64b763b5-3e17-469f-b6c5-69c109529977
# ╠═d5727eea-3021-4c8f-9e15-088305ee096d
# ╟─6e5ef939-ea32-482f-9a2b-ca2cfb8b67b9
# ╠═4cd1ea7f-331b-4b3c-a280-2041de98af2e
# ╟─7ed11855-7609-4c4e-b101-a80c650f243d
# ╠═3f291428-adbc-43a0-8255-728d5a9b5afc
# ╟─8ea71ce4-2959-42b7-b24c-69edbfee2449
# ╠═9115472d-4f45-41fc-b83d-a495a7dcfdfb
# ╟─6dba3536-aae9-43b5-94fd-985da239580a
# ╠═9361b6d9-542a-4f18-9a40-3e0592772117
# ╟─7a7874d3-d19e-4bd0-ab4c-bf78fad88981
# ╠═f4496866-a27b-4d7a-982a-f530bde2b043
# ╟─7aff6f20-d270-4544-88a7-1ef035bedf13
# ╠═97618fbb-9b11-467f-8a63-0c93f337f4a1
# ╟─1312e529-e56a-46e7-8ff5-130437dde938
# ╠═9087ebf6-a9f8-44c4-83ad-63252a4334e8
# ╟─ba716968-6c11-4c90-a09a-efbc94c37f6a
# ╠═a0df8e19-7de9-48ea-9abc-f0a7bc2adb83
# ╟─3d5e45c5-7fd0-4e24-ba8a-b498f628a074
# ╠═f5e58ad0-6998-4eb3-ade7-b4d70b4374df
# ╠═9b00dd03-996e-4bb3-96a1-e6c19190a352
# ╠═f67ff160-bb45-49df-a282-800313f19fdc
# ╠═8ef9fa67-5f5f-4616-9342-6cc484ae4755
# ╟─5ba22a58-e17e-4054-8c60-5990bb4151d1
# ╟─19a15fe3-63c6-405c-8d2e-5e25816608ab
# ╟─44d5b669-03f9-43b7-8248-6b4cfc2f8985
# ╠═2b9a3d6f-afa1-46d2-809a-c33c0f88a408
# ╠═b51af950-d3a6-42cb-8e78-840407d10196
# ╠═f9103597-b7ba-4d6b-a152-b54b6a8ae088
# ╟─b022fe41-000e-440a-9ef9-28697fdb6d72
# ╟─8f324c5d-5a5d-404e-92de-9c4a0aa79397
# ╠═51b551b9-a97d-4a4f-96ca-bc7edfe65814
# ╟─02e1a59c-c747-4bf0-93a8-074ea7f88f9b
# ╠═e4327a21-904b-4bab-9179-442dea00dda5
# ╟─57633811-38aa-4b57-9c5d-2573c4aef2e6
# ╠═fc23f17b-753a-40bd-80aa-cbf61a930717
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
