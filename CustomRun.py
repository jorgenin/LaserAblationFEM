# %% [markdown]
# |Nomenclature||
# |------------|-|
# |$\mathbf{x}$| point in the spatial domain|
# |$t$| time |
# |$p = p(\mathbf{x},t)$| pressure field |
# |$\mathbf{u} = \mathbf{u}(\mathbf{x},t)$| velocity vector field |
# |$T = T(\mathbf{x},t)$| temperature field |
# |$\phi$ | solid volume fraction |
# |$()_t = \frac{\partial}{\partial t}()$| time derivative |
# |$T_r$| central temperature of the regularization |
# |$r$| smoothing parameter of the regularization |
# |$\mu$| constant dynamic viscosity of the fluid|
# |$\mathbf{f}_B(T)$| temperature-dependent buoyancy force|
# |$\mathrm{Pr}$ | Prandtl number|
# |$\mathrm{Ra}$ | Rayleigh number|
# |$\mathrm{Ste}$| Stefan number|
# |$\Omega$| spatial domain |
# |$\mathbf{w} = \begin{pmatrix} p \\ \mathbf{u} \\ T \end{pmatrix}$| system solution|
# |$\mathbf{W}$| mixed finite element function space |
# |$\boldsymbol{\psi} = \begin{pmatrix} \psi_p \\ \boldsymbol{\psi}_u \\ \psi_T \end{pmatrix} $| mixed finite element basis functions|
# |$\gamma$| penalty method stabilization parameter|
# |$T_h$| hot boundary temperature |
# |$T_c$| cold boundary temperature |
# |$\Delta t$| time step size |
# |$\Omega_h$| discrete spatial domain, i.e. the mesh |
# |$M$| goal functional |
# |$\epsilon_M$| error tolerance for goal-oriented AMR |

# %% [markdown]
# ## Governing equations

# %% [markdown]
# Our main equation is just a enthalpy(energy) balance.
# 
# 
# 
# $$\begin{aligned}  \frac{\partial{\rho h}}{\partial t} +\nabla(\rho h U) - \nabla \cdot( K \nabla T) = 0 \end{aligned}$$
# 
# Here $ \frac{\partial{\rho h}}{\partial t}$ is the internal energy term, $\nabla(\rho h U)$ is the mass flow term and lastly $\nabla \cdot( K \nabla T)$ is the term describing the heat transfer. 
# 
# Because we don't have any mass transfer (YET) U will equal 0 
# So the equation becomes
# $$\begin{aligned}  \frac{\partial{\rho h}}{\partial t}  - \nabla \cdot( K \nabla T) = 0\end{aligned}$$
# 
# For now we will assume $\rho = \rho_s = \rho_l = \rho_g$ ei there is no change in density over temperature. We will also assume thermal conducitivity does not change.
# 
# One last argument we will include is the $\phi(T)$ levelset-function. We will have a $\phi_l(T)$ and a $\phi_g(T)$ function which will corepond to the liquid and gas portions respectivly. 
# 
# To ensure differentiability we will approximate the levelset function through:
# $$\begin{aligned} \phi(T) = \frac{1 + \tanh(\frac{T_r - T}{\beta})}{2} \end{aligned}$$
# 
# This gives a smooth function that goes from 0 to 1. $T_r$ is the temereature of the phase change, and  $\beta$ is a regularization parameter which controls the sharpness of the change.
# 

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological, Expression )
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.utils import XDMFFile
from dolfinx.plot import vtk_mesh

from dolfinx import mesh, fem, log

import dolfinx.nls.petsc

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                grad, inner)

from ufl import (FacetNormal, FiniteElement, Identity,TestFunction, TrialFunction, VectorElement,
div, dot, dx, inner, lhs, nabla_grad, rhs, sym, derivative, tanh) 

import ufl


# %% [markdown]
# ## Initial mesh

# %% [markdown]
# Define an intial mesh to have. We'll do refinement at a later date.

# %%
N=100

domain = mesh.create_rectangle(MPI.COMM_WORLD,[[-20,-20],[20,0]],[N,int(N*20/40)],cell_type=mesh.CellType.triangle)

#Left hand side wall
def HotWall(x):
    return np.logical_and(np.logical_and(np.isclose(x[1], 1) , x[0]>0.2), x[0]<0.8)

initial_hot_wall_refinement_cycles = 0

for cycle in range(initial_hot_wall_refinement_cycles):

    #Facets on the hot wall
    hot_wall = mesh.locate_entities_boundary(domain,domain.topology.dim-1,  HotWall) 

    domain = mesh.refine(domain, hot_wall, redistribute=False)

from dolfinx import geometry
#bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# %% [markdown]
# Visualize the mesh

# %%
#plotter = pyvista.Plotter(notebook="true")


#vtkdata = vtk_mesh(domain, domain.topology.dim)
#grid = pyvista.UnstructuredGrid(*vtkdata)
#actor = plotter.add_mesh(grid, show_edges=True)
#plotter.view_xy()
#plotter.show()
#plotter.close()

# %% [markdown]
# ## Material Properties

# %% [markdown]
# We are going to define the different material properties. We'll make the class more official later. But it needs to be looked at.

# %% [markdown]
# We need to transform everything to correct units.
# We will be using ug, um, fs, K

# %%
class Material:
    def __init__(self):
        pass


SS = Material()

SS.BoilingTemp = 3200#K
SS.MeltingTemp = 1727 #K
SS.SolidDensity = 7720 #KG/m^3
SS.LiquidDensity = 7000 #KG/m^3
SS.HeatCapacitySolid = 712 #J/Kg/K
SS.HeatCapacityLiquid = 800 #J/Kg-K
SS.LatentHeatFusion = 2.76e5 #J/kg
SS.LatentHeatVaporization = 7.34e6 #J/Kg
SS.SolidThermalConductivity = 29 #W/(m K)

#ConverstionFactors
DensityConv = 1e-9 #Kg/m^3 to ug/um^3 
HeatCapacityConv = 1 #J/(Kg K) to um^2/(us^2*K)
LatentHeatConv = 1 #J/Kg to um^2/us^2
ThermalConductivityConv = 0.001 #W/(m K) to ug um/(us^3 K)

SS.SolidDensity *= DensityConv
SS.LiquidDensity *= DensityConv

SS.HeatCapacitySolid *= HeatCapacityConv
SS.HeatCapacityLiquid *= HeatCapacityConv

SS.LatentHeatFusion *= LatentHeatConv
SS.LatentHeatVaporization *= LatentHeatConv

SS.SolidThermalConductivity *= ThermalConductivityConv

# %% [markdown]
# ## Mixed finite element function space, test functions, and solution functions
# 

# %% [markdown]
# For now we will just use P1 linear elements for thermal approximation. We will augment this if we begin to add on velocity fields and such.

# %%
P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
#TH = ufl.MixedElement([P1, P2, P1])
TH = P1
#P2 is a vector element to account for velocity
#P1 is a scalar element to account for pressure and temperature
W = FunctionSpace(domain, TH)

# %% [markdown]
# Make the system solution function $\mathbf{w} \in \mathbf{W}$ and obtain references to its components $p$, $\mathbf{u}$, and $T$.

# %%
w = Function(W,dtype=np.float64)
T = w #Technically w is just a single temperature problem

# %% [markdown]
# Now we will make the $\phi(T)$

# %%
regularization_smoothing_parameter = 100

r = Constant(domain,PETSc.ScalarType(regularization_smoothing_parameter))

def phi(T,T_r):
    return 0.5*(1. + tanh((T-T_r)/r))

def phin(T,T_r):
    return 0.5*(1. + np.tanh((T-T_r)/regularization_smoothing_parameter))

T_b = Constant(domain,PETSc.ScalarType(SS.BoilingTemp))

T_m = Constant(domain,PETSc.ScalarType(SS.MeltingTemp))

SS.phi_l = lambda T: phi(T,T_m)
SS.liquid = lambda T: phin(T,SS.MeltingTemp)
SS.phi_b = lambda T: phi(T,T_b)
SS.gas = lambda T: phin(T,SS.BoilingTemp)

# %% [markdown]
# Now we also need to define the latent heat function $h(T)$
# 
# $$\begin{aligned}  h(T) = T*c(T) + \phi_l(T)*(T_m(c_s-c_l)+h_f) +  \phi_b(T)*(T_b(c_l-c_g) + h_v) \end{aligned}$$
# 
# where
# 
# $$\begin{aligned}  c(T) = c_s + \phi_l(T)*(c_l-c_s) +  \phi_b(T)*(c_g-c_l) \end{aligned}$$
# 
# 
# 

# %%
c_s = Constant(domain,PETSc.ScalarType(SS.HeatCapacitySolid))
c_l = Constant(domain,PETSc.ScalarType(SS.HeatCapacityLiquid))
c_g = c_l
h_f = Constant(domain,PETSc.ScalarType(SS.LatentHeatFusion))
h_v = Constant(domain,PETSc.ScalarType(SS.LatentHeatVaporization))

SS.c = lambda T: c_s + SS.phi_l(T)*(c_l-c_s) + SS.phi_b(T)*(c_g-c_l)

SS.h = lambda T: SS.c(T)*T + SS.phi_l(T)*(T_m*(c_s-c_l)+h_f) + SS.phi_b(T)*(T_b*(c_l-c_g)+h_v)
#SS.h = lambda T: SS.c(T)*T + SS.phi_l(T)*(h_f) + SS.phi_b(T)*(h_v)
#SS.h = lambda T: T*c_s

# %% [markdown]
# ## Initial Value Setting
# To solve the initial value problem, we will prescribe the initial values, and then take discrete steps forward in time which solve the governing equations.
# 
# We will set the initial value to just be set at room temperature $T_r$
# 

# %%
RoomTemp = 300 #Deg K 
T_r = Constant(domain,PETSc.ScalarType(RoomTemp))



def InitialTempExp(x):
    return T_r*np.ones(x.size)



#w_n = Function(W)
#p_n, u_n, T_n = w_n.split()
T.interpolate(InitialTempExp)
T_n = Function(W,dtype=np.float64) # T n
T_n.interpolate(InitialTempExp)
T_n1 = Function(W,dtype=np.float64) #T n-1
T_n1.interpolate(InitialTempExp)

# %% [markdown]
# ### Test Plot Initial Value to ensure working state

# %%
import matplotlib.pyplot as plt
#dummy function space for plotting
W_dummy = FunctionSpace(domain, P1)
cells, types, x = vtk_mesh(W_dummy)

grid = pyvista.UnstructuredGrid(cells, types, x)

grid.point_data["T"] = T_n.x.array

#plotter = pyvista.Plotter(notebook="true")

#sargs = dict(title_font_size=25, label_font_size=20, color="black",
#             position_x=0.25, position_y=.9, width=0.5, height=0.1)

#warped = grid.warp_by_scalar("T", factor=0)

#renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
#                            cmap='jet', scalar_bar_args=sargs,
#                            clim=[275, 300])

#actor = plotter.add_mesh(grid, show_edges=True)
#plotter.view_xy()
#plotter.show()
#plotter.close()

# %% [markdown]
# ## Time Integration

# %% [markdown]
# For the time derivative terms $\mathbf{u}_t$, $T_t$, and $\phi_t$, we will apply a second order backwards gear integration.
# 
# Here is what simula says about it. 
# 
# The Gear method is also called BDF (Backward Differentiation Formulas) method and belongs to the class of linear multistep methods as Adams. The applied second-order method is absolutely stable, a property that is not satisfied for higher integration orders of Gear. As a second-order method it is much more accurate than Backward Euler and has comparable accuracy to Adams. Unlike Adams, Gear is especially well suited for stiff problems and does not tend to suffer from numerical oscillations. It is therefore used as default method for standard transient simulations.
# 
# The gear integration formula looks like: 
# $\begin{aligned}  \frac{\partial{X}}{\partial t}  \approxeq  \frac{3X^{n+1} -4X^n+X^{n-1}}{2\delta t}\end{aligned}$

# %%


#u_n is the previous solution essentially
Timestep_size = 1/10 #us

Delta_t = Constant(domain,PETSc.ScalarType(Timestep_size))


#T_t = (T - T_n)/Delta_t

SS.h_t = (3*SS.h(T)-4*SS.h(T_n)+SS.h(T_n1))/(2*Delta_t)
#SS.h_t = ( SS.h(T)-SS.h(T_n))/Delta_t

# %% [markdown]
# ## Nonlinear variational form

# %% [markdown]
# We need to find the variational form of the problem. We can do this by muplibying by a test function $\psi$
# 
# $$\begin{aligned}  (\frac{\partial{\rho h}}{\partial t}  - \nabla \cdot( K \nabla T))\phi = 0\end{aligned}$$
# 
# We will (for now) assume that $K$ and $\rho$ are both constants. 
# 
# Thus when we integrate over the boundary $\Omega$ to try to solve we get:
# 
# $$\begin{aligned}  \int_{\Omega}{ \rho\frac{\partial{ h}}{\partial t}\cdot \psi  - K \nabla \cdot( \nabla T)\cdot \psi} = 0\end{aligned}$$
# 
# We can then integrate by parts to get:
# 
# $$\begin{aligned}  \rho \int_{\Omega}{\frac{\partial{h(T)}}{\partial t}\cdot \psi d\Omega}  + K \int_{\Omega}{\nabla \psi \cdot  \nabla T  d\Omega}  - K \int_{\Gamma}{\psi \cdot  \nabla T d\Gamma } = 0 \end{aligned}$$
# 
# 
# Here because we consider conditions adiabatic the boundary integral $ K \int_{\Gamma}{\psi \cdot  \nabla T d\Gamma } $ is considered 0 and can be removed
# 
# 
# 

# %%
#dxs = ufl.dx(metadata={'quadrature_degree': 8})

rho = Constant(domain,PETSc.ScalarType(SS.SolidDensity))
K = Constant(domain,PETSc.ScalarType(SS.SolidThermalConductivity)) # type: ignore

# %%
psi_T = ufl.TestFunction(W)

enthalpy = rho*SS.h_t*psi_T + dot(grad(psi_T),K*grad(T))
#momentum + enthalpy  
F = (enthalpy)*dx

# %% [markdown]
# ## Boundary conditions

# %% [markdown]
# We need boundary conditions before we can define a nonlinear variational problem (i.e. in this case a boundary value problem).
# 
# We physically consider no slip velocity boundary conditions for all boundaries. These manifest as homogeneous Dirichlet boundary conditions. For the temperature boundary conditions, we consider a constant hot temperature on the left wall, a constant cold temperature on the right wall, and adiabatic (i.e. zero heat transfer) conditions on the top and bottom walls. Because the problem's geometry is simple, we can identify the boundaries with the following piece-wise function.
# 
# \begin{align*} T(\mathbf{x}) &= \begin{cases} T_h , && x_0 = 0 \\ T_c , && x_0 = 1 \end{cases} \end{align*}

# %%
def HotWall1(x):
    #return np.logical_and(np.logical_and(np.isclose(x[1], 0) , x[0]>-30), x[0]<30)
    return np.isclose(x[1], 0)

def ColdWall1(x):
     return np.isclose(x[1], -20)

#def AdiabaticWall(x):
#     return np.logical_or(np.isclose(x[0], 0),np.isclose(x[0], 1))


#def walls(x):
#    return  np.logical_or(np.logical_or(HotWall1(x),ColdWall1(x)),AdiabaticWall(x))

# %% [markdown]
# Here we are marking the cells were the boundary is applied. This way we can add the flux

# %%
boundaries = [(1, HotWall1)]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# %% [markdown]
# We are adding here a boundary integral to the problem. This represents the thermal flux inputed by the laser at the surface of the metal.
# 
# The laser is approximated by a normal distrution of the form
# $$ \frac{1}{\sigma \sqrt{ 2 \pi}} e^{\frac{-1}{2} ( \frac{X}{\sigma})^2} $$

# %%


std = 10

fluence = .27*117.77#j/cm^2
fluenceconv = 10#j/cm^2 to ug/us^2

fluence *= fluenceconv 


g = lambda x: ufl.exp(-2*((x[0])/std)**2)*fluence
#g = lambda x: fluence
x = SpatialCoordinate(domain)

val = g(x)

#domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag,metadata={'quadrature_degree': 16})

LaserOn = Constant(domain, PETSc.ScalarType(1))
#Radiation Heat
STBoltz = 5.67e-8#J/(s · m^2 · K^4)
STBoltzConv = 1e-9 #W / (m^2 *K^4) to ug/(us^3*K^4)
STBoltz *= STBoltzConv
T_a = Constant(domain, PETSc.ScalarType(300)) #ambiant temperature
em = Constant(domain, PETSc.ScalarType(.3)) #Emissivity
Radiation = STBoltz*em*(T**4-T_a**4) # type: ignore
h = 80e-9
Convection = h*(T-T_a) # type: ignore


F += -dot(dot(psi_T,val),LaserOn)*ds(1)+dot(Radiation,psi_T)*ds(1) + dot(Convection,psi_T)*ds(1) # type: ignore

# %%
coldwall = locate_dofs_geometrical(W, ColdWall1)
hotwall = fem.locate_dofs_geometrical(W, HotWall1)

boundary_conditions = []#dirichletbc(PETSc.ScalarType(300), coldwall ,W)]
                       # dirichletbc(PETSc.ScalarType(1000), hotwall ,W)]

# %% [markdown]
# ## Linearization
# 

# %% [markdown]
# To complete the newton method we need to have the jacobian so we cna linearlize the problem. Luckily we can just have FENICS figure this out for us.

# %%
q = TrialFunction(W)
#JF = derivative(F, w, q)

# %%
problem = fem.petsc.NonlinearProblem(F, w, boundary_conditions)

# %% [markdown]
# # Time to solve

# %%
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-12
solver.atol = 1e-12
solver.max_it = 100
solver.report = True

# %% [markdown]
# We can modify the linear solver in each Newton iteration by accessing the underlying PETSc object.
# 
# 

# %%
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
#opts[f"{option_prefix}ksp_type"] = "cg"
#opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 50
#opts[f"{option_prefix}with-precision"] = "__float128"

ksp.setFromOptions()


# %% [markdown]
# We are now ready to solve the non-linear problem. We assert that the solver has converged and print the number of iterations.

# %%
log.set_log_level(log.LogLevel.WARNING)
#n, converged = solver.solve(w)
#print(f"Number of interations: {n:d}")

# %%
import pyvista
t=0

grid = pyvista.UnstructuredGrid(*vtk_mesh(W))

#plotter = pyvista.Plotter(window_size=([3008, 1504]))

#plotter.open_movie("DimensionalTemp.mp4")
#plotter.show_grid()

#grid.point_data["Solid Volume Fraction"] = SS.liquid(w.x.array)+SS.gas(w.x.array)
#warped = grid.warp_by_scalar("Solid Volume Fraction", factor=0)

#sargs = dict(title_font_size=25, label_font_size=20, color="black",
 #           position_x=0.25, position_y=.9, width=0.5, height=0.1)

#renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
  #                          cmap="jet", scalar_bar_args=sargs,
   #                         clim=[0, 2])

#bb_tree = geometry.bb_tree(domain,domain.topology.dim)


# %%
#plotter.view_xy()
#plotter.camera.zoom(1.8)
from datetime import datetime
dt = Timestep_size
t=0
length =20
laserOnLength = 10
gaus = lambda t: 1/(laserOnLength/3 * np.sqrt(2*np.pi))*np.exp(-1/2 *((t-laserOnLength/2)/(laserOnLength/3))**2)
xdmf = XDMFFile(domain.comm, "HeatMelting.xdmf", "w")
xdmf.write_mesh(domain)
T_n.name="Temperature"
xdmf.write_function(T_n,t)

print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

while t<length:
    t += dt
    if t < laserOnLength:
        LaserOn.value = gaus(t)
    else:
        LaserOn.value = 0

    # Solve linear problem
    n, converged = solver.solve(w)
    w.x.scatter_forward()

    # Update solution at previous time step (T_n and T_n1)
    T_n1.x.array[:] = T_n.x.array
    T_n.x.array[:] = w.x.array
    xdmf.write_function(T_n,t)
    #print(max(T.x.array))
    #print(f"Number of interations: {n:d}")
    # Write solution to file
    # Update plot
    #print("Current time")
    #print(t)
    #print("Max Temp")
    #print(np.max(T.x.array[:]))

    #warped = grid.warp_by_scalar("Solid Volume Fraction", factor=0)
    #plotter.update_coordinates(warped.points.copy(), render=True)
    #lotter.update_scalars((SS.liquid(w.x.array)+SS.gas(w.x.array)),render=True)
    #plotter.render()
    #plotter.write_frame()
#plotter.close()
xdmf.close()

print("-----------------------------------------")
print("End computation")                 
# Report elapsed real time for the analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")
