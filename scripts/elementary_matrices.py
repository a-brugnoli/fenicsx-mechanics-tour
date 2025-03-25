import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_box, CellType
from scipy.sparse import csr_matrix, save_npz
import os
results_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Directory '{results_dir}' created.")


# Physical parameters (waveguide 1)
rho = 2700  # density [kg/m^3]
E = 2*1.e11  # Young's modulus [N/m^2]
nu = 0.3  # Poisson's ratio [-]
eta = 0.01  # damping ratio [-]
h_y = 30*1e-2  # height [m]
h_z = 20*1e-2  # width [m]
d = 5*1e-2  # depth [m]

# Lame parameters
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

parametric_cell = create_box(
    MPI.COMM_WORLD,
    [np.array([-1, -1, -1]), np.array([1, 1, 1])],
    [1, 1, 1],
    cell_type=CellType.hexahedron,
)


dim = parametric_cell.topology.dim
print(f"Mesh topology dimension d={dim}.")
shape = (dim, )

def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)

V_cell = fem.functionspace(parametric_cell, ("Lagrange", 1, shape))
u_cell = TrialFunction(V_cell)
v_cell = TestFunction(V_cell)

dx_cell = Measure("dx", domain=parametric_cell)

k_form_cell = fem.form(inner(sigma(u_cell), epsilon(v_cell)) * dx_cell)
m_form_cell = fem.form(rho * inner(u_cell, v_cell) * dx_cell)

k_scipy_cell = fem.assemble_matrix(k_form_cell).to_scipy()
m_scipy_cell = fem.assemble_matrix(m_form_cell).to_scipy()

print(type(k_scipy_cell))

k_cell = k_scipy_cell.todense()
m_cell = m_scipy_cell.todense()

print(k_cell.shape)
print(m_cell.shape)
np.save(results_dir + "k_cell.npy", k_cell)
np.save(results_dir + "m_cell.npy", m_cell)

# Substructure

Ny, Nz = 4, 4
structure = create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([d, h_y, h_z])],
    [1, Ny, Nz],
    cell_type=CellType.hexahedron,
)

V_structure = fem.functionspace(structure, ("Lagrange", 1, shape))

u_structure = TrialFunction(V_structure)
v_structure = TestFunction(V_structure)

dx_structure = Measure("dx", domain=structure)

k_form_structure = fem.form(inner(sigma(u_structure), epsilon(v_structure)) * dx_structure)
m_form_structure = fem.form(rho * inner(u_structure, v_structure) * dx_structure)

k_scipy_structure = fem.assemble_matrix(k_form_structure).to_scipy()
m_scipy_structure = fem.assemble_matrix(m_form_structure).to_scipy()

k_structure = k_scipy_structure.todense()
m_structure = m_scipy_structure.todense()

print(k_structure.shape)    
print(m_structure.shape)
np.save(results_dir + "k_structure.npy", k_structure)
np.save(results_dir + "m_structure.npy", m_structure)