import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

class OneDHeatEquation:
    """ 
    1D Heat Equation solver using Laplacian tridiagonal matrix formulation.
    
    Parameters:
    -----------
    length : float
        Length of the rod
    num_points : int
        Number of spatial discretization points
    alpha : float
        Thermal diffusivity coefficient
    initial_condition : callable or array-like
        Initial temperature distribution. If callable, takes x-coordinates as input.
    boundary_condition_type : str
        'Dirichlet' or 'Neumann'
    left_bc : callable or float
        Left boundary condition. If callable, takes time as input.
    right_bc : callable or float
        Right boundary condition. If callable, takes time as input.
    solver_type : str
        'explicit', 'implicit', or 'crank-nicolson'
    time_step : float
        Time step for simulation
    total_time : float
        Total simulation time
    """
    def __init__(self, length, num_points, alpha, 
                 initial_condition, 
                 boundary_condition_type='Dirichlet', 
                 left_bc=0.0,
                 right_bc=0.0,
                 solver_type='explicit', 
                 time_step=0.01,
                 total_time=1.0):
        
        self.length = length
        self.num_points = num_points
        self.alpha = alpha
        self.initial_condition = initial_condition
        self.bc_type = boundary_condition_type
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.solver_type = solver_type
        self.dt = time_step
        self.total_time = total_time
        
        # Spatial discretization
        self.dx = length / (num_points - 1)
        self.x = np.linspace(0, length, num_points)
        
        # Time discretization
        self.num_time_steps = int(total_time / time_step) + 1
        self.time = np.linspace(0, total_time, self.num_time_steps)
        
        # Check CFL condition for explicit method
        if solver_type == 'explicit':
            cfl = alpha * time_step / (self.dx ** 2)
            if cfl > 0.5:
                raise ValueError(f"CFL condition violated! CFL = {cfl:.4f} > 0.5. "
                               f"Reduce time_step or increase num_points.")
        
        # Build Laplacian matrix
        self.L = self._build_laplacian()
        
    def _build_laplacian(self):
        """Build the Laplacian matrix L with boundary conditions incorporated."""
        n = self.num_points
        
        # Start with standard tridiagonal structure [1, -2, 1] * alpha/dx^2
        diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
        L = diags(diagonals, [-1, 0, 1], shape=(n, n), format='lil')
        L = L * (self.alpha / self.dx**2)
        
        # Modify first and last rows based on boundary condition type
        if self.bc_type == 'Dirichlet':
            # For Dirichlet: u_0 and u_N are fixed, so their time derivatives are 0
            # First row: du_0/dt = 0
            L[0, :] = 0
            # Last row: du_N/dt = 0
            L[-1, :] = 0
            
        elif self.bc_type == 'Neumann':
            # For Neumann with one-sided difference: (u_1 - u_0)/dx = a(t)
            # Rearranging: u_0 = u_1 - dx*a(t)
            # The Laplacian at boundary involves ghost points, modify accordingly
            
            # Left boundary: du_0/dt = alpha/dx^2 * (u_1 - u_0 - dx*a(t))
            # Which simplifies to: alpha/dx^2 * (u_1 - u_0) + forcing term
            L[0, 0] = -self.alpha / self.dx**2
            L[0, 1] = self.alpha / self.dx**2
            
            # Right boundary: du_N/dt = alpha/dx^2 * (u_{N-1} - u_N + dx*b(t))
            L[-1, -2] = self.alpha / self.dx**2
            L[-1, -1] = -self.alpha / self.dx**2
        else:
            raise ValueError(f"Unknown boundary condition type: {self.bc_type}")
        
        return L.tocsr()  # Convert to CSR for efficient operations
    
    def _get_bc_value(self, bc, t):
        """Helper to get boundary condition value at time t."""
        if callable(bc):
            return bc(t)
        else:
            return bc
    
    def _build_forcing_term(self, t):
        """Build the forcing term F(t) from boundary conditions."""
        F = np.zeros(self.num_points)
        
        if self.bc_type == 'Dirichlet':
            # Dirichlet BCs: directly set boundary values
            F[0] = self._get_bc_value(self.left_bc, t)
            F[-1] = self._get_bc_value(self.right_bc, t)
            
        elif self.bc_type == 'Neumann':
            # Neumann BCs: add flux contribution
            left_flux = self._get_bc_value(self.left_bc, t)
            right_flux = self._get_bc_value(self.right_bc, t)
            
            F[0] = -self.alpha / self.dx * left_flux
            F[-1] = self.alpha / self.dx * right_flux
        
        return F
    
    def _initialize_solution(self):
        """Initialize the solution array with initial condition."""
        if callable(self.initial_condition):
            u0 = self.initial_condition(self.x)
        else:
            u0 = np.array(self.initial_condition)
            
        if len(u0) != self.num_points:
            raise ValueError(f"Initial condition length {len(u0)} doesn't match num_points {self.num_points}")
        
        return u0.copy()
    
    def solve(self):
        """Solve the heat equation and return full time evolution."""
        # Initialize solution storage
        solution = np.zeros((self.num_time_steps, self.num_points))
        solution[0, :] = self._initialize_solution()
        
        # Time stepping
        if self.solver_type == 'explicit':
            solution = self._solve_explicit(solution)
        elif self.solver_type == 'implicit':
            solution = self._solve_implicit(solution)
        elif self.solver_type == 'crank-nicolson':
            solution = self._solve_crank_nicolson(solution)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
        
        return solution
    
    def _solve_explicit(self, solution):
        """Explicit Euler time stepping: u^{n+1} = u^n + dt*(L@u^n + F^n)"""
        I = np.eye(self.num_points)
        
        for n in range(self.num_time_steps - 1):
            t = self.time[n]
            u_n = solution[n, :]
            F_n = self._build_forcing_term(t)
            
            # Explicit step
            solution[n+1, :] = u_n + self.dt * (self.L @ u_n + F_n)
            
            # Enforce boundary conditions for Dirichlet
            if self.bc_type == 'Dirichlet':
                solution[n+1, 0] = self._get_bc_value(self.left_bc, self.time[n+1])
                solution[n+1, -1] = self._get_bc_value(self.right_bc, self.time[n+1])
        
        return solution
    
    def _solve_crank_nicolson(self, solution):
        """Crank-Nicolson: (I - dt/2*L)@u^{n+1} = (I + dt/2*L)@u^n + dt/2*(F^{n+1} + F^n)"""
        I = eye(self.num_points, format='csr')
        A = I - (self.dt / 2) * self.L  # Keep as sparse
        B = I + (self.dt / 2) * self.L  # Keep as sparse
        
        for n in range(self.num_time_steps - 1):
            t_next = self.time[n+1]
            t_current = self.time[n]
            u_n = solution[n, :]
            F_next = self._build_forcing_term(t_next)
            F_current = self._build_forcing_term(t_current)
            
            # Right hand side
            rhs = B @ u_n + (self.dt / 2) * (F_next + F_current)
            
            # Solve linear system with sparse solver
            solution[n+1, :] = spsolve(A, rhs)
            
            # Enforce boundary conditions for Dirichlet
            if self.bc_type == 'Dirichlet':
                solution[n+1, 0] = self._get_bc_value(self.left_bc, t_next)
                solution[n+1, -1] = self._get_bc_value(self.right_bc, t_next)
        
        return solution
    
    def _solve_implicit(self, solution):
        """Implicit Euler: (I - dt*L)@u^{n+1} = u^n + dt*F^{n+1}"""
        I = eye(self.num_points, format='csr')
        A = I - self.dt * self.L  # Keep as sparse
        
        for n in range(self.num_time_steps - 1):
            t_next = self.time[n+1]
            u_n = solution[n, :]
            F_next = self._build_forcing_term(t_next)
            
            # Right hand side
            rhs = u_n + self.dt * F_next
            
            # Solve linear system with sparse solver
            solution[n+1, :] = spsolve(A, rhs)
            
            # Enforce boundary conditions for Dirichlet
            if self.bc_type == 'Dirichlet':
                solution[n+1, 0] = self._get_bc_value(self.left_bc, t_next)
                solution[n+1, -1] = self._get_bc_value(self.right_bc, t_next)
        
        return solution


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example 1: Dirichlet BC with simple initial condition
    def initial_temp(x):
        return np.sin(np.pi * x)
    
    solver = OneDHeatEquation(
        length=1.0,
        num_points=50,
        alpha=0.05,
        initial_condition=initial_temp,
        boundary_condition_type='Dirichlet',
        left_bc=0.0,
        right_bc=0.0,
        solver_type='crank-nicolson',
        time_step=0.001,
        total_time=10.0
    )
    
    solution = solver.solve()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in [0, len(solver.time)//4, len(solver.time)//2, -1]:
        plt.plot(solver.x, solution[i, :], label=f't={solver.time[i]:.2f}')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Temperature profiles at different times')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.imshow(solution.T, aspect='auto', origin='lower', 
               extent=[0, solver.total_time, 0, solver.length],
               cmap='hot')
    plt.colorbar(label='Temperature')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('Spatiotemporal evolution')
    
    plt.tight_layout()
    plt.show()