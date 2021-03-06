# The Bidomain model

\begin{equation}
\begin{alignedat}{3}
    \frac{\partial v}{\partial t} &= f(v, s, t) \quad&&\quad x &\in GM \\
    \nabla \cdot(M_i\nabla v) + \nabla \cdot (M_i\nabla u_e) &= \frac{\partial v}{\partial t} \quad&&\quad x &\in GM \\
    \nabla \cdot (M_i\nabla v) + \nabla \cdot ((M_i + M_e)\nabla u_e) &= 0 \quad&&\quad x &\in GM \\
    \nabla \cdot (M_o\nabla u_o) &= 0 \quad&&\quad x &\in CSF \\
    u_e &= u_o \quad&&\quad x &\in \partial GM \\
    (M_i\nabla v + (M_i + M_e)\nabla u_e)\cdot n &= (M_o\nabla u_o)\cdot n \quad&&\quad x &\in \partial GM \\
    (M_i \nabla v + M_i \nabla u_e) \cdot n &= 0 \quad&&\quad x &\in \partial GM \\
    (M_o \nabla u_o)\cdot n &= 0 \quad&&\quad x &\in \partial CSF \\
\end{alignedat}
\end{equation}

## The time discrete system

\begin{align}
    \frac{v_{\theta}^{n + 1} - v^n}{\Delta t} = \theta \nabla \cdot (M_i \nabla v_{\theta}^{n + 1})
     & + (1 - \theta)\nabla \cdot (M_i\nabla v_{\theta}^n) \\
     &+ \nabla \cdot(M_i\nabla u_e^{n + \theta}), \\
     \theta \nabla \cdot (M_i\nabla v_{\theta}^{n + 1}) +
     \nabla \cdot ((M_i + M_e)\nabla u_e^{n + \theta}) &=
     -(1 - \theta)\nabla \cdot (M_i \nabla v_{\theta}^n),
\end{align}

### Some simplified notation

\begin{alignat}{4}
    (\varphi, \phi) &= \int_{GM} \varphi \phi\; dx,\;  &\text{for}\; \varphi, \phi \in V,\\
    a_I(\varphi, \phi) &= \int_{GM} M_i \nabla \varphi \cdot \nabla \phi\; dx,\; &\text{for}\; \varphi, \phi \in V,\\
    a_{I + E}(\varphi, \phi) &= \int_{GM} (M_i + M_e)\nabla \varphi \cdot \nabla \phi\; dx,\; &\text{for}\; \varphi, \phi \in V,
\end{alignat}


## The weak form of the time discrete system

\begin{equation}
\begin{aligned}
    (v_{\theta}^{n + 1}, \psi) &+ \theta\Delta t a_I(v_{\theta}^{n + 1}, \psi) + \Delta t a_I(u_e^{n + \theta}, \psi) \\
    &- \theta\Delta t\int_{\partial GM}\psi[(M_i\nabla v_{\theta}^{n + 1})\cdot n]ds
    - \theta\Delta t\int_{\partial GM}\psi[(M_i\nabla u_e^{n + 1})\cdot n]ds \\
    &= (v_{\theta}, \psi) - (1 - \theta)\Delta t a_I(v_{\theta}^n, \psi) \\
    & + (1 - \theta)\Delta t \int_{\partial GM}\psi[(M_i\nabla v_{\theta}^n)\cdot n]ds
    \quad \text{for all}\; \psi \in V,
\end{aligned}
\end{equation}

\begin{equation}
\begin{aligned}
    -\theta a_I(v_{\theta}^{n + 1}, \psi) &- a_{I + E}(u_e^{n + \theta}, \psi)
    + \theta\int_{\partial GM}\psi[(M_i\nabla v_{\theta}^{n + 1})\cdot n]ds \\
    &+ (1 - \theta)\int_{\partial GM}\psi[(M_i\nabla v_{\theta}^n)\cdot n]ds
    + \int_{\partial GM}\psi[(M_i \nabla u_e^{n + \theta})\cdot n]ds \\
    &+ \int_{\partial GM}\psi[(M_e\nabla u_e^{n + \theta})\cdot n]s
    - \textcolor{red}{(1 - \theta)\nabla \cdot (M_i\nabla v_{\theta}^n)} = 0\; \text{for all}\; \psi \in V
\end{aligned}
\end{equation}

## Discrete boundary conditions

\begin{equation}
    (\theta M_i \nabla v_{\theta}^{n + 1} + (1 - \theta)M_i\nabla v_{\theta}^n + 
    M_i\nabla u_e^{n + \theta}) \cdot n = 0
\end{equation}

## The final system

With this last boundary condition, all the boundary terms cancel, and we are left with

\begin{equation}
\begin{aligned}
    (v_{\theta}^{n + 1}, \psi) &+ \theta\Delta t a_I(v_{\theta}^{n + 1}, \psi) + \Delta t a_I(u_e^{n \ \theta}, \psi) \\
    &= (v_{\theta}, \psi) - (1 - \theta)\Delta t a_I(v_{\theta}^n, \psi) \quad \text{for all}\; \psi \in V,
\end{aligned}
\end{equation}

\begin{equation}
    -\theta a_I(v_{\theta}, \psi) - a_{I + E}(u_e^{n + \theta}, \psi) = -(1 - \theta)\nabla \cdot (M_i\nabla v_{\theta}^n) 
    \; \text{for all}\; \psi \in V
\end{equation}

# Coupling the heart and the torso

## Some simplifying notation

\begin{equation}
    a_{CSF} = \int_{CSF} M_o\nabla \varphi \cdot \nabla \phi\; dx \; \text{for}\; \varphi, \phi \in V(CSF).
\end{equation}


## The time discrete system

The passive conductor is described by

\begin{equation}
    \nabla \cdot (M_o\nabla u_o) = 0
\end{equation}

## The weak formulation

\begin{equation}
\begin{aligned}
    -a_{CSF}(u_o^{n + 1}), \eta &+ \int_{\partial GM} \eta[(M_o\nabla u_o^{n + \theta})\cdot n_{CSF}]ds \\
    &+\int_{\partial CSF} \eta[(M_o \nabla u_o^{n + \theta})\cdot n_{CSF}]ds = 0\quad \text{for all}\;\eta \in V(CSF).
\end{aligned}
\end{equation}

## Unifying the domains

Using the fact that $u_e = u_o \;\text{on}\; \partial GM$ construct a new field $u$ over the complete domain
$GM \cup CSF$ with an associated function space $V(GM \cup CSF)$. Most notably, all functions in $V$ must be
continuous across $\partial GM$. We can now simplify the unified weak form

\begin{equation}
\begin{aligned}
    -\theta a_I(v_{\theta}^{n + 1}, \varphi) &- A_{I + E}(u_e^{n + \theta}, \varphi) \\
    &+ \int_{\partial GM}\varphi[(M_e\nabla u_e^{n + \theta})\cdot n] ds = 0\quad \text{for all}\;\varphi
    \in V(GM \cup CSF)\\
    -a_{CSF}(u_o^{n + \theta}, \varphi) &- \int_{\partial GM}\varphi[(M_o\nabla u_o^{n + \theta})\cdot n]dx = 0
    \quad \text{for all}\; \varphi \in V(GM \cup CSF)\\
\end{aligned}
\end{equation}

Note that the sign has changed in the second integral because the unit normal is pointing into the domain.
When we add these two equations the boundary integral vanishes assuming that the conductivities $M$
are continuous. If not I suspect we need the integral

\begin{equation}
    \int_{\partial GM}\varphi[((M_e - M_o)\nabla u_o^{n + \theta})\cdot n]ds = 0.
\end{equation}


## Splitting the GM into GM and WM --- Discontinouous conductivity

We return to the weak forms, and start by summarising the boundary conditions.

\begin{equation}
    (M_e\nabla u_e)\cdot n = 0, \text{for}\; x \in \partial GM_{WM}
\end{equation}

holds for all time steps. As there is a discontinuity in the extracellular we need to make sure

\begin{equation}
    ((M_e^{GM} - M_e^{WM})\nabla u_e)\cdot n = 0, \text{for}\; x \in \partial GM_{WM}
\end{equation}



We introduce a time discretised version of the other boundary conditions:

\begin{equation}
    (\theta M_i\nabla v_{\theta}^{n + 1} + (1 - \theta)M_i\nabla v_{\theta}^n + M_i \nabla u_e^{n + \theta})\cdot n = 0
    \; \text{for}\; x \in \partial GM_{WM}
\end{equation}

As the only discontinuity is in the extracellular potential, we do not need to modify this equation.
