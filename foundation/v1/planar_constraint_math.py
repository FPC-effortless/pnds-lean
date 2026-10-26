"""Optional *civilian* planar single-integrator CBF-QP mathematical reference.

This is not TAC-OSM's native fast/slow/associative model and is never a
certified hardware controller. Nominal inputs, obstacle velocities, actuator
bounds and clock alignment must be authenticated separately.

Model for one interval: p'=u; o'=w with CONSTANT actual w; sample length dt.
h=||p-o||² - radius². Nominal obstacle speed w_hat may have an explicitly
verified Euclidean error bound eps, ||w-w_hat|| <= eps.
At the start, require 2*r·(u-w_hat) + gamma*h >= 2*||r||*eps.
When h>=0, gamma*dt<=1 and velocity assumptions hold, h stays >=0 during
the interval (via h(t)=h0+2*t*r·(u-w)+t²*||u-w||²).
Failures return an INFEASIBLE result with action=None. Zero is NOT assumed
to be a safe fallback, particularly when an obstacle is approaching.

With a 2D rectangular actuator box and one CBF halfspace, the exact Euclidean
projection is found by enumerating the free point, halfspace projection, and
both ends of each clipped box edge. All feasibility tests use tolerance only
for arithmetic verification, not unreported relaxed constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class ProjectionResult:
    status: str
    action: Vec2 | None
    barrier_before: float
    constraint_margin: float | None
    reason: str
    model: str = "single_integrator_planar_constant_velocity_only"


def dot(x: Vec2, y: Vec2) -> float:
    return x[0]*y[0]+x[1]*y[1]


def squared_norm(x: Vec2) -> float:
    return dot(x, x)


def _validate_pair(name: str, x) -> Vec2:
    if len(x) != 2:
        raise ValueError(name + " must have two coordinates")
    y = (float(x[0]), float(x[1]))
    if not all(math.isfinite(v) for v in y):
        raise ValueError(name + " must contain finite numbers")
    return y


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def project_halfspace_box(
    desired: Vec2,
    normal: Vec2,
    minimum: float,
    lower: Vec2,
    upper: Vec2,
) -> Vec2 | None:
    """Exact mathematical 2D Euclidean projection to n·u>=minimum, l<=u<=h.

    Returns None if no admissible control exists; does not fabricate a fallback.
    Floating point roundoff is NOT a proof of real-world feasibility.
    """
    d, n, l, h = (_validate_pair("desired", desired),
                  _validate_pair("normal", normal),
                  _validate_pair("lower", lower),
                  _validate_pair("upper", upper))
    minimum = float(minimum)
    if not math.isfinite(minimum) or any(a > b for a,b in zip(l,h)):
        raise ValueError("invalid minimum or actuator bounds")

    # Feasible maximum of n·u over the box (closed halfspace∩box).
    max_dot = sum(ni * (hi if ni >= 0 else li)
                  for ni,lo,li,hi in ((n[0],0.0,l[0],h[0]),(n[1],0.0,l[1],h[1])))
    # Avoid falsely accepting an infeasible halfspace.
    if max_dot < minimum:
        return None

    candidates: list[Vec2] = []

    def add(x: Vec2) -> None:
        if (all(l[i] <= x[i] <= h[i] for i in range(2))
                and dot(n,x) >= minimum):
            candidates.append(x)

    box_point = (_clamp(d[0],l[0],h[0]),_clamp(d[1],l[1],h[1]))
    add(box_point)
    n2 = squared_norm(n)
    if n2 > 0:
        shift = max(0.0, (minimum-dot(n,d))/n2)
        add((d[0]+shift*n[0],d[1]+shift*n[1]))

    # If the optimum is on the rectangle boundary, it lies on an edge.
    # Solve each edge's 1D interval exactly, including corners.
    for i in range(2):
        j = 1-i
        for fixed in (l[i],h[i]):
            low, high = l[j],h[j]
            residual = minimum-n[i]*fixed
            if n[j] > 0:
                low = max(low,residual/n[j])
            elif n[j] < 0:
                high = min(high,residual/n[j])
            elif residual > 0:
                continue
            if low > high:
                continue
            q = [0.0,0.0]
            q[i]=fixed
            q[j]=_clamp(d[j],low,high)
            add((q[0],q[1]))

    if not candidates:
        # Float representation may push a true boundary solution just outside;
        # fail closed instead of returning an uncertified velocity.
        return None
    return min(candidates,key=lambda u:squared_norm((u[0]-d[0],u[1]-d[1])))


def conditional_cbf_project(
    position: Vec2, obstacle: Vec2, desired_velocity: Vec2,
    obstacle_velocity_estimate: Vec2, *,
    radius: float, gamma: float, dt: float,
    lower: Vec2, upper: Vec2, obstacle_velocity_error_bound: float = 0.0
) -> ProjectionResult:
    p,o,d,w = (_validate_pair("position",position),_validate_pair("obstacle",obstacle),
               _validate_pair("desired",desired_velocity),
               _validate_pair("obstacle_velocity_estimate",obstacle_velocity_estimate))
    l,h = _validate_pair("lower",lower),_validate_pair("upper",upper)
    vals = (radius,gamma,dt,obstacle_velocity_error_bound)
    if (not all(math.isfinite(v) for v in vals) or radius <= 0 or gamma <= 0 or
        dt <= 0 or gamma*dt > 1 or obstacle_velocity_error_bound < 0):
        raise ValueError("require positive radius,gamma,dt, gamma*dt<=1 and eps>=0")
    if any(lo > hi for lo,hi in zip(l,h)):
        raise ValueError("lower>upper")

    r = (p[0]-o[0],p[1]-o[1])
    barrier = squared_norm(r)-radius*radius
    if barrier < 0:
        return ProjectionResult("UNSAFE_INITIAL",None,barrier,None,
                                "initial state already violates h>=0")
    n = (2*r[0],2*r[1])
    # Worst-case obstacle drift, via ||w-w_hat||<=epsilon and Cauchy-Schwarz.
    margin = 2*math.sqrt(squared_norm(r))*obstacle_velocity_error_bound
    minimum = dot(n,w) - gamma*barrier + margin
    action = project_halfspace_box(d,n,minimum,l,h)
    if action is None:
        return ProjectionResult("INFEASIBLE",None,barrier,None,
                                "empty constrained admissible action set")
    achieved=dot(n,action)-minimum
    # A finite-precision tolerance does not upgrade to a mathematical guarantee.
    if achieved < -1e-10:
        return ProjectionResult("NUMERICAL_FAILURE",None,barrier,achieved,
                                "computed candidate violated source inequality")
    return ProjectionResult("CONDITIONALLY_FEASIBLE",action,barrier,achieved,
                            "valid only for declared single-integrator/constant-motion assumptions")


def conditional_minimum_barrier_at(
    r: Vec2, command: Vec2, true_obstacle_velocity: Vec2,
    radius: float, time: float
) -> float:
    """Actual assumed translation over an interval; diagnostic, not evidence."""
    relative = (command[0]-true_obstacle_velocity[0],
                command[1]-true_obstacle_velocity[1])
    position=(r[0]+time*relative[0], r[1]+time*relative[1])
    return squared_norm(position)-radius*radius
