"""Mechanism diagnostics only; no physical execution, safety certification or real data."""
from __future__ import annotations
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation/v1"))
from planar_constraint_math import (project_halfspace_box,conditional_cbf_project,
    conditional_minimum_barrier_at, dot)


class PlanarConstraintMath(unittest.TestCase):
    def test_unconstrained_action_stays_unchanged(self):
        u=project_halfspace_box((.2,.1),(1.,0.),-.5,(-1.,-1.),(1.,1.))
        self.assertEqual(u,(.2,.1))

    def test_active_halfspace_projection_kkt(self):
        u=project_halfspace_box((0.,0.),(2.,0.),2.,(-3.,-3.),(3.,3.))
        self.assertEqual(u,(1.,0.))
        self.assertAlmostEqual(dot(u,(2.,0.)),2.)

    def test_active_box_edge_projection(self):
        u=project_halfspace_box((.9,-.9),(1.,1.),1.5,(0.,0.),(1.,1.))
        self.assertIsNotNone(u)
        self.assertAlmostEqual(u[0],1.)
        self.assertAlmostEqual(u[1],.5)
        self.assertAlmostEqual(dot((1.,1.),u),1.5)

    def test_box_empty_does_not_return_zero(self):
        u=project_halfspace_box((0.,0.),(2.,0.),3.,(-1.,-1.),(1.,1.))
        self.assertIsNone(u)

    def test_same_location_with_positive_radius_is_preexisting_unsafe(self):
        r=conditional_cbf_project((0.,0.),(0.,0.),(0.,0.),(0.,0.),
                                  radius=1.,gamma=1.,dt=.1,lower=(-2.,-2.),upper=(2.,2.))
        self.assertEqual(r.status,"UNSAFE_INITIAL")
        self.assertIsNone(r.action)

    def test_approaching_obstacle_makes_zero_unsafe(self):
        # At contact and an obstacle moving toward the robot, zero control
        # violates hdot >= 0. A "solver failed -> zero" rule is unsound.
        result=conditional_cbf_project((1.,0.),(0.,0.),(0.,0.),(1.,0.),
          radius=1.,gamma=1.,dt=.1,lower=(-2.,-2.),upper=(2.,2.))
        self.assertEqual(result.status,"CONDITIONALLY_FEASIBLE")
        self.assertGreaterEqual(result.action[0],1.)
        bad=conditional_minimum_barrier_at((1.,0.),(0.,0.),(1.,0.),1.,.1)
        self.assertLess(bad,0.)

    def test_infeasible_dynamic_obstacle_reports_none(self):
        result=conditional_cbf_project((1.,0.),(0.,0.),(0.,0.),(5.,0.),
          radius=1.,gamma=1.,dt=.1,lower=(-1.,-1.),upper=(1.,1.))
        self.assertEqual(result.status,"INFEASIBLE")
        self.assertIsNone(result.action)

    def test_sampled_motion_bound_holds_for_all_real_t_in_simple_model(self):
        result=conditional_cbf_project((1.5,.2),(.1,.1),(.2,-.1),(.4,0.),
            radius=1.,gamma=.8,dt=.2,lower=(-2.,-2.),upper=(2.,2.))
        self.assertEqual(result.status,"CONDITIONALLY_FEASIBLE")
        r=(1.4,.1)
        for t in [i*.2/200 for i in range(201)]:
            self.assertGreaterEqual(conditional_minimum_barrier_at(
                r,result.action,(.4,0.),1.,t),-1e-12)

    def test_uncertainty_requires_extra_clearance(self):
        args=((1.05,0.),(0.,0.),(0.,0.),(.4,0.))
        bare=conditional_cbf_project(*args,radius=1.,gamma=1.,dt=.1,
                  lower=(-4.,-4.),upper=(4.,4.))
        robust=conditional_cbf_project(*args,radius=1.,gamma=1.,dt=.1,
                  lower=(-4.,-4.),upper=(4.,4.),obstacle_velocity_error_bound=.25)
        self.assertGreaterEqual(robust.action[0],bare.action[0]+.25-1e-10)
        for actual_w0 in (.15,.4,.65):
            for t in [i*.1/30 for i in range(31)]:
                self.assertGreaterEqual(conditional_minimum_barrier_at(
                  (1.05,0.),robust.action,(actual_w0,0.),1.,t),-1e-10)

    def test_rejects_false_time_and_positive_gamma_assumptions(self):
        kw=dict(radius=1.,gamma=2.,dt=.6,lower=(-2.,-2.),upper=(2.,2.))
        with self.assertRaises(ValueError):
            conditional_cbf_project((2.,0.),(0.,0.),(0.,0.),(0.,0.),**kw)
        with self.assertRaises(ValueError):
            project_halfspace_box((0.,0.),(1.,0.),math.nan,(-1.,-1.),(1.,1.))

    def test_exact_projection_beats_grid_competitors_under_one_constraint(self):
        rng=random.Random(14177)
        for _ in range(30):
            d=(rng.uniform(-3,3),rng.uniform(-3,3))
            n=(rng.uniform(-2,2),rng.uniform(-2,2))
            lower=(-1.,-1.); upper=(1.,1.)
            maximum=sum(n[i]*(1. if n[i]>=0 else -1.) for i in range(2))
            threshold=maximum-rng.uniform(0.,1.5)
            u=project_halfspace_box(d,n,threshold,lower,upper)
            self.assertIsNotNone(u)
            self.assertGreaterEqual(dot(n,u)+1e-11,threshold)
            ours=sum((u[i]-d[i])**2 for i in range(2))
            for i in range(21):
                for j in range(21):
                    q=(-1+i/10,-1+j/10)
                    if dot(n,q)>=threshold:
                        competitor=sum((q[k]-d[k])**2 for k in range(2))
                        self.assertLessEqual(ours,competitor+1e-9)

if __name__=="__main__":
    unittest.main()
