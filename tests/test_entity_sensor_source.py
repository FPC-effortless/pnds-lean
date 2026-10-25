"""Run the independent full-graph refinement against an immutable TAC source."""
import os,sys,unittest,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
TAC=Path(os.environ.get('TAC_SOURCE_ROOT','/tmp/tac-evidence')).resolve()
sys.path[:0]=[str(TAC),str(TAC/'src')]
from tac_osm.variants.entity_sensor_world import EntitySensorWorld
spec=importlib.util.spec_from_file_location('pnds_entity_reference',ROOT/'foundation/v1/entity_sensor_reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

class SourceTest(unittest.TestCase):
    def test_full_source_refinement(self):
        self.assertEqual((ROOT/'foundation/v1/entity_sensor_reference.py').read_bytes(),(TAC/'scripts/me100_graph_reference.py').read_bytes())
        torch.set_num_threads(1)
        for dtype,tol in [(torch.float32,2e-5),(torch.float64,1e-9)]:
            torch.manual_seed(100);m=EntitySensorWorld().to(dtype)
            with torch.no_grad():m.readout[2].weight[:6].normal_(0,.01)
            x=torch.randn(2,23,13,dtype=dtype);o=torch.rand(2,23,13)>.3;a=torch.rand_like(x);c=torch.randn(2,23,6,dtype=dtype).tanh();ids=torch.tensor([2,9]);hours=torch.arange(23)[None].expand(2,-1)
            p,s=m(x,o,a,c,ids,hours);r=ref.serial(m,x,o,a,c,ids)
            for v,w in zip((p.log_location,p.log_scale,s.source_sum,s.source_mass,s.evidence_sum,s.memory),r):torch.testing.assert_close(v,w,rtol=tol,atol=tol*.1)
            gp=torch.autograd.grad(p.log_location.square().sum()+p.log_scale.square().sum(),tuple(m.parameters()))
            gr=torch.autograd.grad(r[0].square().sum()+r[1].square().sum(),tuple(m.parameters()))
            for v,w in zip(gp,gr):torch.testing.assert_close(v,w,rtol=tol*10,atol=tol)
