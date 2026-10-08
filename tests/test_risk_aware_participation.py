import torch
from graphopf.risk_aware_model import RiskAwareGraphOPF

def test_fixed_range_generators_have_zero_agc_participation():
    torch.manual_seed(0)
    m=RiskAwareGraphOPF(node_dim=8,edge_dim=6,hidden_dim=16,layers=1).double()
    x=torch.zeros((1,4,8),dtype=torch.float64)
    edge_index=torch.tensor([[0,1,2],[1,2,3]],dtype=torch.long)
    edge_attr=torch.zeros((3,6),dtype=torch.float64)
    gen_mask=torch.tensor([True,True,True,False])
    pmin=torch.tensor([0.,0.,5.,0.],dtype=torch.float64)
    pmax=torch.tensor([10.,0.,5.,0.],dtype=torch.float64)
    vmin=torch.full((4,),0.9,dtype=torch.float64)
    vmax=torch.full((4,),1.1,dtype=torch.float64)
    out=m(x,edge_index,edge_attr,gen_mask,pmin,pmax,vmin,vmax)
    a=out["alpha_bus"][0]
    assert a[1].item()==0.0
    assert a[2].item()==0.0
    assert a[3].item()==0.0
    assert torch.allclose(a.sum(),torch.tensor(1.0,dtype=torch.float64))
    assert a[0].item()==1.0
