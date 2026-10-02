"""M11: deployable physics-feature shadow-price sufficiency audit.

M10 showed that a structured price decoder is sufficient when the bus-31
shadow price is known, but that ordinary message passing cannot infer that
shadow price even after targeted M04 support.  M11 tests the next hypothesis:
pre-activation information is present in explicit electrical/local-state
features but is not exposed by the generic representation.

No frozen test or G14 target/dual is used for fitting.  The shadow-price model
is fit on original M01 train plus stratified M04 training-only support.  Model
selection uses frozen validation only.  Test and G14 are final evaluation only.

This is intentionally a sufficiency audit, not yet the final architecture.
Features are deployable proxies available from scenario loads and the frozen
nominal GNN prediction: local/neighbor load, system load, predicted voltage
margin, minimum predicted voltage, and voltage-spread statistics.  If this
simple feature map separates the rare regime, M12 should put the successful
features into a graph/KKT-aware encoder.  If it fails, the next experiment
should compute explicit AC-Jacobian sensitivities rather than tune classifiers.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from graphopf.gnn_baseline import SupervisedGraphOPF


def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def load(p):
    z=np.load(p);return {k:z[k] for k in z.files}
def graph(g,d):
    ef=g['edge_features'].astype(np.float32);em=ef.mean(0,keepdims=True);es=np.maximum(ef.std(0,keepdims=True),1e-6)
    return torch.tensor(g['edge_index'],dtype=torch.long,device=d),torch.tensor((ef-em)/es,dtype=torch.float32,device=d)
def stats(x,y):return x.mean((0,1),keepdims=True),np.maximum(x.std((0,1),keepdims=True),1e-6),y.mean((0,1),keepdims=True),np.maximum(y.std((0,1),keepdims=True),1e-6)
def train_base(tx,ty,vx,vy,g,a,d):
    st=stats(tx,ty);xm,xs,ym,ys=st;TX=torch.tensor((tx-xm)/xs,device=d);TY=torch.tensor((ty-ym)/ys,device=d);VX=torch.tensor((vx-xm)/xs,device=d);VY=torch.tensor((vy-ym)/ys,device=d);ei,ea=graph(g,d);seed(a.seed);m=SupervisedGraphOPF(hidden_dim=a.hidden,layers=a.layers).to(d);o=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
    for ep in range(1,a.epochs+1):
        m.train();o.zero_grad();p=m(TX,ei,ea);loss=F.huber_loss(p,TY,delta=1);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);o.step();m.eval()
        with torch.no_grad():vl=float(F.huber_loss(m(VX,ei,ea),VY,delta=1))
        if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if ep==1 or ep%25==0:print(f'base epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
        if stale>=a.patience:break
    m.load_state_dict(state);return m,st,ei,ea
def pred_base(m,x,st,ei,ea,d,b=256):
    xm,xs,ym,ys=st;o=[];m.eval()
    with torch.no_grad():
        for k in range(0,len(x),b):o.append(m(torch.tensor((x[k:k+b]-xm)/xs,device=d),ei,ea).cpu().numpy())
    return np.concatenate(o)*ys+ym
def stratified(z,n,seedv):
    mu=z['bus31_mu_vmin'].astype(float);active=z['bus31_active'].astype(bool);near=np.where(~active)[0];pos=np.where(active&(mu>0))[0];rng=np.random.default_rng(seedv);groups=[]
    if len(near):groups.append(('near',near))
    if len(pos):
        lm=np.log1p(mu[pos]);qs=np.quantile(lm,[0,.25,.5,.75,1])
        for k in range(4):
            hi=lm<=qs[k+1] if k==3 else lm<qs[k+1];ix=pos[(lm>=qs[k])&hi]
            if len(ix):groups.append((f'active_q{k+1}',ix))
    alloc=[n//len(groups)]*len(groups)
    for k in range(n-sum(alloc)):alloc[k%len(alloc)]+=1
    chosen=[];detail={}
    for (name,ix),want in zip(groups,alloc):
        take=rng.choice(ix,want,replace=want>len(ix));chosen.extend(take.tolist());detail[name]={'available':int(len(ix)),'selected':int(want),'mu_min':float(mu[ix].min()),'mu_max':float(mu[ix].max())}
    rng.shuffle(chosen);return np.array(chosen,int),detail
def adjacency(edge_index,n):
    A=np.zeros((n,n),float)
    for u,v in edge_index.T.astype(int):A[u,v]=A[v,u]=1
    return A
def distances(A,j):
    n=len(A);dist=np.full(n,99,int);dist[j]=0;front=[j]
    while front:
        u=front.pop(0)
        for v in np.where(A[u]>0)[0]:
            if dist[v]>dist[u]+1:dist[v]=dist[u]+1;front.append(int(v))
    return dist
def features(x,bpred,j,dist,vmin):
    pd=x[...,0].astype(float);qd=x[...,1].astype(float);vm=bpred[...,1].astype(float);m=vm[:,j]-vmin
    shells=[]
    for r in (1,2,3):
        w=(dist<=r).astype(float);w/=max(w.sum(),1);shells.extend([pd@w,qd@w])
    return np.column_stack([pd[:,j],qd[:,j],pd.sum(1),qd.sum(1),*shells,m,vm.min(1),vm.mean(1),vm.std(1),vm.max(1)-vm.min(1)])
def standardize_fit(X):
    m=X.mean(0);s=np.maximum(X.std(0),1e-8);return m,s
def ridge_fit(X,y,l2):
    Z=np.column_stack([np.ones(len(X)),X]);R=np.eye(Z.shape[1]);R[0,0]=0;return np.linalg.solve(Z.T@Z+l2*R,Z.T@y)
def ridge_pred(w,X):return np.column_stack([np.ones(len(X)),X])@w
def auc(y,s):
    y=np.asarray(y,bool);p=s[y];n=s[~y]
    if not len(p) or not len(n):return np.nan
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def threshold_from_val(y,s):
    y=np.asarray(y,bool);cand=np.unique(np.quantile(s,np.linspace(0,1,201)));best=(-1,None,None)
    for t in cand:
        p=s>=t;tp=(p&y).sum();fp=(p&~y).sum();fn=(~p&y).sum();prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1);f=2*prec*rec/max(prec+rec,1e-12)
        if f>best[0]:best=(f,float(t),(int(tp),int(fp),int(fn),int((~p&~y).sum())))
    return best
def confusion(y,s,t):
    y=np.asarray(y,bool);p=s>=t;return {'TP':int((p&y).sum()),'FP':int((p&~y).sum()),'FN':int((~p&y).sum()),'TN':int((~p&~y).sum())}
def regimes(data,n):
    r=np.array(['unknown']*n,dtype=object);p=data/'g08_regime_evaluation.csv'
    if p.exists():
        with p.open(newline='') as f:
            for q in csv.DictReader(f):
                i=int(q['scenario']);
                if 0<=i<n:r[i]=q['regime']
    return r
def continuation(data,ex):
    out=[]
    with (data/'g14_active_set_continuation.csv').open(newline='') as f:rr=[q for q in csv.DictReader(f) if str(q['solved']).lower() in ('true','1','yes')]
    for q in rr:
        s=int(q['active_scenario']);c=int(q['control_scenario']);la=float(q['lambda']);x=(ex[c].astype(float)+la*(ex[s].astype(float)-ex[c].astype(float))).astype(np.float32);out.append({'path':s,'lambda':la,'true_lmp':float(q['true_lmp']),'x':x})
    return out
def boot_delta(y0,y1,rng,B):
    d=np.asarray(y1)-np.asarray(y0);vals=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(B)]);return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--augment-n',type=int,default=200);p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=3);p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=45);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--ridge',type=float,default=1e-2);p.add_argument('--alpha-cap',type=float,default=.02);p.add_argument('--high-price',type=float,default=45.);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=20271005);a=p.parse_args();data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');m4=load(data/'m04_dual_regime'/'targeted_dual_regime.npz');g=np.load(data/'graph.npz');ids=te['bus_ids'];j=int(np.where(ids==a.trigger_bus)[0][0]);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('M11 PHYSICS-FEATURE SHADOW-PRICE SUFFICIENCY AUDIT');print(f'device={d} frozen train/val/test={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}');print('Test/G14 labels and duals are evaluation-only.')
    base,st,ei,ea=train_base(tr['x'].astype(np.float32),tr['y'].astype(np.float32),va['x'].astype(np.float32),va['y'].astype(np.float32),g,a,d);btr=pred_base(base,tr['x'].astype(np.float32),st,ei,ea,d);bva=pred_base(base,va['x'].astype(np.float32),st,ei,ea,d);bte=pred_base(base,te['x'].astype(np.float32),st,ei,ea,d)
    ix,detail=stratified(m4,a.augment_n,a.seed+1);bm4=pred_base(base,m4['x'][ix].astype(np.float32),st,ei,ea,d);print('M04 stratification='+json.dumps(detail));A=adjacency(g['edge_index'],len(ids));dist=distances(A,j);vmin=float(np.min(tr['y'][:,j,1]));vmin=min(vmin,float(np.min(va['y'][:,j,1]))) # feature reference only; no test
    X0=features(tr['x'],btr,j,dist,vmin);X4=features(m4['x'][ix],bm4,j,dist,vmin);XV=features(va['x'],bva,j,dist,vmin);XT=features(te['x'],bte,j,dist,vmin);X=np.concatenate([X0,X4]);mu=np.concatenate([tr['mu_vmin'][:,j],m4['bus31_mu_vmin'][ix]]).astype(float);ym=np.log1p(np.maximum(mu,0));xm,xs=standardize_fit(X);Z=(X-xm)/xs;ZV=(XV-xm)/xs;ZT=(XT-xm)/xs
    w=ridge_fit(Z,ym,a.ridge);sv=ridge_pred(w,ZV);stt=ridge_pred(w,ZT);av=va['mu_vmin'][:,j]>1e-6;at=te['mu_vmin'][:,j]>1e-6;f1,thr,vc=threshold_from_val(av,sv);tc=confusion(at,stt,thr);print(f'validation AUC={auc(av,sv):.6f} selected_threshold={thr:.6f} F1={f1:.6f} confusion={vc}');print(f'test AUC={auc(at,stt):.6f} confusion={tc} active_score_mean={stt[at].mean() if at.any() else np.nan:.6f} inactive_score_mean={stt[~at].mean():.6f}')
    # Training-only alpha for the unchanged M10 economic decoder.
    bb=np.concatenate([btr[:,j,0],bm4[:,j,0]]);yy=np.concatenate([tr['y'][:,j,0],m4['y'][ix,j,0]]);act=mu>1e-6;alpha=float((mu[act]@(yy[act]-bb[act]))/(mu[act]@mu[act]+1e-12));alpha=float(np.clip(alpha,-a.alpha_cap,a.alpha_cap));pmu=np.expm1(np.maximum(stt,0));pmu[stt<thr]=0.;structured=bte.copy();structured[:,j,0]=bte[:,j,0]+alpha*pmu;reg=regimes(data,len(te['x']));print(f'training-only alpha={alpha:.9g} active_calibration_n={act.sum()}')
    groups={};rng=np.random.default_rng(a.seed);print('\nFROZEN TEST PRICE')
    for name,mask in [('all',np.ones(len(te['x']),bool)),('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]:
        if not mask.any():continue
        e0=abs(bte[mask,j,0]-te['y'][mask,j,0]);e1=abs(structured[mask,j,0]-te['y'][mask,j,0]);groups[name]={'n':int(mask.sum()),'baseline_trigger_mae':float(e0.mean()),'structured_trigger_mae':float(e1.mean()),'delta_ci95':boot_delta(e0,e1,rng,a.bootstrap)};print(f'{name:14s} n={mask.sum():4d} trigger baseline/structured={e0.mean():.6f}/{e1.mean():.6f} delta95={groups[name]["delta_ci95"]}')
    cont=continuation(data,te['x'].astype(np.float32));cx=np.stack([q['x'] for q in cont]);cb=pred_base(base,cx,st,ei,ea,d);CF=features(cx,cb,j,dist,vmin);cs=ridge_pred(w,(CF-xm)/xs);cmu=np.expm1(np.maximum(cs,0));cmu[cs<thr]=0.;cp=cb[:,j,0]+alpha*cmu;cres={};print('\nFROZEN G14 CONTINUATION')
    for path in sorted(set(q['path'] for q in cont)):
        ii=np.array([k for k,q in enumerate(cont) if q['path']==path and 0<=q['lambda']<=1]);yt=np.array([cont[k]['true_lmp'] for k in ii]);eb=abs(cb[ii,j,0]-yt);es=abs(cp[ii]-yt);hi=yt>=a.high_price;cres[str(path)]={'n':int(len(ii)),'baseline_mae':float(eb.mean()),'structured_mae':float(es.mean()),'high_price_n':int(hi.sum()),'baseline_high_price_mae':float(eb[hi].mean()) if hi.any() else None,'structured_high_price_mae':float(es[hi].mean()) if hi.any() else None,'max_predicted_score':float(cs[ii].max()),'max_predicted_mu':float(cmu[ii].max())};print(f'path={path} MAE base/structured={eb.mean():.4f}/{es.mean():.4f} high_price={eb[hi].mean() if hi.any() else np.nan:.4f}/{es[hi].mean() if hi.any() else np.nan:.4f} max_score={cs[ii].max():.3f} max_mu={cmu[ii].max():.3f}')
    names=['pd31','qd31','total_pd','total_qd','r1_pd','r1_qd','r2_pd','r2_qd','r3_pd','r3_qd','pred_vmargin31','pred_vmin','pred_vmean','pred_vstd','pred_vspread'];coef={n:float(v) for n,v in zip(names,w[1:])};summary={'configuration':vars(a),'m04_stratification':detail,'validation':{'auc':auc(av,sv),'threshold':thr,'f1':f1,'confusion':vc},'test_shadow_price':{'auc':auc(at,stt),'confusion':tc,'active_n':int(at.sum()),'active_score_mean':float(stt[at].mean()) if at.any() else None,'inactive_score_mean':float(stt[~at].mean())},'alpha':alpha,'standardized_coefficients':coef,'frozen_test_price':groups,'g14':cres};out=data/'m11_physics_feature_shadow_price_summary.json';out.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nSummary: {out}');print('Decision: success requires frozen active detection and G14 high-price improvement with no material regular-regime degradation. Failure means hand-crafted primal proxies are insufficient; proceed to explicit AC-Jacobian/KKT sensitivity features rather than additional classifier tuning.')
if __name__=='__main__':main()
