"""Count-supervised local intensity exchanges: standalone NumPy reference.

This is a finite numerical reference, not a native model adapter. Geometry and
inference do not accept targets. Training-only counterfactual labels are isolated.
Native footprint/normalization/scorer parity must be established locally.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def _image(image):
    x=np.asarray(image)
    require(x.ndim==2 and x.size>0 and x.dtype==np.uint8, 'nonempty uint8 HxW image required')
    return x


def _kernel(kernel):
    k=np.asarray(kernel)
    require(k.ndim==2 and k.dtype==np.bool_ and all(v%2==1 for v in k.shape), 'odd boolean kernel required')
    center=tuple(v//2 for v in k.shape)
    require(k[center] and np.array_equal(k,k[::-1,::-1]), 'kernel must contain its center and be centrosymmetric')
    return np.argwhere(k)-np.asarray(center)


def _indices(shape,index,offsets):
    require(isinstance(index,(int,np.integer)) and not isinstance(index,(bool,np.bool_)) and 0<=index<np.prod(shape), 'invalid pixel index')
    h,w=shape; r,c=divmod(int(index),w); rc=offsets+np.array([r,c]);valid=(rc[:,0]>=0)&(rc[:,0]<h)&(rc[:,1]>=0)&(rc[:,1]<w)
    return rc[valid,0]*w+rc[valid,1]


def filter_sum(mask,kernel):
    """Zero-padded integer neighborhood count; no wrapping across borders."""
    a=np.asarray(mask)
    require(a.ndim==2 and a.dtype==np.bool_, 'boolean 2-D mask required')
    off=_kernel(kernel);h,w=a.shape;out=np.zeros(a.shape,np.int32)
    for dr,dc in off:
        r0,r1=max(0,-int(dr)),min(h,h-int(dr));c0,c1=max(0,-int(dc)),min(w,w-int(dc))
        if r1>r0 and c1>c0:out[r0:r1,c0:c1]+=a[r0+dr:r1+dr,c0+dc:c1+dc]
    return out


def max_filter(image,kernel):
    x=_image(image);off=_kernel(kernel);h,w=x.shape;out=np.zeros_like(x)
    for dr,dc in off:
        r0,r1=max(0,-int(dr)),min(h,h-int(dr));c0,c1=max(0,-int(dc)),min(w,w-int(dc))
        if r1>r0 and c1>c0:
            np.maximum(out[r0:r1,c0:c1],x[r0+dr:r1+dr,c0+dc:c1+dc],out=out[r0:r1,c0:c1])
    return out


def _tail(values,weights=None):
    v=np.asarray(values)
    require(v.dtype==np.uint8, 'tail values must be uint8')
    if weights is None:hist=np.bincount(v.ravel(),minlength=256).astype(np.int64)
    else:
        w=np.asarray(weights);require(w.shape==v.shape and w.dtype==np.bool_, 'boolean count weights required')
        hist=np.bincount(v[w],minlength=256).astype(np.int64)
    return hist.sum()-np.cumsum(hist) # Strict >t, including t=255 =>0.


def counts(image,target,kernel):
    """TRAINING/EVALUATION ONLY: all 256 thresholds, no mean-image F1."""
    x=_image(image);y=np.asarray(target)
    require(y.shape==x.shape and y.dtype==np.bool_, 'boolean target congruent with image required')
    near=filter_sum(y,kernel)>0
    return dict(K=_tail(x),T=_tail(x,y),U=_tail(x,near),V=_tail(max_filter(x,kernel),y),G=int(y.sum()))


def pool_count_records(records):
    require(len(records)>0,'nonempty records')
    out={k:np.zeros(256,np.int64) for k in ('K','T','U','V')};out['G']=0
    for r in records:
        for k in ('K','T','U','V'):
            a=np.asarray(r[k]);require(a.shape==(256,) and a.dtype.kind in 'iu' and (a>=0).all(),'invalid count curve')
            out[k]+=a
        require(isinstance(r['G'],(int,np.integer)) and r['G']>=0,'invalid G');out['G']+=int(r['G'])
    require((out['U']<=out['K']).all() and (out['V']<=out['G']).all(),'impossible pooled counts')
    return out


def fcurve(record):
    K,U,V=(np.asarray(record[k],np.float64) for k in ('K','U','V'));G=float(record['G'])
    den=U*G+V*K
    return np.divide(2*U*V,den,out=np.zeros_like(den),where=den>0)


def pooled_slopes(record,threshold):
    """dF/dU,dF/dV at the actual pooled reference counts, not imagewise ratios."""
    require(isinstance(threshold,(int,np.integer)) and 0<=threshold<=255,'invalid threshold')
    U,V,K,G=float(record['U'][threshold]),float(record['V'][threshold]),float(record['K'][threshold]),float(record['G'])
    D=U*G+V*K
    return (2*K*V*V/(D*D),2*G*U*U/(D*D)) if D>0 else (0.,0.)


@dataclass(frozen=True)
class Geometry:
    image_shape: tuple[int, int]
    p: int
    q: int
    threshold: int
    point_ids: np.ndarray
    point_delta: np.ndarray
    coverage_ids: np.ndarray
    coverage_delta: np.ndarray
    influence_ids: np.ndarray
    cost_sse: int


def geometry(image,p,q,threshold,kernel,cover_counts=None):
    """TARGET-FREE sparse changes in S_t and dilation(S_t) for one byte swap.

    Optional cover_counts must be the neighborhood count for this exact image,
    threshold and footprint. The reference verifies it rather than trusting it.
    """
    x=_image(image);off=_kernel(kernel)
    pids=_indices(x.shape,p,off);qids=_indices(x.shape,q,off)
    require(p!=q,'swap endpoints must differ')
    require(isinstance(threshold,(int,np.integer)) and not isinstance(threshold,(bool,np.bool_)) and 0<=threshold<=255,'invalid threshold')
    p,q,t=int(p),int(q),int(threshold)
    selected=x>t;valid_counts=filter_sum(selected,kernel)
    if cover_counts is not None:
        cc=np.asarray(cover_counts);require(cc.shape==x.shape and cc.dtype.kind in 'iu' and np.array_equal(cc,valid_counts),'stale or wrong coverage count map')
    else:cc=valid_counts
    before=np.array([selected.ravel()[p],selected.ravel()[q]],np.int8)
    change=before[::-1]-before
    influence=np.union1d(pids,qids)
    dcount=np.zeros(len(influence),np.int32)
    np.add.at(dcount,np.searchsorted(influence,pids),int(change[0]))
    np.add.at(dcount,np.searchsorted(influence,qids),int(change[1]))
    cb=cc.ravel()[influence];ca=cb+dcount;require((ca>=0).all(),'negative coverage multiplicity')
    dc=(ca>0).astype(np.int8)-(cb>0).astype(np.int8)
    nz=dc!=0;point_ids=np.array([p,q],np.int64);pnz=change!=0
    return Geometry(tuple(x.shape),p,q,t,point_ids[pnz],change[pnz],influence[nz],dc[nz],influence,
                    2*(int(x.ravel()[p])-int(x.ravel()[q]))**2)


def counterfactual_target(g,target,kernel):
    """TRAINING ONLY: (delta near-hit U, delta unique coverage V, delta exact T)."""
    y=np.asarray(target);require(y.ndim==2 and y.dtype==np.bool_,'boolean training target')
    require(tuple(y.shape)==g.image_shape,'target geometry mismatch')
    n=(filter_sum(y,kernel)>0).ravel();yr=y.ravel()
    return np.array([np.dot(g.point_delta.astype(np.int64),n[g.point_ids].astype(np.int64)),
                     np.dot(g.coverage_delta.astype(np.int64),yr[g.coverage_ids].astype(np.int64)),
                     np.dot(g.point_delta.astype(np.int64),yr[g.point_ids].astype(np.int64))],np.int64)


def design_rows(g,features):
    """Shared label-density readout for exact and coverage count changes.

    features is [H*W,D], including an optional constant column. Native integration
    should sum fine features and coarse token weights separately, not tile a
    1536-wide field to native resolution.
    """
    phi=np.asarray(features,np.float64)
    require(phi.ndim==2 and phi.shape[1]>0 and np.isfinite(phi).all(),'finite pixel features required')
    require(phi.shape[0]==np.prod(g.image_shape),'feature geometry mismatch')
    ds=np.sum(phi[g.point_ids]*g.point_delta[:,None],axis=0)
    dc=np.sum(phi[g.coverage_ids]*g.coverage_delta[:,None],axis=0)
    return ds,dc


def ridge(X,y,weights=None,lam=1e-3):
    """Minimize sum normalized_weight*(X beta-y)^2 + lam*||beta||^2."""
    X=np.asarray(X,np.float64);y=np.asarray(y,np.float64)
    require(X.ndim==2 and len(X)>0 and y.shape==(len(X),) and np.isfinite(X).all() and np.isfinite(y).all(),'invalid ridge design')
    require(np.isfinite(lam) and lam>0,'strictly positive ridge required')
    w=np.ones(len(X)) if weights is None else np.asarray(weights,np.float64)
    require(w.shape==y.shape and np.isfinite(w).all() and (w>=0).all() and w.sum()>0,'invalid ridge weights')
    w=w/w.sum();A=X.T@(w[:,None]*X)+float(lam)*np.eye(X.shape[1]);b=X.T@(w*y)
    beta=np.linalg.solve(A,b);require(np.isfinite(beta).all(),'nonfinite ridge solution')
    require(np.linalg.norm(A@beta-b)<=1e-8*(1+np.linalg.norm(b)),'ridge solve residual')
    return beta


def fit_readouts(point_rows,coverage_rows,targets,coverage_scales,weights=None,lam=1e-3):
    """Train two linear count readouts, not calibrated probability heads.

    targets[:,0:3]=(U,V,T). Coverage errors divided by the geometric changed-
    coverage support size, at least1. Each U,T,V loss has the same total weight.
    """
    ds,dc,y=np.asarray(point_rows,np.float64),np.asarray(coverage_rows,np.float64),np.asarray(targets,np.float64)
    scales=np.asarray(coverage_scales,np.float64)
    require(ds.ndim==2 and dc.shape==ds.shape and y.shape==(len(ds),3) and scales.shape==(len(ds),) and np.isfinite(scales).all() and (scales>=1).all(),'invalid readout data')
    w=np.ones(len(ds)) if weights is None else np.asarray(weights,np.float64)
    near=ridge(ds,y[:,0],w,lam)
    # To preserve equal U/T/V terms with a single shared Y penalty, the average
    # of T,V losses has its penalty halved relative to the joint sum objective.
    label=ridge(np.vstack((ds,dc/scales[:,None])),np.r_[y[:,2],y[:,1]/scales],np.r_[w,w],lam/2)
    return near,label


def predict(g,features,near_readout,label_readout):
    """TARGET-FREE bounded signed count predictions. They are not confidence bounds."""
    ds,dc=design_rows(g,features);a,b=np.asarray(near_readout),np.asarray(label_readout)
    require(a.shape==b.shape==ds.shape and np.isfinite(a).all() and np.isfinite(b).all(),'invalid readout')
    raw=np.array([ds@a,dc@b,ds@b],np.float64)
    low=np.array([-np.count_nonzero(g.point_delta<0),-np.count_nonzero(g.coverage_delta<0),-np.count_nonzero(g.point_delta<0)],float)
    high=np.array([np.count_nonzero(g.point_delta>0),np.count_nonzero(g.coverage_delta>0),np.count_nonzero(g.point_delta>0)],float)
    return np.clip(raw,low,high)


def candidate_pairs(image,threshold,step=16):
    """All horizontal/vertical pairs exactly one native bin side apart crossing t.

    This deliberately permits within-bin order changes; it does not reimpose
    the failed prefix allocator. Pair proposal uses no labels or FM scores.
    """
    x=_image(image);require(isinstance(step,int) and not isinstance(step,bool) and step>0,'positive integer step')
    require(isinstance(threshold,(int,np.integer)) and not isinstance(threshold,(bool,np.bool_)) and 0<=threshold<=255,'invalid threshold')
    h,w=x.shape;ii=np.arange(x.size).reshape(x.shape);parts=[]
    if w>step:
        a,b=ii[:,:-step].ravel(),ii[:,step:].ravel();ok=(x.ravel()[a]>threshold)!=(x.ravel()[b]>threshold);parts.append(np.stack((a[ok],b[ok]),1))
    if h>step:
        a,b=ii[:-step,:].ravel(),ii[step:,:].ravel();ok=(x.ravel()[a]>threshold)!=(x.ravel()[b]>threshold);parts.append(np.stack((a[ok],b[ok]),1))
    return np.vstack(parts).astype(np.int64) if parts else np.empty((0,2),np.int64)


def select_and_render(image,pairs,geometries,predictions,slopes,*,rms_limit=1.,max_distance=16,reverse=False):
    """TARGET-FREE one-pass disjoint-influence greedy exchange policy.

    geometries[a][q] binds all screened thresholds to the same action; predictions
    is [actions, thresholds, (U,V,T)]. Threshold0 in this list is the training-
    chosen operating threshold. Require predicted U,V,T >=0 at every screened
    threshold, and positive derivative-weighted utility at the operating point.
    """
    x=_image(image);pairs=np.asarray(pairs);pr=np.asarray(predictions,np.float64);wu,wv=map(float,slopes)
    require(pairs.ndim==2 and pairs.shape[1]==2 and pairs.dtype.kind in 'iu','integer [actions,2] pairs')
    require(pr.ndim==3 and pr.shape[0]==len(pairs) and pr.shape[1]>0 and pr.shape[2]==3 and np.isfinite(pr).all(),'invalid predictions')
    require(len(geometries)==len(pairs) and len(set(map(tuple,pairs)))==len(pairs),'duplicate or unbound candidates')
    require(np.isfinite(rms_limit) and rms_limit>=0 and np.isfinite(wu) and np.isfinite(wv) and min(wu,wv)>=0,'invalid budget or utility weights')
    require(isinstance(reverse,(bool,np.bool_)),'boolean tie convention')
    require(isinstance(max_distance,(int,np.integer)) and not isinstance(max_distance,(bool,np.bool_)) and max_distance>0,'positive integer locality radius')
    require(((pairs>=0)&(pairs<x.size)).all() and (pairs[:,0]!=pairs[:,1]).all(),'invalid candidate endpoint')
    costs=[];supports=[]
    for k,(p,q) in enumerate(pairs):
        require(len(geometries[k])==pr.shape[1],'threshold geometry mismatch')
        a=geometries[k][0];require(a.image_shape==tuple(x.shape) and a.p==p and a.q==q,'geometry bound to different action')
        require(np.array_equal(np.unique(a.influence_ids),a.influence_ids) and (a.influence_ids>=0).all() and (a.influence_ids<x.size).all() and p in a.influence_ids and q in a.influence_ids,'invalid influence support')
        require(all(g.image_shape==tuple(x.shape) and g.p==p and g.q==q and g.cost_sse==a.cost_sse and np.array_equal(g.influence_ids,a.influence_ids) for g in geometries[k]),'geometry differs across thresholds')
        rr,cc=np.array(divmod(int(p),x.shape[1]))-np.array(divmod(int(q),x.shape[1]))
        require(rr*rr+cc*cc<=max_distance*max_distance,'nonlocal exchange')
        require(a.cost_sse==2*(int(x.ravel()[p])-int(x.ravel()[q]))**2,'stale donor geometry')
        costs.append(a.cost_sse);supports.append(a.influence_ids)
    costs=np.asarray(costs,np.int64);score=wu*pr[:,0,0]+wv*pr[:,0,1]
    eligible=(pr>=0).all(axis=(1,2))&(score>0)&(costs>0)
    idx=np.arange(len(pairs));order=np.lexsort(((-idx if reverse else idx),-score))
    used=np.zeros(x.size,bool);out=x.copy();selected=[];sse=0;budget=x.size*float(rms_limit)**2
    for a in order:
        if not eligible[a] or used[supports[a]].any() or sse+costs[a]>budget:continue
        p,q=map(int,pairs[a]);out.ravel()[p],out.ravel()[q]=out.ravel()[q],out.ravel()[p]
        used[supports[a]]=True;sse+=int(costs[a]);selected.append(int(a))
    actual=int(np.square(out.astype(np.int64)-x.astype(np.int64)).sum())
    require(sse==actual and np.array_equal(_tail(out),_tail(x)),'histogram or edit-energy invariant failed')
    return out,dict(selected_indices=selected,selected_count=len(selected),candidate_count=len(pairs),sse=sse,
                    rms=float(np.sqrt(sse/x.size)),histogram_equal=True,all_threshold_budgets_equal=True,
                    target_used_at_inference=False,optimal=False)


def weighted_filter(values,kernel):
    """Zero-padded float64 sum, for preprojected scalar evidence fields only."""
    a=np.asarray(values,np.float64);require(a.ndim==2 and a.size>0 and np.isfinite(a).all(),'finite 2-D field')
    off=_kernel(kernel);h,w=a.shape;out=np.zeros(a.shape,np.float64)
    for dr,dc in off:
        r0,r1=max(0,-int(dr)),min(h,h-int(dr));c0,c1=max(0,-int(dc)),min(w,w-int(dc))
        if r1>r0 and c1>c0:out[r0:r1,c0:c1]+=a[r0+dr:r1+dr,c0+dc:c1+dc]
    return out


def field_predictions(image,pairs,thresholds,kernel,near_field,label_field,*,step=16):
    """TARGET-FREE all-candidate predictions without native-width token expansion.

    Requires the two endpoint neighborhoods WITHIN each action to be disjoint.
    For native radius5 and axis-aligned distance16 this holds. Between-action
    overlap is still handled by the greedy policy. Evidence fields are not
    probabilities. Projection to physical count ranges is not a safety bound.
    """
    x=_image(image);off=_kernel(kernel);pairs=np.asarray(pairs)
    h,v=np.asarray(near_field,np.float64),np.asarray(label_field,np.float64)
    require(h.shape==v.shape==x.shape and np.isfinite(h).all() and np.isfinite(v).all(),'field/image mismatch')
    require(isinstance(step,int) and not isinstance(step,bool) and step>2*int(np.abs(off).max()),'step must exceed footprint diameter')
    require(pairs.ndim==2 and pairs.shape[1]==2 and pairs.dtype.kind in 'iu' and ((pairs>=0)&(pairs<x.size)).all(),'invalid pair endpoints')
    require(len(thresholds)>0 and all(isinstance(t,(int,np.integer)) and not isinstance(t,(bool,np.bool_)) and 0<=t<=255 for t in thresholds),'invalid screened thresholds')
    p,q=pairs[:,0],pairs[:,1];dr=np.abs(p//x.shape[1]-q//x.shape[1]);dc=np.abs(p%x.shape[1]-q%x.shape[1])
    require((((dr==step)&(dc==0))|((dc==step)&(dr==0))).all(),'optimized action must be axis-aligned and exactly step apart')
    out=np.zeros((len(pairs),len(thresholds),3),np.float64)
    for j,t in enumerate(thresholds):
        s=x>int(t);c=filter_sum(s,kernel)
        uncovered=c==0;sole=c==1
        add_value=weighted_filter(v*uncovered,kernel).ravel();remove_value=weighted_filter(v*sole,kernel).ravel()
        add_count=filter_sum(uncovered,kernel).ravel();remove_count=filter_sum(sole,kernel).ravel()
        changed=s.ravel()[p]!=s.ravel()[q]
        add=np.where(s.ravel()[p],q,p);remove=np.where(s.ravel()[p],p,q)
        out[:,j,0]=np.where(changed,np.clip(h.ravel()[add]-h.ravel()[remove],-1,1),0)
        out[:,j,2]=np.where(changed,np.clip(v.ravel()[add]-v.ravel()[remove],-1,1),0)
        gain=add_value[add]-remove_value[remove]
        out[:,j,1]=np.where(changed,np.clip(gain,-remove_count[remove],add_count[add]),0)
    return out


def render_field_policy(image,pairs,thresholds,kernel,near_field,label_field,slopes,*,rms_limit=1.,step=16,reverse=False):
    """TARGET-FREE optimized implementation of the reference policy.

    No targets and no target-derived guard overrides are accepted. Predict all
    candidate deltas from scalar fields, then form influence supports lazily.
    This is a complete numerical renderer, not a private feature/scorer adapter.
    """
    x=_image(image);off=_kernel(kernel);pairs=np.asarray(pairs)
    pr=field_predictions(x,pairs,thresholds,kernel,near_field,label_field,step=step)
    require(len(set(map(tuple,pairs)))==len(pairs),'duplicate candidates')
    wu,wv=map(float,slopes)
    require(np.isfinite(rms_limit) and rms_limit>=0 and np.isfinite(wu) and np.isfinite(wv) and min(wu,wv)>=0,'invalid budget or utility weights')
    require(isinstance(reverse,(bool,np.bool_)),'boolean tie convention')
    a=x.ravel()[pairs[:,0]].astype(np.int64);b=x.ravel()[pairs[:,1]].astype(np.int64);costs=2*(a-b)**2
    score=wu*pr[:,0,0]+wv*pr[:,0,1];eligible=(pr>=0).all(axis=(1,2))&(score>0)&(costs>0)
    idx=np.arange(len(pairs));order=np.lexsort(((-idx if reverse else idx),-score))
    used=np.zeros(x.size,bool);out=x.copy();chosen=[];sse=0;budget=x.size*float(rms_limit)**2
    for ai in order:
        if not eligible[ai] or sse+costs[ai]>budget:continue
        p,q=map(int,pairs[ai]);support=np.union1d(_indices(x.shape,p,off),_indices(x.shape,q,off))
        if used[support].any():continue
        out.ravel()[p],out.ravel()[q]=out.ravel()[q],out.ravel()[p];used[support]=True
        sse+=int(costs[ai]);chosen.append(int(ai))
    actual=int(np.square(out.astype(np.int64)-x.astype(np.int64)).sum())
    require(actual==sse and np.array_equal(_tail(out),_tail(x)),'histogram or energy invariant failed')
    return out,dict(selected_indices=chosen,selected_count=len(chosen),candidate_count=len(pairs),sse=sse,
                    rms=float(np.sqrt(sse/x.size)),histogram_equal=True,all_threshold_budgets_equal=True,
                    target_used_at_inference=False,optimal=False)


def project_evidence_fields(coarse_features,fine_features,near_readout,label_readout,grid_shape):
    """Project [1, coarse_bin, fine_pixel] without tiling the full feature matrix.

    Inputs already carry TRAINING-frozen normalization. Native FM coarse width
    is 1551 (source tokens1536 then cheap15), fine width13; total width1565.
    Cheap-only width is 1+15+13=29. No normalization is fitted in this function.
    """
    coarse=np.asarray(coarse_features,np.float64);fine=np.asarray(fine_features,np.float64)
    require(coarse.ndim==2 and fine.ndim==3 and np.isfinite(coarse).all() and np.isfinite(fine).all(),'finite coarse/fine features')
    require(len(grid_shape)==2 and all(isinstance(z,(int,np.integer)) and z>0 for z in grid_shape),'positive grid')
    gh,gw=map(int,grid_shape);h,w=fine.shape[:2]
    require(h%gh==0 and w%gw==0 and len(coarse)==gh*gw,'feature grid does not tile image')
    d=1+coarse.shape[1]+fine.shape[2];a,b=np.asarray(near_readout,np.float64),np.asarray(label_readout,np.float64)
    require(a.shape==b.shape==(d,) and np.isfinite(a).all() and np.isfinite(b).all(),'readout width')
    fields=[]
    for beta in (a,b):
        low=(beta[0]+coarse@beta[1:1+coarse.shape[1]]).reshape(gh,gw)
        field=np.repeat(np.repeat(low,h//gh,axis=0),w//gw,axis=1)
        field+=fine@beta[1+coarse.shape[1]:];fields.append(field)
    return tuple(fields)


def compressed_design_rows(g,coarse_features,fine_features,grid_shape):
    """Sparse signed feature sums with coarse-bin coefficients accumulated first.

    Equivalent to design_rows on [1, tiled_coarse, fine] but avoids the native
    N-by-1536 expansion. Sums are float64; policy-boundary parity stays local.
    """
    coarse=np.asarray(coarse_features,np.float64);fine=np.asarray(fine_features,np.float64)
    require(coarse.ndim==2 and fine.ndim==3 and tuple(fine.shape[:2])==g.image_shape and np.isfinite(coarse).all() and np.isfinite(fine).all(),'coarse/fine geometry mismatch')
    require(len(grid_shape)==2 and all(isinstance(z,(int,np.integer)) and z>0 for z in grid_shape),'positive grid')
    gh,gw=map(int,grid_shape);h,w=g.image_shape
    require(h%gh==0 and w%gw==0 and len(coarse)==gh*gw,'feature grid does not tile image')
    def row(ids,delta):
        bins=(ids//w//(h//gh))*gw+(ids%w//(w//gw))
        unique,inverse=np.unique(bins,return_inverse=True);weights=np.bincount(inverse,weights=delta,minlength=len(unique))
        low=np.sum(coarse[unique]*weights[:,None],axis=0)
        high=np.sum(fine.reshape(h*w,-1)[ids]*delta[:,None],axis=0)
        return np.r_[float(delta.sum()),low,high]
    return row(g.point_ids,g.point_delta),row(g.coverage_ids,g.coverage_delta)


def unary_statistics(coarse_features,fine_features,target,kernel,grid_shape):
    """TRAINING ONLY exact full-pixel Brier sufficient matrices, no tiled tokens.

    Feature order is [1, coarse_bin, fine_pixel]. Returns pixel-averaged normal
    matrices and RHS for near-label and label readouts. This supplies the matched
    alternative, not the selected count-supervision objective.
    """
    coarse=np.asarray(coarse_features,np.float64);fine=np.asarray(fine_features,np.float64);y=np.asarray(target)
    require(coarse.ndim==2 and fine.ndim==3 and y.shape==fine.shape[:2] and y.dtype==np.bool_ and np.isfinite(coarse).all() and np.isfinite(fine).all(),'invalid unary training arrays')
    require(len(grid_shape)==2 and all(isinstance(z,(int,np.integer)) and z>0 for z in grid_shape),'positive grid')
    gh,gw=map(int,grid_shape);h,w=y.shape
    require(h%gh==0 and w%gw==0 and len(coarse)==gh*gw,'feature grid does not tile image')
    bh,bw=h//gh,w//gw;n=h*w
    C=np.c_[np.ones(len(coarse)),coarse];F=fine.reshape(n,-1)
    def bins(a):return a.reshape(gh,bh,gw,bw,-1).transpose(0,2,1,3,4).reshape(gh*gw,bh*bw,-1)
    fs=bins(fine).sum(1);Gcc=(bh*bw)*(C.T@C);Gcf=C.T@fs;Gff=F.T@F
    gram=np.block([[Gcc,Gcf],[Gcf.T,Gff]])/n
    near=filter_sum(y,kernel)>0
    rhs=[]
    for label in (near,y):
        rhs.append(np.r_[C.T@bins(label[...,None]).sum(1).ravel(),F.T@label.ravel()]/n)
    return gram,rhs[0],rhs[1]


def fit_unary_readouts(statistics,lam=1e-3):
    """TRAINING ONLY streaming equal-image full-label alternative.

    Objective mean[(h-D(Y))^2 + 2*(v-Y)^2] + lam*(||a||^2+||b||^2).
    The duplicated label term matches the selected loss's two uses of the Y
    readout, so its effective penalty is lam/2, not an accidental advantage from
    weaker regularization. Each statistic tuple must be pixel-averaged.
    """
    require(np.isfinite(lam) and lam>0,'positive ridge coefficient')
    total=None;na=nb=None;count=0
    for G,a,b in statistics:
        G,a,b=map(lambda z:np.asarray(z,np.float64),(G,a,b))
        require(G.ndim==2 and G.shape[0]==G.shape[1] and a.shape==b.shape==(len(G),) and np.isfinite(G).all() and np.isfinite(a).all() and np.isfinite(b).all(),'invalid unary statistics')
        if total is None:total=np.zeros_like(G);na=np.zeros_like(a);nb=np.zeros_like(b)
        require(total.shape==G.shape,'unary width changed')
        total+=G;na+=a;nb+=b;count+=1
    require(count>0,'empty unary training population')
    G=total/count;a=na/count;b=nb/count;eye=np.eye(len(G))
    aa=np.linalg.solve(G+lam*eye,a);bb=np.linalg.solve(G+(lam/2)*eye,b)
    require(np.isfinite(aa).all() and np.isfinite(bb).all(),'nonfinite unary readout')
    return aa,bb
