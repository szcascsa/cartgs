"""FP32严格运算顺序的输入打包、FiLM和输出变换；全部提供VJP。"""
import torch
import triton
import triton.language as tl
import triton.language.math as libdevice


@triton.jit
def _pack(Own,Coords,Ids,Freq,Out,K:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B);row=t//55;col=t%55;valid=row<K
    idx=tl.load(Ids+row,valid,other=0)
    value=tl.load(Own+idx*19+col,valid&(col<19),other=0)
    pos=(col-19)%18;axis=pos//6;band=pos%6
    coord=tl.load(Coords+idx*3+axis,valid&(col>=19),other=0)
    freq=tl.load(Freq+band,valid&(col>=19),other=0)
    angle=coord*freq
    value=tl.where(col<19,value,tl.where(col<37,libdevice.sin(angle),libdevice.cos(angle)))
    tl.store(Out+t,value,valid)


@triton.jit
def _pack_backward(G,Coords,Ids,Freq,OwnGrad,CoordGrad,K:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B);row=t//19;col=t%19;valid=row<K
    idx=tl.load(Ids+row,valid,other=0);g=tl.load(G+row*55+col,valid,other=0)
    tl.atomic_add(OwnGrad+idx*19+col,g,valid,sem='relaxed')
    # 每个(row,axis)独占一项；6个频带按PyTorch的末轴归约形成逐点坐标梯度。
    rr=tl.program_id(0)*B+tl.arange(0,B);r=rr//3;a=rr%3;ok=r<K
    ii=tl.load(Ids+r,ok,other=0);coord=tl.load(Coords+ii*3+a,ok,other=0)
    total=tl.full((B,),0,tl.float32)
    for j in tl.static_range(6):
        freq=tl.load(Freq+j);angle=coord*freq
        gs=tl.load(G+r*55+19+a*6+j,ok,other=0)
        gc=tl.load(G+r*55+37+a*6+j,ok,other=0)
        total+=(gs*libdevice.cos(angle)+gc*(-libdevice.sin(angle)))*freq
    tl.atomic_add(CoordGrad+ii*3+a,total,ok,sem='relaxed')


class Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx,own,coords,ids,freq):
        out=torch.empty((len(ids),55),device=own.device,dtype=own.dtype)
        if len(ids):_pack[(triton.cdiv(len(ids)*55,256),)](own,coords,ids,freq,out,len(ids),256,enable_fp_fusion=False)
        ctx.save_for_backward(coords,ids,freq);ctx.own_shape=own.shape
        return out
    @staticmethod
    def backward(ctx,grad):
        coords,ids,freq=ctx.saved_tensors;go=torch.zeros(ctx.own_shape,device=grad.device,dtype=grad.dtype);gc=torch.zeros_like(coords)
        if len(ids):_pack_backward[(triton.cdiv(len(ids)*19,256),)](grad.contiguous(),coords,ids,freq,go,gc,len(ids),256,enable_fp_fusion=False)
        return go,gc,None,None


@triton.jit
def _film(H,G,Bias,Out,N:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B);ok=t<N;c=t%W
    h=tl.load(H+t,ok,other=0);g=tl.load(G+c);b=tl.load(Bias+c)
    v=(1.+g)*h;v=v+b
    tl.store(Out+t,tl.maximum(v,0.),ok)


@triton.jit
def _film_backward(D,H,G,Out,DH,DG,DB,N:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B);ok=t<N;c=t%W
    d=tl.load(D+t,ok,other=0);h=tl.load(H+t,ok,other=0);g=tl.load(G+c);out=tl.load(Out+t,ok,other=0)
    d=tl.where(out>0,d,0.);tl.store(DH+t,d*(1.+g),ok);tl.store(DG+t,d*h,ok);tl.store(DB+t,d,ok)


class Film(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,g,b):
        out=torch.empty_like(h)
        if h.numel():_film[(triton.cdiv(h.numel(),256),)](h,g,b,out,h.numel(),h.shape[1],256,enable_fp_fusion=False)
        ctx.save_for_backward(h,g,out)
        return out
    @staticmethod
    def backward(ctx,d):
        h,g,out=ctx.saved_tensors;dh=torch.empty_like(h);dg=torch.empty_like(h);db=torch.empty_like(h)
        if h.numel():_film_backward[(triton.cdiv(h.numel(),256),)](d.contiguous(),h,g,out,dh,dg,db,h.numel(),h.shape[1],256,enable_fp_fusion=False)
        return dh,dg.sum(0,keepdim=True),db.sum(0,keepdim=True)


@triton.jit
def _output(X,S,R,A,Scale,Out,K:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B);row=t//14;c=t%14;ok=row<K
    x=tl.load(X+row*3+c,ok&(c<3),other=0);scale=tl.load(Scale+row,ok,other=0)
    s=tl.load(S+row*3+c-3,ok&(c>=3)&(c<6),other=0)
    q=tl.load(R+row*4+c-6,ok&(c>=6)&(c<10),other=0)
    a=tl.load(A+row*4+c-10,ok&(c>=10),other=0)
    value=tl.where(c<3,x*scale,tl.where(c<6,s,tl.where(c<10,q,a)))
    tl.store(Out+t,value,ok)


class Output(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,s,q,a,scale):
        out=torch.empty((len(x),14),device=x.device,dtype=x.dtype)
        if len(x):_output[(triton.cdiv(len(x)*14,256),)](x,s,q,a,scale,out,len(x),256,enable_fp_fusion=False)
        ctx.save_for_backward(x,scale)
        return out
    @staticmethod
    def backward(ctx,grad):
        x,scale=ctx.saved_tensors
        return grad[:,:3]*scale,grad[:,3:6],grad[:,6:10],grad[:,10:14],(grad[:,:3]*x).sum(1,keepdim=True)
