// 按只读原二进制duplicateWithKeys的SASS固定FP32舍入与FMA方向。
// 普通C++表达式在不同调用上下文会选择不同的FMA融合方向，改变极薄Gaussian的Tile成员。
__device__ __forceinline__ float orderedTilePower(float2 mean,float4 co,int x,int y) {
  const float xmin=float(x*16),xmax=float((x+1)*16-1),ymin=float(y*16),ymax=float((y+1)*16-1);
  const float xleft=xmin>mean.x,yabove=ymin>mean.y;
  const float outside_x=__fadd_rn(xleft,float(mean.x>xmax)),outside_y=__fadd_rn(yabove,float(mean.y>ymax));
  if(__fadd_rn(outside_x,outside_y)<=0.f)return 0.f;
  const float px=__fmaf_rn(xmin,xleft,__fmul_rn(__fadd_rn(1.f,-xleft),xmax));
  const float py=__fmaf_rn(ymin,yabove,__fmul_rn(__fadd_rn(1.f,-yabove),ymax));
  const float dx=copysignf(15.f,__fadd_rn(xmin,-mean.x)),dy=copysignf(15.f,__fadd_rn(ymin,-mean.y));
  const float diffx=__fadd_rn(mean.x,-px),diffy=__fadd_rn(mean.y,-py);
  const float ax=__fmul_rn(co.x,dx),bx=__fmul_rn(co.y,dx),by=__fmul_rn(co.y,dy),cy=__fmul_rn(co.z,dy);
  const float tx_numerator=__fmaf_rn(diffy,bx,__fmul_rn(diffx,ax));
  const float ty_numerator=__fmaf_rn(by,diffx,__fmul_rn(diffy,cy));
  const float tx=__fmul_rn(outside_y,__saturatef(__fmul_rn(tx_numerator,__frcp_rn(__fmul_rn(225.f,co.x)))));
  const float ty=__fmul_rn(outside_x,__saturatef(__fmul_rn(ty_numerator,__frcp_rn(__fmul_rn(225.f,co.z)))));
  const float maxx=__fmaf_rn(dx,tx,px),maxy=__fmaf_rn(dy,ty,py);
  const float ex=__fadd_rn(mean.x,-maxx),ey=__fadd_rn(mean.y,-maxy);
  const float yy=__fmul_rn(ey,__fmul_rn(co.z,ey));
  const float sum=__fmaf_rn(ex,__fmul_rn(co.x,ex),yy);
  return __fmaf_rn(ey,__fmul_rn(co.y,ex),__fmul_rn(sum,.5f));
}
