import hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path('/home/prmi/zwj/grid-capacity-audit-20261018');MAIN=Path('/home/prmi/zwj/mycartgs')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def archive(path,values):
    m=torch.nn.Identity()
    for k,v in values.items():m.register_buffer(k,v.cpu())
    torch.jit.trace(m,torch.zeros(1)).save(str(path))
def scene(name,checkpoint,dataset):
    out=ROOT/'inputs'/name;out.mkdir(parents=True,exist_ok=True)
    cams=sorted(json.loads((checkpoint/'cameras.json').read_text()),key=lambda c:c['id'])
    stride=4 if len(cams)//4>=8 else 3
    held=[i for i in range(len(cams))if(i+1)%stride==0];train=[i for i in range(len(cams))if i not in held]
    assert len(held)>=8 and train
    values={};images=[];sources={};world=[];projection=[];center=[];params=[]
    for c in cams:
        p=dataset/c['img_name'];image=cv2.imread(str(p));assert image is not None,str(p)
        # fr3 uses zero distortion. Secondary fr1 uses the checkpoint's existing eval loader convention.
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w=image.shape[:2];assert(h,w)==(c['height'],c['width'])
        images.append(torch.from_numpy(image).permute(2,0,1).float()/255)
        twc=torch.eye(4);twc[:3,:3]=torch.tensor(c['rotation']);twc[:3,3]=torch.tensor(c['position'])
        view=torch.linalg.inv(twc).T.contiguous();fx=np.float32(c['fx']);fy=np.float32(c['fy'])
        fovx=float(np.float32(2)*np.arctan(np.float32(w)/(np.float32(2)*fx)));fovy=float(np.float32(2)*np.arctan(np.float32(h)/(np.float32(2)*fy)))
        proj=torch.zeros(4,4);proj[0,0]=1/math.tan(fovx/2);proj[1,1]=1/math.tan(fovy/2);proj[3,2]=1;proj[2,2]=100/(100-.01);proj[2,3]=-(100*.01)/(100-.01)
        world.append(view);projection.append(view@proj.T);center.append(torch.linalg.inv(view)[3,:3]);params.append([h,w,fovx,fovy])
        sources[str(p)]=sha(p)
    archive(out/'cameras.pt',dict(gt=torch.stack(images),world=torch.stack(world),projection=torch.stack(projection),center=torch.stack(center),params=torch.tensor(params)))
    ply=sorted((checkpoint/'point_cloud').glob('*/point_cloud.ply'));assert len(ply)==1
    metadata=ply[0].parent/'selector_metadata.ply';assert metadata.exists()
    files=[ply[0],metadata,checkpoint/'selector.pt',checkpoint/'transform_field.pt',checkpoint/'selector_config.yaml',checkpoint/'cameras.json',checkpoint/'view_bank_orb.json']
    info=dict(name=name,checkpoint=str(checkpoint),master=str(ply[0]),metadata=str(metadata),selector=str(checkpoint/'selector.pt'),tf=str(checkpoint/'transform_field.pt'),config=str(checkpoint/'selector_config.yaml'),
              cameras=str(out/'cameras.pt'),dataset=str(dataset),train=train,heldout=held,stride=stride,camera_ids=[c['id']for c in cams],source_hashes={str(p):sha(p)for p in files},image_hashes=sources)
    save(out/'manifest.json',info);return info
if __name__=='__main__':
    protocol=json.loads((ROOT/'protocol.json').read_text())
    dataset=MAIN/'resultsall/batch_20261016_nolog/data/TUM/fr3_long_office_household'
    info=scene('fr3_long_office_household',Path(protocol['checkpoint']),dataset)
    baseline=json.loads((ROOT/'grid_capacity_source_manifest.json').read_text());checks={}
    for n,v in baseline['production_files'].items():
        p=MAIN/n
        if p.exists():checks[n]=dict(server_sha256=sha(p),lf_match=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==v['lf_sha256'])
    save(ROOT/'server_source_comparison.json',checks)
    required=['improvements/selector/gumbel_network.h','src/gaussian_model.cpp','src/gaussian_renderer.cpp','src/gaussian_rasterizer.cpp','src/transform_field.cpp','include/transform_field.h','src/hexplane_field.cpp','include/selector_handoff.h']
    assert all(checks[n]['lf_match']for n in required),{n:checks[n]for n in required if not checks[n]['lf_match']}
    print(json.dumps(dict(scene=info['name'],cameras=len(info['camera_ids']),train=len(info['train']),heldout=len(info['heldout']),required_source_match=True)),flush=True)
    cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
    commands=[[cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-G','Ninja','-DCMAKE_MAKE_PROGRAM=/home/prmi/.local/bin/ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCARTGS_ROOT='+str(MAIN),
      '-DTorch_DIR='+str(MAIN/'third_party/libtorch-cu118/libtorch/share/cmake/Torch'),'-DEigen3_DIR='+str(MAIN/'third_party/install/eigen-3.3.7/share/eigen3/cmake'),'-DOpenCV_DIR='+str(MAIN/'third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4'),'-Djsoncpp_DIR='+str(MAIN/'third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp')],
      [cmake,'--build',str(ROOT/'build'),'-j1']]
    status={}
    for i,cmd in enumerate(commands):
        start=time.time()
        with(ROOT/f'build_{i}.log').open('w')as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        status[str(i)]=dict(exit=rc,seconds=time.time()-start);save(ROOT/'build_status.json',status);assert rc==0,i
    status['complete']=True;save(ROOT/'build_status.json',status)
