// 独立SVG示意：原始Full渲染作为嵌入底图，Anchor尺寸和位置仅用于概念表达。
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const background=fs.readFileSync(path.join(root,'fr1_desk_full_render.png')).toString('base64');
const origin=[36,447],u=[154,-25],v=[90,-72],w=[0,-130];
const project=(x,y,z)=>[origin[0]+x*u[0]+y*v[0]+z*w[0],origin[1]+x*u[1]+y*v[1]+z*w[1]];
const segments=[];
const edge=(a,b,back=false)=>segments.push({a:project(...a),b:project(...b),back});
// 三列、单层纵深、两层高度，总共六个大尺寸立方体；共边只绘制一次。
for(let y=0;y<=1;y++)for(let z=0;z<=2;z++)edge([0,y,z],[3,y,z],y===1);
for(let x=0;x<=3;x++)for(let z=0;z<=2;z++)edge([x,0,z],[x,1,z],z===0);
for(let x=0;x<=3;x++)for(let y=0;y<=1;y++)edge([x,y,0],[x,y,2],y===1);
const lines=segments.sort((a,b)=>Number(b.back)-Number(a.back)).map(({a,b,back})=>
  `<path d="M${a.join(' ')} L${b.join(' ')}" fill="none" stroke="${back?'#577da7':'#164f92'}" stroke-width="${back?1.65:2.15}" opacity="${back?.54:.94}"/>`).join('\n');
const svg=`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1920" height="1440" viewBox="0 0 640 480">
<title>TUM fr1_desk：六个大尺寸Anchor₀概念线框</title>
<desc>底图是真实Full渲染。蓝色Anchor₀为非真实尺寸的示意，不包含高斯椭球或点标记。</desc>
<image x="0" y="0" width="640" height="480" xlink:href="data:image/png;base64,${background}"/>
<g stroke-linecap="round" stroke-linejoin="round">${lines}</g>
<g font-family="Arial, sans-serif" fill="#164f92">
  <path d="M91 63 L111 83 L126 115" stroke="#164f92" stroke-width="1.55" fill="none"/>
  <text x="27" y="59" font-size="22" font-weight="600" stroke="white" stroke-width="3.5" paint-order="stroke">Anchor<tspan baseline-shift="sub" font-size="14">0</tspan></text>
</g>
</svg>`;
fs.writeFileSync(path.join(root,'fr1_desk_anchor0.svg'),svg);
fs.writeFileSync(path.join(root,'preview.html'),`<!doctype html><meta charset="utf-8"><style>html,body{margin:0;width:1920px;height:1440px;overflow:hidden;background:white}svg{display:block;width:1920px;height:1440px}</style>${svg}`);
console.log('SVG_READY',path.join(root,'fr1_desk_anchor0.svg'));
