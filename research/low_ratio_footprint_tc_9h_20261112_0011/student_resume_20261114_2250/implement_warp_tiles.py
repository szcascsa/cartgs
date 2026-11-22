"""warp独立成员缓存与多像素二次增量，按实际Tile处理、不按ratio分路。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_warp';shutil.copytree(HERE.parent/'source',root);(HERE/'build_warp').mkdir()
    shutil.copy2(HERE/'warp_tile_pixels.cuh',root/'warp_tile_pixels.cuh')
    path=root/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "warp_tile_pixels.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=94')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,(pixels,stop) in enumerate([(2,False),(2,True),(4,False),(4,True),(8,False),(8,True)],71):
        kind=38 if stop else 37;blocks=f'({256//pixels}*tiles+127)/128'
        lines.append(f'      if(pixel_mode_=={mode+18})renderWarpTilePixels<{kind},{pixels}><<<{blocks},128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);')
        diagnosed.append(f'    if(pixel_mode_=={mode+18})renderWarpTilePixels<{kind},{pixels},true><<<{blocks},128,0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,ko,inputs,reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),image.data_ptr<float>());')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines))
    text=text.replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=94')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentWarpTileRasterizer')
    path.write_text(text)
    print('STUDENT_WARP_TILES_SOURCE_READY')


if __name__=='__main__':main()
