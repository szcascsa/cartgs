"""按实际Tile成员量复用已有前256成员；仅剩余成员沿原位表深度顺序遍历。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_overflow_prefix';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_overflow_prefix').mkdir()
    for name in ['encoded_sign_pixels.cuh','encoded_sign_diagnosed.cuh']:
        path=root/name;text=path.read_text()
        text=text.replace('  const int count=counts[tile];\n  if(count<=capacity) {',
            '  const int total=counts[tile],count=min(total,capacity);\n  {')
        anchor='  } else {\n    for(int group=0;group<groups;++group){'
        assert text.count(anchor)==1
        text=text.replace(anchor,'  }\n  if(total>capacity){\n    int enumerated=0;\n    for(int group=0;group<groups;++group){')
        anchor='members&=members-1;const int id=sorted[rank];'
        assert text.count(anchor)==1
        text=text.replace(anchor,'members&=members-1;if(enumerated++<capacity)continue;const int id=sorted[rank];')
        if name.endswith('diagnosed.cuh'):
            lines=[]
            for line in text.splitlines():
                if 'debugOriginalPair(' in line:
                    line='        done[0]=T[0]<0.f;done[1]=T[1]<0.f;T[0]=fabsf(T[0]);T[1]=fabsf(T[1]);\n'+line+'\n        if(done[0])T[0]=-T[0];if(done[1])T[1]=-T[1];'
                lines.append(line)
            text='\n'.join(lines)
        path.write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('StudentEncodedSignRasterizer','StudentOverflowPrefixRasterizer');path.write_text(text)
    print('OVERFLOW_PREFIX_SOURCE_READY')


if __name__=='__main__':main()
