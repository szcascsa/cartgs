"""公开项目/论文直接检索，只存标题、片段和内容哈希。"""
from pathlib import Path
from datetime import datetime
import concurrent.futures
import hashlib
import html
import json
import re
import urllib.request

HERE=Path(__file__).resolve().parent
SOURCES=[('TC-GS','https://arxiv.org/html/2505.24796v1',['tensor core','polynomial','precision']),
    ('Speedy-Splat','https://speedysplat.github.io/',['SnugBox','AccuTile','pixel']),
    ('FlashGS','https://maxwellf1.github.io/flashgs_page/',['pipeline','memory','redundan'])]


def read(spec):
    name,url,terms=spec
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url,timeout=25) as response:raw=response.read(1500000)
    content=raw.decode('utf-8',errors='replace');title=re.search(r'<title>(.*?)</title>',content,re.S)
    text=html.unescape(re.sub(r'<[^>]+>',' ',content));text=re.sub(r'\s+',' ',text)
    snippets={}
    for term in terms:
        matches=list(re.finditer(re.escape(term),text,re.I))[:3]
        snippets[term]=[text[max(0,m.start()-130):m.end()+220] for m in matches]
    return dict(name=name,url=url,title=html.unescape(title.group(1)) if title else '',
        retrieved_at=datetime.now().astimezone().isoformat(),sha256=hashlib.sha256(raw).hexdigest(),snippets=snippets)


def main():
    records=[]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        pending={pool.submit(read,spec):spec for spec in SOURCES}
        for future in concurrent.futures.as_completed(pending):
            spec=pending[future]
            try:records.append(future.result())
            except Exception as error:records.append(dict(name=spec[0],url=spec[1],error=repr(error),title='本次重读失败；此前已核验标题'))
    (HERE/'reports/LITERATURE_REFRESH.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    print('LITERATURE_REFRESH_SAVED',[(r['name'],r['title']) for r in records])


if __name__=='__main__':main()
