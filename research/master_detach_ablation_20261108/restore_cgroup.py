"""仅恢复本批29GiB memory cgroup；由sudo执行，不加入或终止任何进程。"""
from pathlib import Path
import json
import os
import pwd

R=Path('/home/prmi/zwj/realLOD/ablation')
config=json.loads((R/'configs/corridors_stereo_memory.json').read_text())
group=Path(config['cgroup'])
assert os.geteuid()==0
assert str(group)=='/sys/fs/cgroup/memory/cartgs-corridors-stereo-20261108'
assert config['limit_bytes']==29*2**30 and config['system_reserve_bytes']==2*2**30
assert not group.is_symlink()
group.mkdir(exist_ok=True)
assert not (group/'cgroup.procs').read_text().strip(), '内存组非空，禁止改动'
limit=str(config['limit_bytes'])
(group/'memory.limit_in_bytes').write_text(limit)
(group/'memory.memsw.limit_in_bytes').write_text(limit)
assert int((group/'memory.limit_in_bytes').read_text())==int(limit)
assert int((group/'memory.memsw.limit_in_bytes').read_text())==int(limit)
control=dict(line.split() for line in (group/'memory.oom_control').read_text().splitlines())
assert control['oom_kill_disable']=='0'
user=pwd.getpwnam('prmi')
for name in ['cgroup.procs','tasks']:
    os.chown(group/name,user.pw_uid,user.pw_gid)
    os.chmod(group/name,0o644)
print(json.dumps(dict(restored=True,cgroup=str(group),limit_bytes=int(limit),empty=True)))
