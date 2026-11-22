"""按用户网络中断后的明确指令延长剩余预算，保留原时间线。"""
from pathlib import Path
from datetime import datetime,timedelta
import json

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'reports/PROTOCOL.json';protocol=json.loads(path.read_text())
    original=dict(protocol);resume=datetime.fromisoformat('2026-11-12T09:36:53+08:00')
    deadline=resume+timedelta(hours=8)
    record=dict(reason='用户说明网络故障、此前实际运行约1小时，明确要求从恢复后再计8小时',
        previous_protocol=original,resumed_at=resume.isoformat(),remaining_budget_hours=8,
        deadline=deadline.isoformat(),note='保留原起点及已完成验收；中断时段单独记录，不宣称连续运行')
    amendment=HERE/'reports/BUDGET_AMENDMENT.json';assert not amendment.exists()
    amendment.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    protocol.update(deadline=deadline.isoformat(),final_validation_reserve=(deadline-timedelta(hours=1)).isoformat(),
        resumed_at=resume.isoformat(),phase='joint_inference',budget_amended_by_user=True)
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    print('USER_BUDGET_AMENDED',resume.isoformat(),deadline.isoformat())


if __name__=='__main__':main()
