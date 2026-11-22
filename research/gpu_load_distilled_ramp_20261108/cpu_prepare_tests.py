"""仅CPU：检查预算日程、阶段隔离和GPU验收门槛，不导入Torch。"""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phase_gate import validate_phase_intervals
from run_guard import implementation_identity, require_gpu_acceptance, PROTOCOL_ID

HERE=Path(__file__).resolve().parent


class PreparationTests(unittest.TestCase):
    def test_actual_audit_uses_sm_area_and_checks_ledger(self):
        tree=ast.parse((HERE/'experiment.py').read_text(encoding='utf-8'))
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                   and node.name in ('read_csv','audit')]
        namespace={'np':np,'csv':csv,'validate_phase_intervals':validate_phase_intervals,
                   'save':lambda path,value:path.write_text(json.dumps(value))}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'actual_audit','exec'),namespace)
        frames=[];packets=[];windows=[]
        for index in range(2000):
            generation=index+1;duty=.2+.6*index/1999;start=index*.002
            q=duty*1000000;charged=int(np.ceil(q));resident=charged+80000
            frames.append(dict(frame=generation,load_generation=generation,prepare_end_s=start-.001,
                render_phase_start_s=start,render_phase_end_s=start+.001))
            packets.append(dict(generation=generation,submit_monotonic_s=start+.00005,
                drained_monotonic_s=start+.00095,resident_ns=resident,charged_ns=charged,unique_sms=82,
                sm_equivalent_ns=q,slot_equivalent_ns=q))
            windows.append(dict(generation=generation,target_duty=duty,active_ns=1000000,
                                resident_ns=resident,charged_ns=charged))
        charge=sum(p['charged_ns'] for p in packets);area=sum(p['sm_equivalent_ns'] for p in packets)
        active=2000000000;tail=int(np.ceil(max(0.,charge-area)/.8));elapsed=active+tail
        summary=dict(charged_ns=charge,active_ns=active,tail_ns=tail,sm_equivalent_ns=area,
                     actual_sm_duty=area/elapsed,charged_duty=charge/elapsed,
                     weighted_target_duty=(area+.8*tail)/elapsed)
        with tempfile.TemporaryDirectory(prefix='ramp_audit_') as name:
            folder=Path(name)
            def write(filename,rows):
                with (folder/filename).open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            write('PULSES.csv',packets);write('PULSES.csv.windows.csv',windows)
            namespace['audit'](folder,frames,summary,dict(sm_count=82,billing_sm_area=True),True)
            accepted=json.loads((folder/'ACCEPTANCE.json').read_text())
            self.assertTrue(accepted['bin_match'])
            self.assertTrue(all(item['resident_envelope_duty']-item['weighted_target']>.07 for item in accepted['bins']))
            packets[0]['charged_ns']+=1;write('PULSES.csv',packets)
            with self.assertRaises(AssertionError):
                namespace['audit'](folder,frames,summary,dict(sm_count=82,billing_sm_area=True),True)

    def test_actual_schedule_and_method_budgets(self):
        tree=ast.parse((HERE/'experiment.py').read_text(encoding='utf-8'))
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                   and node.name in ('schedule','budget_for')]
        namespace={'np':np}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'actual_schedule','exec'),namespace)
        values=[namespace['schedule'](frame) for frame in range(2000)]
        duty=np.array([item[0] for item in values]); budget=np.array([item[1] for item in values])
        self.assertEqual(values[0],(.2,4000)); self.assertEqual(values[-1],(.8,1000))
        self.assertTrue(np.all(duty<=.8) and np.all(duty>=.2))
        np.testing.assert_allclose(np.diff(duty),.6/1999,rtol=0,atol=2e-16)
        self.assertTrue(np.all(np.diff(budget)<=0))
        self.assertTrue(set(np.diff(budget)).issubset({-1,-2}))
        for frame in (0,500,999,1500,1999):
            self.assertEqual(namespace['budget_for']('student_adaptive',frame),values[frame][1])
            self.assertEqual(namespace['budget_for']('student_fixed1000',frame),1000)
            self.assertEqual(namespace['budget_for']('student_fixed4000',frame),4000)

    def test_protected_phase_accepts_only_render_interval(self):
        row=dict(load_generation=1,prepare_end_s=2.,render_phase_start_s=3.,render_phase_end_s=6.)
        packet=dict(generation=1,submit_monotonic_s=3.5,drained_monotonic_s=5.5)
        self.assertEqual(validate_phase_intervals([row],[packet])['artificial_load_prepare_overlap'],0)
        for start,end in ((1.,4.),(2.5,4.),(3.5,6.1),(5.,4.)):
            with self.assertRaises(AssertionError):
                validate_phase_intervals([row],[dict(packet,submit_monotonic_s=start,drained_monotonic_s=end)])
        with self.assertRaises(AssertionError):
            validate_phase_intervals([row,row],[packet])

    def test_formal_rejects_missing_stale_and_incomplete_gpu_proof(self):
        with tempfile.TemporaryDirectory(prefix='ramp_guard_') as name:
            root=Path(name); (root/'reports').mkdir()
            identity=dict(protocol=PROTOCOL_ID,files={'binary':'current'})
            with self.assertRaises(RuntimeError): require_gpu_acceptance(root,identity)
            proof=dict(passed=True,identity=identity,cases=3,rows=6000,graph_cases=40)
            path=root/'reports/GPU_ACCEPTANCE.json'
            for changed in (dict(passed=False),dict(rows=4000),dict(graph_cases=39),
                            dict(identity=dict(protocol='old_envelope_billing',files={}))):
                path.write_text(json.dumps(dict(proof,**changed)))
                with self.assertRaises(RuntimeError): require_gpu_acceptance(root,identity)
            path.write_text(json.dumps(proof))
            self.assertEqual(require_gpu_acceptance(root,identity),proof)

    def test_implementation_hash_catches_runtime_and_binary_changes(self):
        with tempfile.TemporaryDirectory(prefix='ramp_identity_') as name:
            root=Path(name)
            task=root/'source/research/gpu_load_distilled_ramp_20261108'
            names=['experiment.py','phase_load.py','ramp_load.cu','sm_area.h','run_guard.py',
                   'reference.py','previous_experiment.py','phase_gate.py','duty_budget.h']
            paths=[task/name for name in names]+[root/name for name in (
                'build/libramp_load.so','build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so',
                'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so',
                'reports/PROTECTED_BEFORE.json','source/original_renderer.py','source/python/deployment_runtime.py')]
            for path in paths:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('initial')
            initial=implementation_identity(root)
            self.assertEqual(initial,implementation_identity(root))
            runtime=root/'source/python/deployment_runtime.py';runtime.write_text('changed')
            self.assertNotEqual(initial,implementation_identity(root))
            runtime.write_text('initial')
            binary=root/'build/libramp_load.so';binary.write_text('changed')
            self.assertNotEqual(initial,implementation_identity(root))


if __name__=='__main__':
    unittest.main(verbosity=2)
