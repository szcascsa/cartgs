"""本地验证补丁范围及从实际判定代码提取的标量边界；不代替原生Anchor验收。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import patch_growth as p
import prepare_experiment as prep

SOURCE = Path(__file__).resolve().parents[2]


class GrowthPatchTests(unittest.TestCase):
    def fixtures(self):
        return [dict(id=f'Replica/{s}/{m}/prune1/seed0_repeat1',dataset='Replica',scene=s,mode=m,
                     prune=1,seed=0,repeat=1,frames=2000,environment={},
                     command=['binary','vocabulary','camera','mapping','data','output','no_viewer','--selector-config','config'])
                for s in prep.SCENES for m in ['mono','rgbd']]

    def test_plan_has16_new_full_runs(self):
        jobs=prep.planned_jobs(self.fixtures())
        self.assertEqual(len(jobs),16)
        self.assertEqual(len({j['id'] for j in jobs}),16)
        for job in jobs:
            self.assertEqual(job['environment']['CARTGS_BATCH_FULL_TAIL_ONLY'],'1')
            self.assertEqual(job['state'],'held')
            self.assertTrue(Path(job['output']).is_relative_to(prep.R))
            self.assertNotIn('result',job)

    def test_plan_rejects_missing_scene(self):
        with self.assertRaises(ValueError):prep.planned_jobs(self.fixtures()[:-1])

    def test_plan_rejects_wrong_seed(self):
        jobs=self.fixtures();jobs[0]['seed']=1
        with self.assertRaises(ValueError):prep.planned_jobs(jobs)

    def test_only_requested_threshold_added(self):
        text=(SOURCE/p.HEADER).read_text(encoding='utf-8')
        changed=p.patch(text)
        self.assertEqual(changed.replace(p.AFTER,p.BEFORE),text)

    def test_reject_duplicate_patch(self):
        with self.assertRaises(ValueError):p.patch(p.AFTER)

    def test_reject_unknown_implementation(self):
        with self.assertRaises(ValueError):p.patch('unrelated source')

    def test_apply_only_in_temporary_snapshot(self):
        original=(SOURCE/p.HEADER).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);dest=r/'source'/p.HEADER;dest.parent.mkdir(parents=True)
            dest.write_bytes(original);report=p.apply(r/'source',r/'reports')
            self.assertEqual(report['new_l1_to_l2_tau0_multiplier'],3)
            self.assertIn(p.AFTER,dest.read_text(encoding='utf-8'))
        self.assertEqual((SOURCE/p.HEADER).read_bytes(),original)

    def test_native_scalar_branch_boundaries_from_actual_source(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('g++ unavailable; native scalar test not executed')
        text=(SOURCE/p.HEADER).read_text(encoding='utf-8')
        start=text.index(p.BEFORE)
        end=text.index('    double h=voxel_size_',start)
        actual=text[start:end]
        def function(name,body):
            return ('int '+name+'(int64_t level,double value,double threshold=1.) { '
                    'double alpha=2.; double g[]={value}; int i=0; constexpr int kMaxLevel=2; '
                    'int limited=0; do { '+body+' } while(false); return level; }\n')
        # continue only skips this member; return -1 makes the fixture outcome explicit.
        old=actual.replace('continue;','return -1;')
        new=p.patch(actual).replace('continue;','return -1;')
        code='#include <cmath>\n#include <cstdint>\n#include <cassert>\n'+function('legacy',old)+function('candidate',new)
        code+='''int main() {
          for(double tau:{1.,0.0002}) {
            assert(legacy(1,3.*tau,tau)==1); assert(candidate(1,3.*tau,tau)==2);
            assert(legacy(1,3.5*tau,tau)==1); assert(candidate(1,3.5*tau,tau)==2);
            assert(candidate(1,std::nextafter(3.*tau,0.),tau)==1);
            assert(candidate(1,2.*tau,tau)==1); assert(candidate(1,1.9*tau,tau)==-1);
            for(double v:{0.,.5,1.,1.5,2.,3.,3.9,4.,5.,8.,9.}) {
              assert(legacy(0,v*tau,tau)==candidate(0,v*tau,tau));
              assert(legacy(2,v*tau,tau)==candidate(2,v*tau,tau));
            }
            assert(candidate(1,4.*tau,tau)==2);
          }
        }'''
        code=code.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>')
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);cpp=r/'threshold.cpp';exe=r/'threshold_test.exe';cpp.write_text(code)
            subprocess.run([compiler,'-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
            subprocess.run([str(exe)],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
