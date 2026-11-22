"""固定阈值配置接线和真实C++判定片段的边界测试。"""
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
import unittest
import patch_growth as p
import prepare_experiment as prep

SOURCE=Path(__file__).resolve().parents[2]


class FixedGrowthTests(unittest.TestCase):
    def files(self):return {name:(SOURCE/name).read_text(encoding='utf-8') for name in p.FILES}

    def fixtures(self):
        return [dict(id=f'Replica/{s}/{m}/prune1/seed0_repeat1',dataset='Replica',scene=s,mode=m,
            prune=1,seed=0,repeat=1,frames=2000,environment={},
            command=['binary','vocabulary','camera','mapping','data','output','no_viewer','--selector-config','config'])
            for s in prep.SCENES for m in ['mono','rgbd']]

    def test_six_expected_files_only(self):
        files=self.files();after=p.patch_files(files)
        self.assertEqual(set(files),set(after));self.assertTrue(all(files[x]!=after[x] for x in files))

    def test_old_default_and_manual_propagation(self):
        after=p.patch_files(self.files())
        self.assertIn('anchor_growth_alpha_ = 2.0;',after[p.MODEL])
        self.assertIn('anchor_growth_manual_ = false;',after[p.MODEL])
        self.assertEqual(after[p.CPP].count('if(anchor_growth_threshold_==0)'),2)
        self.assertEqual(after[p.CPP].count('anchor_growth_threshold_,anchor_growth_alpha_,'),2)
        self.assertIn('anchor_refine_requires_full_,anchor_growth_manual_)',after[p.CPP])
        self.assertIn('false,anchor_growth_manual_)',after[p.CPP])
        self.assertIn('gaussians_->setAnchorGrowthPolicy(',after[p.MAPPER_CPP])

    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):p.patch_files(p.patch_files(self.files()))

    def test_reject_unknown_source(self):
        files=self.files();files[p.GROWTH]='unknown'
        with self.assertRaises(ValueError):p.patch_files(files)

    def test_temporary_copy_and_receipt(self):
        original={name:(SOURCE/name).read_bytes() for name in p.FILES}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,value in original.items():
                dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(value)
            result=p.apply(root/'source',root/'reports')
            self.assertEqual(result['manual_threshold'],.001);self.assertEqual(result['alpha'],1.2)
            self.assertTrue(result['strict_upper'])
        self.assertEqual(original,{name:(SOURCE/name).read_bytes() for name in p.FILES})

    def test_sixteen_new_runs(self):
        jobs=prep.planned_jobs(self.fixtures());self.assertEqual(len(jobs),16)
        for job in jobs:
            self.assertEqual(job['state'],'held');self.assertNotIn('result',job)
            self.assertTrue(Path(job['output']).is_relative_to(prep.R))
            self.assertEqual(job['environment']['CARTGS_BATCH_FULL_TAIL_ONLY'],'1')

    def test_reject_missing_or_wrong_seed(self):
        with self.assertRaises(ValueError):prep.planned_jobs(self.fixtures()[:-1])
        jobs=self.fixtures();jobs[0]['seed']=1
        with self.assertRaises(ValueError):prep.planned_jobs(jobs)

    def test_cpp_exact_branch_boundaries_and_legacy(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('g++ unavailable')
        files=self.files();before=files[p.GROWTH];after=p.patch_files(files)[p.GROWTH]
        def extract(text):
            start=text.index('    double low=threshold*std::pow(alpha,level), high=low*alpha;')
            return text[start:text.index('    double h=voxel_size_',start)].replace('continue;','return -1;')
        def function(name,body):
            return ('int '+name+'(int64_t level,double value,double threshold,double alpha,bool strict_upper=false) {'
                    'double g[]={value};int i=0;constexpr int kMaxLevel=2;int limited=0;do {'+body+
                    '} while(false);return level;}\n')
        code='#include <cmath>\n#include <cstdint>\n#include <cassert>\n#include <limits>\n#include <initializer_list>\n'
        code+=function('legacy',extract(before))+function('candidate',extract(after))
        code+='''int main() {
          constexpr double t=.001,alpha=1.2;
          for(int64_t level=0;level<=2;++level) {
            double low=t*std::pow(alpha,level),high=low*alpha;
            assert(candidate(level,std::nextafter(low,0.),t,alpha,true)==-1);
            assert(candidate(level,low,t,alpha,true)==level);
            assert(candidate(level,high,t,alpha,true)==level);
            assert(candidate(level,std::nextafter(high,INFINITY),t,alpha,true)==(level==2?-1:level+1));
            for(double a:{1.2,2.}) for(double value:{0.,.0001,.0005,.001,.0012,.00144,.001728,.002,.004,.008,.1})
              assert(candidate(level,value,t,a,false)==legacy(level,value,t,a));
          }
        }'''
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cpp=root/'boundary.cpp';exe=root/'boundary.exe';cpp.write_text(code)
            subprocess.run([compiler,'-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True,capture_output=True)
            subprocess.run([str(exe)],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
