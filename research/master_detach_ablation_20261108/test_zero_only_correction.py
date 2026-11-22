import copy
import unittest
from correct_zero_only_20261111 import correct,MONO,STEREO


class CorrectionTests(unittest.TestCase):
    def state(self):
        return dict(phase='held',parallel_workers={},protocol={'unchanged':True},jobs=[
            dict(id='complete',state='complete',trained_model='preserved',evaluations=['preserved']),
            dict(id=MONO,state='skipped_health',health_failures=25,attempt_history=[
                dict(health=dict(error='view_budget_compensation_xyz_tail')) for _ in range(25)]),
            dict(id=STEREO,state='retry_pending',health_failures=20,attempt_history=[{} for _ in range(20)])])

    def test_replace_only_invalid_xyz_stops_and_preserve_history(self):
        state=self.state();old=copy.deepcopy(state)
        self.assertEqual(correct(state),[21,22,23,24,25])
        self.assertEqual(state['jobs'][0],old['jobs'][0]);self.assertEqual(state['jobs'][2],old['jobs'][2])
        self.assertEqual(state['jobs'][1]['attempt_history'],old['jobs'][1]['attempt_history'])
        self.assertEqual(state['jobs'][1]['health_failures'],25)
        self.assertEqual(state['jobs'][1]['max_training_attempts'],30)
        self.assertEqual(state['protocol'],old['protocol'])

    def test_successful_training_is_not_requeued(self):
        state=self.state();state['jobs'][1]['state']='trained';old=copy.deepcopy(state['jobs'][1])
        self.assertEqual(correct(state),[]);self.assertEqual(state['jobs'][1],old)

    def test_zero_selection_failure_is_not_replaced(self):
        state=self.state();state['jobs'][1]['attempt_history'][-1]['health']['error']='consecutive_zero_selection_with_candidates'
        with self.assertRaises(AssertionError):correct(state)

    def test_live_workers_cannot_be_overwritten(self):
        state=self.state();state['parallel_workers']={'train':{}}
        with self.assertRaises(AssertionError):correct(state)


if __name__=='__main__':unittest.main()
