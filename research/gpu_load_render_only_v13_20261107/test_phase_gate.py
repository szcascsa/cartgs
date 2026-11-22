import unittest

from phase_gate import validate_phase_intervals


class PhaseGateTests(unittest.TestCase):
    def frame(self):
        return dict(load_generation=1, prepare_end_s=2., render_phase_start_s=2.1,
            render_phase_end_s=3.)

    def test_render_interval_accepts_drained_load(self):
        packet=dict(generation=1, submit_monotonic_s=2.2, drained_monotonic_s=2.8)
        self.assertEqual(validate_phase_intervals([self.frame()], [packet])['artificial_load_prepare_overlap'],0)

    def test_preparation_and_tail_overlap_are_rejected(self):
        for start,end in ((1.9,2.2),(2.2,3.1)):
            with self.assertRaises(AssertionError):
                validate_phase_intervals([self.frame()], [dict(generation=1,
                    submit_monotonic_s=start,drained_monotonic_s=end)])

    def test_duplicate_generation_rejected(self):
        with self.assertRaises(AssertionError):
            validate_phase_intervals([self.frame(),self.frame()], [])


if __name__=='__main__':
    unittest.main()
