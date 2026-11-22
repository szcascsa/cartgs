import math
import os
from pathlib import Path
import tempfile
import unittest

from control import BudgetController, DUTIES, stage_at
from activity import ActivityMonitor


class ControlTests(unittest.TestCase):
    def test_decrease_only_after_full_window(self):
        controller = BudgetController(1.)
        for _ in range(9):
            controller.update(2.)
        self.assertEqual(controller.budget, 4000)
        controller.update(2.)
        self.assertEqual(controller.budget, 3000)

    def test_lower_and_upper_saturation(self):
        controller = BudgetController(1.)
        for _ in range(100):
            controller.update(2.)
        self.assertEqual(controller.budget, 1000)
        for _ in range(200):
            controller.update(.5)
        self.assertEqual(controller.budget, 4000)

    def test_recovery_needs_two_consecutive_windows(self):
        controller = BudgetController(1., index=2)
        for _ in range(10):
            controller.update(.5)
        self.assertEqual(controller.budget, 2000)
        for _ in range(10):
            controller.update(.9)
        for _ in range(10):
            controller.update(.5)
        self.assertEqual(controller.budget, 2000)
        for _ in range(10):
            controller.update(.5)
        self.assertEqual(controller.budget, 3000)

    def test_invalid_values(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                BudgetController(value)
            with self.assertRaises(ValueError):
                BudgetController(1.).update(value)

    def test_schedule_boundaries_and_final_recovery(self):
        self.assertIsNone(stage_at(-1))
        self.assertEqual(stage_at(0), 0)
        self.assertEqual(stage_at(4.999), 0)
        self.assertEqual(stage_at(5), 1)
        self.assertEqual(stage_at(29.999), 5)
        self.assertEqual(DUTIES[5], 0.)
        self.assertIsNone(stage_at(30))

    def test_external_resident_compute_blocks_gpu_admission(self):
        monitor = ActivityMonitor.__new__(ActivityMonitor)
        monitor.samples = [dict(monotonic_s=value, records=[dict(pid=os.getpid()+99999,
            kind='C', sm=0, mem=0, enc=0, dec=0)]) for value in (1., 2., 3.)]
        monitor.unexpected = lambda start, end, allowed: (monitor.samples, [])
        from unittest.mock import patch
        with patch('activity.time.monotonic', return_value=4.):
            self.assertFalse(monitor.quiet())
            for sample in monitor.samples:
                sample['records'][0]['pid'] = os.getpid()
            self.assertTrue(monitor.quiet())

    def test_pmon_parser_preserves_pid_kind_and_timestamps(self):
        monitor = ActivityMonitor.__new__(ActivityMonitor)
        monitor.samples = []
        monitor.identities = {}
        monitor.error = None
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            monitor.folder = Path(folder)
            monitor.process = SimpleNamespace(stdout=[
                '# Time gpu pid type sm mem enc dec fb ccpm command\n',
                '10:00:00 0 41 C 23 4 - - 900 0 python\n',
                '10:00:00 0 42 G - - - - 9 0 Xorg\n',
                '10:00:01 0 41 C 0 0 - - 900 0 python\n',
                '10:00:02 0 41 C 0 0 - - 900 0 python\n',
            ])
            with patch('activity.time.monotonic', side_effect=[10., 11., 12.]):
                monitor._collect()
            self.assertIsNone(monitor.error)
            self.assertEqual(len(monitor.samples), 2)
            self.assertEqual(monitor.samples[0]['records'][0]['pid'], 41)
            self.assertEqual(monitor.samples[0]['records'][0]['sm'], 23)
            self.assertEqual(monitor.samples[0]['records'][1]['kind'], 'G')
            self.assertEqual(monitor.samples[1]['monotonic_s'], 11.)

    def test_only_authorized_background_is_excluded_from_interference(self):
        from types import SimpleNamespace
        monitor = ActivityMonitor.__new__(ActivityMonitor)
        monitor.error = None
        monitor.process = SimpleNamespace(poll=lambda: None)
        monitor.permitted_background = {41}
        monitor.samples = [dict(monotonic_s=1., records=[dict(pid=pid,
            kind='C', sm=30, mem=5, enc=0, dec=0) for pid in (41, 42, 43)])]
        samples, unexpected = monitor.unexpected(0., 2., {42})
        self.assertEqual(len(samples), 1)
        self.assertEqual([r['pid'] for r in unexpected], [43])


if __name__ == '__main__':
    unittest.main()
