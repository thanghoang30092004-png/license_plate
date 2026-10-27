"""Snapshot data regressions; no GPU models required."""
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.inference_worker import InferenceWorker
from tracking.track_cache import TrackCache


class SnapshotWorkerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 220, 3), dtype=np.uint8)
        self.frame[:, 110:] = 200
        self.tracks = [
            SimpleNamespace(track_id='A', to_ltrb=lambda: [0, 0, 100, 100]),
            SimpleNamespace(track_id='B', to_ltrb=lambda: [110, 0, 210, 100]),
        ]
        self.cache = TrackCache(min_frames_to_confirm=1, recheck_interval_found=60)
        self.saved = []
        self.ocr_calls = []

        def read(crops):
            self.ocr_calls.append(len(crops))
            return [dict(text='51A12345', valid=True, confidence=.95) for _ in crops]

        self.worker = InferenceWorker(
            queue.Queue(), queue.Queue(),
            SimpleNamespace(detect=lambda frame: []),
            SimpleNamespace(detect=lambda frame: [[10, 20, 40, 30, .9, 0]]),
            SimpleNamespace(update=lambda dets, frame: self.tracks),
            SimpleNamespace(read_batch=read), self.cache,
            snapshot_writer=SimpleNamespace(submit=lambda *item: self.saved.append(item)))

    def confirm(self, tid, text):
        self.cache.update(tid, dict(text=text, valid=True, confidence=.95), 0)

    def test_confirmation_submits_on_same_frame_with_vehicle_bbox(self):
        self.worker._process((0, self.frame))
        self.assertEqual(len(self.saved), 1)
        crop, tid, text, frame = self.saved[0]
        self.assertEqual((tid, text, frame), ('A', '51A12345', 0))
        self.assertEqual(crop.shape, (100, 100, 3))

    def test_cached_plates_belong_to_correct_tracks_when_ocr_skipped(self):
        self.confirm('A', '51A12345')
        self.confirm('B', '30B67890')
        self.worker._process((1, self.frame))
        self.assertEqual(self.ocr_calls, [])
        self.assertEqual([(item[1], item[2]) for item in self.saved],
                         [('A', '51A12345'), ('B', '30B67890')])
        self.assertEqual([float(item[0].mean()) for item in self.saved], [0, 200])

    def test_confirmed_track_saves_without_current_plate_detection(self):
        self.confirm('A', '51A12345')
        self.worker.plate_detector.detect = lambda frame: []
        self.worker._process((1, self.frame))
        self.assertEqual([item[1] for item in self.saved], ['A'])

    def test_unconfirmed_and_already_saved_tracks_are_skipped(self):
        self.confirm('A', '51A12345')
        self.cache.mark_snapshot_saved('A')
        self.worker.plate_detector.detect = lambda frame: []
        self.worker._process((1, self.frame))
        self.assertEqual(self.saved, [])

    def test_snapshot_does_not_change_when_source_frame_is_drawn(self):
        self.worker._process((0, self.frame))
        crop = self.saved[0][0]
        self.assertFalse(np.shares_memory(crop, self.frame))
        self.frame[:] = 255
        self.assertEqual(float(crop.mean()), 0)

    def test_vehicle_crop_is_clipped_and_outside_boxes_are_skipped(self):
        self.worker._submit_snapshot(self.frame, [-10, -10, 20, 30], 'A', 'P', 0)
        self.assertEqual(self.saved[0][0].shape, (30, 20, 3))
        for box in ([-20, 0, -10, 30], [230, 0, 250, 30], [20, 0, 10, 30]):
            self.worker._submit_snapshot(self.frame, box, 'A', 'P', 0)
        self.assertEqual(len(self.saved), 1)


if __name__ == '__main__':
    unittest.main()
