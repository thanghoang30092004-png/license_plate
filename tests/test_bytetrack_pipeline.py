"""Real ByteTrack + real cache/worker, deterministic detections (no GPU needed).

Run: python -m unittest discover -s tests -p test_bytetrack_pipeline.py -v
"""
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from tracking.bytetrack_tracker import ObjectTracker
from tracking.matching import match_plate_to_tracks
from tracking.track_cache import TrackCache
from pipeline.inference_worker import InferenceWorker


def box(x=20, y=20, score=.9, cls=2):
    return [x, y, x + 80, y + 60, score, cls]


class ByteTrackTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((400, 800, 3), np.uint8)
        self.tracker = ObjectTracker(track_buffer=3)

    def step(self, boxes):
        return self.tracker.update(boxes, self.frame)

    def test_id_stable_under_motion_and_order_reversal(self):
        first = self.step([box(), box(250)])
        ids = [t.track_id for t in first]
        for n in range(1, 30):
            tracks = self.step([box(250+n), box(20+n)])
            tracks.sort(key=lambda t: t.to_ltrb()[0])
            self.assertEqual([t.track_id for t in tracks], ids)

    def test_low_confidence_continues_but_cannot_create_track(self):
        tid = self.step([box()])[0].track_id
        tracks = self.step([box(score=.15), box(250, score=.15)])
        self.assertEqual([t.track_id for t in tracks], [tid])
        self.assertAlmostEqual(tracks[0].confidence, .15, places=5)

    def test_below_low_confidence_is_not_visible(self):
        tid = self.step([box()])[0].track_id
        self.assertEqual(self.step([box(score=.05)]), [])
        self.assertIn(tid, self.tracker.retained_track_ids())

    def test_low_confidence_does_not_reactivate_lost_track(self):
        tid = self.step([box()])[0].track_id
        self.step([])
        self.assertEqual(self.step([box(score=.15)]), [])
        self.assertEqual(self.step([box()])[0].track_id, tid)

    def test_lost_track_not_exposed_but_cache_survives(self):
        tid = self.step([box()])[0].track_id
        cache = TrackCache(min_frames_to_confirm=1, absent_grace=1)
        cache.update(tid, dict(text='51A12345', valid=True, confidence=.9), 0)
        cache.mark_snapshot_saved(tid)
        for _ in range(3):
            self.assertEqual(self.step([]), [])
            cache.retain_tracks(self.tracker.retained_track_ids())
            self.assertEqual(cache.get_text(tid), '51A12345')
            self.assertFalse(cache.should_save_snapshot(tid))
        tracks = self.step([box()])
        cache.retain_tracks(self.tracker.retained_track_ids())
        self.assertEqual(tracks[0].track_id, tid)
        self.assertEqual(cache.get_text(tid), '51A12345')

    def test_expired_track_and_cache_do_not_resurrect(self):
        tid = self.step([box()])[0].track_id
        cache = TrackCache(min_frames_to_confirm=1)
        cache.update(tid, dict(text='51A12345', valid=True), 0)
        for _ in range(4):
            self.step([])
            cache.retain_tracks(self.tracker.retained_track_ids())
        self.assertNotIn(tid, self.tracker.retained_track_ids())
        self.assertIsNone(cache.get_text(tid))
        self.step([box()])
        tracks = self.step([box()])
        self.assertEqual(len(tracks), 1)
        self.assertNotEqual(tracks[0].track_id, tid)

    def test_new_track_after_first_frame_requires_confirmation(self):
        self.step([])
        self.assertEqual(self.step([box()]), [])
        self.assertEqual(len(self.tracker.retained_track_ids()), 1)
        self.assertEqual(len(self.step([box()])), 1)

    def test_unconfirmed_false_positive_disappears(self):
        self.step([])
        self.step([box()])
        self.step([])
        self.assertEqual(self.tracker.retained_track_ids(), set())

    def test_bad_numeric_boxes_do_not_poison_tracker(self):
        invalid = [[0,0,0,20,.9,2], [20,0,10,20,.9,2], [0,0,20,20,float('nan'),2],
                   [0,0,float('inf'),20,.9,2], [0,0,20,20,1.1,2], [0,0,20,20,.9,2.5]]
        tracks = self.step(invalid + [box()])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(self.tracker.rejected_detections, 6)
        self.assertTrue(np.isfinite(tracks[0].to_ltrb()).all())

    def test_wrong_detection_shape_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.step([[1, 2, 3]])

    def test_bbox_result_is_independent_of_future_updates(self):
        track = self.step([box()])[0]
        original = track.to_ltrb()
        altered = track.to_ltrb()
        altered[:] = -100
        self.step([box(25)])
        np.testing.assert_array_equal(track.to_ltrb(), original)

    def test_two_tracker_instances_have_independent_ids(self):
        first = self.step([box()])[0].track_id
        other = ObjectTracker()
        other.update([box()], self.frame)
        self.step([box(), box(250)])
        tracks = self.step([box(), box(250)])
        self.assertEqual(len({t.track_id for t in tracks}), 2)
        self.assertIn(first, {t.track_id for t in tracks})

    def test_many_objects_80_tracks_50_updates(self):
        dets = [box(100*(n%8), 35*(n//8)) for n in range(80)]
        first = {t.track_id for t in self.step(dets)}
        self.assertEqual(len(first), 80)
        for _ in range(50):
            self.assertEqual({t.track_id for t in self.step(list(reversed(dets)))}, first)

    def test_no_detections_200_updates(self):
        self.step([box()])
        for _ in range(200):
            self.assertEqual(self.step([]), [])
        self.assertEqual(self.tracker.retained_track_ids(), set())

    def test_lost_prediction_cannot_match_plate(self):
        self.step([box()])
        tracks = self.step([])
        self.assertIsNone(match_plate_to_tracks([30,40,60,50], tracks))

    def test_invalid_tracker_configuration(self):
        for kw in ({'track_buffer':0}, {'track_buffer':1.5}, {'track_low_thresh':.4},
                   {'match_thresh':float('nan')}, {'fuse_score':'false'}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ObjectTracker(**kw)

    def test_default_30_update_buffer_boundary(self):
        self.tracker = ObjectTracker(track_buffer=30)
        tid = self.step([box()])[0].track_id
        for _ in range(30):
            self.step([])
            self.assertIn(tid, self.tracker.retained_track_ids())
        self.step([])
        self.assertNotIn(tid, self.tracker.retained_track_ids())

    def test_partial_occlusion_does_not_affect_other_vehicle(self):
        tracks = self.step([box(), box(250)])
        ids = [t.track_id for t in tracks]
        for _ in range(2):
            visible = self.step([box(250)])
            self.assertEqual([t.track_id for t in visible], [ids[1]])
            self.assertEqual(self.tracker.retained_track_ids(), set(ids))
        self.assertEqual({t.track_id for t in self.step([box(),box(250)])},set(ids))


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120,220,3),np.uint8)
        self.detections = [box()]
        self.plates = [[30,40,60,50,.9,0]]
        self.calls = 0
        self.snapshots = []
        self.cache = TrackCache(min_frames_to_confirm=1, recheck_interval_found=10000, absent_grace=1)
        self.tracker = ObjectTracker(track_buffer=3)
        def read(crops):
            self.calls += 1
            return [dict(text='51A12345',confidence=.99,valid=True) for _ in crops]
        def save(crop,tid,text,frame):
            self.snapshots.append((tid,text,frame))
            self.cache.mark_snapshot_saved(tid)
        self.worker = InferenceWorker(queue.Queue(),queue.Queue(maxsize=1),
            SimpleNamespace(detect=lambda f:self.detections),
            SimpleNamespace(detect=lambda f:self.plates),self.tracker,
            SimpleNamespace(read_batch=read),self.cache,
            snapshot_writer=SimpleNamespace(submit=save))

    def step(self, frame):
        self.worker._process((frame,self.frame))
        return self.worker.result_queue.get_nowait()[2]

    def test_returning_lost_track_keeps_ocr_and_snapshot_flags(self):
        first = self.step(0)
        tid = next(k for k in first if not k.startswith('__'))
        self.detections = []
        for n in (50,100,150):
            result = self.step(n)
            self.assertNotIn(tid,result)
        self.detections = [box()]
        result = self.step(200)
        self.assertEqual(result[tid]['plate_text'],'51A12345')
        self.assertEqual(self.calls,1)
        self.assertEqual(len(self.snapshots),1)

    def test_expiry_clears_cache_and_new_id_must_ocr_again(self):
        first=self.step(0)
        tid=next(k for k in first if not k.startswith('__'))
        self.detections=[]
        for n in range(1,5): self.step(n)
        self.assertIsNone(self.cache.get_text(tid))
        self.detections=[box()]
        self.step(5); self.step(6)
        self.assertEqual(self.calls,2)
        self.assertNotEqual(self.snapshots[0][0],self.snapshots[-1][0])

    def test_same_frame_duplicate_plate_detections_only_one_vote(self):
        self.cache.min_frames_to_confirm=3
        self.plates *= 3
        self.step(0)
        self.assertFalse(self.snapshots)
        self.assertEqual(len(next(iter(self.cache._cache.values()))['candidates']),1)

    def test_skipped_source_frames_do_not_age_buffer(self):
        self.step(0)
        self.detections=[]
        self.step(5000)
        self.assertTrue(self.tracker.retained_track_ids())

    def test_late_snapshot_callback_cannot_recreate_expired_cache(self):
        self.step(0)
        tid = self.snapshots[0][0]
        self.cache.retain_tracks(set())
        self.cache.mark_snapshot_saved(tid)
        self.assertEqual(self.cache._cache,{})

    def test_two_vehicles_ocr_batch_keeps_plate_assignment(self):
        self.detections=[box(),box(130)]
        self.plates=[[30,40,60,50,.9,0],[140,40,170,50,.9,0]]
        self.worker.ocr_engine.read_batch=lambda crops:[
            dict(text=t,valid=True,confidence=.99) for t in ('51A12345','30B67890')]
        result=self.step(0)
        visible=[row for key,row in result.items() if not key.startswith('__')]
        visible.sort(key=lambda row:row['vehicle_bbox'][0])
        self.assertEqual([r['plate_text'] for r in visible],['51A12345','30B67890'])

    def test_highest_confidence_plate_chosen_for_single_vote(self):
        self.plates=[[30,40,40,50,.2,0],[30,40,60,50,.9,0]]
        shapes=[]
        def read(crops):
            shapes.extend(c.shape for c in crops)
            return [dict(text='51A12345',valid=True,confidence=.9)]
        self.worker.ocr_engine.read_batch=read
        self.step(0)
        self.assertEqual(shapes,[(10,30,3)])


if __name__ == '__main__':
    unittest.main()
