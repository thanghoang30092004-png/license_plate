"""ByteTrack adapter. Buffer age is measured in update calls, not source frame IDs.

Uses the Ultralytics BYTETracker/Boxes API (tested with 8.4.158).
Only currently observed, activated tracks are exposed for OCR and snapshots.
"""
from dataclasses import dataclass
import inspect
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker, STrack


@dataclass(frozen=True)
class ObjectTrack:
    track_id: str
    bbox: tuple
    confidence: float
    class_id: int

    def to_ltrb(self):
        return np.asarray(self.bbox, dtype=float)


class ObjectTracker:
    def __init__(self, track_high_thresh=0.25, track_low_thresh=0.1,
                 new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
                 fuse_score=True):
        if not 0 <= track_low_thresh < track_high_thresh <= new_track_thresh <= 1:
            raise ValueError('Require 0 <= low < high <= new <= 1')
        if isinstance(track_buffer, bool) or not isinstance(track_buffer, int) or track_buffer < 1:
            raise ValueError('track_buffer must be a positive integer (update calls)')
        if not 0 < match_thresh <= 1 or not isinstance(fuse_score, bool):
            raise ValueError('match_thresh must be in (0, 1]; fuse_score must be bool')
        args = SimpleNamespace(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                               new_track_thresh=new_track_thresh, track_buffer=track_buffer,
                               match_thresh=match_thresh, fuse_score=fuse_score)
        # Ultralytics normally uses a process-wide ID counter. Scope IDs to this
        # instance so creating a second tracker cannot reset a running session.
        class LocalTrack(STrack):
            counter = 0

            @classmethod
            def next_id(cls):
                cls.counter += 1
                return cls.counter

        class LocalTracker(BYTETracker):
            track_class = LocalTrack

            @staticmethod
            def reset_id():
                LocalTrack.counter = 0

        kwargs = {'frame_rate': 30} if 'frame_rate' in inspect.signature(BYTETracker).parameters else {}
        self.tracker = LocalTracker(args, **kwargs)
        self.rejected_detections = 0

    def update(self, detections, frame):
        rows = np.asarray(detections, dtype=np.float32)
        if rows.size == 0:
            rows = np.empty((0, 6), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise ValueError('detections must have shape (N, 6): xyxy, confidence, class_id')
        valid = (np.isfinite(rows).all(axis=1) & (rows[:, 2] > rows[:, 0]) &
                 (rows[:, 3] > rows[:, 1]) & (rows[:, 4] >= 0) & (rows[:, 4] <= 1) &
                 (rows[:, 5] >= 0) & (rows[:, 5] == np.floor(rows[:, 5]))) 
        self.rejected_detections += int((~valid).sum())
        boxes = Boxes(rows[valid], orig_shape=frame.shape[:2])
        # Removed entries can linger in lost_stracks for one iteration in some
        # Ultralytics releases. Never let them re-enter association or the cache.
        self.tracker.lost_stracks = [t for t in self.tracker.lost_stracks if t.state == TrackState.Lost]
        self.tracker.update(boxes, frame)
        self.tracker.lost_stracks = [t for t in self.tracker.lost_stracks if t.state == TrackState.Lost]
        return [ObjectTrack(str(t.track_id), tuple(float(v) for v in t.xyxy), float(t.score), int(t.cls))
                for t in self.tracker.tracked_stracks
                if t.state == TrackState.Tracked and t.is_activated and t.frame_id == self.tracker.frame_id]

    def retained_track_ids(self):
        """IDs owned by ByteTrack, including Lost and not-yet-activated tracks."""
        return {str(t.track_id) for t in self.tracker.tracked_stracks + self.tracker.lost_stracks
                if t.state in (TrackState.Tracked, TrackState.Lost)}
