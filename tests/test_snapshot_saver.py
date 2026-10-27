import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from utils.snapshot_saver import SnapshotSaver


class SnapshotSaverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = np.zeros((24, 24, 3), np.uint8)
        self.savers = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for saver in self.savers:
            saver.stop()
            if saver.ident is not None:
                saver.join(3)

    def saver(self, **kw):
        s = SnapshotSaver(self.tmp.name, **kw)
        self.savers.append(s)
        return s

    def finish(self, s):
        s.stop()
        s.join(5)
        self.assertFalse(s.is_alive())
        self.assertEqual(s.queue.unfinished_tasks, 0)

    def test_100_real_images_drained_on_immediate_stop(self):
        callbacks = []
        s = self.saver(queue_max_size=120, on_saved=lambda tid,p:callbacks.append(p))
        for tid in range(100): s.submit(self.image, tid, 'A', tid)
        s.start()
        self.finish(s)
        self.assertEqual(len(callbacks), 100)
        self.assertEqual(s.saved_count, 100)
        self.assertTrue(all(cv2.imread(p) is not None for p in callbacks))

    def test_failed_imwrite_does_not_notify_success_and_allows_retry(self):
        callbacks=[]
        s=self.saver(on_saved=lambda *a:callbacks.append(a))
        s.submit(self.image,1,'A',1)
        with patch('utils.snapshot_saver.cv2.imwrite',return_value=False), self.assertLogs('utils.snapshot_saver',level='ERROR'):
            s.start()
            s.queue.join()
        self.assertEqual(callbacks,[])
        self.assertEqual(s.failed_count,1)
        self.assertTrue(s.submit(self.image,1,'A',2))
        self.finish(s)
        self.assertEqual(len(callbacks),1)

    def test_callback_exception_does_not_stop_other_images(self):
        def callback(tid,path):
            if tid==1: raise RuntimeError('intentional')
        s=self.saver(on_saved=callback)
        s.submit(self.image,1,'A',1); s.submit(self.image,2,'B',2)
        with self.assertLogs('utils.snapshot_saver',level='ERROR'):
            s.start(); self.finish(s)
        self.assertEqual(s.saved_count,2)
        self.assertEqual(s.callback_error_count,1)

    def test_pending_duplicate_is_rejected(self):
        s=self.saver()
        self.assertTrue(s.submit(self.image,1,'A',1))
        self.assertFalse(s.submit(self.image,1,'A',2))
        s.start(); self.finish(s)
        self.assertEqual(s.saved_count,1)

    def test_full_queue_releases_dropped_track_for_retry(self):
        s=self.saver(queue_max_size=2)
        for tid in (1,2,3): s.submit(self.image,tid,'A',tid)
        self.assertEqual(s.dropped_count,1)
        self.assertTrue(s.submit(self.image,1,'A',4))
        s.start(); self.finish(s)
        self.assertEqual(s.saved_count,2)
        self.assertEqual(s.dropped_count,2)

    def test_submit_after_stop_is_rejected(self):
        s=self.saver(); s.stop()
        self.assertFalse(s.submit(self.image,1,'A',1))

    def test_queued_crop_is_independent_of_callers_array(self):
        s=self.saver()
        s.submit(self.image,1,'A',1)
        self.image[:]=255
        s.start(); self.finish(s)
        files=list(Path(self.tmp.name).glob('*.jpg'))
        self.assertEqual(cv2.imread(str(files[0])).mean(),0)

    def test_identical_metadata_does_not_overwrite(self):
        s=self.saver()
        first=s._write_to_disk(self.image,1,'A',1)
        second=s._write_to_disk(self.image,1,'A',1)
        self.assertNotEqual(first,second)
        self.assertTrue(Path(first).is_file() and Path(second).is_file())

    def test_stop_while_writing_still_drains_waiting_item(self):
        s=self.saver(); entered=threading.Event(); release=threading.Event()
        self.addCleanup(release.set)
        original=s._write_to_disk
        def write(*args):
            entered.set()
            if not release.wait(3): raise TimeoutError('test release missing')
            return original(*args)
        s._write_to_disk=write
        s.submit(self.image,1,'A',1); s.start()
        self.assertTrue(entered.wait(3))
        s.submit(self.image,2,'B',2); s.stop(); release.set()
        self.finish(s)
        self.assertEqual(s.saved_count,2)

    def test_parallel_producers_and_stop_account_for_accepted_items(self):
        s=self.saver(queue_max_size=400)
        barrier=threading.Barrier(5)
        def producer(offset):
            barrier.wait()
            for n in range(75): s.submit(self.image,offset+n,'A',n)
        threads=[threading.Thread(target=producer,args=(i*100,)) for i in range(4)]
        s.start()
        for t in threads: t.start()
        barrier.wait()
        for t in threads: t.join(3); self.assertFalse(t.is_alive())
        self.finish(s)
        self.assertEqual(s.saved_count,300)
        self.assertEqual(s.failed_count+s.dropped_count,0)


if __name__=='__main__':
    unittest.main()
