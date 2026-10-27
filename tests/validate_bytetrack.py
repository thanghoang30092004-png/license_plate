"""Run synthetic stress scenarios and the actual configured ANPR video pipeline.

python tests/validate_bytetrack.py --config config/config.yaml
Outputs are isolated under output/bytetrack-validation-<timestamp>/.
Synthetic identity metrics are not real-video accuracy measurements.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.bytetrack_tracker import ObjectTracker


def stats(values):
    return {'samples':len(values), 'mean_ms':float(np.mean(values)),
            'p95_ms':float(np.percentile(values,95)), 'max_ms':float(np.max(values))} if values else None


def synthetic_scenarios():
    import motmetrics as mm
    results={}
    frame=np.zeros((800,1600,3),np.uint8)
    for name in ('smooth_crossing','occlusion_10_updates','occlusion_40_updates',
                 'ambiguous_crossing','crowd_80'):
        tracker=ObjectTracker(track_buffer=30)
        accumulator=mm.MOTAccumulator(auto_id=True)
        timings=[]
        for n in range(100):
            if name=='crowd_80':
                truth=[[30+150*(i%10)+n,30+85*(i//10),60,50] for i in range(80)]
            elif name=='ambiguous_crossing':
                # Abrupt direction change while both objects are hidden: motion
                # alone cannot disambiguate identities after the occlusion.
                x=40+4*n if n<40 else 196-4*(n-40)
                truth=[[x,100,60,50],[432-x,100,60,50]]
            else:
                truth=[[40+4*n,100,60,50],[480-4*n,115,60,50]]
            hidden=((name=='occlusion_10_updates' and 35<=n<45) or
                    (name=='occlusion_40_updates' and 25<=n<65) or
                    (name=='ambiguous_crossing' and 35<=n<48))
            dets=[] if hidden else [[x,y,x+w,y+h,.9,2] for x,y,w,h in truth]
            if n%2: dets.reverse()
            start=time.perf_counter()
            tracks=tracker.update(dets,frame)
            elapsed=(time.perf_counter()-start)*1000
            if n>=10: timings.append(elapsed)
            predicted=[]
            for track in tracks:
                x1,y1,x2,y2=track.to_ltrb()
                predicted.append([x1,y1,x2-x1,y2-y1])
            distances=iou_distances(truth,predicted,.5,np)
            accumulator.update(list(range(len(truth))),[int(t.track_id) for t in tracks],distances)
        metrics=mm.metrics.create().compute(accumulator,metrics=['idf1','mota','num_switches','num_misses','num_false_positives']).iloc[0]
        results[name]={'track_latency':stats(timings),
                       'synthetic_metrics':{key:float(value) if np.isfinite(value) else None for key,value in metrics.items()}}
    return results


def iou_distances(gt_boxes, predicted_boxes, threshold, np):
    a=np.asarray(gt_boxes,dtype=float).reshape(-1,4)
    b=np.asarray(predicted_boxes,dtype=float).reshape(-1,4)
    tl=np.maximum(a[:,None,:2],b[None,:,:2])
    br=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    intersection=np.maximum(br-tl,0).prod(axis=2)
    union=a[:,2:].prod(axis=1)[:,None]+b[:,2:].prod(axis=1)[None,:]-intersection
    iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    return np.where(iou>=threshold,1-iou,np.nan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/config.yaml')
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args()
    out=ROOT/'output'/datetime.now().strftime('bytetrack-validation-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    print(f'OUTPUT={out}',flush=True)
    report={'synthetic':synthetic_scenarios(), 'video_accuracy':'Not measured: no ground truth labels.'}
    if not args.synthetic_only:
        import pipeline.run_realtime as app
        cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
        base=args.config.resolve().parent.parent
        for section,key in [('models','vehicle_detector'),('models','plate_detector'),
                            ('ocr','onnx_model_path'),('ocr','plate_config_path'),('video','source')]:
            cfg[section][key]=str((base/Path(cfg[section][key])).resolve())
        cfg['video']['output']=str(out/'anpr.mp4')
        cfg['snapshot']['output_dir']=str(out/'snapshots')
        test_config=out/'config.yaml'
        test_config.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        report['submissions']=[]
        report['callbacks']=[]
        report['thread_errors']=[]
        original_saver=app.SnapshotSaver
        original_hook=threading.excepthook
        savers=[]
        class ObservedSaver(original_saver):
            def __init__(self,*a,**kw):
                callback=kw.get('on_saved')
                def saved(tid,path):
                    report['callbacks'].append({'track':tid,'path':path})
                    if callback: callback(tid,path)
                kw['on_saved']=saved
                super().__init__(*a,**kw)
                savers.append(self)
            def submit(self,crop,tid,text,frame):
                accepted=super().submit(crop,tid,text,frame)
                report['submissions'].append({'track':tid,'plate':text,'frame':frame,
                                               'shape':list(crop.shape),'accepted':accepted})
                return accepted
        app.SnapshotSaver=ObservedSaver
        threading.excepthook=lambda args:report['thread_errors'].append(str(args.exc_value))
        started=time.perf_counter()
        try:
            report['pipeline']=app.main(str(test_config),display=False)
        except Exception:
            report['error']=traceback.format_exc()
            print(report['error'],flush=True)
        finally:
            app.SnapshotSaver=original_saver
            threading.excepthook=original_hook
            report['elapsed_seconds']=time.perf_counter()-started
            report['snapshot_threads_alive']=[s.is_alive() for s in savers]
            report['snapshot_queue_remaining']=[s.queue.qsize() for s in savers]
        files=list((out/'snapshots').glob('*.jpg'))
        report['snapshots']=len(files)
        report['unreadable_snapshots']=[str(p) for p in files if cv2.imread(str(p)) is None]
        accepted=[s for s in report['submissions'] if s['accepted']]
        report['repeated_track_ids']={k:v for k,v in Counter(s['track'] for s in accepted).items() if v>1}
        report['repeated_plate_texts']={k:v for k,v in Counter(s['plate'] for s in accepted).items() if v>1}
        log=out/'anpr.log'
        if log.exists():
            lines=log.read_text(encoding='utf-8').splitlines()
            frame_lines=[line for line in lines if line.startswith('frame ')]
            report['video_track_latency_after_10_frames']=stats([
                float(re.search(r'\| track ([\d.]+) ms',line).group(1)) for line in frame_lines[10:]])
            indexed={int(line.split()[1]):line for line in frame_lines}
            report['snapshot_log_mismatches']=[s for s in accepted if s['frame'] in indexed and
                f"track {s['track']} hien={s['plate']} " not in indexed[s['frame']]]
        if (out/'anpr.mp4').exists():
            cap=cv2.VideoCapture(str(out/'anpr.mp4')); count=0
            while cap.read()[0]: count+=1
            cap.release()
            report['decoded_output_frames']=count
        pipeline=report.get('pipeline',{})
        report['integrity_ok']=bool(pipeline.get('completed') and
            not report['thread_errors'] and not report['unreadable_snapshots'] and
            not any(report['snapshot_threads_alive']) and not any(report['snapshot_queue_remaining']) and
            not report.get('snapshot_log_mismatches') and
            report.get('decoded_output_frames')==pipeline.get('source_frames') and
            report['snapshots']==pipeline.get('snapshot_saved') and
            pipeline.get('snapshot_failed')==0 and pipeline.get('snapshot_callback_errors')==0)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('submissions','callbacks')},indent=2,ensure_ascii=False),flush=True)
    return 1 if report.get('error') or report.get('thread_errors') or report.get('integrity_ok') is False else 0


if __name__=='__main__':
    raise SystemExit(main())
