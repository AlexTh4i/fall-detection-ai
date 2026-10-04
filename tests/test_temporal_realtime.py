import unittest
import contextlib
import io
import numpy as np
import torch
from realtime_yolo_fall_detection import (TemporalPersonTracker, TemporalBackend,
    TEMPORAL_CHECKPOINT, PROJECT_ROOT, STATE_WARMING, STATE_UNKNOWN,
    STATE_NORMAL, STATE_CONFIRMED)
from gmdcsa_pipeline import windows

class FakeBackend:
    hz=10
    steps=30
    threshold=0.43
    value=0.9
    def predict(self,samples):
        valid=np.array([s[0][:,2].sum()>0 for s in samples[-30:]])
        return (self.value if valid.mean()>=0.5 and valid[-1] else None),float(valid.mean())

class TemporalRealtimeTests(unittest.TestCase):
    def setUp(self):
        self.backend=FakeBackend()
        self.tracker=TemporalPersonTracker(1,self.backend)
        self.pose=np.ones((17,3),np.float32)
        self.pose[:,:2]=300
        self.box=np.array([100,100,500,600],np.float32)
    def feed(self,first,last):
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(first,last):
                self.tracker.update(self.pose,self.box,(720,1280,3),i/30)
    def test_sampling_warmup_persistence_and_low_score(self):
        self.feed(0,75)
        self.assertEqual(self.tracker.state,STATE_WARMING)
        self.feed(75,135)
        self.assertEqual(self.tracker.state,STATE_CONFIRMED)
        self.assertLessEqual(len(self.tracker.samples),31)
        self.backend.value=0.0
        self.feed(135,180)
        self.assertEqual(self.tracker.state,STATE_NORMAL)
    def test_missing_is_unknown_and_long_gap_restarts_history(self):
        self.feed(0,135)
        self.tracker.mark_missing(4.6)
        self.assertEqual(self.tracker.state,STATE_UNKNOWN)
        self.assertIsNone(self.tracker.score)
        self.tracker.update(self.pose,self.box,(720,1280,3),12)
        self.assertEqual(self.tracker.state,STATE_WARMING)
        self.assertEqual(len(self.tracker.samples),1)
    def test_independent_people_and_reset(self):
        self.feed(0,135)
        second=TemporalPersonTracker(2,self.backend)
        second.update(self.pose,self.box,(720,1280,3),4.5)
        self.assertEqual(second.state,STATE_WARMING)
        self.assertEqual(self.tracker.state,STATE_CONFIRMED)
        self.tracker.reset_to_normal()
        self.assertEqual(len(self.tracker.samples),0)
        self.assertIsNone(self.tracker.suspected_start_time)
        self.assertEqual(self.tracker.state,STATE_WARMING)
    def test_live_preprocessing_matches_offline_window(self):
        backend=TemporalBackend(TEMPORAL_CHECKPOINT,PROJECT_ROOT/'yolo26n-pose.pt','cpu')
        # Synthetic COCO skeleton: no downloaded videos or cached keypoints needed.
        anchors=np.array([[320,100],[310,90],[330,90],[300,100],[340,100],
            [280,180],[360,180],[260,250],[380,250],[240,320],[400,320],
            [290,340],[350,340],[290,460],[350,460],[290,620],[350,620]],np.float32)
        pose=np.ones((31,17,3),np.float32)
        pose[:,:,:2]=anchors[None]/1280
        pose[:,:,0]+=np.sin(np.arange(31,dtype=np.float32)/10)[:,None]*0.01
        pose[:,:,2]=0.9
        box=np.tile(np.array([220,60,420,650],np.float32)/1280,(31,1))
        data=dict(pose=pose,box=box,time=np.arange(31,dtype=np.float64)/10)
        samples=list(zip(data['pose'],data['box'],data['time']))
        live,observed=backend.predict(samples)
        x,m,_=windows(data)
        with torch.inference_mode():
            offline=torch.sigmoid(backend.model(torch.tensor(x[-1:]),torch.tensor(m[-1:]))).item()
        self.assertGreaterEqual(observed,0.5)
        self.assertAlmostEqual(live,offline,places=6)

if __name__=='__main__':unittest.main()
