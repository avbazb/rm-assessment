"""Research v8: v7 geometry recovery followed by frozen dark-empty rejection.

Filtering happens at detector output, before all stateful consumers.
Development-scene candidate; this does not claim an independent FP rate.
"""
import copy
from .attachment import FrozenDetectionFusion
from .rejection import features
from .rejection import POLICY,reject

def filter_observations(frame,observations):
    retained=[];removed=[];evaluated=[]
    for index,o in enumerate(observations):
        measurement=features(frame,o)
        discard=bool(reject(o,measurement))
        evaluated.append(dict(original_index=index,features=measurement,rejected=discard))
        if discard:
            removed.append(dict(original_index=index,observation=copy.deepcopy(o),features=measurement,
                                reason='unsupported_dark_empty_large_model_box_v8'))
        else:retained.append(copy.deepcopy(o))
    return retained,removed,evaluated

class FilteredDetectionFusion(FrozenDetectionFusion):
    def detect(self,frame,timestamp=0.,frame_id='camera_optical'):
        # Keep v7's thread-local observation/diagnostic transaction intact.
        with self._call_lock:
            original=super().detect(frame,timestamp,frame_id)
            rows,removed,evaluated=filter_observations(frame,[o.to_dict() for o in original])
            keep={entry['original_index'] for entry in evaluated if not entry['rejected']}
            self.last_diagnostics=dict(self.last_diagnostics,dark_empty_rejection=dict(
                policy='unsupported_dark_empty_large_model_box_v8',parameters=POLICY,
                evaluated_observations=len(evaluated),removed_observations=removed))
            return [o for i,o in enumerate(original) if i in keep]

CandidateDetector=FilteredDetectionFusion
