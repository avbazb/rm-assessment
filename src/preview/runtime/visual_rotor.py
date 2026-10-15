"""Causal image-plane rotor model, independent of metric firing admission.

Robot identity survives visible-plate changes. Wrapped phase fitting retains
multiple speed hypotheses until a distinct temporal fit is available. Every
initial phase model uses measured lamps. Once initialized, fresh image feature
centers can update the joint pixel state without becoming PnP corners or yaw.
"""
import copy
import math
from collections import deque
import numpy as np
from .fast_classical import _iou


def folded(value,period):
    return (value+period/2)%period-period/2


def observed_size(o):
    if o.get('geometry_valid') and o.get('corners') is not None:
        q=np.asarray(o['corners']);return float(np.linalg.norm(q[3]-q[0])),float(np.linalg.norm(q[1]-q[0]))
    box=o['bbox'];return max(4.,(box[3]-box[1])*.55),max(4.,box[2]-box[0])


class Robot:
    def __init__(self,identifier,color,center,timestamp):
        self.id=identifier;self.color=color;self.center=np.array(center,float);self.velocity=np.zeros(2)
        self.seen=timestamp;self.history=deque();self.positions=deque();self.model=None;self.axis=np.array([0.,1.,0.])
        self.indices=[];self.poses=[];self.selected_slot=None;self.height=20.;self.width=40.;self.generation=0
        self.fit_diagnostic={}
        self.axis_history=deque();self.created=timestamp
        self.identity_center=np.array(center,float)
        self.body_history=deque(maxlen=45)

    def update(self,rows,timestamp,calibration,plate_count,radius):
        self.indices=[i for i,o in rows];self.positions.extend((timestamp,np.asarray(o['center'],float)) for i,o in rows)
        while self.positions and timestamp-self.positions[0][0]>.45:self.positions.popleft()
        positions=np.array([p for t,p in self.positions]);measurement=(np.percentile(positions,10,axis=0)+np.percentile(positions,90,axis=0))/2
        dt=timestamp-self.seen
        if dt>0:
            a=1-math.exp(-dt/.12);old=self.center.copy();innovation=measurement-self.center
            # A newly visible face or stray pair must not drag the body center
            # in one update. Huber-weight the body measurement; raw lamp points
            # and the independent phase/pixel fit remain unchanged.
            weight=min(1.,max(12.,.7*self.height)/max(float(np.linalg.norm(innovation)),1e-9))
            self.center+=a*weight*innovation
            self.velocity+=(1-math.exp(-dt/.25))*((self.center-old)/dt-self.velocity)
        self.seen=timestamp;self.height=float(np.median([observed_size(o)[0] for i,o in rows]));self.width=float(np.median([observed_size(o)[1] for i,o in rows]))
        self.body_history.append(dict(timestamp=timestamp,center=self.center.tolist(),observed=True))
        rotations=[np.asarray(h['rotation_matrix'])[:,1] for i,o in rows for h in (o.get('pose') or {}).get('hypotheses',[]) if np.asarray(h.get('rotation_matrix')).shape==(3,3)]
        if rotations:
            self.axis_history.extend((timestamp,r) for r in rotations)
            while self.axis_history and timestamp-self.axis_history[0][0]>1.2:self.axis_history.popleft()
            measured=np.median([r for t,r in self.axis_history],axis=0)
            if measured[1]<0:measured=-measured
            measured/=np.linalg.norm(measured)
            # The lamp-axis direction is an image-derived plane hint, not an
            # IMU attitude. Keep its provenance separate from metric PnP.
            self.axis=measured if timestamp-self.created<.65 else self.axis+(1-math.exp(-max(dt,.001)/.6))*(measured-self.axis)
            self.axis/=np.linalg.norm(self.axis)
        front=np.array([0.,0.,-1.]);front-=self.axis*np.dot(front,self.axis);front/=np.linalg.norm(front)
        side=np.cross(self.axis,front)
        if side[0]<0:side=-side
        k=np.asarray(calibration['camera_matrix']);new=[]
        for index,o in rows:
            pose=o.get('pose') or {};choices=[]
            for h in pose.get('hypotheses',[]):
                n=np.asarray(h['outward_normal'],float);n-=self.axis*np.dot(n,self.axis)
                if np.linalg.norm(n)<.7:continue
                n/=np.linalg.norm(n);position=np.asarray(h['tvec_m']);center=position-radius*n
                if center[2]<=0:continue
                pixel=(k@center)[:2]/center[2];cost=abs(pixel[0]-measurement[0])+.3*abs(pixel[1]-measurement[1])
                choices.append((cost,n,position,np.asarray(h['outward_normal'],float)))
            if not choices:continue
            cost,n,pos,raw_normal=min(choices,key=lambda c:c[0]);yaw=math.atan2(np.dot(n,side),np.dot(n,front))
            if cost>max(100.,6*self.height):continue
            item=dict(t=timestamp,yaw=yaw,normal=raw_normal.tolist(),pixel=list(o['center']),cost=cost,index=index,height=self.height,width=self.width)
            self.history.append(item);new.append(item)
        # A slow rotor exposes less phase in 650 ms, especially when lamps
        # blink. Retain longer measured history until a fast rate is established.
        history_span=.65 if self.model is not None and abs(self.model['omega'])>=4. else 1.2
        while self.history and timestamp-self.history[0]['t']>history_span:self.history.popleft()
        self.poses=new
        if self.model is not None and new:
            old=self.model;age=timestamp-old['timestamp'];phase=old['phase']+old['omega']*age+.5*old['alpha']*age*age
            period=2*math.pi/plate_count;consistent=False
            for item in new:
                phi=phase+round((item['yaw']-phase)/period)*period
                expected=np.array([np.array([1.,age,math.sin(phi)])@old['coefficients'][0],np.array([1.,age,math.cos(phi)])@old['coefficients'][1]])
                if abs(folded(item['yaw']-phase,period))<=.35 or np.linalg.norm(expected-item['pixel'])<=max(12.,old['width']*.7):consistent=True
            # Current image features can validate continuity when PnP chooses
            # the other planar solution. They do not supply physical corners.
            if not consistent:
                for slot in range(plate_count):
                    phi=phase+slot*period
                    if math.cos(phi)<.35:continue
                    expected=np.array([np.array([1.,age,math.sin(phi)])@old['coefficients'][0],np.array([1.,age,math.cos(phi)])@old['coefficients'][1]])
                    if any(np.linalg.norm(expected-np.asarray(o['center']))<max(6.,old['width']*.3) for _,o in rows):consistent=True
            if not consistent:
                self.model=None;self.generation+=1
                while self.history and timestamp-self.history[0]['t']>.12:self.history.popleft()
        # Count capture instants, not boards. Six distinct views allow the
        # short-window fit at ordinary preview rates; phase alias separation,
        # residual and propagated uncertainty still determine admission.
        if len({s['t'] for s in self.history})>=6 and timestamp-self.history[0]['t']>=.35:
            self._fit(timestamp,plate_count)
        if self.model is not None and timestamp-self.model['timestamp']>1e-6:
            self._update_image_state(rows,timestamp,plate_count)

    def _update_image_state(self,rows,timestamp,plate_count):
        """Update the joint image state with fresh pixels, without invented yaw."""
        m=self.model;dt=timestamp-m['timestamp']
        if not 0<dt<=.30 or m.get('joint_covariance') is None:return
        x=np.r_[m['phase'],m['omega'],m['alpha'],m['coefficients'][0],m['coefficients'][1]]
        f=np.eye(9);f[0,1]=dt;f[0,2]=dt*dt/2;f[1,2]=dt;f[3,4]=dt;f[6,7]=dt
        x=f@x;p=f@np.asarray(m['joint_covariance'])@f.T
        jerk=10.**2*np.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
        p[:3,:3]+=jerk
        for position,velocity in ((3,4),(6,7)):
            p[np.ix_([position,velocity],[position,velocity])]+=100.*np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]])
        p[5,5]+=dt*.25;p[8,8]+=dt*.25
        candidates=[]
        for index,o in rows:
            for slot in range(plate_count):
                theta=x[0]+slot*2*math.pi/plate_count;sn=math.sin(theta);cs=math.cos(theta)
                if cs<.35:continue
                predicted=np.array([x[3]+x[5]*sn,x[6]+x[8]*cs]);error=np.asarray(o['center'])-predicted
                if np.linalg.norm(error)>max(8.,m['width']*.40):continue
                h=np.zeros((2,9));h[0,[0,3,5]]=[x[5]*cs,1.,sn];h[1,[0,6,8]]=[-x[8]*sn,1.,cs]
                noise=np.eye(2)*(4. if o.get('geometry_valid') else 6.)**2
                s=h@p@h.T+noise;distance=float(error@np.linalg.solve(s,error))
                if distance<6.:candidates.append((distance,index,slot,error,h,noise,s))
        if not candidates:return
        candidates.sort(key=lambda c:c[0]);best=candidates[0]
        if len(candidates)>1 and candidates[1][2]!=best[2] and candidates[1][0]-best[0]<.5:return
        _,_,_,error,h,noise,s=best;k=np.linalg.solve(s,h@p).T;correction=k@error
        if abs(correction[0])>.12 or abs(correction[1])>.5 or abs(correction[2])>2:return
        x+=correction
        if not .5<=abs(x[1])<=16 or abs(x[2])>20 or x[5]<.5*self.height:return
        identity=np.eye(9)-k@h;p=identity@p@identity.T+k@noise@k.T;p=(p+p.T)/2
        self.model={**m,'timestamp':timestamp,'phase':float(x[0]),'omega':float(x[1]),'alpha':float(x[2]),
            'coefficients':[x[3:6].tolist(),x[6:9].tolist()],
            'joint_covariance':p.tolist(),'angular_covariance':p[:3,:3].tolist(),
            'measurement_kind':'current_image_joint_state','samples':m['samples']+1,'span_s':m['span_s']+dt}

    def _fit(self,timestamp,plate_count):
        history=list(self.history);dt=np.array([s['t']-timestamp for s in history]);angles=np.array([s['yaw'] for s in history]);pixels=np.array([s['pixel'] for s in history])
        reference_height=float(np.percentile([s['height'] for s in history],75))
        front=np.array([0.,0.,-1.]);front-=self.axis*np.dot(front,self.axis);front/=np.linalg.norm(front)
        side=np.cross(self.axis,front)
        if side[0]<0:side=-side
        normals=np.asarray([s['normal'] for s in history]);normals-=np.outer(normals@self.axis,self.axis)
        angles=np.arctan2(normals@side,normals@front)
        weights=np.array([max(.1,1-s['cost']/max(100,6*self.height)) for s in history]);rates=np.arange(-16.,16.001,.1)
        # No greedy unwrap across a missing frame: compare all bounded rates.
        z=np.exp(plate_count*1j*(angles[:,None]-dt[:,None]*rates[None,:]));means=(z*weights[:,None]).sum(axis=0)/weights.sum();scores=np.abs(means)
        best=int(np.argmax(scores));omega=float(rates[best]);score=float(scores[best]);gap=score-float(max(scores[np.abs(rates-omega)>2]))
        self.fit_diagnostic=dict(reason='phase_support',score=score,gap=gap,omega=omega)
        if score<.72 or gap<.08 or abs(omega)<.5:return
        period=2*math.pi/plate_count;phase=float(np.angle(means[best])/plate_count)
        unwrapped=angles+np.rint((phase+omega*dt-angles)/period)*period
        design=np.column_stack((np.ones(len(dt)),dt,dt*dt/2))
        gram=design.T@(design*weights[:,None]);gram[2,2]+=.02
        angular=np.linalg.solve(gram,design.T@(weights*unwrapped))
        self.fit_diagnostic.update(reason='angular_bounds',angular=angular.tolist())
        if abs(angular[1])>16 or abs(angular[2])>20:return
        residual=folded(angles-design@angular,period);sigma=max(.035,float(np.median(abs(residual))*1.4826))
        covariance=np.linalg.inv(gram)*sigma*sigma
        if self.model is not None:
            old=self.model;forecast=old['phase']+old['omega']*(timestamp-old['timestamp'])+.5*old['alpha']*(timestamp-old['timestamp'])**2
            angular[0]+=round((forecast-angular[0])/period)*period
        denoised=design@angular+np.rint((angles-design@angular)/period)*period
        coeff=[];cov=[];stds=[]
        for axis,term in enumerate((np.sin(denoised),np.cos(denoised))):
            matrix=np.column_stack((np.ones(len(dt)),dt,term));robust=weights.copy()
            for _ in range(3):
                normal=matrix.T@(matrix*robust[:,None]);normal[1,1]+=.01
                beta=np.linalg.solve(normal,matrix.T@(pixels[:,axis]*robust));res=pixels[:,axis]-matrix@beta
                std=max(2.,float(np.median(abs(res))*1.4826));robust=weights*np.minimum(1.,2.5*std/np.maximum(abs(res),1e-6))
            self.fit_diagnostic.update(reason='pixel_bounds',axis=axis,beta=beta.tolist(),height=self.height)
            if abs(beta[1])>10*reference_height or abs(beta[2])>(7 if axis==0 else 4)*reference_height:return
            coeff.append(beta.tolist());cov.append((np.linalg.inv(normal)*std*std).tolist());stds.append(std)
        self.fit_diagnostic.update(reason='pixel_residual',coeff=coeff,stds=stds)
        if coeff[0][2]<.5*reference_height or max(stds)>max(8.,reference_height*.65):return
        # Fit angle and pixels together. Their parameter errors are correlated:
        # adding three independent uncertainty radii double-counts the same
        # image error. Preserve the full joint parameter covariance instead.
        prior=angular.copy();precision=np.linalg.inv(covariance+np.eye(3)*1e-9)
        params=np.r_[angular,coeff[0],coeff[1]];slots=np.rint((angles-design@angular)/period)*period
        def equations(beta):
            theta=design@beta[:3]+slots;sn=np.sin(theta);cs=np.cos(theta)
            predictions=np.column_stack((beta[3]+beta[4]*dt+beta[5]*sn,beta[6]+beta[7]*dt+beta[8]*cs))
            jx=np.column_stack((design*(beta[5]*cs)[:,None],np.ones(len(dt)),dt,sn,np.zeros((len(dt),3))))
            jy=np.column_stack((design*(-beta[8]*sn)[:,None],np.zeros((len(dt),3)),np.ones(len(dt)),dt,cs))
            residuals=pixels-predictions;normal=np.zeros((9,9));normal[:3,:3]=precision
            right=np.zeros(9);right[:3]=precision@(prior-beta[:3])
            for axis,jac in enumerate((jx,jy)):
                robust=weights*np.minimum(1.,2.5*stds[axis]/np.maximum(abs(residuals[:,axis]),1e-6))/stds[axis]**2
                normal+=jac.T@(jac*robust[:,None]);right+=jac.T@(robust*residuals[:,axis])
            normal+=np.diag([0,0,0,1e-6,.001,1e-6,1e-6,.001,1e-6])
            return normal,right,residuals
        for _ in range(4):
            normal,right,residuals=equations(params);step=np.linalg.solve(normal,right)
            damp=min(1.,.08/max(abs(step[0]),1e-9),.5/max(abs(step[1]),1e-9),3/max(abs(step[2]),1e-9))
            params+=step*damp
        normal,right,residuals=equations(params);joint_covariance=np.linalg.inv(normal)
        if abs(params[0]-prior[0])>.25 or abs(params[1]-prior[1])>1.5 or abs(params[2])>20:return
        angular=params[:3];coeff=[params[3:6].tolist(),params[6:9].tolist()]
        covariance=joint_covariance[:3,:3]
        self.fit_diagnostic['reason']='accepted'
        self.model=dict(timestamp=timestamp,phase=float(angular[0]),omega=float(angular[1]),alpha=float(angular[2]),
            angular_covariance=covariance.tolist(),coefficients=coeff,pixel_covariance=cov,residual_px=stds,
            joint_covariance=joint_covariance.tolist(),uncertainty_kind='joint_parameter_model_not_calibrated_probability',
            samples=len(history),span_s=timestamp-history[0]['t'],fit_score=score,rate_peak_gap=gap,
            width=min(4*reference_height,float(np.percentile([s['width']/max(.5,math.cos(s['yaw'])) for s in history],75))),
            height=reference_height,shape_kind='past_measured_frontal_reference_for_prediction_only',axis_camera=self.axis.tolist())

    def snapshot(self):
        return dict(robot_id=self.id,color=self.color,center=self.center.tolist(),velocity=self.velocity.tolist(),
            timestamp=self.seen,observation_indices=list(self.indices),model=copy.deepcopy(self.model),height=self.height,width=self.width,
            fit_diagnostic=copy.deepcopy(self.fit_diagnostic),history=list(self.body_history))


class VisualRotorBank:
    def __init__(self,settings):
        self.settings=settings;self.robots=[];self.next_id=1;self.target=None;self.timestamp=None;self.segment=None;self.target_since=None

    def update(self,observations,tracks,timestamp,calibration):
        if self.timestamp==timestamp:return
        if self.timestamp is not None and timestamp<self.timestamp:raise ValueError('rotor measurements must follow capture time')
        segments={t['segment'] for t in tracks if 'segment' in t}
        if len(segments)==1:
            segment=next(iter(segments))
            if self.segment is not None and segment!=self.segment:self.robots=[];self.target=None
            self.segment=segment
        groups=[]
        valid=[]
        for i,original in enumerate(observations):
            o=original
            if o.get('color')=='unknown' and o.get('fusion_evidence',{}).get('current_image_measurement'):
                region=o['fusion_evidence'].get('robot_region');center=(np.asarray(region[:2])+region[2:])/2 if region else np.asarray(o['center'])
                nearby=[r for r in self.robots if timestamp-r.seen<=.3 and np.linalg.norm(center-r.identity_center)<max(40.,3*r.height)]
                if len(nearby)==1:o={**o,'color':nearby[0].color}
            if o['color'] in ('red','blue') and (
               (o.get('geometry_valid') and o.get('corners') is not None and (o.get('geometry_score') or 0)>=.52 or
                o.get('fusion_evidence',{}).get('current_image_measurement') and o.get('score',0)>=.45)):
                valid.append((i,o))
        for index,o in sorted(valid,key=lambda p:-(p[1]['bbox'][2]-p[1]['bbox'][0])*(p[1]['bbox'][3]-p[1]['bbox'][1])):
            h=observed_size(o)[0];center=np.asarray(o['center']);group=next((g for g in groups if g[0][1]['color']==o['color'] and
                (o.get('fusion_evidence',{}).get('robot_region') is not None and g[0][1].get('fusion_evidence',{}).get('robot_region') is not None and _iou(o['fusion_evidence']['robot_region'],g[0][1]['fusion_evidence']['robot_region'])>.15 or
                np.linalg.norm(center-np.mean([q['center'] for i,q in g],axis=0))<7*max(h,np.median([observed_size(q)[0] for i,q in g])))),None)
            if group is None:groups.append([(index,o)])
            else:group.append((index,o))
        self.robots=[r for r in self.robots if timestamp-r.seen<=1.]
        for r in self.robots:r.indices=[]
        used=set()
        for group in groups:
            color=group[0][1]['color'];center=np.mean([o['center'] for i,o in group],axis=0)
            regions=[np.asarray(o['fusion_evidence']['robot_region']) for i,o in group if o.get('fusion_evidence',{}).get('robot_region') is not None]
            identity=np.mean([(b[:2]+b[2:])/2 for b in regions],axis=0) if regions else center
            gate=max(100.,float(np.median([b[2]-b[0] for b in regions]))*.65 if regions else 0.)
            choices=[r for r in self.robots if r.color==color and r.id not in used and np.linalg.norm(identity-r.identity_center)<max(gate,9*r.height)]
            r=min(choices,key=lambda r:np.linalg.norm(identity-r.identity_center)) if choices else Robot(self.next_id,color,center,timestamp)
            if not choices:self.robots.append(r);self.next_id+=1
            r.identity_center=identity.copy()
            r.update(group,timestamp,calibration,self.settings['plate_count'],self.settings['radius_m']);used.add(r.id)
        previous_target=self.target
        requested_robot=self.settings.get('target_robot')
        if requested_robot is not None:self.target=requested_robot
        requested=self.settings.get('target_track')
        if requested is not None and self.target is None:
            index=next((t['observation_index'] for t in tracks if t['track_id']==requested and t.get('observed')),None)
            self.target=next((r.id for r in self.robots if index in r.indices),None)
        if requested_robot is None and (self.target is None or not any(r.id==self.target for r in self.robots)):
            candidates=[r for r in self.robots if r.indices]
            if candidates:self.target=max(candidates,key=lambda r:r.height*r.width).id
        if self.target!=previous_target or self.target_since is None:self.target_since=timestamp
        if requested_robot is None and requested is None:
            current=next((r for r in self.robots if r.id==self.target),None)
            last_fit=current.model['timestamp'] if current and current.model else self.target_since
            alternatives=[r for r in self.robots if r.id!=self.target and r.indices and r.model is not None and timestamp-r.model['timestamp']<=.10]
            # A static/occluded initial target cannot monopolize automatic
            # selection. Explicit user locks never enter this branch.
            if alternatives and timestamp-last_fit>.5 and timestamp-self.target_since>.6:
                self.target=max(alternatives,key=lambda r:r.height*r.width*r.model['fit_score']).id
                self.target_since=timestamp
        self.timestamp=timestamp

    def publish(self,enabled,horizon,physical,now=None,image_size=None):
        robots=[r.snapshot() for r in self.robots if self.timestamp-r.seen<=.30 and len({t for t,p in r.positions})>=3]
        selected=next((r for r in robots if r['robot_id']==self.target),None)
        model=selected.get('model') if selected else None
        state=None;covariance=np.zeros((6,6))
        if model is not None:
            state=[*selected['center'],0.,model['phase'],model['omega'],model['alpha']];covariance[3:,3:]=np.asarray(model['angular_covariance'])
        result=dict(schema='astra.rotor-visual.v1',enabled=enabled,robots=robots,robot_id=self.target,model=model,
            state=state,covariance=covariance.tolist(),state_timestamp=model['timestamp'] if model else self.timestamp,
            last_observation_timestamp=selected['timestamp'] if selected else None,samples=model['samples'] if model else 0,
            measurement_span_s=model['span_s'] if model else 0.,coordinate_frame='image_pixels; phase relative to measured lamp-axis plane',
            filter_kind='robot_body_continuity_and_wrapped_phase_fit',selected_horizon_s=horizon,
            config=dict(plate_count=self.settings['plate_count'],min_observations=6,min_duration_s=.35,max_gap_s=.16,
                max_prediction_horizon_s=.8,unknown_acceleration_bound_rad_s2=10.,max_face_angle_deg=28.),
            decision=dict(allow_fire=False,selected_plate=None,candidates=[],reason='visual_preview_only'),
            physical_rotation=physical,calibration=physical['calibration'],metric_valid=False)
        result['decision']=project_visual(result,self.timestamp if now is None else now,horizon,image_size)
        return result


def project_visual(frame,timestamp,horizon,image_size=None):
    base=dict(enabled=bool(frame and frame.get('enabled')),timestamp=timestamp,target_timestamp=timestamp+horizon,
        candidates=[],selected_plate=None,selected_observation_index=None,allow_fire=False,visual_ready=False,
        fire_admission='visual_pixel_selection_only')
    if not base['enabled']:return dict(base,reason='prediction_disabled')
    robot=next((r for r in frame['robots'] if r['robot_id']==frame['robot_id']),None)
    if robot and robot['observation_indices']:base['selected_observation_index']=robot['observation_indices'][0]
    model=frame.get('model')
    if model is None:return dict(base,reason='rotation_not_stable')
    dt=timestamp+horizon-model['timestamp'];age=timestamp-robot['timestamp']
    if dt<0:return dict(base,reason='clock_mismatch')
    if age>frame['config']['max_gap_s'] or timestamp-model['timestamp']>.16:return dict(base,reason='stale_observation')
    if dt>frame['config']['max_prediction_horizon_s']:return dict(base,reason='prediction_expired')
    phase=model['phase']+model['omega']*dt+.5*model['alpha']*dt*dt
    angular_vector=np.array([1.,dt,.5*dt*dt]);angular_std=math.sqrt(max(0.,angular_vector@np.asarray(model['angular_covariance'])@angular_vector))
    maneuver=.5*frame['config']['unknown_acceleration_bound_rad_s2']*dt*dt
    axis=np.asarray(model['axis_camera']);front=np.array([0.,0.,-1.]);front-=axis*np.dot(front,axis);front/=np.linalg.norm(front)
    side=np.cross(axis,front)
    if side[0]<0:side=-side
    k=np.asarray(frame['calibration']['camera_matrix']);view=-np.linalg.solve(k,np.r_[robot['center'],1.]);view-=axis*np.dot(view,axis);view/=np.linalg.norm(view)
    count=frame['config']['plate_count']
    for slot in range(count):
        phi=phase+slot*2*math.pi/count;normal=front*math.cos(phi)+side*math.sin(phi)
        terms=(math.sin(phi),math.cos(phi));derivatives=(math.cos(phi),-math.sin(phi));pixel=[];bounds=[]
        for i in range(2):
            vector=np.array([1.,dt,terms[i]]);coeff=np.asarray(model['coefficients'][i]);pixel.append(float(vector@coeff))
            sigma=math.sqrt(max(0.,vector@np.asarray(model['pixel_covariance'][i])@vector))
            slope=abs(coeff[2]*derivatives[i])
            if model.get('joint_covariance') is not None:
                jacobian=np.zeros(9);jacobian[:3]=angular_vector*coeff[2]*derivatives[i]
                jacobian[3+i*3:6+i*3]=vector
                sigma=math.sqrt(max(0.,jacobian@np.asarray(model['joint_covariance'])@jacobian))
                bounds.append(2*sigma+slope*maneuver)
            else:bounds.append(2*sigma+2*slope*angular_std+slope*maneuver)
        face=float(normal@view);face_ok=face>=math.cos(math.radians(frame['config']['max_face_angle_deg']))
        future_width=model['width']*max(.3,face)
        uncertain=bounds[0]>max(8.,future_width*.55) or bounds[1]>max(5.,model['height']*.60)
        inside=image_size is None or 0<=pixel[0]<image_size[0] and 0<=pixel[1]<image_size[1]
        valid=face_ok and not uncertain and inside
        base['candidates'].append(dict(plate=slot,center=pixel if valid else None,candidate_center=pixel,
            facing_cos=face,uncertainty_px=bounds,valid=valid,reason='ready' if valid else 'outside_image' if not inside else 'prediction_uncertain' if uncertain else 'waiting_for_window'))
    selected=max(base['candidates'],key=lambda p:p['facing_cos']);base['selected_plate']=selected['plate']
    base['visual_ready']=selected['valid']
    return dict(base,reason=selected['reason'])
