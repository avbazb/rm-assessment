/* Copy-only angular CA projection. Display points never enter a measurement. */
(() => {
  'use strict';
  function strict(result = {tracks:[]}) {
    return {...result,tracks:(result.tracks||[]).map(p=>p.status==='ready'?p:{...p,status:'waiting',center:null,usable:false})};
  }
  function visual(frame,timestamp,horizon,imageSize) {
    const base={enabled:!!frame?.enabled,timestamp,target_timestamp:timestamp+horizon,candidates:[],selected_plate:null,selected_observation_index:null,allow_fire:false,visual_ready:false};
    if(!base.enabled)return {...base,reason:'prediction_disabled'};
    const robot=frame.robots.find(r=>r.robot_id===frame.robot_id),m=frame.model;
    if(robot?.observation_indices.length)base.selected_observation_index=robot.observation_indices[0];
    if(!m)return {...base,reason:'rotation_not_stable'};
    const dt=timestamp+horizon-m.timestamp,age=timestamp-robot.timestamp,cfg=frame.config;
    if(dt<0)return {...base,reason:'clock_mismatch'};
    if(age>cfg.max_gap_s||timestamp-m.timestamp>.16)return {...base,reason:'stale_observation'};
    if(dt>cfg.max_prediction_horizon_s)return {...base,reason:'prediction_expired'};
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),normalize=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);};
    const covariance=(a,p)=>a.reduce((s,x,i)=>s+x*a.reduce((v,y,j)=>v+p[i][j]*y,0),0);
    const a=[1,dt,dt*dt/2],phase=m.phase+m.omega*dt+m.alpha*dt*dt/2;
    const angularStd=Math.sqrt(Math.max(0,covariance(a,m.angular_covariance))),maneuver=.5*cfg.unknown_acceleration_bound_rad_s2*dt*dt;
    const axis=m.axis_camera;let front=[0,0,-1];const v=dot(front,axis);front=normalize(front.map((x,i)=>x-axis[i]*v));
    let side=[axis[1]*front[2]-axis[2]*front[1],axis[2]*front[0]-axis[0]*front[2],axis[0]*front[1]-axis[1]*front[0]];if(side[0]<0)side=side.map(x=>-x);
    const k=frame.calibration.camera_matrix;let view=[-(robot.center[0]-k[0][2])/k[0][0],-(robot.center[1]-k[1][2])/k[1][1],-1];const av=dot(view,axis);view=normalize(view.map((x,i)=>x-axis[i]*av));
    for(let slot=0;slot<cfg.plate_count;slot++){
      const phi=phase+slot*2*Math.PI/cfg.plate_count,sn=Math.sin(phi),cs=Math.cos(phi),normal=front.map((x,i)=>x*cs+side[i]*sn);
      const terms=[sn,cs],derivatives=[cs,-sn],pixel=[],bounds=[];
      for(let i=0;i<2;i++){
        const vector=[1,dt,terms[i]],coeff=m.coefficients[i],slope=Math.abs(coeff[2]*derivatives[i]);pixel.push(dot(vector,coeff));
        let sigma=Math.sqrt(Math.max(0,covariance(vector,m.pixel_covariance[i])));
        if(m.joint_covariance){const jac=Array(9).fill(0);for(let j=0;j<3;j++){jac[j]=a[j]*coeff[2]*derivatives[i];jac[3+i*3+j]=vector[j];}sigma=Math.sqrt(Math.max(0,covariance(jac,m.joint_covariance)));bounds.push(2*sigma+slope*maneuver);}
        else bounds.push(2*sigma+2*slope*angularStd+slope*maneuver);
      }
      const face=dot(normal,view),faceOk=face>=Math.cos(cfg.max_face_angle_deg*Math.PI/180),futureWidth=m.width*Math.max(.3,face);
      const uncertain=bounds[0]>Math.max(8,futureWidth*.55)||bounds[1]>Math.max(5,m.height*.60),inside=!imageSize||pixel[0]>=0&&pixel[0]<imageSize[0]&&pixel[1]>=0&&pixel[1]<imageSize[1];
      const valid=faceOk&&!uncertain&&inside;base.candidates.push({plate:slot,center:valid?pixel:null,candidate_center:pixel,facing_cos:face,uncertainty_px:bounds,valid,reason:valid?'ready':!inside?'outside_image':uncertain?'prediction_uncertain':'waiting_for_window'});
    }
    const selected=base.candidates.reduce((a,b)=>a.facing_cos>b.facing_cos?a:b);base.selected_plate=selected.plate;base.visual_ready=selected.valid;base.reason=selected.reason;return base;
  }
  function project(frame,timestamp,horizon,imageSize) {
    if(frame?.schema==='astra.rotor-visual.v1')return visual(frame,timestamp,horizon,imageSize);
    const base={enabled:!!frame?.enabled, timestamp, target_timestamp:timestamp+horizon, candidates:[],selected_plate:null,allow_fire:false};
    if (!frame?.enabled) return {...base,reason:'prediction_disabled'};
    if (!frame.state || !frame.covariance) return {...base,reason:'waiting_for_observations'};
    const cfg=frame.config,age=timestamp-frame.last_observation_timestamp,dt=timestamp+horizon-frame.state_timestamp;
    if (age<-.000002||dt<0) return {...base,reason:'clock_mismatch'};
    if (age>cfg.max_gap_s) return {...base,reason:'stale_observation'};
    if (dt>cfg.max_prediction_horizon_s) return {...base,reason:'prediction_expired'};
    if (frame.samples<cfg.min_observations || frame.measurement_span_s<cfg.min_duration_s || Math.sqrt(frame.covariance[4][4])>1)
      return {...base,reason:'rotation_not_stable'};
    const s=frame.state,phase=s[3]+s[4]*dt+s[5]*dt*dt/2;
    const g=[1,dt,dt*dt/2];let variance=cfg.angular_jerk_std**2*dt**5/20;
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)variance+=g[i]*frame.covariance[i+3][j+3]*g[j];
    const envelope=.5*(cfg.unknown_acceleration_bound_rad_s2+Math.abs(s[5]))*dt*dt;
    const k=frame.calibration.camera_matrix,dist=frame.calibration.distortion_coefficients;
    const pixel=(p)=>{
      const x=-p[1]/p[0],y=-p[2]/p[0],r=x*x+y*y;
      if(!(p[0]>0))return null;
      const a=1+(dist[0]||0)*r+(dist[1]||0)*r*r+(dist[4]||0)*r*r*r;
      const b=1+(dist[5]||0)*r+(dist[6]||0)*r*r+(dist[7]||0)*r*r*r;
      // Higher-order thin-prism/tilt calibration is handled by OpenCV only.
      if(dist.length>8 || Math.abs(b)<1e-8)return null;
      const xx=x*a/b+2*(dist[2]||0)*x*y+(dist[3]||0)*(r+2*x*x);
      const yy=y*a/b+(dist[2]||0)*(r+2*y*y)+2*(dist[3]||0)*x*y;
      const q=[k[0][0]*xx+k[0][2],k[1][1]*yy+k[1][2]];
      return q.every(Number.isFinite)&&q[0]>=0&&q[0]<imageSize[0]&&q[1]>=0&&q[1]<imageSize[1]?q:null;
    };
    for(let slot=0;slot<cfg.plate_count;slot++) {
      const theta=phase+slot*2*Math.PI/cfg.plate_count,n=[Math.cos(theta),Math.sin(theta),0];
      const p=s.slice(0,3).map((v,i)=>v+cfg.radius_m*n[i]);
      const facing=-(n[0]*p[0]+n[1]*p[1])/Math.hypot(...p);
      const uncertainty=2*Math.sqrt(Math.max(frame.covariance[0][0],frame.covariance[1][1])+cfg.radius_m**2*Math.max(0,variance))+cfg.radius_m*envelope;
      const valid=facing>=Math.cos(cfg.max_face_angle_deg*Math.PI/180)&&uncertainty<cfg.hit_width_m*Math.max(0,facing)/2;
      const q=valid?pixel(p):null;
      base.candidates.push({plate:slot,position:p,center:q,valid:!!q,facing_cos:facing,uncertainty_m:uncertainty});
    }
    const usable=base.candidates.filter(c=>c.valid).sort((a,b)=>b.facing_cos-a.facing_cos);
    const selected=usable.find(c=>c.plate===frame.decision?.selected_plate)||usable[0];
    base.selected_plate=selected?.plate??null;
    // This field is the original ballistic-arrival decision, not the user-
    // horizon projection; it is invalidated as soon as its source frame ages.
    const decisionAge=timestamp-frame.decision?.timestamp;
    base.allow_fire=!!frame.decision?.allow_fire&&decisionAge>=-.005&&decisionAge<=.05;
    base.reason=!usable.length?'waiting_for_window':!frame.metric_valid?'camera_calibration_required':base.allow_fire?'ready':'waiting_for_window';
    return base;
  }
  const api={strict,project};if(typeof module!=='undefined'&&module.exports)module.exports=api;else globalThis.AstraSpin=api;
})();
