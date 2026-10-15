/* Copy-only angular CA projection. Display points never enter a measurement. */
(() => {
  'use strict';
  function strict(result = {tracks:[]}) {
    return {...result,tracks:(result.tracks||[]).map(p=>p.status==='ready'?p:{...p,status:'waiting',center:null,usable:false})};
  }
  function project(frame,timestamp,horizon,imageSize) {
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
