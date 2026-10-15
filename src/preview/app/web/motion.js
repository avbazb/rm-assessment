/* Propagate the returned Kalman mixtures on the media clock, without new measurements. */
(() => {
  "use strict";
  const finite = Number.isFinite;
  const eye = () => Array.from({length: 6}, (_, i) => Array.from({length: 6}, (_, j) => Number(i === j)));
  function propagate(model, dt, e) {
    const f = eye(), q = Array.from({length: 6}, () => Array(6).fill(0));
    f[0][2] = f[1][3] = dt;
    let g, indices, variance;
    if (model.kind === "ca") {
      f[0][4] = f[1][5] = dt * dt / 2; f[2][4] = f[3][5] = dt;
      g = [dt ** 3 / 6, dt * dt / 2, dt]; indices = [[0, 2, 4], [1, 3, 5]];
      variance = (e.jerk_std_px_s3 * e.scale) ** 2;
    } else {
      f[4].fill(0); f[5].fill(0);
      g = [dt * dt / 2, dt]; indices = [[0, 2], [1, 3]];
      variance = (e.acceleration_std_px_s2 * e.scale) ** 2;
      q[4][4] = q[5][5] = variance;
    }
    for (const axis of indices) for (let i = 0; i < g.length; i++) for (let j = 0; j < g.length; j++) q[axis[i]][axis[j]] = g[i] * g[j] * variance;
    const mean = f.map(row => row.reduce((sum, value, i) => sum + value * model.mean[i], 0));
    // Only the two-dimensional position covariance is needed for publication.
    const covariance = [0, 1].map(i => [0, 1].map(j => {
      let sum = q[i][j];
      for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) sum += f[i][a] * model.covariance[a][b] * f[j][b];
      return sum;
    }));
    return {mean, covariance, weight: model.weight};
  }
  function forecast(estimates, targetTimestamp, imageSize) {
    const rows = [];
    for (const e of estimates || []) {
      if (!Array.isArray(e.models) || !e.models.length || !finite(e.state_timestamp)) continue;
      const dt = targetTimestamp - e.state_timestamp;
      let reason = dt < -0.000001 ? "clock_mismatch" : dt > e.max_horizon_s + 0.000001 ? "prediction_expired" : !e.confirmed ? "tentative" : e.samples < e.min_samples || e.measurement_span_s + 0.000000001 < e.min_span_s ? "insufficient_measurements" : null;
      const base = {track_id: e.track_id, segment: e.segment, target_timestamp: targetTimestamp, observation_timestamp: e.last_observation_timestamp, horizon_s: dt};
      if (["clock_mismatch", "prediction_expired"].includes(reason)) { rows.push({...base, status: "waiting", reason, center: null}); continue; }
      const models = e.models.map(m => propagate(m, Math.max(0, dt), e));
      const x = Array.from({length: 6}, (_, i) => models.reduce((sum, m) => sum + m.weight * m.mean[i], 0));
      const p = [0, 1].map(i => [0, 1].map(j => models.reduce((sum, m) => sum + m.weight * (m.covariance[i][j] + (m.mean[i] - x[i]) * (m.mean[j] - x[j])), 0)));
      const eigen = (p[0][0] + p[1][1] + Math.sqrt((p[0][0] - p[1][1]) ** 2 + 4 * p[0][1] * p[1][0])) / 2;
      const radius = Math.sqrt(Math.max(0, eigen)) * 2.44774683068;
      const anchor = e.last_observed_center || e.center;
      const displacement = Math.hypot(x[0] - anchor[0], x[1] - anchor[1]);
      const limit = e.max_extrapolation_displacement_px ?? Math.max(24 * e.scale, Math.hypot(...e.size));
      if (!reason && displacement > limit) reason = "unsupported_displacement";
      if (!reason && radius > Math.max(4, Math.min(...e.size) / 2)) reason = "uncertain";
      const inside = !imageSize || x[0] >= 0 && x[1] >= 0 && x[0] < imageSize[0] && x[1] < imageSize[1];
      if (!reason && !inside) reason = "outside_image";
      const visible = inside && (!reason || ["uncertain", "unsupported_displacement"].includes(reason) && displacement <= Math.max(3 * limit, 240 * e.scale));
      rows.push({...base, status: !reason ? "ready" : visible ? "estimated" : "waiting", usable: !reason, reason,
        center: visible ? x.slice(0, 2) : null, candidate_center: x.slice(0, 2), velocity: x.slice(2, 4), acceleration: x.slice(4),
        uncertainty_radius_px: radius, position_covariance: p, displacement_px: displacement, max_extrapolation_displacement_px: limit});
    }
    return {schema: "astra.pixel-forecast.v1", target_timestamp: targetTimestamp, tracks: rows};
  }
  function alignTrack(track, timestamp, imageSize) {
    const dt = timestamp - (track.state_timestamp ?? track.timestamp);
    if (!finite(dt) || dt < -0.000001 || dt > 0.25 || !track.center?.every(finite) || !track.velocity?.every(finite)) return null;
    const dx = track.velocity[0] * Math.max(0, dt), dy = track.velocity[1] * Math.max(0, dt);
    const limit = track.max_coasting_displacement_px ?? Math.max(12, Math.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]) / 2);
    const center = [track.center[0] + dx, track.center[1] + dy];
    if (Math.hypot(dx, dy) > limit || center[0] < 0 || center[1] < 0 || center[0] >= imageSize[0] || center[1] >= imageSize[1]) return null;
    return {...track, center, bbox: track.bbox.map((v, i) => v + (i % 2 ? dy : dx)), display_age_s: dt};
  }
  // Pixel matching is for drawing only. It never creates tracks, measured
  // lamp endpoints, PnP inputs, or hit-point predictions.
  function overlayPixels(rgba, width, height, imageWidth, imageHeight) {
    const red = new Float32Array(width * height), blue = new Float32Array(width * height);
    for (let i = 0, j = 0; i < red.length; i++, j += 4) {
      const light = (rgba[j] + 2 * rgba[j + 1] + rgba[j + 2]) / 4;
      const color = .4 * (rgba[j] - rgba[j + 2]);
      red[i] = light + color; blue[i] = light - color;
    }
    return {red, blue, width, height, scaleX: width / imageWidth, scaleY: height / imageHeight, templates: new Map()};
  }
  function templateFor(reference, obs) {
    const key = `${obs.color}:${obs.bbox.join(',')}`;
    if (reference.templates.has(key)) return reference.templates.get(key);
    const values = obs.color === 'red' ? reference.red : reference.blue;
    const [a,b,c,d] = obs.bbox, sx = reference.scaleX, sy = reference.scaleY;
    const padX = Math.max(2, (c-a)*sx*.12), padY = Math.max(2, (d-b)*sy*.12);
    const x0 = Math.max(1, Math.floor(a*sx-padX)), y0 = Math.max(1, Math.floor(b*sy-padY));
    const x1 = Math.min(reference.width-2, Math.ceil(c*sx+padX)), y1 = Math.min(reference.height-2, Math.ceil(d*sy+padY));
    const points = [], used = new Set(); let sum = 0, square = 0;
    for (let y = 0; y < 7; y++) for (let x = 0; x < 9; x++) {
      const px = Math.round(x0+(x1-x0)*x/8), py = Math.round(y0+(y1-y0)*y/6), index = py*reference.width+px;
      if (used.has(index)) continue;
      used.add(index); const v = values[index]; points.push([px,py,v]); sum += v; square += v*v;
    }
    const variance = square-sum*sum/points.length;
    const result = points.length >= 12 && variance/points.length >= 16 ? {points,sum,variance,x0,y0,x1,y1} : null;
    reference.templates.set(key, result);
    return result;
  }
  function matchOverlay(reference, current, obs, dt, track) {
    if (!reference || !current || reference.width !== current.width || reference.height !== current.height) return null;
    const template = templateFor(reference, obs);
    if (!template) return null;
    const values = obs.color === 'red' ? current.red : current.blue, n = template.points.length;
    const hint = track?.velocity?.every(finite) ? [track.velocity[0]*dt*reference.scaleX, track.velocity[1]*dt*reference.scaleY] : [0,0];
    const radius = Math.min(48, Math.max(12, Math.ceil(12+100*dt), Math.ceil(Math.hypot(...hint)+6)));
    const minX = Math.max(-radius, -template.x0), maxX = Math.min(radius, current.width-1-template.x1);
    const minY = Math.max(-radius, -template.y0), maxY = Math.min(radius, current.height-1-template.y1);
    function score(dx,dy) {
      let sum = 0, square = 0, cross = 0;
      for (const [x,y,a] of template.points) { const b = values[(y+dy)*current.width+x+dx]; sum += b; square += b*b; cross += a*b; }
      const variance = square-sum*sum/n;
      if (variance/n < 16) return -1;
      return (cross-template.sum*sum/n)/Math.sqrt(template.variance*variance);
    }
    let best = {score:-1,rank:-Infinity,dx:0,dy:0};
    function consider(dx,dy) {
      if (dx < minX || dx > maxX || dy < minY || dy > maxY) return;
      const value = score(dx,dy), rank = value-.0002*Math.hypot(dx-hint[0],dy-hint[1]);
      if (rank > best.rank) best = {score:value,rank,dx,dy};
    }
    consider(0,0); consider(Math.round(hint[0]),Math.round(hint[1]));
    for (let dy=minY;dy<=maxY;dy+=3) for (let dx=minX;dx<=maxX;dx+=3) consider(dx,dy);
    const coarse = best;
    for (let dy=coarse.dy-2;dy<=coarse.dy+2;dy++) for (let dx=coarse.dx-2;dx<=coarse.dx+2;dx++) consider(dx,dy);
    if (best.score < .65) return null;
    return {dx:best.dx/reference.scaleX,dy:best.dy/reference.scaleY,score:best.score};
  }
  function alignObservations(observations, tracks, timestamp, referenceTimestamp, imageSize, reference, current) {
    const dt = timestamp-referenceTimestamp;
    if (!finite(dt) || dt < -.000002 || dt > .40) return [];
    return (observations || []).map((obs,index) => {
      if (!obs.center?.every(finite) || !obs.bbox?.every(finite)) return null;
      const track = (tracks || []).find(t => t.observation_index === index || obs.track_id != null && t.track_id === obs.track_id);
      const measured = dt <= .000002;
      const matched = measured ? {dx:0,dy:0,score:1} : matchOverlay(reference,current,obs,dt,track);
      const predicted = !matched && track ? alignTrack(track,timestamp,imageSize) : null;
      if (!matched && !predicted) return null;
      const dx = matched ? matched.dx : predicted.center[0]-obs.center[0];
      const dy = matched ? matched.dy : predicted.center[1]-obs.center[1];
      const center = [obs.center[0]+dx,obs.center[1]+dy];
      if (center[0]<0 || center[1]<0 || center[0]>=imageSize[0] || center[1]>=imageSize[1]) return null;
      return {...obs,center,bbox:obs.bbox.map((v,i)=>v+(i%2?dy:dx)),display_index:index,
        display_alignment:measured?'measured':matched?'image_translation':'kalman',display_age_s:dt,
        display_match_score:matched?.score ?? null,
        display_corners:matched && obs.geometry_valid && Array.isArray(obs.corners) ? obs.corners.map(p=>[p[0]+dx,p[1]+dy]) : null,
        geometry_valid:measured && obs.geometry_valid,corners:measured?obs.corners:null};
    }).filter(Boolean);
  }
  globalThis.AstraMotion = {forecast, alignTrack, overlayPixels, alignObservations};
})();
