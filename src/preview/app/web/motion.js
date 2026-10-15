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
  globalThis.AstraMotion = {forecast, alignTrack};
})();
