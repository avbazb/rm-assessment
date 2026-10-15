/* ASTRA portable preview. No external runtime or network resources. */
(() => {
  "use strict";
  const page = document.body.dataset.page || "detection";
  const impactPage = page === "impact";
  const motionPage = page === "motion";
  const posePage = page === "pose";
  const predictionPage = motionPage || page === "detection";
  const pageTitle = {detection: "检测预览", motion: "运动预测", pose: "三维位姿", impact: "落点分析"}[page] || "检测预览";
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const pair = value => Array.isArray(value) && value.length >= 2 && value.slice(0, 2).every(finite);
  const num = (value, decimals = 1) => finite(value) ? value.toFixed(decimals) : "—";
  const xy = (value, decimals = 1) => pair(value) ? `${num(value[0], decimals)}, ${num(value[1], decimals)}` : "—";
  const clock = value => {
    const ticks = finite(value) ? Math.max(0, Math.floor(value * 10 + 0.000001)) : 0;
    return `${Math.floor(ticks / 600).toString().padStart(2, "0")}:${Math.floor(ticks / 10 % 60).toString().padStart(2, "0")}.${ticks % 10}`;
  };
  function validateCalibrationInput(value) {
    const invalid = () => { throw new Error("标定参数无效：需要图像尺寸、3×3 内参、畸变系数及灯条规格"); };
    if (!value || typeof value !== "object" || Array.isArray(value)) invalid();
    const width = value.image_width, height = value.image_height, matrix = value.camera_matrix;
    if (![width, height].every(n => Number.isInteger(n) && n >= 8 && n <= 16384)) invalid();
    if (!Array.isArray(matrix) || matrix.length !== 3 || !matrix.every(row => Array.isArray(row) && row.length === 3 && row.every(finite))) invalid();
    if (matrix[0][0] <= 0 || matrix[1][1] <= 0 || matrix[0][1] !== 0 || matrix[1][0] !== 0 || matrix[0][2] < 0 || matrix[0][2] >= width || matrix[1][2] < 0 || matrix[1][2] >= height || matrix[2].some((n, i) => Math.abs(n - (i === 2 ? 1 : 0)) > 0.00000001)) invalid();
    let distortion = value.distortion_coefficients;
    if (!Array.isArray(distortion)) invalid();
    if (distortion.length === 1 && Array.isArray(distortion[0])) distortion = distortion[0];
    else if (distortion.every(row => Array.isArray(row) && row.length === 1)) distortion = distortion.map(row => row[0]);
    if (![4, 5, 8, 12, 14].includes(distortion.length) || !distortion.every(finite)) invalid();
    if (!["rm_vision_small", "rm_vision_large", "custom"].includes(value.reference_profile)) invalid();
    if (value.reference_profile === "custom" && (!Array.isArray(value.lamp_reference_mm) || value.lamp_reference_mm.length !== 2 || !value.lamp_reference_mm.every(n => finite(n) && n > 0 && n <= 1000))) invalid();
    if (value.max_reprojection_px != null && (!finite(value.max_reprojection_px) || value.max_reprojection_px <= 0 || value.max_reprojection_px > 10)) invalid();
    if (value.intrinsics_source != null && !["provided", "assumed_fov"].includes(value.intrinsics_source)) invalid();
    if (value.intrinsics_source === "assumed_fov" && (!finite(value.assumed_horizontal_fov_deg) || value.assumed_horizontal_fov_deg <= 1 || value.assumed_horizontal_fov_deg >= 179)) invalid();
    return {...value, distortion_coefficients: [...distortion]};
  }
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const state = {
    media: [], item: null, source: null, epoch: 0, sequence: 0, busy: false,
    session: typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    wantsFrame: false, scrubbing: false, wasPlaying: false, result: null,
    snapshot: null, presentedAt: null, selectedIndex: -1, selectedTrack: null,
    calibration: null, calibrationName: "默认参数", stream: null, pointMode: false,
    impactPoint: null, impactResult: null, cameraStart: null, lastCaptureTime: null, mediaFrame: null,
    processed: 0, startedAt: null, completions: [], lastRequest: null, requestHistory: [], timingHistory: [],
    ready: false, preparedMode: null, preparingMode: null, imageReady: false, exporting: false,
    horizon: 0.15, predictionEnabled: false, analysisController: null, failed: false,
    view: null, displayCompletions: [], displayCount: 0, lastDisplayTime: null, lastDetailAt: 0
  };
  try {
    const saved = JSON.parse(localStorage.getItem("astra-calibration") || "null");
    if (saved && saved.value) { state.calibration = validateCalibrationInput(saved.value); state.calibrationName = saved.name || "已导入"; }
    const savedHorizon = Number(localStorage.getItem("astra-prediction-horizon-ms") ?? 150);
    if (finite(savedHorizon) && savedHorizon >= 0 && savedHorizon <= 500) state.horizon = savedHorizon / 1000;
    state.predictionEnabled = localStorage.getItem("astra-prediction-enabled") === "1";
  } catch (_) { /* An unavailable browser store does not block preview. */ }

  document.getElementById("app").innerHTML = `
    <header class="topbar">
      <a class="brand" href="/">ASTRA <span>视觉实验</span></a>
      <nav class="navigation" aria-label="主导航">
        <a href="/" ${page === "detection" ? 'aria-current="page"' : ""}>检测预览</a>
        <a href="/motion.html" ${motionPage ? 'aria-current="page"' : ""}>运动预测</a>
        <a href="/pose.html" ${posePage ? 'aria-current="page"' : ""}>三维位姿</a>
        <a href="/impact.html" ${impactPage ? 'aria-current="page"' : ""}>落点分析</a>
      </nav>
      <div class="top-actions"><div id="serviceStatus" class="status" data-kind="loading" role="status">正在连接</div><button class="text-button" id="exitApp">退出</button></div>
    </header>
    <main class="workspace">
      <aside class="library" aria-label="素材库">
        <div class="section-heading"><h2>素材库</h2><span id="mediaCount" class="count">0</span></div>
        <div class="library-actions"><button class="primary" id="uploadButton">上传素材</button><button id="cameraButton">摄像头</button></div>
        <input id="mediaUpload" type="file" accept="video/*,image/*,.mkv,.avi,.mov,.webm,.mp4" multiple hidden>
        <progress id="uploadProgress" class="upload-progress" hidden></progress>
        <div id="mediaList" class="media-list"></div>
      </aside>
      <section class="main-view" aria-label="${pageTitle}">
        <div class="view-heading"><div><h1>${pageTitle}</h1><div id="sourceHeading" class="source-heading">—</div></div><div class="view-heading-actions"><input id="detectionMode" type="hidden" value="fusion"><button id="retryButton" disabled>重新检测</button><button id="calibrationButton">相机参数</button></div></div>
        <div class="player">
          <div id="stage" class="stage"><canvas id="preview" width="640" height="480" aria-label="检测画面"></canvas><div class="stage-empty" id="stageEmpty">选择素材</div></div>
          <div class="player-controls">
            <div class="timeline"><input id="timeline" type="range" min="0" max="1" step="0.001" value="0" aria-label="视频进度" disabled><span id="timeDisplay" class="time">00:00.0 / 00:00.0</span></div>
            <div class="control-row"><button id="playButton" class="primary" disabled>播放</button><button id="backButton" disabled>前一帧</button><button id="forwardButton" disabled>后一帧</button><span class="spacer"></span><button id="exportButton" disabled>保存画面</button></div>
          </div>
          <div class="overlay-controls">
            <label class="checkbox-label"><input id="showBoxes" type="checkbox" checked>检测框</label>
            <label class="checkbox-label"><input id="showTraditional" type="checkbox" checked>传统框</label>
            <label class="checkbox-label"><input id="showLamps" type="checkbox" checked>灯条</label>
            <label class="checkbox-label"><input id="showPrediction" type="checkbox" ${state.predictionEnabled ? "checked" : ""}><span id="predictionLabel">预测 +0.150 s</span></label>
            <label class="checkbox-label"><input id="showHistory" type="checkbox" ${predictionPage ? "checked" : ""}>历史轨迹</label>
            <label class="checkbox-label"><input id="showFiltered" type="checkbox" ${predictionPage ? "checked" : ""}>平滑中心</label>
            <label class="checkbox-label"><input id="showAxes" type="checkbox" ${posePage ? "checked" : ""}>坐标轴</label>
            ${predictionPage ? '<label class="horizon-control" for="horizonMs">预测提前量 · ms<input id="horizonMs" type="number" min="0" max="500" step="10" value="150"></label>' : ''}
          </div>
        </div>
        <div class="metrics" aria-live="off"><div class="metric"><span class="metric-label">目标</span><span class="metric-value" id="metricTargets">—</span></div><div class="metric"><span class="metric-label">处理耗时</span><span class="metric-value" id="metricLatency">—</span><span class="metric-label">ms</span></div><div class="metric"><span class="metric-label">检测频率</span><span class="metric-value" id="metricRate">—</span><span class="metric-label">Hz</span></div><div class="metric"><span class="metric-label">画面时间</span><span class="metric-value" id="metricTime">—</span><span class="metric-label">s</span></div><div class="metric"><span class="metric-label">播放帧率</span><span class="metric-value" id="metricPlayback">—</span><span class="metric-label">fps</span></div></div>
        <div id="notice" class="notice" role="status"></div>
        ${posePage ? '<section class="pose-view"><div class="section-heading"><h2>三维视图</h2><span id="poseViewState" class="target-state">等待灯条</span></div><div class="pose-controls"><label for="poseView">视角<select id="poseView"><option value="detail">目标近景</option><option value="scene">相机全景</option></select></label><label for="poseYaw">水平<input id="poseYaw" type="range" min="-180" max="180" value="20"></label><label for="posePitch">垂直<input id="posePitch" type="range" min="-80" max="80" value="-15"></label><button id="resetPoseView">复位视角</button></div><canvas id="poseCanvas" width="640" height="480" aria-label="相机坐标系装甲板三维位姿"></canvas></section>' : ''}
      </section>
      <aside class="inspector" aria-label="目标状态">
        <div class="section-heading"><h2>目标状态</h2><span id="targetCount" class="count">0</span></div>
        <table class="target-table"><thead><tr><th>ID</th><th>评分 / 置信度</th><th>位置 · px</th></tr></thead><tbody id="targetRows"><tr class="empty-row"><td colspan="3">等待检测</td></tr></tbody></table>
        <section id="targetDetail" class="target-detail" hidden>
          <div class="detail-heading"><h3 id="selectedTitle">目标</h3><span class="target-state" id="selectedState">—</span></div>
          <dl class="measurements">
            <div class="measurement"><dt>位置 · px</dt><dd id="detailPosition">—</dd></div>
            <div class="measurement"><dt>速度 · px/s</dt><dd id="detailVelocity">—</dd></div>
            <div class="measurement"><dt>估算 XYZ · m</dt><dd id="detailXYZ">—</dd></div>
            <div class="measurement"><dt>估算距离 · m</dt><dd id="detailRange">—</dd></div>
            <div class="measurement"><dt>旋转向量 · rad</dt><dd id="detailRotation">—</dd></div>
            <div class="measurement"><dt id="detailHorizon">预测 +0.150 s</dt><dd id="detailPrediction">—</dd></div>
            ${predictionPage ? '<div class="measurement"><dt>预测状态</dt><dd id="detailPredictionState">—</dd></div><div class="measurement"><dt>观测 / 预测 · s</dt><dd id="detailPredictionTime">—</dd></div><div class="measurement"><dt>样本 / 时长</dt><dd id="detailSamples">—</dd></div><div class="measurement"><dt>加速度 · px/s²</dt><dd id="detailAcceleration">—</dd></div><div class="measurement"><dt>不确定半径 · px</dt><dd id="detailUncertainty">—</dd></div>' : ''}
            ${posePage ? '<div class="measurement"><dt>位姿状态</dt><dd id="detailPoseState">—</dd></div><div class="measurement"><dt>三维速度 · m/s</dt><dd id="detailPoseVelocity">—</dd></div><div class="measurement"><dt>重投影误差 · px</dt><dd id="detailReprojection">—</dd></div>' : ''}
          </dl>
        </section>
        ${impactPage ? `<section class="impact-controls"><div class="section-heading"><h2>落点</h2><span class="target-state" id="impactStatus">等待完整灯条</span></div><div class="control-row"><button class="primary" id="pointButton" disabled>选取落点</button><button id="clearPointButton" disabled>清除</button></div><div id="impactResult" class="impact-result" hidden><div id="impactState" class="impact-state">—</div><dl class="measurements"><div class="measurement"><dt>落点 · px</dt><dd id="impactPoint">—</dd></div><div class="measurement"><dt>目标 · px</dt><dd id="impactTarget">—</dd></div><div class="measurement"><dt>水平修正</dt><dd id="impactDX">—</dd></div><div class="measurement"><dt>垂直修正</dt><dd id="impactDY">—</dd></div><div class="measurement"><dt>边缘距离 · px</dt><dd id="impactEdge">—</dd></div></dl></div></section>` : ""}
      </aside>
    </main>
    <dialog id="calibrationDialog" class="dialog"><div class="dialog-head"><h2>相机参数</h2><button id="closeCalibration" class="text-button">关闭</button></div><div id="calibrationState" class="calibration-status">默认参数</div><dl class="measurements"><div class="measurement"><dt>水平视场角</dt><dd id="calibrationFov">60°</dd></div><div class="measurement"><dt>参考分辨率</dt><dd id="calibrationResolution">640 × 480</dd></div></dl><div class="dialog-actions"><button id="defaultCalibration">默认参数</button><button class="primary" id="importCalibration">导入 JSON</button></div><input type="file" id="calibrationUpload" accept="application/json,.json" hidden></dialog>
    <video id="sourceVideo" class="hidden-media" muted playsinline preload="auto"></video>
    <img id="sourceImage" class="hidden-media" alt="">
  `;
  const video = $("sourceVideo");
  if (predictionPage) { $("horizonMs").value = Math.round(state.horizon * 1000); $("horizonMs").disabled = !$("showPrediction").checked; }
  $("predictionLabel").textContent = $("detailHorizon").textContent = `预测 +${state.horizon.toFixed(3)} s`;
  const sourceImage = $("sourceImage");
  const canvas = $("preview");
  const ctx = canvas.getContext("2d");
  const capture = document.createElement("canvas");
  const captureCtx = capture.getContext("2d", {willReadFrequently: true});
  const decodedFrame = document.createElement("canvas");
  const decodedContext = decodedFrame.getContext("2d");
  const snapshots = Array.from({length: 2}, () => {
    const frame = document.createElement("canvas");
    return {frame, context: frame.getContext("2d")};
  });
  let nextSnapshot = 0;
  if (typeof video.requestVideoFrameCallback === "function") {
    const onVideoFrame = (_now, metadata) => {
      if (!video.seeking) {
        resizeFrameCanvas(decodedFrame, video.videoWidth, video.videoHeight);
        decodedContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        state.mediaFrame = {timestamp: metadata.mediaTime, epoch: state.epoch, sourceURL: video.currentSrc, captured: true};
        renderLiveFrame(state.source === "camera" ? sourceTimestamp() : metadata.mediaTime);
      }
      video.requestVideoFrameCallback(onVideoFrame);
    };
    video.requestVideoFrameCallback(onVideoFrame);
  }
  const palette = {};
  const css = getComputedStyle(document.documentElement);
  ["lamp-red", "lamp-blue", "prediction", "geometry", "stage", "stage-text", "neutral-400"].forEach(key => { palette[key] = css.getPropertyValue(`--color-${key}`).trim(); });
  window.debugState = () => ({epoch: state.epoch, pending: state.busy, sequence: state.sequence, latestTimestamp: state.presentedAt, displayTimestamp: state.view?.timestamp ?? state.presentedAt, displayCount: state.displayCount, sourceTimestamp: sourceTimestamp(), sourceId: state.item?.id ?? null, resultTimestamp: state.result?.timestamp ?? null, sourceType: state.source, selectedIndex: state.selectedIndex, ready: state.ready, preparedMode: state.preparedMode, preparingMode: state.preparingMode, failed: state.failed, horizon: state.horizon, displayedPrediction: state.view?.future_prediction ?? state.result?.future_prediction, requestHistory: state.requestHistory.slice(-20)});

  function notice(message = "") { $("notice").textContent = message; }
  function status(text, kind = "loading") { $("serviceStatus").textContent = text; $("serviceStatus").dataset.kind = kind; }
  async function request(url, options = {}, timeoutMs = 0) {
    const controller = !options.signal && timeoutMs > 0 && typeof AbortController === "function" ? new AbortController() : null;
    const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
      const response = await fetch(url, controller ? {...options, signal: controller.signal} : options);
      if (!response.ok) {
        let message = `请求失败 · ${response.status}`;
        try { const body = await response.json(); message = body.error?.message || body.detail || body.error || message; } catch (_) { /* Keep the HTTP status. */ }
        const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
        error.status = response.status;
        throw error;
      }
      return await response.json();
    } catch (error) { if (controller?.signal.aborted) throw new Error("请求超时，请重试"); throw error; }
    finally { if (timer != null) clearTimeout(timer); }
  }
  let preparationTask = null;
  let rawFramesSupported = true;
  const analysisIdleWaiters = [];
  function waitForAnalysisIdle() {
    return state.busy ? new Promise(resolve => analysisIdleWaiters.push(resolve)) : Promise.resolve();
  }
  function prepareDetectionMode() {
    if (state.ready && state.preparedMode === $("detectionMode").value) {
      askFrame();
      return preparationTask || Promise.resolve(true);
    }
    state.ready = false;
    status("正在准备");
    if (preparationTask) return preparationTask;
    preparationTask = (async () => {
      while (true) {
        // Preparation uses the same backend worker as detection. No frame may
        // enter until the current analysis has released it and warmup finishes.
        await waitForAnalysisIdle();
        const mode = $("detectionMode").value;
        state.preparingMode = mode;
        try {
          await request(`/api/warmup?mode=${encodeURIComponent(mode)}`, {method: "POST"}, 35000);
        } catch (error) {
          if (mode !== $("detectionMode").value) continue;
          status("连接失败", "error"); notice(error.message);
          return false;
        }
        if (mode !== $("detectionMode").value) continue;
        state.preparedMode = mode;
        state.ready = true;
        status("已就绪 · CPU", "ready");
        askFrame();
        return true;
      }
    })().finally(() => { state.preparingMode = null; preparationTask = null; });
    return preparationTask;
  }
  function resetEpoch(clear = true) {
    state.analysisController?.abort();
    state.epoch++;
    state.wantsFrame = false;
    state.lastCaptureTime = null;
    state.mediaFrame = null;
    state.impactPoint = null;
    state.impactResult = null;
    state.pointMode = false;
    state.failed = false;
    state.view = null; state.lastDisplayTime = null; state.displayCompletions = []; state.displayCount = 0;
    $("stage").dataset.pointMode = "false";
    if (clear) { state.result = null; state.snapshot = null; state.presentedAt = null; state.selectedIndex = -1; state.selectedTrack = null; clearCanvas(); renderTargets(); }
    renderImpact();
  }
  function stopSource() {
    state.imageReady = false;
    video.pause();
    if (state.stream) { state.stream.getTracks().forEach(track => track.stop()); state.stream = null; }
    video.srcObject = null;
    video.removeAttribute("src");
    video.load();
    sourceImage.removeAttribute("src");
  }
  function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    delete canvas.dataset.frameTimestamp;
    delete canvas.dataset.epoch;
    $("stageEmpty").hidden = false;
    $("exportButton").disabled = true;
    $("metricTargets").textContent = "—";
    $("metricLatency").textContent = "—";
    $("metricRate").textContent = "—";
    $("metricTime").textContent = "—";
    $("metricPlayback").textContent = "—";
  }
  function renderMedia() {
    $("mediaCount").textContent = state.media.length;
    $("mediaList").innerHTML = state.media.length ? state.media.map(item => `<button class="media-item" data-media-id="${escape(item.id)}" aria-pressed="${state.item?.id === item.id}"><span class="media-thumb">${item.thumbnail ? `<img src="${escape(item.thumbnail)}" alt="" loading="lazy">` : ""}<span class="media-tag">${item.kind === "image" ? "图片" : clock(item.duration || 0)}</span></span><span class="media-item-body"><span class="media-name">${escape(item.title)}</span></span></button>`).join("") : '<div class="media-empty">暂无素材</div>';
    $("mediaList").querySelectorAll("button[data-media-id]").forEach(button => button.addEventListener("click", () => selectMedia(state.media.find(item => String(item.id) === button.dataset.mediaId))));
  }
  async function selectMedia(item) {
    if (!item) return;
    resetEpoch();
    stopSource();
    state.item = item;
    state.source = item.kind;
    $("cameraButton").textContent = "摄像头";
    state.processed = 0;
    state.startedAt = null;
    state.completions = [];
    $("sourceHeading").textContent = item.title;
    $("stageEmpty").textContent = "正在加载";
    notice(); renderMedia(); updateControls();
    if (item.kind === "image") sourceImage.src = item.url;
    else { video.src = item.url; video.load(); }
  }
  function sourceTimestamp() {
    if (state.source === "camera") return state.cameraStart === null ? 0 : (performance.now() - state.cameraStart) / 1000;
    if (state.source !== "video") return 0;
    const fps = finite(state.item?.fps) && state.item.fps > 0 ? state.item.fps : 30;
    const finalIndex = finite(video.duration) ? Math.max(0, Math.round(video.duration * fps) - 1) : Infinity;
    const index = Math.min(finalIndex, Math.max(0, Math.floor((video.currentTime + 0.000000501) * fps)));
    const cfrTimestamp = index / fps;
    // mediaTime is decoded-frame PTS. The CFR map also covers seek completion before rVFC,
    // the final paused frame, and a compositor that stops callbacks for background tabs.
    const media = state.mediaFrame;
    if (media?.captured && media.epoch === state.epoch && media.sourceURL === video.currentSrc && !video.seeking) return media.timestamp;
    if (media && media.epoch === state.epoch && media.sourceURL === video.currentSrc && finite(media.timestamp) && Math.abs(media.timestamp - cfrTimestamp) < 0.25 / fps) return media.timestamp;
    return cfrTimestamp;
  }
  function updateControls() {
    const isVideo = state.source === "video";
    const isCamera = state.source === "camera";
    $("playButton").disabled = !(isVideo || isCamera) || !video.videoWidth;
    $("playButton").textContent = (isVideo || isCamera) && !video.paused ? "暂停" : "播放";
    $("backButton").disabled = !isVideo || !video.videoWidth;
    $("forwardButton").disabled = !isVideo || !video.videoWidth;
    $("timeline").disabled = !isVideo || !finite(video.duration);
    $("timeline").max = isVideo && finite(video.duration) ? video.duration : 1;
    if (!state.scrubbing) $("timeline").value = isVideo ? video.currentTime : 0;
    $("timeDisplay").textContent = isCamera ? `${clock(sourceTimestamp())} / 实时` : isVideo ? `${clock(video.currentTime)} / ${clock(video.duration)}` : "图片";
    $("retryButton").disabled = state.busy || !state.source;
  }
  function showLoadedFrame() {
    const source = state.source === "image" ? sourceImage : video;
    const width = state.source === "image" ? sourceImage.naturalWidth : video.videoWidth;
    const height = state.source === "image" ? sourceImage.naturalHeight : video.videoHeight;
    if (!width || !height) return;
    canvas.width = width; canvas.height = height;
    ctx.drawImage(source, 0, 0, width, height);
    $("stageEmpty").hidden = true;
  }
  function askFrame() { state.wantsFrame = true; }
  function resizeFrameCanvas(frame, width, height) {
    if (frame.width !== width) frame.width = width;
    if (frame.height !== height) frame.height = height;
  }
  function snapshotForFrame(width, height) {
    if (snapshots[nextSnapshot].frame === state.snapshot) nextSnapshot = 1 - nextSnapshot;
    const selected = snapshots[nextSnapshot];
    nextSnapshot = 1 - nextSnapshot;
    resizeFrameCanvas(selected.frame, width, height);
    selected.context.drawImage(capture, 0, 0);
    return selected.frame;
  }
  async function pngFramePayload() {
    const body = await new Promise((resolve, reject) => capture.toBlob(value => value ? resolve(value) : reject(new Error("无法读取画面")), "image/png"));
    return {body, headers: {"Content-Type": "image/png"}, encoding: "png", bytes: body.size};
  }
  async function framePayload(width, height) {
    const bytes = width * height * 4;
    if (rawFramesSupported && Number.isInteger(width) && Number.isInteger(height) && bytes <= 32 * 1024 * 1024 && typeof captureCtx.getImageData === "function") {
      try {
        const data = captureCtx.getImageData(0, 0, width, height).data;
        if (ArrayBuffer.isView(data) && data.BYTES_PER_ELEMENT === 1 && data.byteLength === bytes) {
          const body = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
          if (body.byteLength === bytes) return {body, encoding: "rgba8", bytes,
            headers: {"Content-Type": "application/octet-stream", "X-Frame-Encoding": "rgba8", "X-Frame-Width": String(width), "X-Frame-Height": String(height)}};
        }
      } catch (_) { /* Preserve the same captured pixels through PNG when raw reads are unavailable. */ }
    }
    return pngFramePayload();
  }
  async function analyzeFrame() {
    if (state.busy || !state.ready || !state.source || state.scrubbing || state.failed) return;
    const isImage = state.source === "image";
    if (isImage && (!state.imageReady || !sourceImage.complete)) return;
    const width = isImage ? sourceImage.naturalWidth : video.videoWidth;
    const height = isImage ? sourceImage.naturalHeight : video.videoHeight;
    if (!width || !height || (!isImage && (video.seeking || video.readyState < 2))) return;
    const timestamp = sourceTimestamp();
    if (!state.wantsFrame && state.lastCaptureTime === timestamp) return;
    state.busy = true;
    $("stage").dataset.pending = "true";
    state.wantsFrame = false;
    state.lastCaptureTime = timestamp;
    const epoch = state.epoch;
    const sequence = ++state.sequence;
    const mode = $("detectionMode").value;
    const horizon = state.horizon;
    const predictionEnabled = $("showPrediction").checked;
    const sourceId = state.item?.id ?? null;
    const started = performance.now();
    let captureMs = 0, payloadMs = 0, requestMs = 0;
    try {
      resizeFrameCanvas(capture, width, height);
      const captured = !isImage && state.source === "video" && state.mediaFrame?.captured
        && state.mediaFrame.epoch === epoch && state.mediaFrame.sourceURL === video.currentSrc;
      captureCtx.drawImage(isImage ? sourceImage : captured ? decodedFrame : video, 0, 0, width, height);
      // Only the non-presented buffer may be written while a response is pending.
      const snapshot = snapshotForFrame(width, height);
      captureMs = performance.now() - started;
      let payloadStarted = performance.now();
      let payload = await framePayload(width, height);
      payloadMs += performance.now() - payloadStarted;
      if (epoch !== state.epoch) return;
      const headersFor = value => {
        const headers = {...value.headers};
        if (state.calibration) headers["X-Calibration"] = encodeURIComponent(JSON.stringify(state.calibration));
        return headers;
      };
      const query = new URLSearchParams({session: state.session, epoch: String(epoch), sequence: String(sequence), timestamp: timestamp.toFixed(6), horizon: String(horizon), prediction: predictionEnabled ? "1" : "0", mode});
      state.lastRequest = {epoch, sequence, timestamp};
      const sendPayload = async value => {
        const requestStarted = performance.now();
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        state.analysisController = controller;
        const timer = controller ? setTimeout(() => controller.abort(), 10000) : null;
        try { return await request(`/api/analyze?${query}`, {method: "POST", headers: headersFor(value), body: value.body, ...(controller ? {signal: controller.signal} : {})}); }
        catch (error) { if (controller?.signal.aborted && epoch === state.epoch) throw new Error("检测超时，请重新检测"); throw error; }
        finally { if (timer != null) clearTimeout(timer); if (state.analysisController === controller) state.analysisController = null; requestMs += performance.now() - requestStarted; }
      };
      let result;
      try {
        result = await sendPayload(payload);
      } catch (error) {
        if (payload.encoding !== "rgba8" || ![415, 501].includes(error.status)) throw error;
        rawFramesSupported = false;
        if (epoch !== state.epoch || mode !== $("detectionMode").value) return;
        payloadStarted = performance.now();
        payload = await pngFramePayload();
        payloadMs += performance.now() - payloadStarted;
        if (epoch !== state.epoch || mode !== $("detectionMode").value) return;
        result = await sendPayload(payload);
      }
      const record = {epoch, sequence, timestamp, mode, sourceId, encoding: payload.encoding, payload_bytes: payload.bytes, responseTimestamp: result.timestamp, accepted: false};
      state.requestHistory.push(record);
      if (state.requestHistory.length > 40) state.requestHistory.shift();
      if (epoch !== state.epoch || mode !== $("detectionMode").value) return;
      if (horizon !== state.horizon) { askFrame(); return; }
      if (!$("showPrediction").checked) {
        result.motion_estimates = []; result.future_prediction = disabledPrediction(); result.prediction_enabled = false;
      }
      if (predictionEnabled !== $("showPrediction").checked) askFrame();
      if (finite(result.timestamp) && Math.abs(result.timestamp - timestamp) > 0.000002) throw new Error("画面时间校验失败");
      const renderStarted = performance.now();
      state.snapshot = snapshot;
      state.result = result;
      state.presentedAt = timestamp;
      canvas.dataset.frameTimestamp = timestamp.toFixed(6);
      canvas.dataset.epoch = String(epoch);
      if (state.startedAt === null) state.startedAt = started;
      state.processed++;
      $("metricTime").textContent = num(timestamp, 3);
      $("metricTargets").textContent = (result.observations || []).length;
      $("exportButton").disabled = state.exporting;
      $("stageEmpty").hidden = true;
      state.view = null;
      renderTargets();
      if (isPlaying()) renderLiveFrame(sourceTimestamp()); else drawFrame();
      notice();
      const completedAt = performance.now();
      record.accepted = true;
      record.server_ms = finite(result.processing_ms) ? result.processing_ms : null;
      record.body_ms = finite(result.request_body_ms) ? result.request_body_ms : null;
      record.capture_ms = captureMs;
      record.payload_ms = payloadMs;
      record.request_ms = requestMs;
      record.render_ms = completedAt - renderStarted;
      record.end_to_end_ms = completedAt - started;
      state.timingHistory = [...state.timingHistory, {...record}].slice(-40);
      canvas.dataset.timingHistory = JSON.stringify(state.timingHistory);
      state.completions = [...state.completions.filter(time => completedAt - time < 5000), completedAt];
      const rate = state.completions.length > 1 ? (state.completions.length - 1) * 1000 / Math.max(1, completedAt - state.completions[0]) : 1000 / Math.max(1, completedAt - started);
      $("metricRate").textContent = num(rate, 1);
      $("metricLatency").textContent = num(record.end_to_end_ms, 1);
    } catch (error) {
      if (epoch === state.epoch) {
        state.failed = true; state.lastCaptureTime = null; state.wantsFrame = false;
        video.pause(); notice(error.message || "检测失败，请重新检测");
      }
    } finally {
      state.busy = false; $("stage").dataset.pending = "false";
      analysisIdleWaiters.splice(0).forEach(resolve => resolve());
    }
  }
  function frameLoop() {
    const running = (state.source === "video" || state.source === "camera") && !video.paused && !video.ended;
    if ((state.wantsFrame || running) && !state.busy) void analyzeFrame();
    updateControls();
    // Keep inference scheduling independent from the compositor's background-tab RAF throttle.
    // The busy flag remains the only admission gate: this timer never queues captured frames.
    setTimeout(frameLoop, 40);
  }
  function isPlaying() { return (state.source === "video" || state.source === "camera") && !video.paused && !video.ended && !video.seeking && !state.scrubbing; }
  function rawTracks() { return [...(state.result?.tracks || []), ...(state.result?.tentative_tracks || [])]; }
  function disabledPrediction() { return {enabled: false, status: "disabled", target_timestamp: null, tracks: []}; }
  function renderLiveFrame(timestamp) {
    if (!isPlaying() || !finite(timestamp) || !video.videoWidth) return;
    const imageSize = [video.videoWidth, video.videoHeight];
    const age = timestamp - (state.result?.timestamp ?? -Infinity);
    const valid = age >= -0.000002 && age <= 0.25;
    state.view = {timestamp,
      tracks: valid ? rawTracks().map(track => AstraMotion.alignTrack(track, timestamp, imageSize)).filter(Boolean) : [],
      future_prediction: $("showPrediction").checked && valid && state.result?.prediction_enabled !== false
        ? AstraMotion.forecast(state.result?.motion_estimates, timestamp + state.horizon, imageSize) : disabledPrediction()};
    drawFrame();
    const now = performance.now();
    if (state.lastDisplayTime !== timestamp) {
      state.lastDisplayTime = timestamp; state.displayCount++;
      state.displayCompletions = [...state.displayCompletions.filter(t => now - t < 2000), now];
    }
    canvas.dataset.displayCount = String(state.displayCount);
    canvas.dataset.prediction = JSON.stringify({enabled: $("showPrediction").checked, target_timestamp: $("showPrediction").checked ? timestamp + state.horizon : null,
      tracks: state.view.future_prediction.tracks.map(p => ({track_id:p.track_id,status:p.status,center:p.center,usable:p.usable}))});
    if (now - state.lastDetailAt >= 100) {
      state.lastDetailAt = now; renderDetail(); $("metricTime").textContent = num(timestamp, 3);
      const frames = state.displayCompletions;
      if (frames.length > 1) $("metricPlayback").textContent = num((frames.length - 1) * 1000 / Math.max(1, frames.at(-1) - frames[0]), 1);
    }
  }
  if (typeof video.requestVideoFrameCallback !== "function" && typeof requestAnimationFrame === "function") {
    const displayLoop = () => { renderLiveFrame(sourceTimestamp()); requestAnimationFrame(displayLoop); };
    requestAnimationFrame(displayLoop);
  }
  function tracks() { return state.view?.tracks ?? rawTracks(); }
  function trackFor(obs, index) {
    return tracks().find(track => track.observation_index === index) || (obs.track_id != null ? tracks().find(track => track.track_id === obs.track_id || track.id === obs.track_id) : null);
  }
  function idFor(obs, index) { const track = trackFor(obs, index); return track?.track_id ?? track?.id ?? obs.track_id ?? null; }
  function futureFor(id) { return !$("showPrediction").checked || id == null ? null : ((state.view?.future_prediction ?? state.result?.future_prediction)?.tracks || []).find(track => (track.track_id ?? track.id) === id); }
  function selectedTrackState() { return tracks().find(track => (track.track_id ?? track.id) === state.selectedTrack) ?? null; }
  function motionFor(id) { return (state.result?.motion_estimates || []).find(track => track.track_id === id); }
  const predictionReasons = {tentative: "目标待确认", insufficient_measurements: "连续观测不足", unsupported_displacement: "提前量过大", uncertain: "预测不确定", outside_image: "预测超出画面", prediction_expired: "观测已过期", clock_mismatch: "时间不匹配"};
  const poseReasons = {estimated: "已解算", ambiguous: "朝向待确认", pose_temporal_waiting: "三维重建中", calibration_required: "等待相机参数", resolution_mismatch: "参数分辨率不符", lamp_corners_required: "等待完整灯条", incomplete_lamp_extent: "灯条不完整", reprojection_rejected: "角点误差过大", invalid_corners: "角点无效", invalid_corner_order: "角点顺序无效", corners_outside_image: "角点超出画面", solver_failed: "解算失败"};
  function selectedObservation() { return state.result?.observations?.[state.selectedIndex] ?? null; }
  function guarded(obs) { return obs?.pose_geometry_guard?.status === "rejected" || obs?.pose_geometry_guard === "rejected" || obs?.pose_geometry_guard?.rejected === true; }
  function poseFor(obs, index) {
    if (!obs || guarded(obs)) return null;
    const pose = trackFor(obs, index)?.pose ?? obs.pose;
    return pose?.status === "estimated" && Array.isArray(pose.tvec_m) && pose.tvec_m.length === 3 && pose.tvec_m.every(finite) ? pose : null;
  }
  function renderTargets() {
    const observations = state.result?.observations || [];
    if (state.selectedTrack != null) {
      const retained = observations.findIndex((obs, index) => idFor(obs, index) === state.selectedTrack);
      state.selectedIndex = retained;
    }
    const coasting = tracks().filter(track => !track.observed);
    if ((state.selectedIndex < 0 || state.selectedIndex >= observations.length) && !selectedTrackState()) state.selectedIndex = observations.length ? 0 : -1;
    state.selectedTrack = state.selectedIndex >= 0 ? idFor(observations[state.selectedIndex], state.selectedIndex) : selectedTrackState()?.track_id ?? coasting[0]?.track_id ?? null;
    $("targetCount").textContent = observations.length + coasting.length;
    $("targetRows").innerHTML = observations.length ? observations.map((obs, index) => {
      const id = idFor(obs, index);
      const center = trackFor(obs, index)?.center ?? obs.center;
      const color = obs.color === "red" || obs.color === "blue" ? obs.color : "";
      return `<tr class="target-row" tabindex="0" data-index="${index}" aria-selected="${index === state.selectedIndex}"><td><span class="target-color ${color}"></span>${id == null ? "—" : escape(id)}</td><td>${finite(obs.score) ? `${num(obs.score * 100, 1)}%` : "—"}</td><td>${xy(center)}</td></tr>`;
    }).join("") : "";
    $("targetRows").innerHTML += coasting.map(track => `<tr class="target-row" tabindex="0" data-coasting-id="${escape(track.track_id)}" aria-selected="${state.selectedTrack === track.track_id}"><td>${escape(track.track_id)}</td><td>短时外推</td><td>${xy(track.center)}</td></tr>`).join("");
    if (!observations.length && !coasting.length) $("targetRows").innerHTML = '<tr class="empty-row"><td colspan="3">暂无目标</td></tr>';
    $("targetRows").querySelectorAll("[data-index]").forEach(row => {
      const select = () => { state.selectedIndex = Number(row.dataset.index); state.selectedTrack = idFor(observations[state.selectedIndex], state.selectedIndex); state.impactResult = null; state.impactPoint = null; renderTargets(); drawFrame(); };
      row.addEventListener("click", select);
      row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
    });
    $("targetRows").querySelectorAll("[data-coasting-id]").forEach(row => {
      const select = () => { state.selectedIndex = -1; state.selectedTrack = coasting.find(track => String(track.track_id) === row.dataset.coastingId)?.track_id ?? null; renderTargets(); drawFrame(); };
      row.addEventListener("click", select);
      row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
    });
    renderDetail(); renderImpact();
  }
  function renderDetail() {
    const obs = selectedObservation();
    const track = obs ? trackFor(obs, state.selectedIndex) : selectedTrackState();
    $("targetDetail").hidden = !obs && !track;
    if (!obs && !track) { if (posePage) drawPoseView(); return; }
    const id = track?.track_id ?? (obs ? idFor(obs, state.selectedIndex) : null);
    const pose = poseFor(obs, state.selectedIndex);
    const prediction = futureFor(id);
    $("selectedTitle").textContent = id == null ? "目标" : `目标 ${id}`;
    $("selectedState").textContent = track ? (!track.observed ? "短时外推" : track.confirmed === false || track.status === "tentative" ? "待确认" : "跟踪中") : "已检测";
    $("detailPosition").textContent = xy(track?.center ?? obs?.center);
    $("detailVelocity").textContent = xy(track?.velocity);
    $("detailXYZ").textContent = pose ? pose.tvec_m.map(value => num(value, 3)).join(", ") : "—";
    $("detailRange").textContent = pose ? num(pose.range_m, 3) : "—";
    const rotation = pose?.orientation_readout ? pose.orientation_readout.rvec_for_display : pose?.orientation_status === "waiting" ? null : pose?.rvec_rad;
    $("detailRotation").textContent = Array.isArray(rotation) && rotation.every(finite) ? rotation.map(value => num(value, 3)).join(", ") : "—";
    $("detailPrediction").textContent = !$("showPrediction").checked ? "—" : pair(prediction?.center) ? xy(prediction.center) : "等待";
    if (predictionPage) {
      const estimate = motionFor(id);
      $("detailPredictionState").textContent = !$("showPrediction").checked ? "已关闭" : prediction?.status === "ready" ? "可用" : prediction?.status === "estimated" ? `估计 · ${predictionReasons[prediction.reason] || "待确认"}` : predictionReasons[prediction?.reason] || "等待连续观测";
      $("detailPredictionTime").textContent = prediction ? `${num(prediction.observation_timestamp, 3)} / ${num(prediction.target_timestamp, 3)}` : "—";
      $("detailSamples").textContent = estimate ? `${estimate.samples} / ${num(estimate.measurement_span_s, 3)} s` : "—";
      $("detailAcceleration").textContent = xy(estimate?.acceleration);
      $("detailUncertainty").textContent = num(prediction?.uncertainty_radius_px);
    }
    if (posePage) {
      const raw = track?.pose ?? obs?.pose;
      $("detailPoseState").textContent = poseReasons[raw?.status] || "等待完整灯条";
      $("detailPoseVelocity").textContent = raw?.velocity_m_s?.every(finite) ? raw.velocity_m_s.map(value => num(value, 3)).join(", ") : "—";
      $("detailReprojection").textContent = num(raw?.reprojection_px, 3);
      drawPoseView();
    }
  }
  function strokeLine(a, b, color, width) { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); }
  function drawPrediction(center, prediction, unit) {
    if (!$("showPrediction").checked || !["ready", "estimated"].includes(prediction?.status) || !pair(prediction.center)) return;
    const [x, y] = prediction.center;
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;
    if (pair(center)) { ctx.setLineDash([4 * unit, 4 * unit]); strokeLine(center, [x, y], palette.prediction, unit); ctx.setLineDash([]); }
    strokeLine([x - 6 * unit, y], [x + 6 * unit, y], palette.prediction, 2 * unit);
    strokeLine([x, y - 6 * unit], [x, y + 6 * unit], palette.prediction, 2 * unit);
    ctx.fillStyle = palette.prediction;
    ctx.fillText(`+${Math.round(state.horizon * 1000)} ms${prediction.status === "estimated" ? " 估计" : ""}`, Math.min(x + 8 * unit, canvas.width - 112 * unit), Math.max(16 * unit, y - 8 * unit));
  }
  function drawPoseView() {
    if (!posePage) return;
    const view = $("poseCanvas"), painter = view.getContext("2d");
    const obs = selectedObservation(), pose = obs ? poseFor(obs, state.selectedIndex) : null;
    painter.clearRect(0, 0, view.width, view.height);
    painter.fillStyle = css.getPropertyValue("--color-neutral-50").trim(); painter.fillRect(0, 0, view.width, view.height);
    painter.font = `400 14px "Segoe UI", "Microsoft YaHei", sans-serif`;
    painter.fillStyle = css.getPropertyValue("--color-neutral-500").trim();
    if (!pose || !Array.isArray(pose.corners_camera_m) || !pose.corners_camera_m.every(point => point.length === 3 && point.every(finite))) {
      const status = poseReasons[(selectedTrackState()?.pose ?? obs?.pose)?.status] || "等待完整灯条";
      $("poseViewState").textContent = status; painter.fillText(status, 24, 32); return;
    }
    const orientation = pose.orientation_readout;
    const reliable = orientation ? Array.isArray(orientation.rvec_for_display) : !["waiting", "unavailable"].includes(pose.orientation_status);
    $("poseViewState").textContent = reliable ? "已解算" : "位置已解算 · 朝向待确认";
    const yaw = Number($("poseYaw").value) * Math.PI / 180, pitch = Number($("posePitch").value) * Math.PI / 180;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const detail = $("poseView").value === "detail";
    const span = Math.max(...pose.lamp_reference_mm) / 1000;
    const focus = detail ? pose.tvec_m : pose.tvec_m.map(value => value / 2);
    const scale = Math.min(view.width, view.height) * .60 / (detail ? span : Math.max(pose.range_m, span));
    const transform = point => {
      const [x, y, z] = point.map((value, i) => value - focus[i]);
      const horizontal = cy * x + sy * z, depth = -sy * x + cy * z;
      return [view.width / 2 + horizontal * scale, view.height / 2 + (cp * y - sp * depth) * scale];
    };
    const line = (points, color, dashed = false) => {
      painter.strokeStyle = color; painter.lineWidth = 2; painter.setLineDash(dashed ? [8, 4] : []);
      painter.beginPath(); points.map(transform).forEach((point, i) => i ? painter.lineTo(...point) : painter.moveTo(...point)); painter.stroke(); painter.setLineDash([]);
    };
    const label = (point, text, color) => { const [x, y] = transform(point); painter.fillStyle = color; painter.fillText(text, x + 8, y - 8); };
    const axisColors = [palette["lamp-red"], palette.geometry, palette["lamp-blue"]];
    if (!detail) {
      const origin = [0, 0, 0], length = Math.max(pose.range_m * .15, span);
      axisColors.forEach((color, i) => { const end = origin.map((_, j) => i === j ? length : 0); line([origin, end], color); label(end, ["X 右", "Y 下", "Z 前"][i], color); });
      line([origin, pose.tvec_m], palette["neutral-400"], true); label(origin, "相机", palette["neutral-400"]);
    }
    if (reliable) {
      const points = pose.corners_camera_m;
      line([...points, points[0]], obs.color === "red" ? palette["lamp-red"] : palette["lamp-blue"]);
      if (Array.isArray(pose.axes_camera_m) && pose.axes_camera_m.length === 4) axisColors.forEach((color, i) => {
        line([pose.axes_camera_m[0], pose.axes_camera_m[i + 1]], color); label(pose.axes_camera_m[i + 1], ["X", "Y", "Z"][i], color);
      });
    } else {
      const point = transform(pose.tvec_m); painter.fillStyle = palette.geometry;
      painter.beginPath(); painter.arc(...point, 4, 0, Math.PI * 2); painter.fill();
    }
    label(pose.tvec_m, `ID ${state.selectedTrack ?? "—"}`, palette.geometry);
  }
  function drawFrame() {
    const live = state.view && isPlaying();
    const source = live ? video : state.snapshot;
    if (!source) return;
    const width = live ? video.videoWidth : source.width, height = live ? video.videoHeight : source.height;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0, width, height);
    if (live) { canvas.dataset.frameTimestamp = state.view.timestamp.toFixed(6); canvas.dataset.presentation = "media-clock-filtered-state"; $("stageEmpty").hidden = true; }
    else { state.view = null; canvas.dataset.frameTimestamp = num(state.presentedAt, 6); canvas.dataset.presentation = "measured-frame"; }
    const unit = Math.max(1, canvas.width / 640);
    const labelSize = 14 * unit;
    ctx.font = `600 ${labelSize}px "Segoe UI", "Microsoft YaHei", sans-serif`;
    tracks().forEach(track => {
      const color = track.color === "red" ? palette["lamp-red"] : palette["lamp-blue"];
      if ($("showHistory").checked) {
        const history = (track.history || []).filter(point => pair(point.center));
        for (let i = 1; i < history.length; i++) {
          if (!history[i].observed || !history[i - 1].observed) ctx.setLineDash([4 * unit, 4 * unit]);
          strokeLine(history[i - 1].center, history[i].center, color, 1.5 * unit); ctx.setLineDash([]);
        }
      }
      if ($("showFiltered").checked && pair(track.center)) {
        ctx.strokeStyle = palette.geometry; ctx.lineWidth = 2 * unit;
        ctx.beginPath(); ctx.arc(...track.center, 4 * unit, 0, Math.PI * 2); ctx.stroke();
      }
      if (!track.observed) drawPrediction(track.center, futureFor(track.track_id), unit);
    });
    const observations = (state.result?.observations || []).map((obs, index) => {
      if (!live) return obs;
      const track = trackFor(obs, index);
      if (!track) return null;
      const dx = track.center[0] - obs.center[0], dy = track.center[1] - obs.center[1];
      return {...obs, bbox: obs.bbox.map((v, i) => v + (i % 2 ? dy : dx)), center: track.center,
        geometry_valid: false, corners: null, display_track: track, display_index: index};
    }).filter(Boolean);
    observations.forEach((obs, displayIndex) => {
      const index = obs.display_index ?? displayIndex;
      const color = obs.color === "red" ? palette["lamp-red"] : obs.color === "blue" ? palette["lamp-blue"] : palette["geometry"];
      const selected = index === state.selectedIndex;
      if ($("showBoxes").checked && Array.isArray(obs.bbox) && obs.bbox.length === 4 && obs.bbox.every(finite)) {
        const [x1, y1, x2, y2] = obs.bbox;
        ctx.strokeStyle = color; ctx.lineWidth = (selected ? 2.5 : 1.5) * unit;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const id = idFor(obs, index);
        const label = `${live ? "跟踪 " : obs.source === "classical" ? "传统 " : ""}${id != null ? `#${id}  ` : ""}${finite(obs.score) ? `${num(obs.score * 100, 1)}%` : ""}`;
        if (label) {
          const padding = 4 * unit, textWidth = ctx.measureText(label).width;
          const top = Math.max(0, y1 - labelSize - padding * 2), left = Math.min(Math.max(0, x1), Math.max(0, canvas.width - textWidth - padding * 2));
          ctx.fillStyle = palette.stage; ctx.fillRect(left, top, textWidth + padding * 2, labelSize + padding * 2);
          ctx.fillStyle = palette["stage-text"]; ctx.fillText(label, left + padding, top + labelSize + padding / 2);
        }
      }
      if ($("showLamps").checked && obs.geometry_valid && !guarded(obs) && Array.isArray(obs.corners) && obs.corners.length === 4 && obs.corners.every(pair)) {
        ctx.strokeStyle = palette.geometry; ctx.lineWidth = 2 * unit;
        ctx.beginPath(); obs.corners.forEach((point, n) => n ? ctx.lineTo(...point) : ctx.moveTo(...point)); ctx.closePath(); ctx.stroke();
        ctx.fillStyle = palette.geometry;
        obs.corners.forEach(point => { ctx.beginPath(); ctx.arc(point[0], point[1], 2.5 * unit, 0, Math.PI * 2); ctx.fill(); });
      }
      const pose = poseFor(obs, index);
      const orientation = pose?.orientation_readout;
      const axes = orientation ? orientation.axes_for_display : pose?.axes_image_px;
      const orientationValid = orientation ? Array.isArray(orientation.axes_for_display) : pose?.orientation_status != null && !["waiting", "unavailable"].includes(pose.orientation_status);
      if (!live && $("showAxes").checked && orientationValid && Array.isArray(axes) && axes.length === 4 && axes.every(pair)) {
        strokeLine(axes[0], axes[1], palette["lamp-red"], 2 * unit);
        strokeLine(axes[0], axes[2], palette.geometry, 2 * unit);
        strokeLine(axes[0], axes[3], palette["lamp-blue"], 2 * unit);
      }
      drawPrediction(trackFor(obs, index)?.center ?? obs.center, futureFor(idFor(obs, index)), unit);
    });
    if (!live && $("showTraditional").checked && state.result?.detection_mode !== "classical") {
      (state.result?.traditional_observations || []).forEach(obs => {
        if (!Array.isArray(obs.bbox) || obs.bbox.length !== 4 || !obs.bbox.every(finite)) return;
        const [x1,y1,x2,y2] = obs.bbox;
        ctx.strokeStyle = palette.geometry; ctx.lineWidth = unit;
        ctx.setLineDash([4 * unit, 4 * unit]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
        if (pair(obs.center)) { ctx.fillStyle=palette.geometry; ctx.beginPath(); ctx.arc(...obs.center,2.5*unit,0,Math.PI*2); ctx.fill(); }
        const label=`传统 ${num(obs.score * 100,1)}%`;
        ctx.fillStyle=palette.stage; ctx.fillRect(x1,Math.min(canvas.height-labelSize-4*unit,y2),ctx.measureText(label).width+8*unit,labelSize+4*unit);
        ctx.fillStyle=palette.geometry; ctx.fillText(label,x1+4*unit,Math.min(canvas.height-2*unit,y2+labelSize));
      });
    }
    if (state.impactPoint) {
      const [x, y] = state.impactPoint;
      ctx.strokeStyle = palette.prediction; ctx.lineWidth = 2 * unit;
      ctx.beginPath(); ctx.arc(x, y, 7 * unit, 0, Math.PI * 2); ctx.stroke();
      strokeLine([x - 10 * unit, y], [x + 10 * unit, y], palette.prediction, unit);
      strokeLine([x, y - 10 * unit], [x, y + 10 * unit], palette.prediction, unit);
    }
  }
  function renderImpact() {
    if (!impactPage) return;
    const obs = selectedObservation();
    const valid = obs?.geometry_valid && !guarded(obs) && Array.isArray(obs.corners) && obs.corners.length === 4 && obs.corners.every(pair);
    $("pointButton").disabled = !valid;
    $("pointButton").textContent = state.pointMode ? "取消选取" : "选取落点";
    $("clearPointButton").disabled = !state.impactPoint;
    $("impactStatus").textContent = valid ? state.pointMode ? "选取中" : "可选取" : "等待完整灯条";
    $("impactResult").hidden = !state.impactResult;
    if (!state.impactResult) return;
    const result = state.impactResult;
    $("impactState").textContent = result.state === "edge" ? "命中边缘" : result.hit ? "命中" : "未命中";
    $("impactState").dataset.hit = String(!!result.hit);
    $("impactPoint").textContent = xy(result.point ?? state.impactPoint);
    $("impactTarget").textContent = xy(result.target);
    $("impactDX").textContent = finite(result.dx) ? `${result.dx < 0 ? "向左" : result.dx > 0 ? "向右" : "居中"}${result.dx === 0 ? "" : ` ${num(Math.abs(result.dx))} px`}` : "—";
    $("impactDY").textContent = finite(result.dy) ? `${result.dy < 0 ? "向上" : result.dy > 0 ? "向下" : "居中"}${result.dy === 0 ? "" : ` ${num(Math.abs(result.dy))} px`}` : "—";
    $("impactEdge").textContent = num(result.distance_to_edge_px);
  }
  async function toggleCamera() {
    if (state.source === "camera") { resetEpoch(); stopSource(); state.source = null; state.item = null; $("cameraButton").textContent = "摄像头"; $("sourceHeading").textContent = "—"; updateControls(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
      resetEpoch(); stopSource(); state.stream = stream; state.source = "camera"; state.item = null; state.cameraStart = performance.now(); state.processed = 0; state.startedAt = null; state.completions = [];
      video.srcObject = stream; $("sourceHeading").textContent = "摄像头"; $("cameraButton").textContent = "关闭相机"; renderMedia(); notice(); await video.play();
    } catch (error) { notice(error.name === "NotAllowedError" ? "摄像头权限未开启" : `摄像头不可用 · ${error.message}`); }
  }
  function calibrationDisplay() {
    $("calibrationState").textContent = state.calibrationName;
    $("calibrationFov").textContent = state.calibration ? "—" : "60°";
    const calibration = state.calibration;
    const size = calibration?.image_size || calibration?.resolution;
    const width = calibration?.image_width ?? calibration?.width ?? size?.[0];
    const height = calibration?.image_height ?? calibration?.height ?? size?.[1];
    $("calibrationResolution").textContent = calibration ? finite(width) && finite(height) ? `${width} × ${height}` : "—" : "640 × 480";
  }
  function saveCalibration() {
    try { localStorage.setItem("astra-calibration", JSON.stringify({value: state.calibration, name: state.calibrationName})); } catch (_) { /* Calibration remains active in this session. */ }
    calibrationDisplay(); resetEpoch(); showLoadedFrame(); askFrame();
  }

  $("uploadButton").addEventListener("click", () => $("mediaUpload").click());
  $("mediaUpload").addEventListener("change", async event => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    $("uploadButton").disabled = true; $("uploadProgress").hidden = false;
    $("uploadProgress").max = files.length * 100; $("uploadProgress").value = 0;
    let lastItem = null;
    try {
      for (const [index, file] of files.entries()) {
        let data = await request(`/api/media?filename=${encodeURIComponent(file.name)}&async=1`, {method: "POST", headers: {"Content-Type": file.type || "application/octet-stream"}, body: file}, 120000);
        if (data.job) {
          const id = data.job.id;
          if (typeof id !== "string" || !/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error("素材导入状态无效");
          let deadline = performance.now() + 30 * 60 * 1000;
          while (data.job.status === "processing") {
            $("uploadProgress").value = index * 100 + Math.max(0, Math.min(100, Number(data.job.progress) || 0));
            if (performance.now() > deadline) throw new Error("素材导入超时，请重新上传");
            await new Promise(resolve => setTimeout(resolve, 500));
            data = await request(`/api/imports/${encodeURIComponent(id)}`, {}, 10000);
          }
          if (data.job.status !== "ready") throw new Error(data.job.error || "素材导入失败，请重新上传");
          data = {item: data.job.item};
        }
        if (!data.item) throw new Error("素材导入失败");
        state.media = [data.item, ...state.media.filter(item => item.id !== data.item.id)]; lastItem = data.item;
        $("uploadProgress").value = (index + 1) * 100; renderMedia();
      }
      notice();
    } catch (error) { notice(error.message); }
    finally { $("uploadButton").disabled = false; $("uploadProgress").hidden = true; event.target.value = ""; if (lastItem) await selectMedia(lastItem); }
  });
  $("cameraButton").addEventListener("click", () => void toggleCamera());
  $("retryButton").addEventListener("click", async () => {
    if (state.busy) return;
    state.failed = false; state.lastCaptureTime = null; notice();
    if (!state.ready) await prepareDetectionMode();
    askFrame();
  });
  $("playButton").addEventListener("click", async () => {
    if (video.paused) { state.failed = false; state.impactResult = null; state.impactPoint = null; state.pointMode = false; state.completions = []; renderImpact(); try { if (video.ended) { resetEpoch(); video.currentTime = 0; } await video.play(); } catch (error) { notice(error.message); } }
    else { video.pause(); state.view = null; showLoadedFrame(); askFrame(); }
    updateControls();
  });
  function step(direction) {
    if (state.source !== "video" || !finite(video.duration)) return;
    video.pause(); resetEpoch();
    const fps = finite(state.item?.fps) && state.item.fps > 0 ? state.item.fps : 30;
    const finalIndex = Math.max(0, Math.round(video.duration * fps) - 1);
    const nextIndex = Math.max(0, Math.min(finalIndex, Math.round(sourceTimestamp() * fps) + direction));
    video.currentTime = nextIndex / fps;
    askFrame(); updateControls();
  }
  $("backButton").addEventListener("click", () => step(-1));
  $("forwardButton").addEventListener("click", () => step(1));
  $("timeline").addEventListener("input", event => {
    if (!state.scrubbing) { state.wasPlaying = !video.paused; state.scrubbing = true; video.pause(); }
    resetEpoch(); video.currentTime = Number(event.target.value); updateControls();
  });
  $("timeline").addEventListener("change", async () => {
    state.scrubbing = false; askFrame();
    if (state.wasPlaying) { try { await video.play(); } catch (error) { notice(error.message); } }
    state.wasPlaying = false;
  });
  video.addEventListener("loadeddata", () => { showLoadedFrame(); askFrame(); updateControls(); });
  video.addEventListener("seeked", () => { if (!state.scrubbing) { showLoadedFrame(); askFrame(); } });
  video.addEventListener("ended", () => { askFrame(); updateControls(); });
  video.addEventListener("error", () => { if (state.source === "video" && video.getAttribute("src")) notice("视频无法播放"); });
  sourceImage.addEventListener("load", async () => {
    const epoch = state.epoch, url = sourceImage.currentSrc;
    try { if (typeof sourceImage.decode === "function") await sourceImage.decode(); }
    catch (_) { if (epoch === state.epoch) notice("图片无法读取"); return; }
    if (epoch !== state.epoch || state.source !== "image" || sourceImage.currentSrc !== url) return;
    state.imageReady = true; showLoadedFrame(); askFrame(); updateControls();
  });
  sourceImage.addEventListener("error", () => { if (sourceImage.getAttribute("src")) notice("图片无法读取"); });
  ["showBoxes", "showTraditional", "showLamps", "showHistory", "showFiltered", "showAxes"].forEach(id => $(id).addEventListener("change", drawFrame));
  $("showPrediction").addEventListener("change", () => {
    const enabled = $("showPrediction").checked;
    if (predictionPage) $("horizonMs").disabled = !enabled;
    try { localStorage.setItem("astra-prediction-enabled", enabled ? "1" : "0"); } catch (_) { /* Keep the current switch. */ }
    if (!enabled) {
      if (state.result) { state.result.motion_estimates = []; state.result.future_prediction = disabledPrediction(); state.result.prediction_enabled = false; }
      if (state.view) state.view.future_prediction = disabledPrediction();
      canvas.dataset.prediction = JSON.stringify(disabledPrediction());
    }
    renderDetail(); drawFrame(); askFrame();
  });
  function selectHorizon(event) {
    if (event.target.value.trim() === "") return;
    const value = Number(event.target.value);
    if (!finite(value) || value < 0 || value > 500) { event.target.value = Math.round(state.horizon * 1000); return; }
    state.horizon = value / 1000;
    try { localStorage.setItem("astra-prediction-horizon-ms", String(value)); } catch (_) { /* Keep the current selection when storage is unavailable. */ }
    $("predictionLabel").textContent = $("detailHorizon").textContent = `预测 +${state.horizon.toFixed(3)} s`;
    askFrame();
  }
  if (predictionPage) ["input", "change"].forEach(name => $("horizonMs").addEventListener(name, selectHorizon));
  if (posePage) {
    ["poseView", "poseYaw", "posePitch"].forEach(id => $(id).addEventListener("input", drawPoseView));
    $("resetPoseView").addEventListener("click", () => { $("poseYaw").value = 20; $("posePitch").value = -15; drawPoseView(); });
  }
  $("detectionMode").addEventListener("change", () => {
    resetEpoch(); showLoadedFrame();
    void prepareDetectionMode();
  });
  $("exportButton").addEventListener("click", async () => {
    if (state.exporting || !state.snapshot) return;
    state.exporting = true;
    $("exportButton").disabled = true;
    const filename = `ASTRA-${impactPage ? "落点" : "检测"}-${num(state.presentedAt, 3)}s.png`;
    const exportFrame = document.createElement("canvas");
    exportFrame.width = canvas.width; exportFrame.height = canvas.height;
    exportFrame.getContext("2d").drawImage(canvas, 0, 0);
    try {
      const blob = await new Promise((resolve, reject) => exportFrame.toBlob(value => value ? resolve(value) : reject(new Error("无法保存画面")), "image/png"));
      const saved = await request(`/api/export?filename=${encodeURIComponent(filename)}`, {method: "POST", headers: {"Content-Type": "image/png"}, body: blob});
      if (typeof saved.url !== "string" || !saved.url.startsWith("/exports/")) throw new Error("保存画面失败");
      const anchor = document.createElement("a");
      anchor.download = saved.filename || filename; anchor.href = saved.url; anchor.hidden = true;
      document.body.appendChild(anchor); anchor.click(); anchor.remove(); notice();
    } catch (error) { notice(error.message || "保存画面失败"); }
    finally { state.exporting = false; $("exportButton").disabled = !state.snapshot; }
  });
  $("calibrationButton").addEventListener("click", () => { calibrationDisplay(); $("calibrationDialog").showModal(); });
  $("closeCalibration").addEventListener("click", () => $("calibrationDialog").close());
  $("importCalibration").addEventListener("click", () => $("calibrationUpload").click());
  $("defaultCalibration").addEventListener("click", () => { state.calibration = null; state.calibrationName = "默认参数"; saveCalibration(); });
  $("calibrationUpload").addEventListener("change", async event => {
    const file = event.target.files?.[0]; if (!file) return;
    try {
      const value = validateCalibrationInput(JSON.parse(await file.text()));
      state.calibration = value; state.calibrationName = file.name; saveCalibration(); notice();
    } catch (error) { notice(`导入失败 · ${error.message}`); }
    event.target.value = "";
  });
  $("exitApp").addEventListener("click", async () => {
    video.pause(); stopSource(); $("exitApp").disabled = true;
    try { await request("/api/shutdown", {method: "POST"}); status("已退出"); state.ready = false; }
    catch (error) { notice(error.message); $("exitApp").disabled = false; }
  });
  if (impactPage) {
    $("pointButton").addEventListener("click", () => {
      const selecting = !state.pointMode;
      video.pause();
      // Retain the presented image and its observation; invalidate the frame in flight.
      resetEpoch(false);
      state.pointMode = selecting;
      $("stage").dataset.pointMode = String(selecting);
      renderImpact(); updateControls();
    });
    $("clearPointButton").addEventListener("click", () => { state.impactPoint = null; state.impactResult = null; renderImpact(); drawFrame(); });
    canvas.addEventListener("click", async event => {
      if (!state.pointMode || !selectedObservation()) return;
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
      const left = rect.left + (rect.width - canvas.width * scale) / 2;
      const top = rect.top + (rect.height - canvas.height * scale) / 2;
      const point = [(event.clientX - left) / scale, (event.clientY - top) / scale];
      if (point[0] < 0 || point[1] < 0 || point[0] >= canvas.width || point[1] >= canvas.height) return;
      state.impactPoint = point; state.impactResult = null; state.pointMode = false; $("stage").dataset.pointMode = "false"; drawFrame(); renderImpact();
      const epoch = state.epoch, selected = state.selectedIndex;
      try {
        const result = await request("/api/impact", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({observation: selectedObservation(), point})});
        if (epoch !== state.epoch || selected !== state.selectedIndex) return;
        state.impactResult = result; renderImpact(); notice();
      } catch (error) { notice(error.message); }
    });
  }
  window.addEventListener("beforeunload", () => { state.analysisController?.abort(); if (state.stream) state.stream.getTracks().forEach(track => track.stop()); });
  async function initialize() {
    frameLoop(); renderMedia(); calibrationDisplay();
    try {
      const [health, media] = await Promise.all([request("/api/health"), request("/api/media")]);
      const initialMedia = media.items || [];
      state.media = [...state.media, ...initialMedia.filter(item => !state.media.some(existing => existing.id === item.id))];
      renderMedia();
      status("模型加载中");
      if (state.media.length && !state.source) await selectMedia(state.media[0]);
      await prepareDetectionMode();
    } catch (error) { status("连接失败", "error"); notice(error.message); }
  }
  void initialize();
})();
