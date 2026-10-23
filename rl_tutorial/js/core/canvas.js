/* canvas.js — 高清画布、"只在可见时运行"的动画循环、迷你绘图工具 */
(function () {
  'use strict';
  const RL = window.RL = window.RL || {};

  RL.css = (name, el) => getComputedStyle(el || document.documentElement).getPropertyValue(name).trim();
  RL.lerp = (a, b, t) => a + (b - a) * t;
  RL.ease = { inOut: t => (t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2), out: t => 1 - (1 - t) ** 3 };
  RL.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches || /[?&]still\b/.test(location.search);
  RL.alpha = function (hex, a) {
    const h = hex.replace('#', ''), n = parseInt(h.length === 3 ? h.replace(/./g, c => c + c) : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  RL.font = (size, weight, mono) => `${weight || 500} ${size}px ${RL.css(mono ? '--font-mono' : '--font-sans')}`;

  // 高清画布：宽度跟随容器，高度 = height(w) 或 w * aspect
  RL.canvas = function (parent, opts) {
    const cv = document.createElement('canvas');
    if (opts.label) { cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', opts.label); }
    parent.appendChild(cv);
    const ctx = cv.getContext('2d');
    const api = { cv, ctx, w: 0, h: 0, draw: opts.draw || (() => {}) };
    let lastW = -1;
    const resize = () => {
      const w = parent.clientWidth;
      if (w === lastW || w === 0) return;
      lastW = w;
      const h = opts.height ? (typeof opts.height === 'function' ? opts.height(w) : opts.height) : Math.round(w * (opts.aspect || .5));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.width = w + 'px'; cv.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      api.w = w; api.h = h;
      api.redraw();
    };
    api.redraw = () => { if (!api.w) return; ctx.clearRect(0, 0, api.w, api.h); api.draw(ctx, api.w, api.h); };
    new ResizeObserver(resize).observe(parent);
    resize();
    return api;
  };

  // 只在"可见且播放中"时运行。tick(dt) 返回 false 表示播完。
  RL.loop = function (el, tick, opts) {
    let visible = false, playing = !!(opts && opts.autoplay) && !RL.reducedMotion, raf = 0, last = 0;
    const frame = t => {
      raf = 0;
      if (!visible || !playing) return;
      const dt = last ? Math.min(0.05, (t - last) / 1000) : 0;
      last = t;
      if (tick(dt) === false) { api.pause(); return; }
      raf = requestAnimationFrame(frame);
    };
    const kick = () => { if (!raf && visible && playing) { last = 0; raf = requestAnimationFrame(frame); } };
    new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; kick(); }, { threshold: 0.02 }).observe(el);
    const api = {
      play() { playing = true; kick(); if (api.onchange) api.onchange(true); },
      pause() { playing = false; if (api.onchange) api.onchange(false); },
      toggle() { if (playing) api.pause(); else api.play(); },
      get playing() { return playing; }
    };
    return api;
  };

  // ---------- 绘图 ----------
  RL.scale = function (d0, d1, r0, r1) {
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  };
  // 坐标轴与网格。box = {x, y, w, h}；o = {x:[a,b], y:[a,b], xticks, yticks, xfmt, yfmt, xlabel, ylabel}
  RL.axes = function (ctx, box, o) {
    const sx = RL.scale(o.x[0], o.x[1], box.x, box.x + box.w), sy = RL.scale(o.y[0], o.y[1], box.y + box.h, box.y);
    const ink3 = RL.css('--ink-3'), line = RL.css('--line');
    ctx.save();
    ctx.lineWidth = 1; ctx.strokeStyle = line; ctx.fillStyle = ink3; ctx.font = RL.font(11, 500, true);
    (o.yticks || []).forEach(v => {
      const y = Math.round(sy(v)) + .5;
      ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText((o.yfmt || String)(v), box.x - 6, y);
    });
    (o.xticks || []).forEach(v => {
      const x = Math.round(sx(v)) + .5;
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText((o.xfmt || String)(v), x, box.y + box.h + 6);
    });
    ctx.strokeStyle = RL.css('--line-2'); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(box.x, box.y); ctx.lineTo(box.x, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h); ctx.stroke();
    ctx.font = RL.font(12, 600); ctx.fillStyle = RL.css('--ink-2');
    if (o.xlabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(o.xlabel, box.x + box.w / 2, box.y + box.h + 24); }
    if (o.ylabel) { ctx.save(); ctx.translate(box.x - (o.ylabelOffset || 40), box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.ylabel, 0, 0); ctx.restore(); }
    ctx.restore();
    return { sx, sy };
  };
  RL.path = function (ctx, pts, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
    ctx.restore();
  };
  RL.sample = (f, a, b, n) => Array.from({ length: n + 1 }, (_, i) => { const x = a + (b - a) * i / n; return [x, f(x)]; });
  RL.rrect = function (ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
  };
  RL.text = function (ctx, str, x, y, o) {
    o = o || {};
    ctx.save();
    ctx.font = RL.font(o.size || 12, o.weight || 500, o.mono);
    ctx.fillStyle = o.color || RL.css('--ink-2');
    ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'middle';
    ctx.fillText(str, x, y);
    ctx.restore();
  };
  RL.dot = function (ctx, x, y, r, fill, stroke) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
  };
})();
