/* lineage.js — 首屏：方法演化路线图（SVG）与背景粒子网络 */
(function () {
  'use strict';
  const RL = window.RL;
  const NS = 'http://www.w3.org/2000/svg';
  RL.svg = function (tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (parent) parent.appendChild(e);
    return e;
  };
  const S = RL.svg;

  const NODES = [
    { href: '#ch1', label: 'Bellman 方程', year: '1957', c: '--c-rl' },
    { href: '#ch2', label: 'REINFORCE', year: '1992', c: '--c-pg' },
    { href: '#ch3', label: 'Actor-Critic · GAE', year: '1983 / 2015', c: '--c-ac' },
    { href: '#ch4-trpo', label: 'TRPO', year: '2015', c: '--c-ppo' },
    { href: '#ch4-clip', label: 'PPO', year: '2017', c: '--c-ppo' },
    { href: '#ch5', label: 'RLHF', year: '2017 / 2022', c: '--c-rlhf' },
    { href: '#ch6', label: 'GRPO', year: '2024', c: '--c-grpo' },
    { href: '#ch6-r1', label: 'R1 · RLVR', year: '2025', c: '--c-grpo' },
    { href: '#ch7', label: 'RLCD', year: '2026', c: '--c-rlcd' }
  ];

  RL.figure('lineage', function (el) {
    el.setAttribute('role', 'navigation');
    el.setAttribute('aria-label', '方法演化路线图');
    let state = null, t = 0;
    const build = () => {
      el.innerHTML = '';
      const W = el.clientWidth, vertical = W < 680, n = NODES.length;
      const H = vertical ? n * 54 + 16 : 210;
      const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, el);
      const defs = S('defs', {}, svg);
      const grad = S('linearGradient', vertical ? { id: 'lg', x1: 0, y1: 0, x2: 0, y2: 1 } : { id: 'lg', x1: 0, y1: 0, x2: 1, y2: 0 }, defs);
      NODES.forEach((nd, i) => S('stop', { offset: i / (n - 1), 'stop-color': RL.css(nd.c) }, grad));
      const f = S('filter', { id: 'glow', x: '-50%', y: '-50%', width: '200%', height: '200%' }, defs);
      S('feGaussianBlur', { stdDeviation: 4 }, f);

      const pts = NODES.map((_, i) => vertical
        ? [28 + (i % 2) * 22, 26 + i * 54]
        : [44 + i * (W - 88) / (n - 1), H / 2 + (i % 2 ? 34 : -34)]);
      let d = `M${pts[0][0]},${pts[0][1]}`;
      for (let i = 1; i < n; i++) {
        const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
        d += vertical ? ` C${x0},${(y0 + y1) / 2} ${x1},${(y0 + y1) / 2} ${x1},${y1}` : ` C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}`;
      }
      S('path', { d, fill: 'none', stroke: 'rgba(255,255,255,.12)', 'stroke-width': 2 }, svg);
      const prog = S('path', { d, fill: 'none', stroke: 'url(#lg)', 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
      const len = prog.getTotalLength();
      prog.setAttribute('stroke-dasharray', len);
      // 每个节点在路径上的弧长位置
      const samples = Array.from({ length: 300 }, (_, k) => { const p = prog.getPointAtLength(len * k / 299); return [p.x, p.y, len * k / 299]; });
      const at = pts.map(([x, y]) => samples.reduce((b, s) => ((s[0] - x) ** 2 + (s[1] - y) ** 2 < (b[0] - x) ** 2 + (b[1] - y) ** 2 ? s : b))[2]);

      const parts = [0, 1, 2].map(() => S('circle', { r: 4, fill: '#fff', filter: 'url(#glow)', opacity: 0 }, svg));
      const nodes = NODES.map((nd, i) => {
        const [x, y] = pts[i], col = RL.css(nd.c), up = !vertical && y < H / 2;
        const a = S('a', { href: nd.href, 'aria-label': `${nd.label}（${nd.year}）` }, svg);
        const halo = S('circle', { cx: x, cy: y, r: 16, fill: col, opacity: 0, filter: 'url(#glow)' }, a);
        const ring = S('circle', { class: 'ring', cx: x, cy: y, r: 13, fill: '#10142e', stroke: col, 'stroke-width': 2 }, a);
        const core = S('circle', { cx: x, cy: y, r: 4.5, fill: col }, a);
        const lx = vertical ? 86 : x, ly = vertical ? y - 7 : (up ? y - 40 : y + 30);
        const tl = S('text', { class: 'node-label', x: lx, y: ly, 'text-anchor': vertical ? 'start' : 'middle' }, a);
        tl.textContent = nd.label;
        const ty = S('text', { class: 'node-year', x: lx, y: ly + 16, 'text-anchor': vertical ? 'start' : 'middle' }, a);
        ty.textContent = nd.year;
        return { halo, ring, core, tl, ty, g: a };
      });
      state = { prog, len, at, nodes, parts };
      render();
    };
    const render = () => {
      if (!state) return;
      const { prog, len, at, nodes, parts } = state;
      const draw = RL.reducedMotion ? 1 : RL.ease.inOut(Math.min(1, t / 2.6));
      prog.setAttribute('stroke-dashoffset', len * (1 - draw));
      const heads = parts.map((p, k) => {
        if (draw < 1 || RL.reducedMotion) { p.setAttribute('opacity', 0); return -1e9; }
        const s = ((t - 2.6) * 70 + k * len / 3) % len, q = prog.getPointAtLength(s);
        p.setAttribute('cx', q.x); p.setAttribute('cy', q.y); p.setAttribute('opacity', .9);
        return s;
      });
      nodes.forEach((nd, i) => {
        const lit = draw * len >= at[i] - 1;
        const near = Math.max(0, 1 - Math.min(...heads.map(h => Math.abs(h - at[i]))) / 40);
        nd.g.style.opacity = lit ? 1 : .32;
        nd.halo.setAttribute('opacity', lit ? .25 + .5 * near : 0);
        nd.ring.setAttribute('r', 13 + 3 * near);
      });
    };
    let lastW = 0;
    new ResizeObserver(() => { if (el.clientWidth !== lastW) { lastW = el.clientWidth; build(); } }).observe(el);
    const loop = RL.loop(el, dt => { t += dt; render(); }, { autoplay: true });
    if (RL.reducedMotion) render();
    return loop;
  });

  // 背景：缓慢漂移的粒子网络
  RL.figure('herobg', function (el) {
    const cols = ['--c-rl', '--c-pg', '--c-ppo', '--c-grpo', '--c-rlcd'].map(c => RL.css(c));
    const r = window.RLCore.rng(3);
    const P = Array.from({ length: 70 }, () => ({ x: r(), y: r(), vx: (r() - .5) * .012, vy: (r() - .5) * .012, c: cols[Math.floor(r() * cols.length)] }));
    const cv = RL.canvas(el, {
      height: () => el.clientHeight || 600,
      draw(ctx, w, h) {
        P.forEach((p, i) => {
          for (let j = i + 1; j < P.length; j++) {
            const q = P[j], dx = (p.x - q.x) * w, dy = (p.y - q.y) * h, d = Math.hypot(dx, dy);
            if (d < 130) { ctx.strokeStyle = RL.alpha(p.c, .18 * (1 - d / 130)); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x * w, p.y * h); ctx.lineTo(q.x * w, q.y * h); ctx.stroke(); }
          }
          RL.dot(ctx, p.x * w, p.y * h, 1.8, RL.alpha(p.c, .7));
        });
      }
    });
    RL.loop(el, dt => {
      P.forEach(p => { p.x += p.vx * dt; p.y += p.vy * dt; if (p.x < 0 || p.x > 1) p.vx *= -1; if (p.y < 0 || p.y > 1) p.vy *= -1; });
      cv.redraw();
    }, { autoplay: true });
  });
})();
