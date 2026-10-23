/* proper.js — 图 7.2 报多大的概率最划算？RLVR（线性）vs 严格恰当评分规则（在 p = q 处取最大） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const star = (ctx, x, y, r, col) => {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * .45 : r; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
    ctx.closePath(); ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
  };

  RL.figure('proper', function (el) {
    const L = window.RLLLM;
    let q = .7, p = .4, rule = 'brier', geo = [], sL = null, sR = null;
    const F = {
      rlvr: { f: v => L.rlvrExpectedBinary(v, q), y: [0, 1], ticks: [0, .25, .5, .75, 1], title: '只奖励答对（RLVR：按 p 采样作答，对得 1）', col: '--c-grpo' },
      brier: { f: v => L.brierExpected(L.bin(v), L.bin(q)), y: [-1, 1], ticks: [-1, -.5, 0, .5, 1], title: 'Brier 分数（严格恰当）', col: '--c-rlcd' },
      log: { f: v => L.logScoreExpected(L.bin(C.clip(v, .005, .995)), L.bin(q)), y: [-3, 0], ticks: [-3, -2, -1, 0], title: '对数分数（严格恰当）', col: '--c-rlcd' }
    };
    const argmax = f => { let b = 0, bv = -Infinity; for (let k = 0; k <= 1000; k++) { const v = f(k / 1000); if (v > bv + 1e-12) { bv = v; b = k / 1000; } } return b; };
    const layout = w => (w >= 640
      ? { plots: [{ x: 48, y: 34, w: w / 2 - 70, h: 220 }, { x: w / 2 + 48, y: 34, w: w / 2 - 70, h: 220 }], H: 300 }
      : { plots: [{ x: 48, y: 34, w: w - 62, h: 190 }, { x: 48, y: 314, w: w - 62, h: 190 }], H: 550 });
    const drawPlot = (ctx, b, spec) => {
      const col = RL.css(spec.col), { sx, sy } = RL.axes(ctx, b, { x: [0, 1], y: spec.y, xticks: [0, .25, .5, .75, 1], yticks: spec.ticks, yfmt: v => v.toFixed(2).replace(/\.?0+$/, '') || '0', xlabel: '报告的概率 p', ylabel: '期望得分', ylabelOffset: 38 });
      RL.text(ctx, spec.title, b.x, b.y - 18, { size: 12.5, weight: 750, color: col });
      RL.path(ctx, [[sx(q), b.y], [sx(q), b.y + b.h]], RL.css('--ink-3'), 1.2, [4, 4]);
      RL.text(ctx, `p = q = ${q.toFixed(2)}`, sx(q) + 5, b.y + 10, { size: 10.5, mono: true, color: RL.css('--ink-3') });
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      RL.path(ctx, RL.sample(spec.f, 0, 1, 200).map(([x, y]) => [sx(x), sy(y)]), col, 3);
      ctx.restore();
      const flat = Math.abs(spec.f(0) - spec.f(1)) < 1e-9 && spec === F.rlvr;
      const bx = argmax(spec.f);
      if (!flat) star(ctx, sx(bx), sy(spec.f(bx)), 9, RL.css('--ink'));
      const v = spec.f(p), py = sy(C.clip(v, spec.y[0], spec.y[1]));
      RL.dot(ctx, sx(p), py, 6.5, RL.css('--surface'), col);
      RL.text(ctx, v.toFixed(3), sx(p) + (p > .8 ? -10 : 10), py + 14, { size: 11, mono: true, weight: 700, color: RL.css('--ink'), align: p > .8 ? 'right' : 'left' });
      return { sx, best: flat ? null : bx, v };
    };
    const cv = RL.canvas(el, {
      label: '两条期望得分曲线：左边对报告概率是直线，右边在报告概率等于真实概率时最高',
      height: w => layout(w).H,
      draw(ctx, w) { const Lo = layout(w); geo = [drawPlot(ctx, Lo.plots[0], F.rlvr), drawPlot(ctx, Lo.plots[1], F[rule])]; geo.forEach((g, i) => { g.b = Lo.plots[i]; }); sync(); }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: '<span>★ 期望得分最高的报告</span><span>○ 你当前报告的 p</span><span>虚线：如实报告 p = q</span>' }));
    const ctr = RL.controls(el), st = RL.stats(el);
    RL.slider(ctr, { label: '真实概率 q', min: 0, max: 1, step: .01, value: q, fmt: v => v.toFixed(2), onInput: v => { q = v; cv.redraw(); } });
    const ps = RL.slider(ctr, { label: '报告概率 p', min: 0, max: 1, step: .01, value: p, fmt: v => v.toFixed(2), onInput: v => { p = v; cv.redraw(); } });
    RL.seg(ctr, { label: '右图的评分规则', value: rule, options: [{ value: 'brier', label: 'Brier' }, { value: 'log', label: '对数分数' }], onChange: v => { rule = v; cv.redraw(); } });
    sL = RL.stat(st, 'RLVR 最划算的报告：'); sR = RL.stat(st, '恰当评分最划算的报告：', 'pos');
    function sync() {
      if (!geo.length || !sL) return;
      sL.set(geo[0].best === null ? '任意 p 都一样（q = 0.5）' : `p = ${geo[0].best.toFixed(2)}（全押${geo[0].best > .5 ? '"是"' : '"否"'}）`);
      sR.set(`p = ${geo[1].best.toFixed(2)}（= q，说实话）`);
    }
    let drag = false;
    const setP = e => {
      const r = cv.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const g = geo.find(o => y >= o.b.y - 24 && y <= o.b.y + o.b.h + 30 && x >= o.b.x - 10 && x <= o.b.x + o.b.w + 10) || geo[0];
      p = C.clip(g.sx.inv(x), 0, 1); ps.set(p); cv.redraw();
    };
    cv.cv.addEventListener('pointerdown', e => { drag = true; cv.cv.setPointerCapture(e.pointerId); setP(e); });
    cv.cv.addEventListener('pointermove', e => { if (drag) setP(e); });
    cv.cv.addEventListener('pointerup', () => { drag = false; });
    sync();
  });
})();
