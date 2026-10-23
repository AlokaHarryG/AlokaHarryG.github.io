/* calib.js — 图 7.1 自动执行还是转人工？过度自信 vs 校准（合成数据，指标实时计算） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('calib', function (el) {
    const L = window.RLLLM, N = 400;
    const MODELS = [['overconfident', '过度自信（类似 RLVR / GRPO 训练后）', '--c-grpo'], ['calibrated', '校准（RLCD 追求的状态）', '--c-rlcd']];
    let seed = 7, tau = .8, data, jit, sx = null;
    const make = () => { data = L.simulateCalibration(N, seed); const r = C.rng(seed + 99); jit = Array.from({ length: N }, () => r()); };
    make();
    const split = k => {
      const auto = data[k].map((c, i) => (c >= tau ? i : -1)).filter(i => i >= 0);
      const err = auto.length ? auto.filter(i => !data.correct[i]).length / auto.length : 0;
      return { cov: auto.length / N, err, n: auto.length };
    };
    const layout = w => {
      if (w >= 760) { const rw = w * .36, s = Math.min(140, (rw - 72) / 2); return { strips: { x: 0, y: 0, w: w * .64 - 10, h: 268 }, rel: [{ x: w * .64 + 40, y: 44, s }, { x: w * .64 + 40 + s + 26, y: 44, s, noY: true }], H: 296 }; }
      const s = Math.min(150, (w - 110) / 2);
      return { strips: { x: 0, y: 0, w, h: 268 }, rel: [{ x: 36, y: 318, s }, { x: 36 + s + 52, y: 318, s }], H: 318 + s + 46 };
    };

    const drawStrips = (ctx, b) => {
      const ax = { x: b.x + 12, w: b.w - 24 }, rowH = (b.h - 40) / 2, ink3 = RL.css('--ink-3');
      sx = RL.scale(0, 1, ax.x, ax.x + ax.w);
      MODELS.forEach(([k, name, cvar], row) => {
        const y0 = b.y + row * rowH, col = RL.css(cvar), s = split(k);
        RL.rrect(ctx, ax.x, y0 + 22, ax.w, rowH - 28, 9); ctx.fillStyle = RL.css('--surface-2'); ctx.fill();
        ctx.save(); RL.rrect(ctx, ax.x, y0 + 22, ax.w, rowH - 28, 9); ctx.clip();
        ctx.fillStyle = RL.alpha(col, .1); ctx.fillRect(sx(tau), y0 + 22, ax.x + ax.w - sx(tau), rowH - 28);
        data[k].forEach((c, i) => RL.dot(ctx, sx(c), y0 + 30 + jit[i] * (rowH - 44), 2.5, RL.alpha(data.correct[i] ? RL.css('--pos') : RL.css('--neg'), .72)));
        ctx.restore();
        RL.text(ctx, name, ax.x, y0 + 10, { size: 12.5, weight: 750, color: col });
        RL.text(ctx, `自动执行 ${(s.cov * 100).toFixed(0)}% · 其中错误率 ${(s.err * 100).toFixed(1)}%`, ax.x + ax.w, y0 + 10, { size: 12, weight: 650, mono: true, color: RL.css('--ink'), align: 'right' });
      });
      const yb = b.y + b.h - 18;
      [0, .2, .4, .6, .8, 1].forEach(v => RL.text(ctx, v.toFixed(1), sx(v), yb + 14, { size: 10, mono: true, color: ink3, align: 'center' }));
      RL.text(ctx, '模型给出的置信度 →', ax.x + ax.w, yb + 30, { size: 11, color: ink3, align: 'right' });
      const tx = sx(tau);
      RL.path(ctx, [[tx, b.y + 18], [tx, b.y + b.h - 28]], RL.css('--ink'), 2);
      RL.rrect(ctx, tx - 34, b.y + b.h - 34, 68, 18, 9); ctx.fillStyle = RL.css('--ink'); ctx.fill();
      RL.text(ctx, `阈值 ${tau.toFixed(2)}`, tx, b.y + b.h - 25, { size: 10.5, weight: 700, color: '#fff', align: 'center' });
      RL.text(ctx, '← 转人工', tx - 42, b.y + b.h - 25, { size: 11, weight: 650, color: ink3, align: 'right' });
      RL.text(ctx, '自动执行 →', tx + 42, b.y + b.h - 25, { size: 11, weight: 650, color: ink3 });
    };
    const drawRel = (ctx, r, k, name, col) => {
      const rel = L.reliability(data[k], data.correct, 10), box = { x: r.x, y: r.y, w: r.s, h: r.s };
      const { sx: X, sy: Y } = RL.axes(ctx, box, { x: [0, 1], y: [0, 1], xticks: [0, .5, 1], yticks: [0, .5, 1], yfmt: v => (r.noY ? '' : v.toFixed(1)), xlabel: '置信度', ylabel: r.noY ? '' : '实际正确率', ylabelOffset: 30 });
      rel.bins.forEach(bn => { if (bn.n < 4) return; ctx.fillStyle = RL.alpha(col, .72); ctx.fillRect(X(bn.lo) + 1, Y(bn.acc), X(bn.hi) - X(bn.lo) - 2, Y(0) - Y(bn.acc)); });
      RL.path(ctx, [[X(0), Y(0)], [X(1), Y(1)]], RL.css('--ink-2'), 1.3, [4, 4]);
      RL.text(ctx, name, r.x, r.y - 26, { size: 12, weight: 750, color: col });
      RL.text(ctx, `ECE = ${rel.ece.toFixed(3)}`, r.x, r.y - 10, { size: 11, mono: true, color: RL.css('--ink-2') });
    };
    const cv = RL.canvas(el, {
      label: '两行散点：每个点是一道题，横坐标是置信度，颜色表示对错；竖线是自动执行的阈值。右侧是两个模型的可靠性图',
      height: w => layout(w).H,
      draw(ctx, w) {
        const Lo = layout(w);
        drawStrips(ctx, Lo.strips);
        drawRel(ctx, Lo.rel[0], 'overconfident', '过度自信', RL.css('--c-grpo'));
        drawRel(ctx, Lo.rel[1], 'calibrated', '校准', RL.css('--c-rlcd'));
      }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: `<span><i class="dot" style="background:${RL.css('--pos')}"></i>答对</span><span><i class="dot" style="background:${RL.css('--neg')}"></i>答错</span><span>两个模型面对同一批题、对错完全相同，只是报告的置信度不同</span>` }));
    const ctr = RL.controls(el), st = RL.stats(el);
    const ts = RL.slider(ctr, { label: '自动执行阈值', min: .5, max: .99, step: .01, value: tau, fmt: v => v.toFixed(2), onInput: v => { tau = v; cv.redraw(); sync(); } });
    RL.button(ctr, '换一批题目', () => { seed++; make(); cv.redraw(); sync(); });
    const sA = RL.stat(st, '两者准确率 ='), sO = RL.stat(st, '过度自信：', 'neg'), sC = RL.stat(st, '校准：', 'pos');
    const sync = () => {
      sA.set((C.mean(data.correct) * 100).toFixed(1) + '%');
      const o = split('overconfident'), c = split('calibrated');
      sO.set(`自动执行 ${o.n} 题，错 ${(o.err * 100).toFixed(1)}%`);
      sC.set(`自动执行 ${c.n} 题，错 ${(c.err * 100).toFixed(1)}%`);
    };
    let drag = false;
    const setT = e => { if (!sx) return; const r = cv.cv.getBoundingClientRect(); tau = C.clip(sx.inv(e.clientX - r.left), .5, .99); ts.set(tau); cv.redraw(); sync(); };
    cv.cv.addEventListener('pointerdown', e => { const r = cv.cv.getBoundingClientRect(); if (e.clientY - r.top > 268) return; drag = true; cv.cv.setPointerCapture(e.pointerId); setT(e); });
    cv.cv.addEventListener('pointermove', e => { if (drag) setT(e); });
    cv.cv.addEventListener('pointerup', () => { drag = false; });
    sync();
  });
})();
