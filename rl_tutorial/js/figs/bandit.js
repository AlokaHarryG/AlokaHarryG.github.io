/* bandit.js — 图 2.1 三臂老虎机上的 REINFORCE：有基线 vs 无基线（各 20 个随机种子） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('bandit', function (el) {
    const MEANS = [0.2, 0.5, 1.0], RUNS = 20, STEPS = 300;
    const colNo = RL.css('--c-grpo'), colB = RL.css('--c-pg');
    let offset = 3, lr = 0.05, seedBase = 1, data, t = 0;
    const run = () => {
      data = [false, true].map(useBaseline => Array.from({ length: RUNS }, (_, i) =>
        C.reinforceBandit({ means: MEANS, noise: 1, offset, lr, steps: STEPS, useBaseline, baselineRate: 0.05, seed: seedBase + i })));
      t = RL.reducedMotion ? STEPS : 0;
    };
    run();
    // 第 i 步时，20 次运行的统计量
    const statAt = (k, i, f) => { const xs = data[k].map(h => f(h[i])); return { m: C.mean(xs), sd: C.std(xs) }; };

    const layout = w => (w >= 760
      ? { bars: { x: 0, y: 0, w: w * .34, h: 290 }, plot: { x: w * .34 + 62, y: 8, w: w * .66 - 76, h: 244 }, H: 300 }
      : { bars: { x: 0, y: 0, w, h: 280 }, plot: { x: 54, y: 300, w: w - 66, h: 210 }, H: 560 });

    const drawGroups = (ctx, b) => {
      const i = Math.floor(t), ink = RL.css('--ink');
      [0, 1].forEach(k => {
        const gy = b.y + k * b.h / 2, col = k ? colB : colNo;
        RL.text(ctx, k ? '有基线：权重 = G − b' : '无基线：权重 = G', b.x, gy + 12, { size: 13, weight: 750, color: col });
        MEANS.forEach((mu, a) => {
          const { m, sd } = statAt(k, i, x => x.p[a]);
          const y = gy + 34 + a * 33, x0 = b.x + 96, bw = Math.max(60, b.w - 96 - 44);
          RL.text(ctx, `动作 ${a + 1}`, b.x, y + 8, { size: 12, weight: 600, color: RL.css('--ink-2') });
          RL.text(ctx, `μ=${mu}`, b.x + 50, y + 8, { size: 11, mono: true, color: RL.css('--ink-3') });
          RL.rrect(ctx, x0, y, bw, 16, 5); ctx.fillStyle = RL.css('--surface-2'); ctx.fill();
          RL.rrect(ctx, x0, y, Math.max(3, bw * m), 16, 5); ctx.fillStyle = RL.alpha(col, .78); ctx.fill();
          // 须线：20 次运行之间 ±1 个标准差
          const xl = x0 + bw * C.clip(m - sd, 0, 1), xr = x0 + bw * C.clip(m + sd, 0, 1);
          ctx.strokeStyle = ink; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(xl, y + 8); ctx.lineTo(xr, y + 8); ctx.moveTo(xl, y + 3); ctx.lineTo(xl, y + 13); ctx.moveTo(xr, y + 3); ctx.lineTo(xr, y + 13); ctx.stroke();
          RL.text(ctx, m.toFixed(2), x0 + bw + 6, y + 8, { size: 11, mono: true, color: ink });
        });
      });
    };
    const drawPlot = (ctx, b) => {
      const { sx, sy } = RL.axes(ctx, b, { x: [0, STEPS], y: [0.1, 1.08], xticks: [0, 100, 200, 300], yticks: [0.2, 0.4, 0.6, 0.8, 1.0], yfmt: v => v.toFixed(1), xlabel: '训练步数', ylabel: '期望奖励（不含平移）', ylabelOffset: 44 });
      RL.path(ctx, [[sx(0), sy(1)], [sx(STEPS), sy(1)]], RL.css('--ink-3'), 1, [4, 4]);
      RL.text(ctx, '最优 = 1.0', sx(STEPS) - 4, sy(1) - 9, { size: 11, color: RL.css('--ink-3'), align: 'right' });
      const i = Math.floor(t);
      [0, 1].forEach(k => {
        const col = k ? colB : colNo;
        data[k].forEach(h => RL.path(ctx, h.slice(0, i + 1).map((x, j) => [sx(j), sy(x.expReward - offset)]), RL.alpha(col, .16), 1.1));
      });
      [0, 1].forEach(k => {
        const pts = Array.from({ length: i + 1 }, (_, j) => [sx(j), sy(statAt(k, j, x => x.expReward).m - offset)]);
        RL.path(ctx, pts, k ? colB : colNo, 3);
      });
    };

    const cv = RL.canvas(el, {
      label: '两种 REINFORCE 的对比：左侧是三个动作的平均选择概率及其跨种子标准差，右侧是 20 次运行的学习曲线',
      height: w => layout(w).H,
      draw(ctx, w) { const L = layout(w); drawGroups(ctx, L.bars); drawPlot(ctx, L.plot); }
    });
    const legend = RL.el('div', { class: 'legend', html: `<span><i style="background:${colNo}"></i>无基线（粗线 = 20 次平均，细线 = 单次运行）</span><span><i style="background:${colB}"></i>有基线（基线 b = 奖励的滑动平均）</span><span>须线 = 20 次运行间 ±1 个标准差</span>` });
    el.appendChild(legend);
    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, dt => {
      if (t >= STEPS) t = 0;
      t = Math.min(STEPS, t + dt * 60);
      cv.redraw(); sync();
      return t < STEPS;
    }, { autoplay: true });
    RL.playButton(ctr, loop);
    RL.button(ctr, '换一批随机种子', () => { seedBase += RUNS; run(); cv.redraw(); sync(); if (!RL.reducedMotion) loop.play(); });
    const restart = () => { run(); cv.redraw(); sync(); if (!RL.reducedMotion) loop.play(); };
    RL.slider(ctr, { label: '奖励整体平移', min: 0, max: 8, step: .5, value: offset, fmt: v => '+' + v.toFixed(1), onInput: v => { offset = v; restart(); } });
    RL.slider(ctr, { label: '学习率', min: .01, max: .15, step: .01, value: lr, fmt: v => v.toFixed(2), onInput: v => { lr = v; restart(); } });
    const sNo = RL.stat(st, '无基线 E[r] =', 'neg'), sB = RL.stat(st, '有基线 E[r] =', 'pos'), sT = RL.stat(st, '步数');
    const sync = () => {
      const i = Math.floor(t), f = k => { const s = statAt(k, i, x => x.expReward - offset); return `${s.m.toFixed(3)} ± ${s.sd.toFixed(3)}`; };
      sNo.set(f(0)); sB.set(f(1)); sT.set(i);
    };
    sync();
  });
})();
