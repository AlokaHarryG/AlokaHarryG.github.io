/* clip.js — 图 4.1 PPO 裁剪目标（对照 PPO 论文 Figure 1），可拖动比例 ρ */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('clip', function (el) {
    let eps = .2, rho = 1.35, geo = [];
    const layout = w => (w >= 640
      ? { plots: [{ x: 50, y: 34, w: w / 2 - 74, h: 230 }, { x: w / 2 + 50, y: 34, w: w / 2 - 74, h: 230 }], H: 300 }
      : { plots: [{ x: 50, y: 34, w: w - 66, h: 200 }, { x: 50, y: 314, w: w - 66, h: 200 }], H: 550 });

    const drawPlot = (ctx, b, A) => {
      const ppo = RL.css('--c-ppo'), org = RL.css('--c-grpo'), ink3 = RL.css('--ink-3');
      const { sx, sy } = RL.axes(ctx, b, { x: [0, 2], y: A > 0 ? [-.1, 2.1] : [-2.1, .1], xticks: [0, .5, 1, 1.5, 2], yticks: A > 0 ? [0, .5, 1, 1.5, 2] : [-2, -1.5, -1, -.5, 0], yfmt: v => v.toFixed(1), xlabel: '比例 ρ = π_θ / π_old', ylabel: '目标贡献', ylabelOffset: 40 });
      RL.text(ctx, A > 0 ? 'Â > 0（好动作）' : 'Â < 0（坏动作）', b.x, b.y - 18, { size: 13, weight: 750, color: A > 0 ? RL.css('--pos') : RL.css('--neg') });
      // 灰色：该样本已不提供梯度
      const z0 = A > 0 ? 1 + eps : 0, z1 = A > 0 ? 2 : 1 - eps;
      ctx.fillStyle = RL.alpha(RL.css('--ink-3'), .1); ctx.fillRect(sx(z0), b.y, sx(z1) - sx(z0), b.h);
      RL.text(ctx, '梯度 = 0', (sx(z0) + sx(z1)) / 2, b.y + b.h - 14, { size: 11, weight: 650, color: ink3, align: 'center' });
      // 未裁剪与裁剪之间：被"悲观下界"削掉的部分
      ctx.beginPath();
      RL.sample(x => x * A, z0, z1, 40).forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
      RL.sample(x => C.ppoClipObjective(x, A, eps), z1, z0, 40).forEach(([x, y]) => ctx.lineTo(sx(x), sy(y)));
      ctx.closePath(); ctx.fillStyle = RL.alpha(org, .14); ctx.fill();
      [1 - eps, 1, 1 + eps].forEach((v, i) => {
        RL.path(ctx, [[sx(v), b.y], [sx(v), b.y + b.h]], i === 1 ? RL.css('--line-2') : RL.alpha(ppo, .5), 1, i === 1 ? null : [4, 4]);
        if (i !== 1) RL.text(ctx, i ? '1+ε' : '1−ε', sx(v), b.y - 5, { size: 10, mono: true, color: ppo, align: 'center', baseline: 'bottom' });
      });
      RL.path(ctx, RL.sample(x => x * A, 0, 2, 2).map(([x, y]) => [sx(x), sy(y)]), RL.alpha(ppo, .75), 1.6, [6, 5]);
      RL.path(ctx, RL.sample(x => C.ppoClipObjective(x, A, eps), 0, 2, 200).map(([x, y]) => [sx(x), sy(y)]), org, 3.2);
      RL.text(ctx, 'ρ·Â（未裁剪）', sx(A > 0 ? 1.62 : .2), sy(A > 0 ? 1.95 : -.05), { size: 11, color: ppo, align: 'center' });
      RL.text(ctx, 'L^CLIP', sx(A > 0 ? 1.75 : .4), sy(A > 0 ? 1 + eps : -(1 - eps)) + (A > 0 ? 12 : -12), { size: 11, weight: 700, color: org, align: 'center' });
      // 当前 ρ 处的点与梯度箭头（梯度上升会把 ρ 往哪推）
      const L = C.ppoClipObjective(rho, A, eps), g = C.ppoClipGradCoef(rho, A, eps), px = sx(rho), py = sy(L);
      RL.dot(ctx, px, py, 7, RL.css('--surface'), RL.css('--ink'));
      if (g !== 0) {
        const dx = Math.sign(g) * 42;
        RL.path(ctx, [[px + Math.sign(g) * 10, py], [px + dx, py]], RL.css('--ink'), 2.4);
        ctx.beginPath(); ctx.moveTo(px + dx + Math.sign(g) * 7, py); ctx.lineTo(px + dx, py - 5); ctx.lineTo(px + dx, py + 5); ctx.closePath(); ctx.fillStyle = RL.css('--ink'); ctx.fill();
      } else RL.text(ctx, '不再推动', px, py + (A > 0 ? 18 : -18), { size: 11, weight: 700, color: RL.css('--ink'), align: 'center' });
      return sx;
    };

    const cv = RL.canvas(el, {
      label: 'PPO 裁剪目标随比例 ρ 变化的两条曲线：左为优势为正，右为优势为负。可拖动圆点',
      height: w => layout(w).H,
      draw(ctx, w) { const L = layout(w); geo = L.plots.map((b, i) => ({ b, sx: drawPlot(ctx, b, i ? -1 : 1) })); }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: `<span><i style="background:${RL.css('--c-grpo')};height:4px"></i>裁剪目标 L<sup>CLIP</sup> = min(ρÂ, clip(ρ)Â)</span><span><i style="background:${RL.css('--c-ppo')};height:2px"></i>未裁剪 ρÂ</span><span><i style="background:${RL.alpha(RL.css('--c-grpo'), .3)};height:10px"></i>被 min 削掉的"乐观收益"</span><span>箭头：梯度上升推动 ρ 的方向</span>` }));
    const setRho = (clientX, clientY) => {
      const r = cv.cv.getBoundingClientRect(), x = clientX - r.left, y = clientY - r.top;
      const hit = geo.find(g => y >= g.b.y - 20 && y <= g.b.y + g.b.h + 30 && x >= g.b.x - 10 && x <= g.b.x + g.b.w + 10) || geo[0];
      rho = C.clip(hit.sx.inv(x), .02, 1.98); rs.set(rho); cv.redraw(); sync();
    };
    let drag = false;
    cv.cv.addEventListener('pointerdown', e => { drag = true; cv.cv.setPointerCapture(e.pointerId); setRho(e.clientX, e.clientY); });
    cv.cv.addEventListener('pointermove', e => { if (drag) setRho(e.clientX, e.clientY); });
    cv.cv.addEventListener('pointerup', () => { drag = false; });
    const ctr = RL.controls(el), st = RL.stats(el);
    const rs = RL.slider(ctr, { label: '比例 ρ', min: .02, max: 1.98, step: .01, value: rho, fmt: v => v.toFixed(2), onInput: v => { rho = v; cv.redraw(); sync(); } });
    RL.slider(ctr, { label: '裁剪范围 ε', min: .05, max: .5, step: .01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; cv.redraw(); sync(); } });
    const sP = RL.stat(st, 'Â = +1：'), sN = RL.stat(st, 'Â = −1：');
    const f = A => `L = ${C.ppoClipObjective(rho, A, eps).toFixed(3)}，∂L/∂ρ = ${C.ppoClipGradCoef(rho, A, eps)}`;
    const sync = () => { sP.set(f(1)); sN.set(f(-1)); };
    sync();
  });
})();
