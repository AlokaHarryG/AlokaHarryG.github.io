/* ppo.js — 图 4.2 在同一批数据上做多轮更新：裁剪 vs 不裁剪（真实梯度上升） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('ppo', function (el) {
    const MEANS = [1.0, .8, .5, .2, 0], K = 5, N = 16, EPOCHS = 40, OLD = [0, 0, 0, 0, 0];
    let seed = 1, lr = 1, eps = .2, batch, runs, e = 0;
    const recompute = () => {
      batch = C.sampleBanditBatch(OLD, MEANS, 1, N, seed);
      runs = { clip: C.ppoBatchUpdate(OLD, batch, { eps, lr, epochs: EPOCHS, clip: true }), noclip: C.ppoBatchUpdate(OLD, batch, { eps, lr, epochs: EPOCHS, clip: false }) };
    };
    const restart = () => { recompute(); e = RL.reducedMotion ? EPOCHS : 0; cv.redraw(); sync(); if (!RL.reducedMotion) loop.play(); };
    recompute();
    if (RL.reducedMotion) e = EPOCHS;
    const colC = RL.css('--c-ppo'), colU = RL.css('--c-grpo');
    const layout = w => (w >= 760
      ? { bars: { x: 0, y: 0, w: w * .42, h: 318 }, kl: { x: w * .42 + 60, y: 22, w: w * .58 - 74, h: 140 }, rat: { x: w * .42 + 60, y: 222, w: w * .58 - 74, h: 96 }, H: 330 }
      : { bars: { x: 0, y: 0, w, h: 300 }, kl: { x: 50, y: 334, w: w - 64, h: 130 }, rat: { x: 50, y: 520, w: w - 64, h: 96 }, H: 630 });

    const drawBars = (ctx, b) => {
      const i = Math.floor(e), pc = runs.clip[i].p, pu = runs.noclip[i].p, x0 = b.x + 150, bw = b.w - 150 - 44;
      RL.text(ctx, `第 ${i} / ${EPOCHS} 轮后的策略`, b.x, b.y + 10, { size: 13, weight: 750, color: RL.css('--ink') });
      for (let a = 0; a < K; a++) {
        const y = b.y + 32 + a * 57, cnt = batch.filter(s => s.a === a), mA = cnt.length ? C.mean(cnt.map(s => s.A)) : 0;
        RL.text(ctx, `动作 ${a + 1}  μ=${MEANS[a]}`, b.x, y + 8, { size: 12, weight: 650, color: RL.css('--ink-2') });
        RL.text(ctx, cnt.length ? `${cnt.length} 个样本 · 平均Â ${mA >= 0 ? '+' : ''}${mA.toFixed(2)}` : '批内没采到', b.x, y + 26, { size: 11, color: cnt.length ? (mA >= 0 ? RL.css('--pos') : RL.css('--neg')) : RL.css('--ink-3') });
        [[pc[a], colC], [pu[a], colU]].forEach(([p, col], k) => {
          const yy = y + k * 17;
          RL.rrect(ctx, x0, yy, bw, 13, 4); ctx.fillStyle = RL.css('--surface-2'); ctx.fill();
          RL.rrect(ctx, x0, yy, Math.max(2, bw * p), 13, 4); ctx.fillStyle = RL.alpha(col, .85); ctx.fill();
          RL.text(ctx, p.toFixed(2), x0 + bw + 6, yy + 7, { size: 10, mono: true, color: RL.css('--ink') });
        });
      }
      const xo = x0 + bw * .2;
      RL.path(ctx, [[xo, b.y + 26], [xo, b.y + 32 + K * 57 - 22]], RL.css('--ink-3'), 1.2, [3, 3]);
      RL.text(ctx, 'π_old = 0.2', xo + 4, b.y + 22, { size: 10, mono: true, color: RL.css('--ink-3') });
    };
    const drawKL = (ctx, b) => {
      const maxKL = Math.max(.1, ...runs.noclip.map(s => s.kl)) * 1.08, i = Math.floor(e);
      const { sx, sy } = RL.axes(ctx, b, { x: [0, EPOCHS], y: [0, maxKL], xticks: [0, 10, 20, 30, 40], yticks: [0, maxKL / 2, maxKL].map(v => +v.toFixed(2)), yfmt: v => v.toFixed(2), xlabel: '在同一批数据上的更新轮数', ylabel: 'KL(π_old‖π)', ylabelOffset: 44 });
      [['noclip', colU], ['clip', colC]].forEach(([k, col]) => {
        RL.path(ctx, runs[k].slice(0, i + 1).map((s, j) => [sx(j), sy(s.kl)]), col, 2.6);
        RL.dot(ctx, sx(i), sy(runs[k][i].kl), 4, col);
      });
    };
    const drawRatios = (ctx, b) => {
      const i = Math.floor(e), lx = v => b.x + (Math.log2(v) + 2) / 4 * b.w, ink3 = RL.css('--ink-3');
      RL.text(ctx, '每个样本的比例 ρ = π_θ(a)/π_old(a)（对数刻度）', b.x, b.y - 12, { size: 12, weight: 650, color: RL.css('--ink-2') });
      ctx.fillStyle = RL.alpha(colC, .1); ctx.fillRect(lx(1 - eps), b.y, lx(1 + eps) - lx(1 - eps), b.h - 18);
      [.25, .5, 1, 2, 4].forEach(v => { RL.path(ctx, [[lx(v), b.y], [lx(v), b.y + b.h - 18]], RL.css('--line'), 1); RL.text(ctx, String(v), lx(v), b.y + b.h - 6, { size: 10, mono: true, color: ink3, align: 'center' }); });
      [['clip', '裁剪', colC], ['noclip', '不裁剪', colU]].forEach(([k, name, col], row) => {
        const yc = b.y + 18 + row * 38;
        RL.text(ctx, name, b.x - 8, yc, { size: 11, weight: 650, color: col, align: 'right' });
        runs[k][i].ratios.forEach((r, n) => {
          const A = batch[n].A, off = r < .25 || r > 4, x = lx(C.clip(r, .25, 4)), y = yc + ((n % 4) - 1.5) * 5.5;
          const c = A >= 0 ? RL.css('--pos') : RL.css('--neg'), dead = k === 'clip' && C.ppoClipGradCoef(r, A, eps) === 0;
          if (off) { ctx.fillStyle = c; ctx.beginPath(); const d = r > 4 ? 1 : -1; ctx.moveTo(x + d * 6, y); ctx.lineTo(x, y - 4); ctx.lineTo(x, y + 4); ctx.closePath(); ctx.fill(); }
          else if (dead) RL.dot(ctx, x, y, 3.6, RL.css('--surface'), c);
          else RL.dot(ctx, x, y, 3.6, c);
        });
      });
    };

    const cv = RL.canvas(el, {
      label: 'PPO 多轮更新的对比：左侧是裁剪与不裁剪两种策略的动作概率，右上是 KL 散度曲线，右下是每个样本的概率比',
      height: w => layout(w).H,
      draw(ctx, w) { const L = layout(w); drawBars(ctx, L.bars); drawKL(ctx, L.kl); drawRatios(ctx, L.rat); }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: `<span><i style="background:${colC}"></i>PPO 裁剪（ε = 0.2）</span><span><i style="background:${colU}"></i>不裁剪：直接最大化 ρÂ</span><span><i class="dot" style="background:${RL.css('--pos')}"></i>Â > 0 的样本</span><span><i class="dot" style="background:${RL.css('--neg')}"></i>Â < 0 的样本</span><span><i class="dot" style="border:2px solid ${RL.css('--ink-3')};background:none"></i>空心：已被裁剪、不再提供梯度</span><span>▶ 超出刻度范围</span>` }));
    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, dt => { if (e >= EPOCHS) e = 0; e = Math.min(EPOCHS, e + dt * 8); cv.redraw(); sync(); return e < EPOCHS; }, { autoplay: true });
    RL.playButton(ctr, loop);
    RL.button(ctr, '重新采样一批', () => { seed++; restart(); });
    RL.slider(ctr, { label: '学习率', min: .2, max: 2, step: .1, value: lr, fmt: v => v.toFixed(1), onInput: v => { lr = v; restart(); } });
    RL.slider(ctr, { label: 'ε', min: .05, max: .5, step: .01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; restart(); } });
    const sK = RL.stat(st, 'KL：'), sR = RL.stat(st, '真实期望奖励：'), sF = RL.stat(st, '被裁剪样本比例 =');
    const sync = () => {
      const i = Math.floor(e), c = runs.clip[i], u = runs.noclip[i];
      sK.set(`裁剪 ${c.kl.toFixed(3)} · 不裁剪 ${u.kl.toFixed(3)}`);
      sR.set(`裁剪 ${C.dot(c.p, MEANS).toFixed(3)} · 不裁剪 ${C.dot(u.p, MEANS).toFixed(3)}（π_old 为 0.500，最优 1.0）`);
      sF.set((c.clippedFrac * 100).toFixed(0) + '%');
    };
    sync();
  });
})();
