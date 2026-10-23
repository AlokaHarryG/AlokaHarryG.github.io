/* collapse.js — 图 7.4 两种失败模式示意：System-One 坍缩（依据论文 Fig. 1 的描述）与梯度淹没（论文报告 7–12×） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const smooth = x => { x = C.clip(x, 0, 1); return x * x * (3 - 2 * x); };

  RL.figure('collapse', function (el) {
    const S = 300, STOP = 150, U0 = [.95, .08, .9, .12], UK = [.78, .3, .72, .36];
    // 示意的动力学：无 KL 时四条推理在 ~100 步内缩为空串、读数一起变成"含糊的 0.55"；有 KL 时推理保留、读数学会在推理之后适度对冲
    const runs = {
      nokl: { len: (i, s) => Math.pow(C.clip(1 - s / (84 + 8 * i), 0, 1), 1.2), u: (i, s) => RL.lerp(U0[i], .55, smooth(s / 100)) },
      kl: { len: (i, s) => .82 + .14 * Math.cos(s / 23 + i * 1.3), u: (i, s) => RL.lerp(U0[i], UK[i], smooth(s / 220)) }
    };
    const variance = (k, s) => { const us = [0, 1, 2, 3].map(i => runs[k].u(i, s)), m = C.mean(us); return C.mean(us.map(u => 2 * (u - m) ** 2)); };
    const words = [0, 1, 2, 3].map(i => { const r = C.rng(40 + i), ws = []; for (let k = 0; k < 40; k++) ws.push(14 + r() * 32); return ws; });
    let s = RL.reducedMotion ? S : 0;
    const layout = w => (w >= 760
      ? { pan: [{ x: 0, y: 0, w: w * .6, h: 152 }, { x: 0, y: 164, w: w * .6, h: 152 }], vp: { x: 62, y: 352, w: w * .6 - 74, h: 104 }, gb: { x: w * .6 + 34, y: 14, w: w * .4 - 40 }, H: 492 }
      : { pan: [{ x: 0, y: 0, w, h: 152 }, { x: 0, y: 164, w, h: 152 }], vp: { x: 62, y: 352, w: w - 74, h: 104 }, gb: { x: 0, y: 512, w }, H: 760 });

    const drawPanel = (ctx, b, k) => {
      const nok = k === 'nokl', col = nok ? RL.css('--neg') : RL.css('--c-rl'), ss = nok ? Math.min(s, STOP) : s;
      RL.rrect(ctx, b.x, b.y, b.w, b.h, 12); ctx.fillStyle = RL.alpha(col, .05); ctx.fill(); ctx.strokeStyle = RL.alpha(col, .3); ctx.lineWidth = 1; ctx.stroke();
      RL.text(ctx, nok ? '无 KL 锚（端到端训练）' : '有 KL 锚 β = 0.04', b.x + 12, b.y + 16, { size: 12.5, weight: 750, color: col });
      RL.text(ctx, `第 ${Math.round(ss)} 步`, b.x + b.w - 12, b.y + 16, { size: 11, mono: true, color: RL.css('--ink-3'), align: 'right' });
      const x0 = b.x + 36, maxW = b.w - 36 - 150, bx = b.x + b.w - 100;
      for (let i = 0; i < 4; i++) {
        const y = b.y + 36 + i * 28, L = runs[k].len(i, ss) * maxW, u = runs[k].u(i, ss);
        RL.text(ctx, 'r' + '₁₂₃₄'[i], b.x + 12, y + 8, { size: 12, mono: true, weight: 700, color: RL.css('--ink-2') });
        let x = x0;
        for (const ww of words[i]) {
          if (x - x0 >= L) break;
          const vis = Math.min(ww, L - (x - x0));
          RL.rrect(ctx, x, y + 3, vis, 10, 3); ctx.fillStyle = RL.alpha(RL.css('--ink-3'), .38); ctx.fill();
          x += ww + 5;
        }
        const ax = x0 + Math.min(L, maxW) + (L > 1 ? 6 : 0);
        RL.rrect(ctx, ax, y, 58, 16, 5); ctx.fillStyle = RL.css('--ink'); ctx.fill();
        RL.text(ctx, 'Answer:', ax + 29, y + 8.5, { size: 10, mono: true, weight: 700, color: '#fff', align: 'center' });
        RL.rrect(ctx, bx, y + 2, 88, 12, 4); ctx.fillStyle = RL.alpha(RL.css('--c-rlcd'), .75); ctx.fill();
        RL.rrect(ctx, bx, y + 2, 88 * u, 12, 4); ctx.fillStyle = RL.alpha(RL.css('--c-pg'), .85); ctx.fill();
      }
      if (nok && s > STOP) {
        RL.rrect(ctx, b.x + b.w / 2 - 150, b.y + b.h - 26, 300, 20, 10); ctx.fillStyle = RL.alpha(RL.css('--ink'), .85); ctx.fill();
        RL.text(ctx, '已坍缩：论文中该运行在第 150 步终止', b.x + b.w / 2, b.y + b.h - 16, { size: 11, weight: 650, color: '#fff', align: 'center' });
      }
    };
    const drawVar = (ctx, b) => {
      const vmax = Math.max(...Array.from({ length: 61 }, (_, j) => Math.max(variance('nokl', j * 5), variance('kl', j * 5)))) * 1.1;
      const { sx, sy } = RL.axes(ctx, b, { x: [0, S], y: [0, vmax], xticks: [0, 100, 150, 200, 300], yticks: [0, +(vmax / 2).toFixed(2), +vmax.toFixed(2)], yfmt: v => v.toFixed(2), xlabel: '训练步数', ylabel: '读数方差', ylabelOffset: 50 });
      RL.path(ctx, RL.sample(t => variance('nokl', t), 0, Math.min(s, STOP), 80).map(([x, y]) => [sx(x), sy(y)]), RL.css('--neg'), 2.6);
      RL.path(ctx, RL.sample(t => variance('kl', t), 0, s, 120).map(([x, y]) => [sx(x), sy(y)]), RL.css('--c-rl'), 2.6);
      RL.path(ctx, [[sx(s), b.y], [sx(s), b.y + b.h]], RL.css('--ink-3'), 1, [3, 3]);
    };
    const drawGrad = (ctx, b) => {
      const u = (b.w - 30) / 12.5, pg = RL.css('--c-pg'), og = RL.css('--c-grpo');
      RL.text(ctx, '梯度淹没：两个梯度分量的范数（论文：训练过程中的 RMS）', b.x, b.y + 6, { size: 12.5, weight: 750, color: RL.css('--ink') });
      RL.text(ctx, '路径项 ∇θ Rᵢ（教读数说准概率）', b.x, b.y + 36, { size: 12, color: RL.css('--ink-2') });
      RL.rrect(ctx, b.x, b.y + 48, u, 20, 5); ctx.fillStyle = pg; ctx.fill();
      RL.text(ctx, '1×', b.x + u + 8, b.y + 58, { size: 12, mono: true, weight: 700, color: pg });
      RL.text(ctx, '得分函数项（REINFORCE，强化推理）', b.x, b.y + 92, { size: 12, color: RL.css('--ink-2') });
      RL.rrect(ctx, b.x, b.y + 104, 7 * u, 20, 5); ctx.fillStyle = og; ctx.fill();
      ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = og; ctx.lineWidth = 1.5; RL.rrect(ctx, b.x + 7 * u, b.y + 104, 5 * u, 20, 5); ctx.stroke(); ctx.restore();
      ctx.fillStyle = RL.alpha(og, .25); ctx.fillRect(b.x + 7 * u, b.y + 104, 5 * u, 20);
      RL.text(ctx, '7–12×', b.x + 12 * u + 8, b.y + 114, { size: 12, mono: true, weight: 800, color: og });
      ['Adam 按梯度二阶矩的平方根来缩放步长。', '大而嘈杂的得分函数项主导了这个尺度，', '有用、低方差的读数信号只能迈出很小的一步。', '这一项权重越大，各项指标越差（论文 Table 4）。']
        .forEach((t, i) => RL.text(ctx, t, b.x, b.y + 156 + i * 21, { size: 12, color: RL.css('--ink-2') }));
    };
    const cv = RL.canvas(el, {
      label: '左侧：无 KL 锚与有 KL 锚时四条推理的长度和读数随训练步数的变化；右侧：两个梯度分量的相对大小',
      height: w => layout(w).H,
      draw(ctx, w) { const L = layout(w); drawPanel(ctx, L.pan[0], 'nokl'); drawPanel(ctx, L.pan[1], 'kl'); drawVar(ctx, L.vp); drawGrad(ctx, L.gb); }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: `<span><i style="background:${RL.alpha(RL.css('--ink-3'), .5)};height:8px"></i>推理文本（示意）</span><span><i style="background:${RL.css('--c-pg')};height:8px"></i>读数 u(是)</span><span><i style="background:${RL.css('--c-rlcd')};height:8px"></i>读数 u(否)</span><span><i style="background:${RL.css('--neg')}"></i>无 KL 锚</span><span><i style="background:${RL.css('--c-rl')}"></i>有 KL 锚</span>` }));
    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, dt => { if (s >= S) s = 0; s = Math.min(S, s + dt * 55); ss.set(s); cv.redraw(); sync(); return s < S; }, { autoplay: true });
    RL.playButton(ctr, loop);
    const ss = RL.slider(ctr, { label: '训练步数', min: 0, max: S, step: 1, value: s, fmt: v => Math.round(v), onInput: v => { s = v; loop.pause(); cv.redraw(); sync(); } });
    const s1 = RL.stat(st, '无 KL：读数方差 ='), s2 = RL.stat(st, '有 KL：读数方差 =');
    const sync = () => { s1.set(variance('nokl', Math.min(s, STOP)).toFixed(3) + (s >= 100 ? '（四条推理已相同）' : '')); s2.set(variance('kl', s).toFixed(3)); };
    sync();
  });
})();
