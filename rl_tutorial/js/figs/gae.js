/* gae.js — 图 3.1 从 TD 误差到 GAE：一条 8 步、只有终点奖励的轨迹 */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const SUB = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);

  RL.figure('gae', function (el) {
    const T = 8, R = [0, 0, 0, 0, 0, 0, 0, 1];
    let gamma = .99, lambda = .95, seed = 5, V, sel = 0, sweep = RL.reducedMotion ? null : { i: T - 1, c: 0 }, geo = null;
    // 不完美的评论家：真实回报 + 噪声（终止之后的自举值为 0）
    const makeCritic = () => { const r = C.rng(seed); V = Array.from({ length: T }, (_, t) => C.clip(0.99 ** (T - 1 - t) + .32 * C.randn(r), -.2, 1.25)); V.push(0); };
    makeCritic();
    const calc = () => {
      const g = C.gae(R, V, gamma, lambda), G = C.discountedReturns(R, gamma);
      return { ...g, mc: G.map((x, t) => x - V[t]) };
    };

    const layout = w => { const lw = w < 560 ? 58 : 92; return { lw, cw: (w - lw - 6) / T, rows: [{ k: 'r', y: 6, h: 36 }, { k: 'V', y: 50, h: 74 }, { k: 'd', y: 132, h: 100 }, { k: 'A', y: 240, h: 128 }], H: 374 }; };
    const LABELS = { r: ['奖励', 'r'], V: ['评论家', 'V(s)'], d: ['TD 误差', 'δ'], A: ['优势', 'Â'] };

    const cv = RL.canvas(el, {
      label: '8 步轨迹上的奖励、评论家估值、TD 误差与 GAE 优势条形图',
      height: w => layout(w).H,
      draw(ctx, w) {
        const L = layout(w), res = calc(), ink = RL.css('--ink'), ink3 = RL.css('--ink-3'), ac = RL.css('--c-ac');
        const colX = t => L.lw + t * L.cw + L.cw / 2, showNum = L.cw >= 46;
        const absMax = Math.max(.5, ...res.deltas.map(Math.abs), ...res.adv.map(Math.abs), ...res.mc.map(Math.abs));
        geo = L;
        ctx.fillStyle = RL.alpha(ac, .07);
        RL.rrect(ctx, L.lw + sel * L.cw + 2, 2, L.cw - 4, L.H - 4, 8); ctx.fill();
        L.rows.forEach(row => {
          RL.rrect(ctx, L.lw, row.y, w - L.lw - 6, row.h, 9); ctx.fillStyle = RL.alpha(RL.css('--surface-2'), .7); ctx.fill();
          RL.text(ctx, LABELS[row.k][0], 4, row.y + row.h / 2 - 8, { size: 12, weight: 700, color: RL.css('--ink-2') });
          RL.text(ctx, LABELS[row.k][1], 4, row.y + row.h / 2 + 9, { size: 12, mono: true, color: ink3 });
          const mid = row.y + row.h / 2, half = row.h / 2 - 14;
          if (row.k === 'd' || row.k === 'A') { ctx.strokeStyle = RL.css('--line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.lw + 6, mid + .5); ctx.lineTo(w - 12, mid + .5); ctx.stroke(); }
          for (let t = 0; t < T; t++) {
            const x = colX(t), bw = Math.min(30, L.cw * .52);
            if (row.k === 'r') {
              const on = R[t] !== 0;
              RL.rrect(ctx, x - 13, row.y + 6, 26, 24, 7); ctx.fillStyle = on ? RL.css('--pos') : RL.css('--surface'); ctx.fill();
              RL.text(ctx, on ? '+1' : '0', x, row.y + 18, { size: 12, mono: true, weight: 700, color: on ? '#fff' : ink3, align: 'center' });
            } else if (row.k === 'V') {
              const v = V[t], hh = Math.max(1, (row.h - 26) * Math.max(0, v) / 1.25), base = row.y + row.h - 8;
              ctx.fillStyle = RL.alpha(ac, .55); ctx.fillRect(x - bw / 2, base - hh, bw, hh);
              if (showNum) RL.text(ctx, v.toFixed(2), x, base - hh - 8, { size: 10, mono: true, color: ink, align: 'center' });
            } else {
              let v = row.k === 'd' ? res.deltas[t] : res.adv[t], s = 1;
              if (row.k === 'A' && sweep) { if (t < sweep.i) continue; if (t === sweep.i) s = RL.ease.out(Math.min(1, sweep.c / .35)); }
              const hh = half * Math.abs(v) / absMax * s, col = row.k === 'd' ? (v >= 0 ? RL.css('--pos') : RL.css('--neg')) : ac;
              ctx.fillStyle = RL.alpha(col, .82); ctx.fillRect(x - bw / 2, v >= 0 ? mid - hh : mid, bw, Math.max(1, hh));
              if (row.k === 'A') { // 虚线框：λ=1（蒙特卡洛）时的值
                const mh = half * Math.abs(res.mc[t]) / absMax;
                ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = ink3; ctx.lineWidth = 1.2;
                ctx.strokeRect(x - bw / 2 - 3, res.mc[t] >= 0 ? mid - mh : mid, bw + 6, Math.max(1, mh)); ctx.restore();
              }
              if (showNum && s === 1) RL.text(ctx, v.toFixed(2), x, v >= 0 ? mid - hh - 8 : mid + hh + 8, { size: 10, mono: true, color: ink, align: 'center' });
            }
          }
        });
        for (let t = 0; t < T; t++) RL.text(ctx, 't=' + t, colX(t), L.H - 3, { size: 10, mono: true, color: ink3, align: 'center', baseline: 'bottom' });
        if (sweep && sweep.i < T - 1) { // 回溯箭头：Â_{t+1} ×γλ → Â_t
          const A = L.rows[3], x1 = colX(sweep.i + 1), x0 = colX(sweep.i), y = A.y + 12;
          ctx.save(); ctx.strokeStyle = RL.css('--c-ppo'); ctx.fillStyle = RL.css('--c-ppo'); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x1, y + 6); ctx.quadraticCurveTo((x0 + x1) / 2, y - 14, x0 + 6, y + 4); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x0 + 2, y + 8); ctx.lineTo(x0 + 12, y + 6); ctx.lineTo(x0 + 6, y - 2); ctx.closePath(); ctx.fill(); ctx.restore();
          RL.text(ctx, '×γλ', (x0 + x1) / 2, y - 12, { size: 11, weight: 700, color: RL.css('--c-ppo'), align: 'center' });
        }
      }
    });
    el.appendChild(RL.el('div', { class: 'legend', html: `<span><i style="background:${RL.css('--pos')}"></i>δ > 0：比评论家预想的好</span><span><i style="background:${RL.css('--neg')}"></i>δ < 0：比预想的差</span><span><i style="background:${RL.css('--c-ac')}"></i>GAE 优势 Âₜ</span><span><i style="border:1.5px dashed ${RL.css('--ink-3')};height:10px;background:none"></i>虚线框：λ = 1 时的值 Gₜ − V(sₜ)</span>` }));

    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, dt => {
      if (!sweep) return false;
      sweep.c += dt;
      if (sweep.c > .6) { sweep.c = 0; sweep.i--; if (sweep.i < 0) { sweep = null; sel = 0; } }
      if (sweep) sel = sweep.i;
      cv.redraw(); sync();
    }, { autoplay: true });
    RL.button(ctr, '逐步回溯', () => { sweep = { i: T - 1, c: 0 }; loop.play(); }, 'primary');
    RL.button(ctr, '换一个评论家', () => { seed++; makeCritic(); cv.redraw(); sync(); });
    RL.slider(ctr, { label: 'λ', min: 0, max: 1, step: .01, value: lambda, fmt: v => v.toFixed(2), onInput: v => { lambda = v; cv.redraw(); sync(); } });
    RL.slider(ctr, { label: 'γ', min: .8, max: 1, step: .01, value: gamma, fmt: v => v.toFixed(2), onInput: v => { gamma = v; cv.redraw(); sync(); } });
    const sEq = RL.stat(st, ''), sCmp = RL.stat(st, '');
    const sync = () => {
      const res = calc(), t = sel, nxt = t + 1 < T ? res.adv[t + 1] : 0;
      sEq.set(`Â${SUB(t)} = δ${SUB(t)} + γλ·Â${SUB(t + 1)} = ${res.deltas[t].toFixed(3)} + ${(gamma * lambda).toFixed(3)} × ${nxt.toFixed(3)} = ${res.adv[t].toFixed(3)}`);
      sCmp.set(`对照：λ=0 → ${res.deltas[t].toFixed(3)}；λ=1 → ${res.mc[t].toFixed(3)}`);
    };
    cv.cv.addEventListener('click', e => {
      if (!geo) return;
      const r = cv.cv.getBoundingClientRect(), t = Math.floor((e.clientX - r.left - geo.lw) / geo.cw);
      if (t >= 0 && t < T && !sweep) { sel = t; cv.redraw(); sync(); }
    });
    sync();
  });
})();
