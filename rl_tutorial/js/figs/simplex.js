/* simplex.js — 图 7.3 在概率单纯形上看方差恒等式 J(p,q) = E_r J(u(r),q) + E_r‖u(r) − p‖²（RLCD 论文 Prop. 1） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const PRESETS = {
    contradict: { label: '互相矛盾', us: [[.88, .06, .06], [.05, .9, .05], [.1, .1, .8], [.77, .05, .18]] },
    calibrated: { label: '各自校准', us: 'q' },
    mode: { label: '一致但过度自信', us: 'mode' },
    random: { label: '随机', us: 'random' }
  };

  RL.figure('simplex', function (el) {
    const L = window.RLLLM, COLS = ['--c-pg', '--c-grpo', '--c-rl', '--c-ppo'].map(c => RL.css(c));
    const mean = us => [0, 1, 2].map(k => C.mean(us.map(u => u[k])));
    let us = PRESETS.contradict.us.map(u => u.slice()), q = mean(us), drag = null, tri = null, preset = 'contradict', seed = 1;
    const apply = name => {
      preset = name; const P = PRESETS[name].us;
      if (P === 'q') us = us.map(() => q.slice());
      else if (P === 'mode') { const m = q.indexOf(Math.max(...q)); us = us.map(() => [0, 1, 2].map(k => (k === m ? .98 : .01))); }
      else if (P === 'random') { const r = C.rng(++seed); us = us.map(() => { const a = [r(), r(), r()].map(x => -Math.log(x + 1e-9)), s = C.sum(a); return a.map(x => x / s); }); }
      else { us = P.map(u => u.slice()); q = mean(us); }
    };
    const wrap = RL.el('div', { class: 'fig-grid' }), left = RL.el('div', { class: 'fig-stage' }), right = RL.el('div', { class: 'side-panel' });
    wrap.append(left, right); el.appendChild(wrap);
    const toXY = b => [b[0] * tri.A[0] + b[1] * tri.B[0] + b[2] * tri.C[0], b[0] * tri.A[1] + b[1] * tri.B[1] + b[2] * tri.C[1]];
    const toBary = (x, y) => {
      const { A, B, C: Cc } = tri, d = (B[1] - Cc[1]) * (A[0] - Cc[0]) + (Cc[0] - B[0]) * (A[1] - Cc[1]);
      let a = ((B[1] - Cc[1]) * (x - Cc[0]) + (Cc[0] - B[0]) * (y - Cc[1])) / d, b = ((Cc[1] - A[1]) * (x - Cc[0]) + (A[0] - Cc[0]) * (y - Cc[1])) / d;
      let c = 1 - a - b;
      [a, b, c] = [a, b, c].map(v => Math.max(.005, v)); const s = a + b + c;
      return [a / s, b / s, c / s];
    };

    const cv = RL.canvas(left, {
      label: '三选项概率单纯形：四个可拖动的彩色点是四条推理的读数，十字是它们的平均，星号是真实分布',
      height: w => Math.min(380, Math.round(w * .82)),
      draw(ctx, w, h) {
        const side = Math.min(w - 60, (h - 60) / .866), cx = w / 2, top = 30;
        tri = { A: [cx, top], B: [cx - side / 2, top + side * .866], C: [cx + side / 2, top + side * .866] };
        const ink3 = RL.css('--ink-3');
        ctx.beginPath(); [tri.A, tri.B, tri.C].forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath();
        ctx.fillStyle = RL.css('--surface-2'); ctx.fill(); ctx.strokeStyle = RL.css('--line-2'); ctx.lineWidth = 1.5; ctx.stroke();
        for (let k = 0; k < 3; k++) for (let t = .25; t < 1; t += .25) {
          const a = [0, 0, 0], b = [0, 0, 0]; a[k] = t; b[k] = t; a[(k + 1) % 3] = 1 - t; b[(k + 2) % 3] = 1 - t;
          RL.path(ctx, [toXY(a), toXY(b)], RL.css('--line'), 1);
        }
        [['A', tri.A, 0, -14], ['B', tri.B, -14, 12], ['C', tri.C, 14, 12]].forEach(([n, p, dx, dy]) => RL.text(ctx, `100% 选 ${n}`, p[0] + dx, p[1] + dy, { size: 11, weight: 700, color: ink3, align: dx < 0 ? 'right' : dx > 0 ? 'left' : 'center' }));
        const p = mean(us), P = toXY(p);
        us.forEach((u, i) => RL.path(ctx, [toXY(u), P], RL.alpha(COLS[i], .6), 1.6, [4, 4]));
        us.forEach((u, i) => { const [x, y] = toXY(u), o = i * 2.2 - 3.3; RL.dot(ctx, x + o, y + o, 9, COLS[i], '#fff'); RL.text(ctx, String(i + 1), x + o, y + o + 1, { size: 10, weight: 800, color: '#fff', align: 'center' }); });
        ctx.strokeStyle = RL.css('--ink'); ctx.lineWidth = 2.6;
        ctx.beginPath(); ctx.moveTo(P[0] - 11, P[1]); ctx.lineTo(P[0] + 11, P[1]); ctx.moveTo(P[0], P[1] - 11); ctx.lineTo(P[0], P[1] + 11); ctx.stroke();
        const Q = toXY(q); // 星号画在十字之上：两者重合时仍能看见 q
        ctx.save(); ctx.translate(Q[0], Q[1]); ctx.beginPath();
        for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? 4 : 9; ctx.lineTo(r * Math.cos(a), r * Math.sin(a)); }
        ctx.closePath(); ctx.fillStyle = RL.css('--c-rlcd'); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();
      }
    });
    left.appendChild(RL.el('div', { class: 'legend', html: `<span><i class="dot" style="background:${COLS[0]}"></i>uᵢ：第 i 条推理的读数（可拖动）</span><span>✚ 混合分布 p = 平均 uᵢ</span><span style="color:var(--c-rlcd)">★ 真实分布 q（可拖动）</span>` }));
    const panel = RL.el('div', { class: 'ident', 'aria-live': 'polite' });
    right.appendChild(panel);
    const bar = v => `<span class="ib"><i class="${v >= 0 ? 'p' : 'n'}" style="width:${Math.min(50, Math.abs(v) * 50)}%;${v >= 0 ? 'left:50%' : `left:${50 - Math.min(50, Math.abs(v) * 50)}%`}"></i></span>`;
    const render = () => {
      const v = L.varianceIdentity(us, q), per = us.map(u => L.brierExpected(u, q));
      panel.innerHTML = `<div class="ident-row mix"><span class="il">混合得分 J(p, q)</span>${bar(v.mixture)}<b class="mono">${v.mixture.toFixed(3)}</b></div>
        <div class="ident-op">=</div>
        <div class="ident-row per"><span class="il">逐条得分的平均 E J(uᵢ, q)</span>${bar(v.perRationale)}<b class="mono">${v.perRationale.toFixed(3)}</b></div>
        <div class="ident-op">+</div>
        <div class="ident-row var"><span class="il">分歧奖金 E‖uᵢ − p‖²</span>${bar(v.variance)}<b class="mono">${v.variance.toFixed(3)}</b></div>
        <div class="ident-chips">${per.map((s, i) => `<span style="--c:${COLS[i]}">J(u${'₁₂₃₄'[i]}, q) = <b class="mono">${s.toFixed(3)}</b></span>`).join('')}</div>
        <p class="ident-note">RLCD 只优化中间那一项（逐条推理目标）。验证：等式两边之差 = <span class="mono">${Math.abs(v.mixture - v.perRationale - v.variance).toExponential(1)}</span></p>`;
    };
    const ctr = RL.controls(el);
    const seg = RL.seg(ctr, { label: '预设', value: preset, options: Object.entries(PRESETS).map(([k, o]) => ({ value: k, label: o.label })), onChange: v => { apply(v); cv.redraw(); render(); } });
    const pick = (x, y) => {
      const cands = us.map((u, i) => [i, toXY(u)]).concat([['q', toXY(q)]]);
      let best = null, bd = 22;
      cands.forEach(([i, [px, py]]) => { const d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; best = i; } });
      return best;
    };
    const move = e => {
      const r = cv.cv.getBoundingClientRect(), b = toBary(e.clientX - r.left, e.clientY - r.top);
      if (drag === 'q') q = b; else us[drag] = b;
      cv.redraw(); render();
    };
    cv.cv.addEventListener('pointerdown', e => { if (!tri) return; const r = cv.cv.getBoundingClientRect(); drag = pick(e.clientX - r.left, e.clientY - r.top); if (drag === null) return; cv.cv.setPointerCapture(e.pointerId); seg.set(null); move(e); });
    cv.cv.addEventListener('pointermove', e => { if (drag !== null) move(e); });
    cv.cv.addEventListener('pointerup', () => { drag = null; });
    render();
  });
})();
