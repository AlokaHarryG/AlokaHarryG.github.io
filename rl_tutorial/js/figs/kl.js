/* kl.js — 图 6.2 两种逐样本 KL 估计器：k1 = −log x 与 k3 = x − log x − 1（x = π_ref/π_θ） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('kl', function (el) {
    const L = window.RLLLM, TOKS = ['的', '是', '答案', '因此'];
    let x = .45, seed = 4, sx = null, P, Q;
    const dists = () => { const r = C.rng(seed * 131); P = C.softmax(TOKS.map(() => 1.6 * C.randn(r))); Q = C.softmax(TOKS.map(() => 1.6 * C.randn(r))); };
    dists();
    const wrap = RL.el('div', { class: 'fig-grid' }), left = RL.el('div', { class: 'fig-stage' }), right = RL.el('div', { class: 'side-panel' });
    wrap.append(left, right); el.appendChild(wrap);

    const cv = RL.canvas(left, {
      label: '两种 KL 估计器随比例 x 变化的曲线，k1 可以为负，k3 恒非负',
      height: w => Math.min(330, Math.round(w * .7)),
      draw(ctx, w, h) {
        const b = { x: 44, y: 14, w: w - 58, h: h - 58 }, ax = RL.axes(ctx, b, { x: [0, 3], y: [-1.6, 3], xticks: [0, .5, 1, 1.5, 2, 2.5, 3], yticks: [-1, 0, 1, 2, 3], xlabel: 'x = π_ref(o) / π_θ(o)', ylabel: '单样本估计值', ylabelOffset: 34 });
        sx = ax.sx; const sy = ax.sy, k1c = RL.css('--c-ppo'), k3c = RL.css('--c-grpo'), ink3 = RL.css('--ink-3');
        ctx.fillStyle = RL.alpha(RL.css('--neg'), .07); ctx.fillRect(sx(1), sy(0), sx(3) - sx(1), sy(-1.6) - sy(0));
        RL.text(ctx, 'k1 < 0 的区域', sx(2.2), sy(-1.35), { size: 11, color: RL.css('--neg'), align: 'center' });
        RL.path(ctx, [[sx(0), sy(0)], [sx(3), sy(0)]], RL.css('--ink-2'), 1);
        RL.path(ctx, RL.sample(t => t - 1, .02, 3, 2).map(([a, v]) => [sx(a), sy(v)]), ink3, 1.2, [5, 4]);
        RL.path(ctx, RL.sample(Math.log, .2, 3, 120).map(([a, v]) => [sx(a), sy(v)]), ink3, 1.2);
        RL.text(ctx, 'x − 1（切线）', sx(2.55), sy(1.75) - 10, { size: 10, color: ink3, align: 'center' });
        RL.text(ctx, 'log x', sx(2.75), sy(Math.log(2.75)) + 12, { size: 10, color: ink3, align: 'center' });
        RL.path(ctx, RL.sample(L.k1, .2, 3, 160).map(([a, v]) => [sx(a), sy(v)]), k1c, 2.6);
        RL.path(ctx, RL.sample(L.k3, .07, 3, 160).map(([a, v]) => [sx(a), sy(v)]), k3c, 3);
        // 标记：k3 正好是切线与 log 曲线之间的竖直距离
        const v1 = L.k1(x), v3 = L.k3(x);
        RL.path(ctx, [[sx(x), sy(x - 1)], [sx(x), sy(Math.log(x))]], RL.alpha(k3c, .55), 5);
        RL.path(ctx, [[sx(x), b.y], [sx(x), b.y + b.h]], RL.css('--line-2'), 1, [2, 3]);
        RL.dot(ctx, sx(x), sy(v1), 6, RL.css('--surface'), k1c);
        RL.dot(ctx, sx(x), sy(v3), 6, RL.css('--surface'), k3c);
        RL.text(ctx, `k1 = ${v1.toFixed(3)}`, sx(x) + 10, sy(v1) - 10, { size: 11, mono: true, weight: 700, color: k1c });
        RL.text(ctx, `k3 = ${v3.toFixed(3)}`, sx(x) + 10, sy(v3) + 12, { size: 11, mono: true, weight: 700, color: k3c });
      }
    });
    left.appendChild(RL.el('div', { class: 'legend', html: `<span><i style="background:${RL.css('--c-ppo')}"></i>k1 = −log x</span><span><i style="background:${RL.css('--c-grpo')}"></i>k3 = x − log x − 1（GRPO 使用）</span><span><i style="background:${RL.alpha(RL.css('--c-grpo'), .5)};width:5px;height:12px"></i>k3 = 切线与 log 之间的距离</span>` }));

    const table = RL.el('div', { class: 'backup kl-table', 'aria-live': 'polite' });
    right.appendChild(table);
    const renderTable = () => {
      let e1 = 0, e3 = 0;
      const rows = TOKS.map((t, i) => {
        const xi = Q[i] / P[i], a = L.k1(xi), c = L.k3(xi); e1 += P[i] * a; e3 += P[i] * c;
        return `<tr><td>${t}</td><td>${P[i].toFixed(3)}</td><td>${Q[i].toFixed(3)}</td><td>${xi.toFixed(2)}</td><td class="${a < 0 ? 'negv' : ''}">${a.toFixed(3)}</td><td>${c.toFixed(3)}</td></tr>`;
      }).join('');
      table.innerHTML = '<div class="bk-title">精确验证：4 个 token 的小词表</div><table><thead><tr><th>token</th><th>π_θ</th><th>π_ref</th><th>x</th><th>k1</th><th>k3</th></tr></thead><tbody>' + rows +
        `</tbody></table><p>按 π_θ 加权求期望：<br>E[k1] = <b class="mono">${e1.toFixed(4)}</b>，E[k3] = <b class="mono">${e3.toFixed(4)}</b><br>精确 KL(π_θ‖π_ref) = <b class="mono">${C.kl(P, Q).toFixed(4)}</b></p><p class="muted">两者都无偏；但 k1 的单个样本（红色）可以是负数，k3 不会。</p>`;
    };
    renderTable();
    const ctr = RL.controls(el);
    const xs = RL.slider(ctr, { label: '比例 x', min: .07, max: 3, step: .01, value: x, fmt: v => v.toFixed(2), onInput: v => { x = v; cv.redraw(); } });
    RL.button(ctr, '换一组分布', () => { seed++; dists(); renderTable(); });
    let drag = false;
    const setX = e => { if (!sx) return; const r = cv.cv.getBoundingClientRect(); x = C.clip(sx.inv(e.clientX - r.left), .07, 3); xs.set(x); cv.redraw(); };
    cv.cv.addEventListener('pointerdown', e => { drag = true; cv.cv.setPointerCapture(e.pointerId); setX(e); });
    cv.cv.addEventListener('pointermove', e => { if (drag) setX(e); });
    cv.cv.addEventListener('pointerup', () => { drag = false; });
  });
})();
