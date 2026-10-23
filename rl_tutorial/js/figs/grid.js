/* grid.js — 图 1.2 网格世界中的价值迭代（可点击查看任一格子的 Bellman 备份） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  RL.figure('grid', function (el) {
    const W = RL.WORLD, N = W.w * W.h, xy = s => [s % W.w, Math.floor(s / W.w)];
    const fmtR = r => (r > 0 ? '+1' : r < 0 ? '−1' : '0');
    let gamma = 0.9, V, iter, delta, sel = 4, walk = null, geo = null, acc = 0;
    const reset = () => { V = new Array(N).fill(0); iter = 0; delta = NaN; walk = null; };
    const sweep = () => { const r = C.valueIterationSweep(W, V, gamma); V = r.V; delta = r.maxDelta; iter++; };
    // 相对当前估计 V 的贪心动作；四个动作完全并列时不画箭头
    const best = s => {
      if (W.walls.has(s) || W.terminals.has(s)) return [];
      const q = C.bellmanBackups(W, V, s, gamma), m = Math.max(...q), as = [0, 1, 2, 3].filter(a => q[a] > m - 1e-9);
      return as.length === 4 ? [] : as;
    };
    reset();

    const wrap = RL.el('div', { class: 'fig-grid' }), left = RL.el('div', { class: 'fig-stage' }), side = RL.el('div', { class: 'side-panel' });
    wrap.append(left, side); el.appendChild(wrap);
    const cv = RL.canvas(left, {
      label: '6×4 网格世界的价值热力图与贪心策略箭头。可点击格子，或聚焦后用方向键选择格子',
      aspect: W.h / W.w,
      draw(ctx, w, h) {
        geo = RL.drawWorld(ctx, W, { x: 0, y: 0, w, h }, { V, arrows: Array.from({ length: N }, (_, s) => best(s)), sel, robot: walk && walk.pos, trail: walk && walk.trail });
      }
    });
    cv.cv.tabIndex = 0;
    const panel = RL.el('div', { class: 'backup', 'aria-live': 'polite' });
    side.appendChild(panel);

    const renderPanel = () => {
      const [x, y] = xy(sel);
      let html = `<div class="bk-title">选中格子 (${x}, ${y})</div>`;
      if (W.walls.has(sel)) html += '<p>墙：不能进入。朝墙（或边界）走的动作会让智能体原地不动。</p>';
      else if (W.terminals.has(sel)) {
        const r = W.terminals.get(sel);
        html += `<p>${r > 0 ? '终点' : '岩浆'}：一进入就获得 ${fmtR(r)}，回合结束。之后再没有未来回报，所以邻居的备份里这一项只算 r，不再加 γ·V。</p>`;
      } else {
        const q = C.bellmanBackups(W, V, sel, gamma), m = Math.max(...q);
        html += '<table><thead><tr><th>动作</th><th>r</th><th>γ·V(s′)</th><th>备份值</th></tr></thead><tbody>';
        q.forEach((v, a) => {
          const { s2, r, done } = C.gridStep(W, sel, a);
          html += `<tr class="${v > m - 1e-9 ? 'best' : ''}"><td>${RL.ACTION_NAMES[a]}${s2 === sel ? '<span class="muted">（原地）</span>' : ''}</td><td>${fmtR(r)}</td><td>${done ? '0 <span class="muted">结束</span>' : (gamma * V[s2]).toFixed(3)}</td><td><b>${v.toFixed(3)}</b></td></tr>`;
        });
        html += `</tbody></table><p class="bk-sum">下一轮：V<sub>${iter + 1}</sub>(s) = max = <b>${m.toFixed(3)}</b><br><span class="muted">当前 V<sub>${iter}</sub>(s) = ${V[sel].toFixed(3)}</span></p>`;
      }
      panel.innerHTML = html;
    };

    const ctr = RL.controls(el), st = RL.stats(el);
    const sK = RL.stat(st, '迭代轮数 k ='), sD = RL.stat(st, '最大变化 max|ΔV| ='), sW = RL.stat(st, '机器人：');
    const refresh = () => {
      cv.redraw(); renderPanel();
      sK.set(iter); sD.set(isNaN(delta) ? '—' : delta < 1e-6 ? '0（已收敛）' : delta.toFixed(4));
      sW.set(walk ? walk.msg || '行走中…' : '待命');
    };

    const iterLoop = RL.loop(el, dt => { acc += dt; if (acc > .45) { acc = 0; sweep(); refresh(); if (delta < 1e-6) return false; } });
    const walkLoop = RL.loop(el, dt => {
      if (!walk) return false;
      walk.t += dt / .36;
      if (walk.t >= 1) {
        walk.pos = xy(walk.to);
        if (walk.to !== walk.trail[walk.trail.length - 1]) walk.trail.push(walk.to);
        const s = walk.to;
        if (W.terminals.has(s)) { walk.msg = W.terminals.get(s) > 0 ? `${walk.n} 步到达终点` : '掉进了岩浆'; refresh(); return false; }
        const as = best(s);
        if (!as.length || walk.n >= 20) { walk.msg = as.length ? '20 步内没走到：价值还没传到这里' : '这里还没有方向：再多迭代几轮'; refresh(); return false; }
        walk.from = s; walk.to = C.gridStep(W, s, as[0]).s2; walk.n++; walk.t = 0;
      }
      const [x0, y0] = xy(walk.from), [x1, y1] = xy(walk.to), m = RL.ease.inOut(walk.t);
      walk.pos = [RL.lerp(x0, x1, m), RL.lerp(y0, y1, m)];
      cv.redraw();
    });

    RL.button(ctr, '迭代一轮', () => { sweep(); refresh(); }, 'primary');
    const autoBtn = RL.button(ctr, '自动迭代', () => iterLoop.toggle());
    iterLoop.onchange = p => { autoBtn.textContent = p ? '暂停' : '自动迭代'; };
    RL.button(ctr, '重置', () => { iterLoop.pause(); reset(); refresh(); });
    RL.slider(ctr, { label: '折扣 γ', min: 0, max: .99, step: .01, value: gamma, fmt: v => v.toFixed(2), onInput: v => { gamma = v; reset(); refresh(); iterLoop.play(); } });
    RL.button(ctr, '让机器人走一遍', () => { walk = { from: W.start, to: W.start, t: 1, n: 0, trail: [W.start], pos: xy(W.start), msg: '' }; refresh(); walkLoop.play(); });

    cv.cv.addEventListener('click', e => {
      const r = cv.cv.getBoundingClientRect(), s = geo ? geo.cellAt(e.clientX - r.left, e.clientY - r.top) : -1;
      if (s >= 0) { sel = s; refresh(); }
    });
    cv.cv.addEventListener('keydown', e => {
      const d = { ArrowUp: [0, -1], ArrowRight: [1, 0], ArrowDown: [0, 1], ArrowLeft: [-1, 0] }[e.key];
      if (!d) return;
      e.preventDefault();
      const [x, y] = xy(sel);
      sel = C.clip(y + d[1], 0, W.h - 1) * W.w + C.clip(x + d[0], 0, W.w - 1);
      refresh();
    });
    refresh();
  });
})();
