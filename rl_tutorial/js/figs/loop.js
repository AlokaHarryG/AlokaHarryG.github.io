/* loop.js — 图 1.1 智能体–环境循环与折扣回报；同时定义两张网格图共用的小世界与绘制函数 */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;

  // 6×4 网格：S 起点，终点 +1，岩浆 −1，# 墙；其余每步奖励 0
  RL.WORLD = { w: 6, h: 4, start: 18, stepReward: 0, walls: new Set([7, 8, 20]), terminals: new Map([[5, 1], [10, -1], [16, -1]]) };
  RL.ACTION_NAMES = ['上', '右', '下', '左'];

  RL.arrow = function (ctx, cx, cy, a, len, color, width) {
    const [dx, dy] = C.ACTIONS[a], hx = cx + dx * len / 2, hy = cy + dy * len / 2, h = Math.max(4, len * .32);
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = width || 2; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx - dx * len / 2, cy - dy * len / 2); ctx.lineTo(hx - dx * h * .6, hy - dy * h * .6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx - dx * h - dy * h * .6, hy - dy * h + dx * h * .6); ctx.lineTo(hx - dx * h + dy * h * .6, hy - dy * h - dx * h * .6); ctx.closePath(); ctx.fill();
    ctx.restore();
  };

  // o: { V, arrows, sel, robot:[fx,fy], trail:[s...] }
  RL.drawWorld = function (ctx, world, box, o) {
    o = o || {};
    const cs = Math.min(box.w / world.w, box.h / world.h);
    const ox = box.x + (box.w - cs * world.w) / 2, oy = box.y + (box.h - cs * world.h) / 2;
    const at = s => [ox + (s % world.w) * cs, oy + Math.floor(s / world.w) * cs];
    const teal = RL.css('--c-rl'), neg = RL.css('--neg'), good = RL.css('--pos'), ink = RL.css('--ink');
    const big = cs >= 52, p = Math.max(2, cs * .045), rad = Math.min(9, cs * .16);
    for (let s = 0; s < world.w * world.h; s++) {
      const [x, y] = at(s);
      RL.rrect(ctx, x + p, y + p, cs - 2 * p, cs - 2 * p, rad);
      if (world.walls.has(s)) { ctx.fillStyle = '#2b3245'; ctx.fill(); continue; }
      const term = world.terminals.get(s);
      if (term !== undefined) {
        ctx.fillStyle = term > 0 ? good : neg; ctx.fill();
        RL.text(ctx, term > 0 ? '+1' : '−1', x + cs / 2, y + cs / 2 - (big ? 7 : 0), { color: '#fff', size: Math.round(cs * .26), weight: 800, align: 'center' });
        if (big) RL.text(ctx, term > 0 ? '终点' : '岩浆', x + cs / 2, y + cs / 2 + cs * .2, { color: 'rgba(255,255,255,.88)', size: 11, weight: 600, align: 'center' });
        continue;
      }
      const v = o.V ? o.V[s] : 0;
      ctx.fillStyle = !o.V ? RL.css('--surface-2') : v >= 0 ? RL.alpha(teal, .06 + .6 * Math.min(1, v)) : RL.alpha(neg, .06 + .55 * Math.min(1, -v));
      ctx.fill();
      if (s === world.start) RL.text(ctx, 'S', x + p + 7, y + p + 10, { size: 11, weight: 750, color: RL.css('--ink-3') });
      if (o.trail && o.trail.includes(s)) RL.dot(ctx, x + cs / 2, y + cs / 2, Math.max(2.5, cs * .06), RL.alpha(RL.css('--c-pg'), .35));
      // 箭头：所有并列最优的动作都画出来（只有一个时画在正中）
      const as = o.arrows && o.arrows[s], ay = y + cs / 2 - (o.V && big ? 7 : 0);
      if (as && as.length === 1) RL.arrow(ctx, x + cs / 2, ay, as[0], cs * .36, RL.alpha(ink, .72), 2.2);
      else if (as) as.forEach(a => RL.arrow(ctx, x + cs / 2 + C.ACTIONS[a][0] * cs * .13, ay + C.ACTIONS[a][1] * cs * .13, a, cs * .2, RL.alpha(ink, .6), 1.8));
      if (o.V && big) RL.text(ctx, v.toFixed(2), x + cs / 2, y + cs - p - 10, { size: 11, mono: true, color: ink, align: 'center' });
      if (o.sel === s) { ctx.save(); ctx.lineWidth = 2.5; ctx.strokeStyle = ink; RL.rrect(ctx, x + p, y + p, cs - 2 * p, cs - 2 * p, rad); ctx.stroke(); ctx.restore(); }
    }
    if (o.robot) {
      const cx = ox + (o.robot[0] + .5) * cs, cy = oy + (o.robot[1] + .5) * cs, R = cs * .26;
      ctx.save(); ctx.shadowColor = 'rgba(0,0,0,.28)'; ctx.shadowBlur = 10; ctx.shadowOffsetY = 3;
      RL.dot(ctx, cx, cy, R, RL.css('--c-pg')); ctx.restore();
      RL.dot(ctx, cx - R * .36, cy - R * .12, R * .17, '#fff'); RL.dot(ctx, cx + R * .36, cy - R * .12, R * .17, '#fff');
    }
    return { cellAt(px, py) { const gx = Math.floor((px - ox) / cs), gy = Math.floor((py - oy) / cs); return gx >= 0 && gy >= 0 && gx < world.w && gy < world.h ? gy * world.w + gx : -1; } };
  };

  RL.figure('loop', function (el) {
    const W = RL.WORLD, STEP = 0.9, MAXT = 14;
    // 先求出最优策略；智能体以 ε = 0.25 的概率随机乱走，这样每个回合都不一样
    let V = new Array(W.w * W.h).fill(0), pol;
    for (let k = 0; k < 60; k++) { const r = C.valueIterationSweep(W, V, 0.9); V = r.V; pol = r.policy; }
    let gamma = 0.9, ep = 0, steps, s, mode, clock, revealed, rnd, robot, cur;
    const xy = st => [st % W.w, Math.floor(st / W.w)];
    const newEpisode = () => { ep++; rnd = C.rng(97 + ep * 13); s = W.start; steps = []; mode = 'run'; clock = 0; revealed = 0; robot = xy(s); cur = null; };

    const layout = w => {
      if (w >= 700) return { diag: { x: 0, y: 0, w: w * .36, h: 250 }, grid: { x: w * .39, y: 4, w: w * .61, h: 242 }, tape: { x: 0, y: 272, w, h: 118 } };
      const gh = Math.round(w * .6);
      return { diag: { x: 0, y: 0, w, h: 170 }, grid: { x: 0, y: 178, w, h: gh }, tape: { x: 0, y: 194 + gh, w, h: 118 } };
    };
    const bez = (a, c, b, t) => [(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t * t * b[0], (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t * t * b[1]];
    const pill = (ctx, x, y, str, col) => {
      ctx.font = RL.font(12, 700); const tw = ctx.measureText(str).width + 16;
      RL.rrect(ctx, x - tw / 2, y - 12, tw, 24, 12); ctx.fillStyle = col; ctx.fill();
      RL.text(ctx, str, x, y + 1, { color: '#fff', size: 12, weight: 700, align: 'center' });
    };
    const drawLoop = (ctx, b) => {
      const cx = b.x + b.w / 2, bw = Math.min(176, b.w * .6), bh = 52, ay = b.y + 14, ey = b.y + b.h - 14 - bh, my = (ay + ey + bh) / 2, bulge = Math.min(64, b.w * .17);
      const box = (y, title, sub, col) => {
        RL.rrect(ctx, cx - bw / 2, y, bw, bh, 12); ctx.fillStyle = RL.alpha(col, .1); ctx.fill(); ctx.strokeStyle = RL.alpha(col, .6); ctx.lineWidth = 1.5; ctx.stroke();
        RL.text(ctx, title, cx, y + 19, { size: 13, weight: 750, color: RL.css('--ink'), align: 'center' });
        RL.text(ctx, sub, cx, y + 37, { size: 11, mono: true, color: RL.css('--ink-3'), align: 'center' });
      };
      const R0 = [cx + bw / 2, ay + bh / 2], Rc = [cx + bw / 2 + bulge, my], R1 = [cx + bw / 2, ey + bh / 2];
      const L0 = [cx - bw / 2, ey + bh / 2], Lc = [cx - bw / 2 - bulge, my], L1 = [cx - bw / 2, ay + bh / 2];
      [[R0, Rc, R1, RL.css('--c-pg')], [L0, Lc, L1, RL.css('--c-rl')]].forEach(([a, c, e, col]) => {
        const pts = Array.from({ length: 30 }, (_, i) => bez(a, c, e, i / 29));
        RL.path(ctx, pts, RL.alpha(col, .55), 2, [5, 5]);
        const [px, py] = pts[27]; const ang = Math.atan2(e[1] - py, e[0] - px);
        ctx.save(); ctx.fillStyle = RL.alpha(col, .8); ctx.translate(e[0], e[1]); ctx.rotate(ang);
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-9, -5); ctx.lineTo(-9, 5); ctx.closePath(); ctx.fill(); ctx.restore();
      });
      box(ay, '智能体 Agent', 'π(a | s)', RL.css('--c-pg'));
      box(ey, '环境 Environment', '网格世界', RL.css('--c-rl'));
      RL.text(ctx, '动作 a', Rc[0] - 10, my, { size: 12, weight: 650, color: RL.css('--c-pg'), align: 'right' });
      RL.text(ctx, '新状态 s′ · 奖励 r', Lc[0] + 10, my, { size: 12, weight: 650, color: RL.css('--c-rl') });
      if (mode === 'run' && cur) {
        const f = clock / STEP;
        if (f < .3) { const [x, y] = bez(R0, Rc, R1, RL.ease.inOut(f / .3)); pill(ctx, x, y, '动作 ' + RL.ACTION_NAMES[cur.a], RL.css('--c-pg')); }
        else if (f > .6) { const [x, y] = bez(L0, Lc, L1, RL.ease.inOut(Math.min(1, (f - .6) / .3))); pill(ctx, x, y, 'r = ' + (cur.r > 0 ? '+1' : cur.r < 0 ? '−1' : '0'), cur.r > 0 ? RL.css('--pos') : cur.r < 0 ? RL.css('--neg') : RL.css('--c-rl')); }
      }
    };
    const drawTape = (ctx, b) => {
      const lw = 40, cw = Math.min(56, (b.w - lw - 10) / MAXT);
      const G = C.discountedReturns(steps.map(x => x.r), gamma);
      RL.rrect(ctx, b.x, b.y, b.w, b.h, 12); ctx.fillStyle = RL.css('--surface-2'); ctx.fill();
      const rowT = b.y + 16, rowA = b.y + 36, rowR = b.y + 56, base = b.y + 92;
      [['t', rowT], ['a', rowA], ['r', rowR], ['G', base]].forEach(([lab, y]) => RL.text(ctx, lab, b.x + 16, y, { size: 12, weight: 700, mono: true, color: RL.css('--ink-3') }));
      ctx.strokeStyle = RL.css('--line-2'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(b.x + lw, base + .5); ctx.lineTo(b.x + b.w - 10, base + .5); ctx.stroke();
      steps.forEach((st, i) => {
        const x = b.x + lw + i * cw + cw / 2;
        RL.text(ctx, String(i), x, rowT, { size: 11, mono: true, color: RL.css('--ink-3'), align: 'center' });
        RL.text(ctx, RL.ACTION_NAMES[st.a], x, rowA, { size: 12, weight: 600, color: RL.css('--c-pg'), align: 'center' });
        RL.text(ctx, st.r > 0 ? '+1' : st.r < 0 ? '−1' : '0', x, rowR, { size: 12, mono: true, weight: 700, color: st.r > 0 ? RL.css('--pos') : st.r < 0 ? RL.css('--neg') : RL.css('--ink-3'), align: 'center' });
        if (i >= steps.length - revealed) {
          const g = G[i], hgt = Math.max(1, Math.abs(g) * 22), col = g >= 0 ? RL.css('--c-rl') : RL.css('--neg');
          ctx.fillStyle = RL.alpha(col, .78);
          ctx.fillRect(x - cw * .3, g >= 0 ? base - hgt : base, cw * .6, hgt);
          if (cw >= 30) RL.text(ctx, g.toFixed(2), x, g >= 0 ? base + 12 : base - 10, { size: 10, mono: true, color: RL.css('--ink'), align: 'center' });
        }
      });
    };

    const tick = dt => {
      clock += dt;
      if (mode === 'run') {
        if (!cur) {
          const a = rnd() < 0.25 ? Math.floor(rnd() * 4) : pol[s], res = C.gridStep(W, s, a);
          cur = { s, a, r: res.r, s2: res.s2, done: res.done }; clock = 0;
        }
        const f = Math.min(1, clock / STEP), [x0, y0] = xy(cur.s), [x1, y1] = xy(cur.s2), m = RL.ease.inOut(C.clip((f - .3) / .3, 0, 1));
        robot = [RL.lerp(x0, x1, m), RL.lerp(y0, y1, m)];
        if (clock >= STEP) {
          steps.push(cur); s = cur.s2;
          const end = cur.done || steps.length >= MAXT;
          cur = null; clock = 0;
          if (end) mode = 'back';
        }
      } else if (mode === 'back') {
        if (clock >= .3) { clock = 0; revealed++; if (revealed >= steps.length) mode = 'done'; }
      } else if (clock >= 4.5) newEpisode();
      cv.redraw(); syncStats();
    };

    newEpisode();
    const cv = RL.canvas(el, {
      label: '智能体与环境交互循环的动画：机器人在网格中行动，下方轨迹带记录每一步的动作、奖励与折扣回报',
      height: w => { const L = layout(w); return L.tape.y + L.tape.h; },
      draw(ctx, w) {
        const L = layout(w);
        drawLoop(ctx, L.diag);
        RL.drawWorld(ctx, W, L.grid, { robot, trail: steps.map(x => x.s) });
        drawTape(ctx, L.tape);
      }
    });
    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, tick, { autoplay: true });
    RL.playButton(ctr, loop);
    RL.button(ctr, '新回合', () => { newEpisode(); cv.redraw(); syncStats(); });
    RL.slider(ctr, { label: '折扣 γ', min: 0, max: 1, step: .01, value: gamma, fmt: v => v.toFixed(2), onInput: v => { gamma = v; cv.redraw(); syncStats(); } });
    const sEp = RL.stat(st, '回合'), sT = RL.stat(st, '步数'), sG = RL.stat(st, 'G₀ ='), sMsg = RL.stat(st, '');
    const syncStats = () => {
      sEp.set(ep); sT.set(steps.length);
      sG.set(mode === 'done' ? C.discountedReturns(steps.map(x => x.r), gamma)[0].toFixed(3) : '—');
      const last = steps[steps.length - 1];
      sMsg.set(mode === 'run' ? '正在交互（25% 的概率随机探索）' : mode === 'back' ? '回合结束：从后往前算 Gₜ = rₜ + γ·Gₜ₊₁'
        : last && last.done ? '本回合结束，可拖动 γ 观察回报变化' : `达到 ${MAXT} 步上限，截断（之后的回报记为 0）`);
    };
    syncStats();
  });
})();
