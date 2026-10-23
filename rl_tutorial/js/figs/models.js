/* models.js — 图 5.2 一步 RLHF 训练中的数据流：PPO（四个模型）vs GRPO（去掉价值模型），对照 DeepSeekMath Figure 4 */
(function () {
  'use strict';
  const RL = window.RL;
  const BACK = 'M840,287 L840,322 Q840,334 828,334 L217,334 Q205,334 205,322 L205,121';
  const NODES = [
    { id: 'q', x: 60, y: 90, w: 86, h: 46, k: 'data', t: ['问题 q'] },
    { id: 'pol', x: 205, y: 90, w: 128, h: 58, k: 'train', t: ['策略模型', 'π_θ · 训练'] },
    { id: 'o', x: 360, y: 90, w: 86, h: 46, k: 'data', t: ['回答 o'], m: 'ppo' },
    { id: 'og', x: 360, y: 90, w: 86, h: 46, k: 'data stack', t: ['o₁ … o_G'], m: 'grpo' },
    { id: 'ref', x: 525, y: 40, w: 134, h: 52, k: 'frozen', t: ['参考模型', 'π_ref · 冻结'] },
    { id: 'rm', x: 525, y: 145, w: 134, h: 52, k: 'frozen', t: ['奖励模型', 'r_φ · 冻结'] },
    { id: 'val', x: 525, y: 250, w: 134, h: 52, k: 'train', t: ['价值模型', 'V_ψ · 训练'], m: 'ppo' },
    { id: 'valx', x: 525, y: 250, w: 134, h: 52, k: 'ghost', t: ['价值模型', '不再需要'], m: 'grpo' },
    { id: 'kl', x: 692, y: 40, w: 104, h: 42, k: 'op', t: ['逐 token KL'], m: 'ppo' },
    { id: 'klg', x: 692, y: 40, w: 104, h: 42, k: 'op', t: ['KL（k3 估计）'], m: 'grpo' },
    { id: 'r', x: 692, y: 145, w: 104, h: 50, k: 'op', t: ['逐 token 奖励', 'r_φ − β·KL'], m: 'ppo' },
    { id: 'rg', x: 692, y: 145, w: 104, h: 50, k: 'op', t: ['G 个奖励', 'r₁ … r_G'], m: 'grpo' },
    { id: 'gae', x: 840, y: 145, w: 104, h: 50, k: 'op', t: ['GAE', '→ 逐 token Âₜ'], m: 'ppo' },
    { id: 'norm', x: 840, y: 145, w: 104, h: 50, k: 'op', t: ['组内归一化', 'Âᵢ = (rᵢ − μ)/σ'], m: 'grpo' },
    { id: 'upd', x: 840, y: 262, w: 104, h: 50, k: 'loss', t: ['PPO 损失', '更新参数'], m: 'ppo' },
    { id: 'updg', x: 840, y: 262, w: 104, h: 50, k: 'loss', t: ['GRPO 损失', '裁剪项 − β·KL'], m: 'grpo' }
  ];
  // [起点, 终点, 阶段, 模式, 标签, 自定义路径]
  const EDGES = [
    ['q', 'pol', 0, 'both'], ['pol', 'o', 1, 'ppo', '采样'], ['pol', 'og', 1, 'grpo', '采样 G 个'],
    ['o', 'ref', 2, 'ppo'], ['o', 'rm', 2, 'ppo'], ['o', 'val', 2, 'ppo'], ['og', 'ref', 2, 'grpo'], ['og', 'rm', 2, 'grpo'],
    ['ref', 'kl', 3, 'ppo'], ['rm', 'r', 3, 'ppo'], ['kl', 'r', 4, 'ppo'], ['r', 'gae', 5, 'ppo'], ['val', 'gae', 5, 'ppo'], ['gae', 'upd', 6, 'ppo'],
    ['upd', 'pol', 7, 'ppo', '更新策略', BACK], ['upd', 'val', 7, 'ppo', '回归'],
    ['ref', 'klg', 3, 'grpo'], ['rm', 'rg', 3, 'grpo'], ['rg', 'norm', 4, 'grpo'], ['norm', 'updg', 5, 'grpo'],
    ['klg', 'updg', 5, 'grpo', 'KL 直接进损失', 'M744,40 C905,40 905,262 892,262'], ['updg', 'pol', 6, 'grpo', '更新策略', BACK]
  ];
  const STAGE = .7, PERIOD = 8 * STAGE;
  let uid = 0;

  RL.figure('models', function (el) {
    const S = RL.svg, byId = Object.fromEntries(NODES.map(n => [n.id, n]));
    let mode = el.dataset.mode || 'ppo', clock = 0;
    const root = RL.el('div', { class: 'models', 'data-mode': mode });
    const scroller = RL.el('div', { class: 'models-scroll' });
    root.appendChild(scroller); el.appendChild(root);
    const svg = S('svg', { viewBox: '0 0 920 350', class: 'models-svg', role: 'img', 'aria-label': 'RLHF 训练中策略、参考、奖励、价值模型之间的数据流示意图' }, scroller);
    const mid = 'mArrow' + (++uid);
    const mk = S('marker', { id: mid, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, S('defs', {}, svg));
    S('path', { d: 'M0,0 L10,5 L0,10 z', class: 'arrowhead' }, mk);

    // 边：从框的边界出发、到框的边界结束
    const edge = (a, b) => {
      const A = byId[a], B = byId[b], dx = B.x - A.x, dy = B.y - A.y;
      const cut = n => Math.min(n.w / 2 / Math.abs(dx || 1e-9), n.h / 2 / Math.abs(dy || 1e-9));
      const ta = cut(A), tb = cut(B);
      return `M${A.x + dx * ta},${A.y + dy * ta} L${B.x - dx * tb},${B.y - dy * tb}`;
    };
    const gEdges = S('g', {}, svg), gNodes = S('g', {}, svg), gDots = S('g', {}, svg);
    const edges = EDGES.map(([a, b, stage, m, label, custom]) => {
      const g = S('g', { class: 'me m-' + m }, gEdges);
      const p = S('path', { d: custom || edge(a, b), class: 'me-line', 'marker-end': `url(#${mid})` }, g);
      if (label) {
        const len = p.getTotalLength(), pt = p.getPointAtLength(custom ? len * .5 : len / 2);
        const tx = S('text', { x: pt.x + (custom === BACK ? 0 : 4), y: pt.y - 7, class: 'me-label', 'text-anchor': 'middle' }, g);
        tx.textContent = label;
      }
      const dot = S('circle', { r: 5, class: 'me-dot m-' + m, opacity: 0 }, gDots);
      return { p, dot, stage, m, len: p.getTotalLength() };
    });
    NODES.forEach(n => {
      const g = S('g', { class: `mn ${n.k} m-${n.m || 'both'}` }, gNodes);
      if (n.k.includes('stack')) [8, 4].forEach(o => S('rect', { x: n.x - n.w / 2 + o, y: n.y - n.h / 2 - o, width: n.w, height: n.h, rx: 9, class: 'stack-back' }, g));
      S('rect', { x: n.x - n.w / 2, y: n.y - n.h / 2, width: n.w, height: n.h, rx: 10 }, g);
      n.t.forEach((line, i) => {
        const tx = S('text', { x: n.x, y: n.y + (n.t.length === 1 ? 5 : i ? 14 : -4), 'text-anchor': 'middle', class: i ? 'sub' : 'main' }, g);
        tx.textContent = line;
      });
      if (n.k === 'ghost') S('line', { x1: n.x - n.w / 2 + 10, y1: n.y, x2: n.x + n.w / 2 - 10, y2: n.y, class: 'strike' }, g);
    });

    const mem = RL.el('div', { class: 'mem' });
    root.appendChild(mem);
    const MEM = { ppo: [['策略 π_θ', 'train'], ['价值 V_ψ', 'train'], ['参考 π_ref', 'frozen'], ['奖励 r_φ', 'frozen']], grpo: [['策略 π_θ', 'train'], ['参考 π_ref', 'frozen'], ['奖励 r_φ', 'frozen']] };
    const render = () => {
      root.dataset.mode = mode;
      const items = MEM[mode], nt = items.filter(x => x[1] === 'train').length;
      mem.innerHTML = `<div class="mem-title">显存中的大模型：<b>${items.length}</b> 个，其中要训练（存梯度与优化器状态）的 <b>${nt}</b> 个</div><div class="mem-row">` +
        items.map(([name, k]) => `<span class="mem-chip ${k}">${name}<small>${k === 'train' ? '训练' : '冻结'}</small></span>`).join('') +
        (mode === 'grpo' ? '<span class="mem-note">若改用规则判卷（RLVR），奖励模型也可以省掉</span>' : '') + '</div>';
    };
    const ctr = RL.controls(el);
    const loop = RL.loop(el, dt => {
      clock = (clock + dt) % PERIOD;
      edges.forEach(e => {
        const on = e.m === 'both' || e.m === mode, f = (clock - e.stage * STAGE) / STAGE;
        if (!on || f < 0 || f > 1) { e.dot.setAttribute('opacity', 0); return; }
        const pt = e.p.getPointAtLength(RL.ease.inOut(f) * e.len);
        e.dot.setAttribute('cx', pt.x); e.dot.setAttribute('cy', pt.y); e.dot.setAttribute('opacity', 1);
      });
    }, { autoplay: true });
    RL.playButton(ctr, loop);
    RL.seg(ctr, { label: '算法', value: mode, options: [{ value: 'ppo', label: 'PPO（RLHF）' }, { value: 'grpo', label: 'GRPO' }], onChange: v => { mode = v; clock = 0; render(); } });
    ctr.appendChild(RL.el('span', { class: 'legend', html: '<span><i style="background:var(--c-rlhf)"></i>实线框：训练</span><span><i style="border:1.5px dashed var(--ink-3);background:none;height:10px"></i>虚线框：冻结</span><span><i style="background:var(--c-grpo)"></i>计算步骤</span>' }));
    render();
  });
})();
