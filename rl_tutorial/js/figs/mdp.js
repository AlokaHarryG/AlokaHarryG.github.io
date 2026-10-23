/* mdp.js — 图 5.1 逐 token 生成：每一步都是一次动作选择（概率为示意；奖励、KL、GAE 实时计算） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  // [选中的 token, 概率, 备选1, 概率, 备选2, 概率] —— 为演示设定，不是真实模型输出
  const TOK = [['先', .62, '3', .21, '我们', .09], ['算', .71, '计算', .18, '看', .05], ['乘法', .88, '5', .06, '括号', .03], ['：', .74, '，', .2, '。', .03],
    ['5', .93, '2', .04, '我们', .01], ['×', .95, '*', .03, '乘', .01], ['2', .97, '3', .01, '5', .01], ['=', .96, '得', .02, '是', .01], ['10', .94, '7', .03, '12', .01],
    ['，', .81, '；', .1, '。', .06], ['再', .77, '然后', .17, '加', .03], ['加', .83, '算', .09, '加上', .06], ['3', .95, '10', .02, '上', .02], ['，', .58, '得', .3, '=', .08],
    ['答案', .66, '结果', .24, '所以', .06], ['是', .71, '：', .25, '为', .02], ['13', .9, '16', .07, '11', .01], ['⟨eos⟩', .92, '。', .07, '！', .005]];

  RL.figure('mdp', function (el) {
    const N = TOK.length, beta = .05, RM = 1, r = C.rng(11);
    const pref = TOK.map(t => C.clip(t[1] * (.8 + .35 * r()), .01, .99));      // 参考模型给同一 token 的概率（示意）
    const logRatio = TOK.map((t, i) => Math.log(t[1] / pref[i]));
    const rewards = logRatio.map((lr, i) => -beta * lr + (i === N - 1 ? RM : 0));  // DeepSeekMath 式(2)
    const V = TOK.map((_, i) => C.clip(.58 + .34 * i / N + .07 * Math.sin(i * 1.7), 0, 1)).concat([0]);
    const adv = C.gae(rewards, V, 1, .95).adv;
    let view = 'reward', i = RL.reducedMotion ? N : 0, clock = 0, chosenShown = false;

    const root = RL.el('div', { class: 'mdp' });
    root.appendChild(RL.el('div', { class: 'mdp-prompt', html: '<span class="lab">提示词 q</span>问：3 + 5 × 2 = ？请一步步计算。' }));
    const seq = RL.el('div', { class: 'mdp-seq' });
    const chips = TOK.map((t, k) => {
      const c = RL.el('div', { class: 'tok' + (k === N - 1 ? ' last' : ''), html: `<span class="tx">${t[0]}</span><span class="num"></span><span class="bar"><i></i></span>` });
      seq.appendChild(c); return c;
    });
    root.appendChild(seq);
    const panel = RL.el('div', { class: 'mdp-panel', 'aria-live': 'polite' });
    root.appendChild(panel);
    el.appendChild(root);

    const renderPanel = () => {
      if (i >= N) {
        panel.innerHTML = view === 'reward'
          ? `<div class="mp-title">回合结束：判卷程序检查答案 <b>13</b>，正确 → 奖励 <b class="pos">+${RM}</b> 只加在最后一个 token 上</div><div class="mp-note">其余每个 token 的奖励只有很小的 KL 惩罚 −β·log(π_θ/π_ref)（β = ${beta}）。这就是"稀疏奖励"：模型要自己弄清楚是哪些 token 立了功。</div>`
          : `<div class="mp-title">PPO 视角：评论家 V 给每个位置一个估值，再用 GAE（γ = 1，λ = 0.95）把终点奖励分摊成逐 token 的优势 Âₜ</div><div class="mp-note">每个 token 的 Âₜ 都不同，而且完全依赖评论家估得准不准。下一章的 GRPO 会去掉这个评论家。</div>`;
        return;
      }
      const t = TOK[i], text = TOK.slice(0, i).map(x => x[0]).join('');
      const rows = [[t[0], t[1], true], [t[2], t[3]], [t[4], t[5]]].sort((a, b) => b[1] - a[1]);
      panel.innerHTML = `<div class="mp-title">第 ${i + 1} 步 · 状态 sₜ = q + 「${text || '（空）'}」 · 策略 π_θ 给出下一个 token 的分布</div>` +
        rows.map(([tok, p, ch]) => `<div class="mp-row ${ch ? 'is-chosen' : ''}"><span class="mp-tok">${tok}</span><span class="mp-bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span class="mp-p">${p.toFixed(2)}</span></div>`).join('') +
        '<div class="mp-note">按概率采样一个 token 作为动作 aₜ，接到末尾，得到新状态。</div>';
      chosenShown = false;
    };
    const renderChips = () => {
      const vals = view === 'reward' ? rewards : adv, nums = view === 'reward' ? rewards : V, max = Math.max(...vals.map(Math.abs));
      chips.forEach((c, k) => {
        c.classList.toggle('on', k < i);
        c.classList.toggle('cur', k === i);
        const done = i >= N;
        c.querySelector('.num').textContent = done ? (nums[k] >= 0 && view === 'reward' ? '+' : '') + nums[k].toFixed(view === 'reward' && k < N - 1 ? 3 : 2) : '';
        const bar = c.querySelector('.bar i'), v = vals[k], h = done ? 22 * Math.abs(v) / max : 0;
        bar.style.height = Math.max(done ? 1 : 0, h) + 'px';
        bar.style.top = v >= 0 ? (22 - h) + 'px' : '22px';
        bar.className = v >= 0 ? 'p' : 'n';
      });
      root.classList.toggle('done', i >= N);
      root.dataset.view = view;
    };
    const ctr = RL.controls(el), st = RL.stats(el);
    const loop = RL.loop(el, dt => {
      clock += dt;
      if (i < N && !chosenShown && clock > .45) { const row = panel.querySelector('.is-chosen'); if (row) row.classList.add('chosen'); chosenShown = true; }
      if (i < N && clock > .95) { i++; clock = 0; renderPanel(); renderChips(); sync(); }
      return i < N;
    }, { autoplay: true });
    RL.button(ctr, '重新生成', () => { i = 0; clock = 0; renderPanel(); renderChips(); sync(); loop.play(); }, 'primary');
    RL.seg(ctr, { label: '视角', value: view, options: [{ value: 'reward', label: '逐 token 奖励 rₜ' }, { value: 'ppo', label: 'PPO 视角：V 与 Âₜ' }], onChange: v => { view = v; renderPanel(); renderChips(); sync(); } });
    const s1 = RL.stat(st, '已生成'), s2 = RL.stat(st, 'Σ rₜ ='), s3 = RL.stat(st, '');
    const sync = () => {
      s1.set(`${Math.min(i, N)} / ${N} 个 token`);
      s2.set(i >= N ? `${C.sum(rewards).toFixed(3)}（奖励模型 +${RM}，KL 部分 ${(C.sum(rewards) - RM).toFixed(3)}）` : '—');
      s3.set(view === 'reward' ? '柱 = rₜ，数字 = rₜ' : '柱 = Âₜ，数字 = V(sₜ)');
    };
    renderPanel(); renderChips(); sync();
  });
})();
