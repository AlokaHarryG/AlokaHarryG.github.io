/* rlcd.js — 图 7.5 RLCD 的一步训练，逐步拆解（OpenJev-RLCD 算法 1；奖励与留一优势实时计算） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const ROWS = [ // u = 读数中"是"的概率（示意）
    { t: '先涨 20% 再跌 20%：100 × 1.2 = 120，120 × 0.8 = 96 ≠ 100。', u: .03 },
    { t: '涨 20%、跌 20%，一正一负正好抵消，最终还是原价。', u: .97 },
    { t: '设原价为 x：x × 1.2 × 0.8 = 0.96x，比原价低 4%。', u: .05 },
    { t: '涨跌幅度相同……但基数好像不一样？不太确定。', u: .45 }
  ];
  const NAMES = ['① 采样推理', '② 读出概率', '③ 揭晓结果', '④ 恰当评分', '⑤ 阶段 1：校准', '⑥ 阶段 2：强化'];

  RL.figure('rlcd', function (el) {
    const L = window.RLLLM, Y = 1, M = ROWS.length; // 选项 0 = 是，1 = 否；真实结果为"否"
    const rows = ROWS.map(r => ({ ...r }));
    let step = RL.reducedMotion ? 5 : 0, typed = RL.reducedMotion ? 99 : 0, auto = !RL.reducedMotion, clock = 0;
    const root = RL.el('div', { class: 'rlcd' });
    const chips = RL.el('div', { class: 'rc-steps', role: 'tablist', 'aria-label': 'RLCD 训练步骤' });
    const btns = NAMES.map((n, k) => { const b = RL.el('button', { type: 'button', role: 'tab', text: n }); b.addEventListener('click', () => { auto = false; go(k); }); chips.appendChild(b); return b; });
    root.appendChild(chips);
    root.appendChild(RL.el('div', { class: 'rc-q', html: '<span class="lab">问题 x</span>一件商品先涨价 20%，再降价 20%。候选解答声称"最终价格等于原价"。这个解答对吗？<span class="opt">选项：是 / 否</span><span class="y-reveal">真实结果 Y = <b>否</b></span>' }));
    root.appendChild(RL.el('div', { class: 'rc-head', 'aria-hidden': 'true', html: '<span>#</span><span>推理 rᵢ（示意）</span><span>读数 uᵢ（是 / 否）</span><span>Rᵢ</span><span>Aᵢ</span>' }));
    const els = rows.map((r, i) => {
      const e = RL.el('div', { class: 'rc-row' });
      e.innerHTML = `<span class="rc-i mono">r${'₁₂₃₄'[i]}</span><span class="rc-text"><span class="tw"></span><span class="ans mono">Answer:</span><span class="rc-badge"></span></span>
        <span class="rc-read"><span class="rb"><i class="yes"></i><i class="no"></i></span><input type="range" min="0.01" max="0.99" step="0.01" value="${r.u}" aria-label="第 ${i + 1} 条推理的读数：是的概率"><small class="mono"></small><span class="rc-grad mono"></span></span>
        <span class="rc-R mono"></span><span class="rc-A mono"></span>`;
      e.querySelector('input').addEventListener('input', ev => { rows[i].u = +ev.target.value; auto = false; update(); });
      root.appendChild(e);
      return e;
    });
    const expl = RL.el('div', { class: 'rc-expl', 'aria-live': 'polite' });
    root.appendChild(expl); el.appendChild(root);

    const compute = () => {
      const us = rows.map(r => [r.u, 1 - r.u]), R = us.map(u => L.brier(u, Y));
      return { us, R, A: L.rlcdStage2Advantages(R, null, .04) };
    };
    const text = k => {
      const { us, R } = compute(), f = v => (v >= 0 ? '+' : '') + v.toFixed(3);
      return [
        `对同一道题，从当前策略采样 $M=${M}$ 条推理 $r_i\\sim\\pi_\\theta(\\cdot\\mid x)$，每条都写到标记 <code>Answer:</code> 为止。和 GRPO 一样是"同题多采样"，但接下来<b>不采样答案</b>。`,
        `在 <code>Answer:</code> 之后读出下一个 token 在两个选项上的分布 $u_i=u_\\theta(x,r_i)$，而不是采样。拖动每行的滑块可以改变读数（示意）。注意 r₂ 推理错了，却很自信。`,
        `世界揭晓真实结果 $Y=$ <b>否</b>（120 × 0.8 = 96 ≠ 100）。和所有基线一样，训练只看到这一个结果。`,
        `奖励是读数的 Brier 分数 $R_i=J(u_i,Y)=2u_i[Y]-\\|u_i\\|^2$。对照 RLVR：按 $u_i$ 采样作答，期望奖励 $=u_i[Y]$。自信答错的 r₂：RLVR 期望得 ${us[1][Y].toFixed(2)}，与"只是没答对"区别不大；Brier 给 <b>${f(R[1])}</b>，重罚"自信地错"。诚实地拿不准的 r₄：Brier 给 ${f(R[3])}，不会被重罚。`,
        `损失 $\\mathcal L=-\\frac1M\\sum_i R_i$，梯度<b>只穿过读数</b>（路径项）：推理照常按策略采样，但不被强化。二分类时 $\\partial R_i/\\partial u_i[Y]=4(1-u_i[Y])$，读数越错被拉得越狠（每行的箭头）。在许多题上平均后期望梯度为 $4(q-u)$，在 $u=q$ 处为零：读数学到的是<b>真实概率</b>，而不是一味变自信。`,
        `再用同一个恰当分数当奖励强化推理：$A_i=\\frac1M\\big(R_i-\\beta\\log\\frac{\\pi_\\theta(r_i)}{\\pi_{\\theta_1}(r_i)}\\big)-$留一均值（刚从阶段 1 模型 $\\theta_1$ 出发，KL 项为 0）。$A_i>0$ 的推理被加强，$A_i<0$ 的被削弱。得分函数项降权 $c=0.3$，KL 锚 $\\beta=0.04$ 保护阶段 1 学到的读数。$\\sum_i A_i=0$。`
      ][k];
    };
    const update = () => {
      const { us, R, A } = compute();
      for (let k = 1; k <= 5; k++) root.classList.toggle('ge' + k, step >= k);
      root.classList.toggle('at4', step === 4);
      btns.forEach((b, k) => { b.setAttribute('aria-selected', String(k === step)); b.classList.toggle('done', k < step); });
      els.forEach((e, i) => {
        const u = us[i], t = rows[i].t;
        e.querySelector('.tw').textContent = t.slice(0, Math.floor(typed));
        e.querySelector('.yes').style.width = (u[0] * 100) + '%'; e.querySelector('.no').style.width = (u[1] * 100) + '%';
        e.querySelector('small').textContent = `是 ${u[0].toFixed(2)} · 否 ${u[1].toFixed(2)}`;
        e.querySelector('.rc-grad').textContent = `→ 否 拉力 ${(4 * (1 - u[Y])).toFixed(2)}`;
        e.querySelector('.rc-R').textContent = (R[i] >= 0 ? '+' : '') + R[i].toFixed(3);
        e.querySelector('.rc-A').textContent = (A[i] >= 0 ? '+' : '') + A[i].toFixed(3);
        e.querySelector('.rc-badge').textContent = step === 4 ? '推理：不强化' : step === 5 ? (A[i] > 0 ? '推理：加强 ↑' : '推理：削弱 ↓') : '';
        e.classList.toggle('up', step === 5 && A[i] > 0); e.classList.toggle('down', step === 5 && A[i] < 0);
      });
      expl.innerHTML = `<div class="rc-expl-k">${NAMES[step]}</div><p>${text(step)}</p><div class="stats"><span class="stat">平均奖励 R̄ = <b>${C.mean(R).toFixed(3)}</b></span><span class="stat">阶段 1 损失 ℒ = −R̄ = <b>${(-C.mean(R)).toFixed(3)}</b></span><span class="stat">ΣAᵢ = <b>${Math.abs(C.sum(A)).toFixed(3)}</b></span></div>`;
      RL.math(expl);
    };
    const go = k => { step = C.clip(k, 0, 5); if (step === 0) { typed = 0; loop.play(); } clock = 0; update(); };
    const loop = RL.loop(el, dt => {
      let busy = false;
      if (typed < 64) { typed += dt * 26; busy = true; update(); }
      if (auto && step < 5) { clock += dt; busy = true; if (clock > (step === 0 ? 2.6 : 4.2)) { step++; clock = 0; update(); } }
      return busy;
    }, { autoplay: true });
    const ctr = RL.controls(el);
    RL.button(ctr, '上一步', () => { auto = false; go(step - 1); });
    RL.button(ctr, '下一步', () => { auto = false; go(step + 1); }, 'primary');
    RL.button(ctr, '从头自动播放', () => { auto = true; go(0); });
    RL.button(ctr, '恢复默认读数', () => { rows.forEach((r, i) => { r.u = ROWS[i].u; }); els.forEach((e, i) => { e.querySelector('input').value = ROWS[i].u; }); update(); });
    update();
  });
})();
