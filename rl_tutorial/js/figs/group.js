/* group.js — 图 6.1 GRPO：一组 8 个回答，从判卷到组内相对优势（实时计算，可点击切换对错） */
(function () {
  'use strict';
  const RL = window.RL, C = window.RLCore;
  const SUB = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);
  const CORRECT = ['因式分解：(x−2)(x−3)=0，两根为 2 和 3，较大根是 3。', '求根公式：x=(5±√(25−24))/2=(5±1)/2，得 3 或 2，答案 3。',
    '韦达定理：两根之和 5、之积 6，只能是 2 与 3，较大者为 3。', '代入检验 x=3：9−15+6=0 ✓，x=2 也成立，较大根为 3。',
    '配方：(x−2.5)²=0.25，x−2.5=±0.5，较大根 x=3。', '判别式 Δ=1，两根为 (5±1)/2，较大的是 3。'];
  const WRONG = [['因式分解：(x−1)(x−6)=0，较大根是 6。', '6'], ['求根公式：x=(5±√(25+24))/2=(5±7)/2，较大根 6。', '6'],
    ['两根之和 −5、之积 6，所以根是 −2 与 −3，较大根 −2。', '−2'], ['配方：(x−5)²=19，较大根约为 9.36。', '9.36'],
    ['两边除以 x 得 x=5−6/x，猜 x=2 成立，答案 2。', '2'], ['判别式 Δ=25−24=1，所以 x=1。', '1']];

  RL.figure('group', function (el) {
    const L = window.RLLLM, G = 8, MAXA = Math.sqrt(G - 1), pos = RL.css('--pos'), neg = RL.css('--neg');
    let p = .5, seed = 10, rows = [], token = 0;
    const root = RL.el('div', { class: 'grp' });
    root.innerHTML = '<div class="grp-q"><span class="lab">题目 q</span>方程 x² − 5x + 6 = 0 的较大根是多少？<span class="muted">（标准答案 3；判卷程序只比对最终答案）</span></div>' +
      '<div class="grp-head" aria-hidden="true"><span>#</span><span>回答 oᵢ<small>（整段底色 = 这个回答所有 token 共享的 Âᵢ）</small></span><span>答案</span><span>判卷</span><span>rᵢ</span><span>优势 Âᵢ</span></div>';
    const list = RL.el('div', { class: 'grp-list' }), foot = RL.el('div', { class: 'grp-foot stats', 'aria-live': 'polite' });
    root.append(list, foot); el.appendChild(root);

    // 对错按 p 独立采样；文本在各自的池里不放回地抽（池用完才会重复）
    const sample = () => {
      const r = C.rng(seed * 7919), perm = () => { const a = [0, 1, 2, 3, 4, 5]; for (let i = 5; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
      const pc = perm(), pw = perm(); let nc = 0, nw = 0;
      rows = Array.from({ length: G }, () => { const ok = r() < p; return { ok, k: ok ? pc[nc++ % 6] : pw[nw++ % 6] }; });
    };
    sample();
    const els = rows.map((_, i) => {
      const b = RL.el('button', { type: 'button', class: 'g-row', style: `--d:${i * 85}ms` });
      b.innerHTML = `<span class="g-i mono">o${SUB(i + 1)}</span><span class="g-text"><span class="sn"></span></span><span class="g-ans mono"></span><span class="g-mark"></span><span class="g-r mono"></span><span class="g-adv"><em class="mono"></em><span class="trk"><i class="zero"></i><i class="bar"></i></span></span>`;
      b.addEventListener('click', () => { rows[i].ok = !rows[i].ok; update(); });
      list.appendChild(b);
      return b;
    });
    const update = () => {
      const rewards = rows.map(r => (r.ok ? 1 : 0)), g = L.grpoAdvantages(rewards);
      rows.forEach((row, i) => {
        const [txt, ans] = row.ok ? [CORRECT[row.k], '3'] : WRONG[row.k], A = g.adv[i], e = els[i], w = Math.abs(A) / MAXA * 50;
        e.querySelector('.sn').textContent = txt;
        e.querySelector('.g-ans').textContent = ans;
        e.querySelector('.g-mark').textContent = row.ok ? '✓' : '✗';
        e.querySelector('.g-r').textContent = rewards[i];
        e.classList.toggle('ok', row.ok); e.classList.toggle('bad', !row.ok);
        e.setAttribute('aria-label', `回答 ${i + 1}：${row.ok ? '答对' : '答错'}，奖励 ${rewards[i]}，优势 ${A.toFixed(2)}。点击切换对错`);
        const bar = e.querySelector('.bar');
        bar.className = 'bar ' + (A > 0 ? 'p' : A < 0 ? 'n' : '');
        bar.style.width = w + '%'; bar.style.left = A >= 0 ? '50%' : (50 - w) + '%';
        e.querySelector('em').textContent = (A > 0 ? '+' : '') + A.toFixed(2);
        e.style.setProperty('--tint', A > 0 ? RL.alpha(pos, Math.min(.3, .06 + .09 * A)) : A < 0 ? RL.alpha(neg, Math.min(.28, .06 + .09 * -A)) : 'transparent');
      });
      const k = C.sum(rewards), zero = g.std < 1e-8;
      foot.innerHTML = `<span class="stat">答对 <b>${k} / ${G}</b></span><span class="stat">组均值 μ = <b>${g.mean.toFixed(3)}</b></span><span class="stat">组标准差 σ = <b>${g.std.toFixed(3)}</b></span>` +
        (zero ? '<span class="stat neg">σ = 0 → 所有 Âᵢ = 0：<b>这道题本轮没有学习信号</b></span>'
          : `<span class="stat">答对者 Â = <b>${k ? '+' + ((1 - g.mean) / g.std).toFixed(2) : '—'}</b>，答错者 Â = <b>${k < G ? (-g.mean / g.std).toFixed(2) : '—'}</b></span><span class="stat">ΣÂᵢ = <b>${Math.abs(C.sum(g.adv)).toFixed(2)}</b>（恒为 0）</span>`);
    };
    const STAGES = ['st-in', 'st-judge', 'st-stats', 'st-adv'];
    const play = () => {
      const my = ++token;
      root.classList.remove(...STAGES);
      update();
      if (RL.reducedMotion) { root.classList.add(...STAGES); return; }
      void root.offsetWidth;
      [30, 1000, 1600, 2200].forEach((ms, k) => setTimeout(() => { if (my === token) root.classList.add(STAGES[k]); }, ms));
    };

    const ctr = RL.controls(el);
    RL.button(ctr, '重新采样一组', () => { seed++; sample(); play(); }, 'primary');
    RL.button(ctr, '全部答对', () => { rows.forEach(r => { r.ok = true; }); update(); });
    RL.button(ctr, '全部答错', () => { rows.forEach(r => { r.ok = false; }); update(); });
    RL.slider(ctr, { label: '模型正确率 p', min: .05, max: .95, step: .05, value: p, fmt: v => v.toFixed(2), onInput: v => { p = v; sample(); update(); } });
    update();
    if (RL.reducedMotion) play();
    else {
      const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); play(); } }, { threshold: .3 });
      io.observe(root);
    }
  });
})();
