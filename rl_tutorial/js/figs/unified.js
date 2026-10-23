/* unified.js — 图 8.1 梯度系数的演化：∇J = E[ GC · ∇θ log πθ ]（DeepSeekMath §5.2.1 的统一视角） */
(function () {
  'use strict';
  const RL = window.RL;
  const H = (s, c) => `{\\color{${c}}${s}}`; // 高亮"这一代改了什么"
  const CARDS = [
    { n: 'REINFORCE', c: '--c-pg', gc: 'G_t', solved: '只用"分数"就能直接优化策略，不需要知道环境模型。', left: '方差极大：所有奖励为正时，每个动作都会被加强。' },
    { n: '+ 基线', c: '--c-pg', gc: `G_t ${H('-\\,b(s_t)', '#d93a63')}`, solved: '减去与动作无关的参照，期望不变，方差大降。', left: '基线只是全局平均，不知道"在这个状态下"的平均水平。' },
    { n: 'Actor-Critic + GAE', c: '--c-ac', gc: `${H('\\hat A_t=\\sum_l(\\gamma\\lambda)^l\\delta_{t+l}', '#4b53d6')}`, solved: '评论家 V(s) 给出逐状态的基线；λ 在偏差与方差之间连续调节。', left: '一批数据只能用一次，步子一大策略就崩。' },
    { n: 'PPO', c: '--c-ppo', gc: `${H('\\mathbb 1[\\text{未越界}]\\cdot\\rho_t', '#7b4ce0')}\\,\\hat A_t`, solved: '裁剪让同一批数据可以安全地更新多轮，只用一阶优化。', left: '仍需评论家；搬到大模型上要同时放四个大网络。' },
    { n: 'RLHF-PPO', c: '--c-rlhf', gc: `\\mathbb 1[\\cdot]\\,\\rho_t\\,\\hat A_t^{\\mathrm{GAE}},\\quad r_t=${H('r_\\phi-\\beta\\log\\tfrac{\\pi_\\theta}{\\pi_{\\mathrm{ref}}}', '#b0469e')}`, solved: '把每个 token 当作动作，用人类偏好训练出的奖励模型对齐大模型；KL 防止奖励作弊。', left: '价值模型与策略同样大；奖励只在最后一个 token，逐 token 估值很难学准。' },
    { n: 'GRPO', c: '--c-grpo', gc: `${H('\\frac{r_i-\\mathrm{mean}(\\mathbf r)}{\\mathrm{std}(\\mathbf r)}', '#e0782b')} + ${H('\\beta\\big(\\tfrac{\\pi_{\\mathrm{ref}}}{\\pi_\\theta}-1\\big)', '#d93a63')}`, solved: '用同题多采样的组均值当基线，去掉价值模型；KL 直接进损失；天然适合判卷式奖励（RLVR）。', left: '只奖励"对不对"会让模型过度自信；全对或全错的组没有信号。' },
    { n: 'RLCD', c: '--c-rlcd', gc: `\\underbrace{${H('\\nabla_\\theta R_i', '#2f6fdf')}}_{\\text{路径项}}+\\big(${H('R_i', '#d93a63')}-\\tfrac{1}{M-1}\\textstyle\\sum_{j\\ne i}R_j\\big)\\nabla_\\theta\\log\\pi_\\theta(r_i),\\ \\ ${H('R_i=J(u_i,Y)', '#d93a63')}`, solved: '奖励换成读出分布的严格恰当评分：既要答对，也要让说出的概率与正确率一致；先校准、再强化。', left: '需要可读出的选项分布；对标注者分歧这类偶然不确定性帮助有限；实验规模尚小。' }
  ];

  RL.figure('unified', function (el) {
    let k = 0;
    const root = RL.el('div', { class: 'uni', tabindex: '0', 'aria-roledescription': '轮播', 'aria-label': '梯度系数的演化，使用左右方向键切换' });
    const track = RL.el('div', { class: 'uni-track' }), dots = RL.el('div', { class: 'uni-dots' });
    const cards = CARDS.map((c, i) => {
      const card = RL.el('div', { class: 'uni-card', style: `--c:${RL.css(c.c)}`, role: 'group', 'aria-label': `${i + 1} / ${CARDS.length}：${c.n}` });
      card.innerHTML = `<div class="uc-top"><span class="uc-n mono">${String(i + 1).padStart(2, '0')}</span><span class="uc-name">${c.n}</span></div><div class="uc-gc"><span class="uc-lab">梯度系数 GC</span><div class="uc-tex"></div></div><div class="uc-row"><span class="uc-tag ok">解决了</span><p>${c.solved}</p></div><div class="uc-row"><span class="uc-tag left">遗留问题</span><p>${c.left}</p></div>`;
      RL.tex(card.querySelector('.uc-tex'), c.gc, true);
      track.appendChild(card);
      const d = RL.el('button', { type: 'button', 'aria-label': c.n, style: `--c:${RL.css(c.c)}` });
      d.addEventListener('click', () => go(i));
      dots.appendChild(d);
      return card;
    });
    const nav = RL.el('div', { class: 'uni-nav' });
    const prev = RL.button(nav, '←', () => go(k - 1)), next = RL.button(nav, '→', () => go(k + 1), 'primary');
    prev.setAttribute('aria-label', '上一代'); next.setAttribute('aria-label', '下一代');
    nav.insertBefore(dots, next);
    root.append(track, nav); el.appendChild(root);
    const go = i => {
      k = Math.max(0, Math.min(CARDS.length - 1, i));
      cards.forEach((c, j) => { c.classList.toggle('on', j === k); c.classList.toggle('past', j < k); c.setAttribute('aria-hidden', String(j !== k)); });
      [...dots.children].forEach((d, j) => { d.setAttribute('aria-current', String(j === k)); d.classList.toggle('past', j < k); });
      prev.disabled = k === 0; next.disabled = k === CARDS.length - 1;
    };
    root.addEventListener('keydown', e => { if (e.key === 'ArrowRight') { e.preventDefault(); go(k + 1); } if (e.key === 'ArrowLeft') { e.preventDefault(); go(k - 1); } });
    go(0);
  });
})();
