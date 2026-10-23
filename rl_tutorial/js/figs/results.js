/* results.js — 图 7.6 OpenJev-RLCD 论文 Table 1（3 个种子的均值 ± 标准差；温度缩放之后的单次查询） */
(function () {
  'use strict';
  const RL = window.RL;
  // [方法, 类别, Acc, sd, Brier, sd, AURC, sd, Cov@ε, sd, T]；base 行无标准差
  const DATA = {
    gsm: { name: 'GSM8K-Verify', eps: '5%', rows: [
      ['base（零样本推理）', 'base', 84.1, 0, .211, 0, .073, 0, 14.9, 0, 9.1],
      ['SFT + TS（直接作答）', 'sft', 70.4, .2, .382, .009, .174, .010, 6.2, 2.0, 2.1],
      ['RFT/STaR + TS', 'rft', 89.0, 1.6, .207, .020, .103, .013, .9, .7, 7.3],
      ['RFT/STaR + KL + TS', 'rft', 91.4, .5, .160, .006, .075, .010, 11.6, 8.1, 6.6],
      ['GRPO + KL + TS', 'grpo', 91.8, .6, .150, .009, .061, .006, 19.4, 7.3, 7.2],
      ['RLCD 阶段 1（Brier 读数）', 'rlcd1', 90.4, 2.0, .164, .020, .047, .010, 62.6, 21.3, 1.6],
      ['RLCD 阶段 1（对数分数读数）', 'rlcd1', 91.3, .4, .149, .009, .038, .002, 72.2, 7.2, 1.8],
      ['RLCD 两阶段', 'rlcd', 92.2, .7, .135, .009, .034, .001, 81.1, 7.2, 1.4]] },
    mmlu: { name: 'MMLU-Pro', eps: '20%', rows: [
      ['base（零样本推理）', 'base', 38.9, 0, .786, 0, .516, 0, 0, 0, 10.9],
      ['SFT + TS（直接作答）', 'sft', 42.3, .4, .711, .002, .386, .005, 15.6, .7, 1.6],
      ['RFT/STaR + TS', 'rft', 32.7, .9, .830, .004, .619, .011, 0, 0, 17.6],
      ['RFT/STaR + KL + TS', 'rft', 40.9, .6, .761, .006, .493, .008, 0, 0, 11.0],
      ['GRPO + KL + TS', 'grpo', 42.4, .5, .753, .006, .481, .015, 0, 0, 11.1],
      ['RLCD 阶段 1（Brier 读数）', 'rlcd1', 47.0, 1.3, .667, .014, .312, .019, 27.3, 2.1, 1.4],
      ['RLCD 阶段 1（对数分数读数）', 'rlcd1', 46.5, .7, .668, .009, .319, .012, 24.8, 1.6, 1.5],
      ['RLCD 两阶段', 'rlcd', 47.7, 2.0, .658, .015, .304, .022, 27.6, 3.3, 1.6]] }
  };
  const METRICS = {
    cov: { label: 'Cov@ε', col: 8, hint: '越高越好：在错误预算内能自动处理的比例（%）', max: 100, fmt: v => v.toFixed(1) + '%' },
    acc: { label: '准确率', col: 2, hint: '越高越好（%）', max: 100, fmt: v => v.toFixed(1) + '%' },
    brier: { label: 'Brier', col: 4, hint: '越低越好', max: .9, fmt: v => v.toFixed(3) },
    aurc: { label: 'AURC', col: 6, hint: '越低越好：风险–覆盖曲线下面积', max: .65, fmt: v => v.toFixed(3) },
    t: { label: '温度 T', col: 10, hint: '越接近 1 越好：训练出来就是校准的', max: 18, fmt: v => v.toFixed(1) }
  };
  const NOTES = {
    gsm: { cov: '准确率几乎相同的 GRPO（91.8%）与 RLCD（92.2%），在 5% 错误预算下能自动处理的题目分别是 19.4% 与 81.1%。', acc: '几种推理方法的准确率非常接近；差别在于置信度能不能用来做决定。', brier: 'RLCD 两阶段的 Brier 分数最低（0.135）。', aurc: 'AURC 衡量"按置信度排序有多好"：RLCD 0.034，GRPO 0.061（配对 bootstrap p < 10⁻³）。', t: 'GRPO 与 RFT 需要 T ≈ 7 才能校准；RLCD 只要 1.4。' },
    mmlu: { cov: '在 20% 错误预算下，所有推理基线都无法自动处理任何题目（0%）；RLCD 两阶段为 27.6%，SFT + TS 为 15.6%。', acc: '论文报告 RLCD 比 SFT+TS 高 5.3 个点、比 GRPO+TS 高 5.2 个点（p < 10⁻³）。', brier: 'RLCD 两阶段 0.658，比 SFT+TS 低 0.053、比 GRPO+TS 低 0.095（p < 10⁻³）。无 KL 的 RFT 坍缩成空推理，Brier 0.830 比基座模型还差。', aurc: 'RLCD 0.304，而推理基线都在 0.48 以上。', t: 'RFT（无 KL）需要 T = 17.6；RLCD 为 1.6。' }
  };
  const COL = { base: '--ink-3', sft: '--ink-3', rft: '--c-ppo', grpo: '--c-grpo', rlcd1: '--c-rlcd', rlcd: '--c-rlcd' };

  RL.figure('results', function (el) {
    let ds = 'gsm', metric = 'cov';
    const root = RL.el('div', { class: 'res' });
    const hint = RL.el('div', { class: 'res-hint' }), list = RL.el('div', { class: 'res-list' }), note = RL.el('div', { class: 'res-note', 'aria-live': 'polite' });
    root.append(hint, list, note); el.appendChild(root);
    const els = DATA.gsm.rows.map(() => {
      const r = RL.el('div', { class: 'res-row' });
      r.innerHTML = '<span class="rn"></span><span class="rt"><i class="rb"></i><i class="rw"></i><i class="r1"></i></span><b class="rv mono"></b>';
      list.appendChild(r); return r;
    });
    const render = () => {
      const D = DATA[ds], M = METRICS[metric];
      hint.innerHTML = `<b>${D.name}</b> · ${M.label}${metric === 'cov' ? `（ε = ${D.eps}）` : ''} · <span class="muted">${M.hint}</span>`;
      D.rows.forEach((row, i) => {
        const e = els[i], v = row[M.col], sd = metric === 't' ? 0 : row[M.col + 1], pct = x => Math.max(0, Math.min(100, x / M.max * 100));
        e.className = 'res-row k-' + row[1];
        e.style.setProperty('--c', RL.css(COL[row[1]]));
        e.querySelector('.rn').textContent = row[0];
        e.querySelector('.rb').style.width = Math.max(.6, pct(v)) + '%';
        const w = e.querySelector('.rw');
        w.style.left = pct(v - sd) + '%'; w.style.width = (pct(v + sd) - pct(v - sd)) + '%'; w.style.opacity = sd ? 1 : 0;
        const one = e.querySelector('.r1');
        one.style.left = pct(1) + '%'; one.style.opacity = metric === 't' ? 1 : 0;
        e.querySelector('.rv').textContent = M.fmt(v) + (sd ? ` ± ${metric === 'cov' || metric === 'acc' ? sd.toFixed(1) : sd.toFixed(3)}` : '');
      });
      note.textContent = NOTES[ds][metric];
    };
    const ctr = RL.controls(el);
    RL.seg(ctr, { label: '数据集', value: ds, options: [{ value: 'gsm', label: 'GSM8K-Verify' }, { value: 'mmlu', label: 'MMLU-Pro' }], onChange: v => { ds = v; render(); } });
    RL.seg(ctr, { label: '指标', value: metric, options: Object.entries(METRICS).map(([k, m]) => ({ value: k, label: m.label })), onChange: v => { metric = v; render(); } });
    render();
  });
})();
