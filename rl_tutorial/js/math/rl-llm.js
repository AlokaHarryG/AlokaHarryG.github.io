/* rl-llm.js — 大模型 RL 部分的数学：GRPO、KL 估计器、恰当评分规则、RLCD、校准
 * 公式均对照原论文：DeepSeekMath (arXiv 2402.03300) 式(3)(4)；OpenJev-RLCD (arXiv 2609.38850) §2–§3。 */
(function (root) {
  'use strict';
  const C = (typeof module !== 'undefined' && module.exports) ? require('./rl-core.js') : root.RLCore;
  const L = {};

  // ---------- GRPO：组内相对优势 Â_i = (r_i − mean(r)) / std(r) ----------
  // 论文只写"除以组内标准差"；不同开源实现在无偏/有偏 std 与 ε 上略有差异，这里用总体标准差。
  L.grpoAdvantages = function (rewards, opts) {
    const o = opts || {}, m = C.mean(rewards), sd = C.std(rewards, o.unbiased);
    if (sd < 1e-8) return { mean: m, std: sd, adv: rewards.map(() => 0) }; // 全对/全错：没有学习信号
    return { mean: m, std: sd, adv: rewards.map(r => (r - m) / (sd + (o.eps || 0))) };
  };

  // ---------- KL 估计器（Schulman 2020 "Approximating KL Divergence"） ----------
  // 采样 x ~ π_θ，ρ = π_ref(x) / π_θ(x)
  L.k1 = rho => -Math.log(rho);              // 无偏，但单个样本可能为负
  L.k3 = rho => rho - Math.log(rho) - 1;     // 无偏，且恒 ≥ 0（GRPO 式(4)）

  // ---------- 恰当评分规则 ----------
  // RLCD 论文使用的 Brier 形式：J(u, y) = 2u_y − ‖u‖² = 1 − ‖u − e_y‖²
  L.brier = (u, y) => 2 * u[y] - C.dot(u, u);
  // 对结果分布 q 取期望：J(u, q) = 2uᵀq − ‖u‖²，唯一最大值在 u = q
  L.brierExpected = (u, q) => 2 * C.dot(u, q) - C.dot(u, u);
  L.logScoreExpected = (u, q) => q.reduce((s, qi, i) => s + (qi > 0 ? qi * Math.log(Math.max(u[i], 1e-12)) : 0), 0);
  // 二分类便捷版本：报告 "是" 的概率 p，真实概率 q
  L.bin = p => [p, 1 - p];
  // RLVR 视角：从 p 中采样一个答案，答对得 1 → 期望 = Pr(答对)，对 p 是线性的
  L.rlvrExpectedBinary = (p, q) => p * q + (1 - p) * (1 - q);

  // ---------- 方差恒等式（RLCD 论文 Proposition 1） ----------
  // J(p, q) = E_r J(u(r), q) + E_r ‖u(r) − p‖²，其中 p = E_r u(r)
  L.varianceIdentity = function (us, q) {
    const K = q.length, M = us.length;
    const p = new Array(K).fill(0).map((_, k) => C.mean(us.map(u => u[k])));
    const mixture = L.brierExpected(p, q);
    const perRationale = C.mean(us.map(u => L.brierExpected(u, q)));
    const variance = C.mean(us.map(u => u.reduce((s, x, k) => s + (x - p[k]) ** 2, 0)));
    return { p, mixture, perRationale, variance, M };
  };

  // ---------- RLCD 算法 1 ----------
  // 第一阶段（校准决策）：ℒ = −(1/M) Σ R_i，只走 readout 的路径梯度，不强化推理
  L.rlcdStage1Loss = R => -C.mean(R);
  // 第二阶段（强化推理）：A_i = (1/M)(R_i − β log π_θ/π_θ1) − 留一均值
  L.rlcdStage2Advantages = function (R, logRatio, beta) {
    const M = R.length, s = R.map((r, i) => (r - beta * (logRatio ? logRatio[i] : 0)) / M);
    const tot = C.sum(s);
    return s.map(x => x - (tot - x) / (M - 1));
  };

  // ---------- 校准：可靠性图、风险–覆盖曲线 ----------
  L.reliability = function (conf, correct, nBins) {
    const bins = Array.from({ length: nBins }, (_, i) => ({ lo: i / nBins, hi: (i + 1) / nBins, n: 0, acc: 0, conf: 0 }));
    conf.forEach((c, i) => {
      const b = bins[Math.min(nBins - 1, Math.floor(c * nBins))];
      b.n++; b.acc += correct[i]; b.conf += c;
    });
    let ece = 0;
    bins.forEach(b => { if (b.n) { b.acc /= b.n; b.conf /= b.n; ece += (b.n / conf.length) * Math.abs(b.acc - b.conf); } });
    return { bins, ece };
  };
  // 按置信度从高到低"自动处理"，覆盖率 = 处理比例，风险 = 其中的错误率
  L.riskCoverage = function (conf, correct) {
    const idx = conf.map((c, i) => i).sort((a, b) => conf[b] - conf[a]);
    let err = 0;
    return idx.map((i, k) => { err += 1 - correct[i]; return { coverage: (k + 1) / idx.length, risk: err / (k + 1), conf: conf[i] }; });
  };
  L.aurc = curve => C.mean(curve.map(p => p.risk));
  L.coverageAtRisk = (curve, eps) => curve.reduce((best, p) => (p.risk <= eps ? p.coverage : best), 0);

  // 示意用的合成数据：同一批题目，"过度自信"模型与"校准"模型给出的置信度
  // 过度自信：置信度普遍接近 1，且与对错只弱相关（论文 Fig.2 中 GRPO/RFT 的风险曲线几乎是平的）
  L.simulateCalibration = function (n, seed) {
    const r = C.rng(seed), sig = x => 1 / (1 + Math.exp(-x));
    const truth = [], calibrated = [], overconfident = [], correct = [];
    for (let i = 0; i < n; i++) {
      const z = 1.0 + 1.6 * C.randn(r), c = sig(z);
      truth.push(c); correct.push(r() < c ? 1 : 0);
      calibrated.push(c);
      overconfident.push(sig(2.6 + 1.1 * (0.35 * (z - 1) + 0.94 * C.randn(r))));
    }
    return { truth, calibrated, overconfident, correct };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = L;
  else root.RLLLM = L;
})(typeof window !== 'undefined' ? window : globalThis);
