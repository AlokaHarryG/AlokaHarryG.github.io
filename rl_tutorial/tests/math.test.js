// 运行：node --test tests/
// 目的：保证页面动画背后的数值计算与论文中的定义一致。
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../js/math/rl-core.js');
const L = require('../js/math/rl-llm.js');
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} ≉ ${b}`);

test('softmax 归一化且对平移不变', () => {
  const p = C.softmax([1, 2, 3]), q = C.softmax([101, 102, 103]);
  close(C.sum(p), 1);
  p.forEach((x, i) => close(x, q[i]));
});

test('折扣回报 G_t = r_t + γ G_{t+1}', () => {
  const G = C.discountedReturns([0, 0, 1], 0.9);
  close(G[0], 0.81); close(G[1], 0.9); close(G[2], 1);
});

test('价值迭代：无步长惩罚时 V = γ^(到终点步数−1)', () => {
  // 1×4 走廊，最右端为 +1 终点
  const grid = { w: 4, h: 1, walls: new Set(), terminals: new Map([[3, 1]]), stepReward: 0 };
  let V = new Array(4).fill(0), res;
  for (let k = 0; k < 20; k++) { res = C.valueIterationSweep(grid, V, 0.9); V = res.V; }
  close(V[2], 1); close(V[1], 0.9); close(V[0], 0.81);
  assert.equal(res.policy[0], 1); // 向右
  close(res.maxDelta, 0);
});

test('GAE：λ=0 退化为 TD 误差，λ=1 退化为蒙特卡洛回报减基线', () => {
  const r = [0, 0, 0, 1], v = [0.3, 0.5, 0.2, 0.6, 0], g = 0.95;
  const a0 = C.gae(r, v, g, 0), a1 = C.gae(r, v, g, 1), G = C.discountedReturns(r, g);
  a0.adv.forEach((x, t) => close(x, r[t] + g * v[t + 1] - v[t]));
  a1.adv.forEach((x, t) => close(x, G[t] - v[t]));
});

test('PPO 裁剪：只在"有利方向"越界后梯度为 0', () => {
  assert.equal(C.ppoClipGradCoef(1.3, +1, 0.2), 0);   // A>0 且已涨过 1+ε
  assert.equal(C.ppoClipGradCoef(0.7, +1, 0.2), 1);   // A>0 但比例变小：仍然有梯度把它拉回
  assert.equal(C.ppoClipGradCoef(0.7, -1, 0.2), 0);   // A<0 且已降过 1−ε
  assert.equal(C.ppoClipGradCoef(1.5, -1, 0.2), -1);  // A<0 但比例变大：仍然有梯度
  close(C.ppoClipObjective(1.5, 2, 0.2), 2.4);
  close(C.ppoClipObjective(0.5, -2, 0.2), -1.6);
});

test('PPO 多轮更新：裁剪把 KL 控制住，不裁剪则越跑越远', () => {
  const means = [1.0, 0.8, 0.5, 0.2, 0.0];
  for (let seed = 1; seed <= 20; seed++) {
    const batch = C.sampleBanditBatch([0, 0, 0, 0, 0], means, 1.0, 16, seed);
    const c = C.ppoBatchUpdate([0, 0, 0, 0, 0], batch, { eps: 0.2, lr: 1, epochs: 40, clip: true });
    const u = C.ppoBatchUpdate([0, 0, 0, 0, 0], batch, { eps: 0.2, lr: 1, epochs: 40, clip: false });
    assert.ok(c[c.length - 1].kl < 0.12, `seed ${seed} clipped KL ${c[c.length - 1].kl}`);
    assert.ok(u[u.length - 1].kl > 3 * c[c.length - 1].kl, `seed ${seed}`);
  }
});

test('REINFORCE：奖励整体平移后，基线能消除额外的方差', () => {
  const finals = (useBaseline) => Array.from({ length: 30 }, (_, i) => {
    const h = C.reinforceBandit({ means: [0.2, 0.5, 1.0], noise: 1, offset: 5, lr: 0.05, steps: 300, useBaseline, baselineRate: 0.05, seed: i + 1 });
    return h[h.length - 1].expReward;
  });
  assert.ok(C.std(finals(false)) > 4 * C.std(finals(true)));
});

test('GRPO 组内优势：均值 0、标准差 1；全对时没有信号', () => {
  const { adv } = L.grpoAdvantages([1, 0, 1, 1, 0, 0, 0, 1]);
  close(C.mean(adv), 0); close(C.std(adv), 1);
  assert.deepEqual(L.grpoAdvantages([1, 1, 1, 1]).adv, [0, 0, 0, 0]);
});

test('k3 KL 估计器：逐样本非负，期望等于精确 KL', () => {
  const pTheta = [0.5, 0.3, 0.2], pRef = [0.2, 0.5, 0.3];
  let est = 0;
  pTheta.forEach((p, i) => { const k = L.k3(pRef[i] / p); assert.ok(k >= 0); est += p * k; });
  close(est, C.kl(pTheta, pRef));
  assert.ok(L.k1(2) < 0); // k1 单个样本可以为负
});

test('Brier 恰当性：期望得分在 u = q 处取最大', () => {
  const q = 0.7;
  let best = 0, bestP = -1;
  for (let p = 0; p <= 1.0001; p += 0.01) { const s = L.brierExpected(L.bin(p), L.bin(q)); if (s > best) { best = s; bestP = p; } }
  close(bestP, q, 1e-6);
  // 对照：RLVR 的期望奖励对 p 线性，q>0.5 时在 p=1 处取最大 → 鼓励过度自信
  assert.ok(L.rlvrExpectedBinary(1, q) > L.rlvrExpectedBinary(q, q));
});

test('方差恒等式（Prop. 1）与推论 1：RLVR = 混合目标去掉多样性项', () => {
  const us = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.1, 0.4]], q = [0, 1, 0];
  const v = L.varianceIdentity(us, q);
  close(v.mixture, v.perRationale + v.variance);
  // 采样出独热答案 e_A，A ~ p：E_A J(e_A, q) = 2 Pr(A=Y) − 1
  const p = [0.2, 0.5, 0.3], qq = [0.1, 0.6, 0.3];
  const lhs = p.reduce((s, pa, a) => s + pa * L.brierExpected([0, 1, 2].map(k => (k === a ? 1 : 0)), qq), 0);
  close(lhs, 2 * C.dot(p, qq) - 1);
});

test('RLCD 第二阶段留一优势：总和为 0', () => {
  const A = L.rlcdStage2Advantages([0.9, -0.5, 0.2, 0.6], [0.1, -0.2, 0, 0.05], 0.04);
  close(C.sum(A), 0, 1e-12);
});

test('校准模拟：校准模型 ECE、AURC 更低，低风险下覆盖率更高', () => {
  const s = L.simulateCalibration(2000, 7);
  const cal = L.riskCoverage(s.calibrated, s.correct), over = L.riskCoverage(s.overconfident, s.correct);
  assert.ok(L.reliability(s.calibrated, s.correct, 10).ece < L.reliability(s.overconfident, s.correct, 10).ece);
  assert.ok(L.aurc(cal) < L.aurc(over));
  assert.ok(L.coverageAtRisk(cal, 0.1) > L.coverageAtRisk(over, 0.1));
});
