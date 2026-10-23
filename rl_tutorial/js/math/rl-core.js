/* rl-core.js — 经典强化学习部分的数学（浏览器与 Node 通用，便于单元测试）
 * 页面上所有"实时计算"的动画都调用这里的函数，而不是写死的数字。 */
(function (root) {
  'use strict';
  const M = {};

  // ---------- 可复现的随机数 ----------
  M.rng = function (seed) { // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  M.randn = function (r) { // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  M.sampleCategorical = function (p, r) {
    const x = r(); let c = 0;
    for (let i = 0; i < p.length; i++) { c += p[i]; if (x < c) return i; }
    return p.length - 1;
  };

  // ---------- 基础工具 ----------
  M.sum = a => a.reduce((s, x) => s + x, 0);
  M.mean = a => M.sum(a) / a.length;
  M.dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  M.std = function (a, unbiased) {
    const m = M.mean(a), d = unbiased ? a.length - 1 : a.length;
    return Math.sqrt(M.sum(a.map(x => (x - m) ** 2)) / d);
  };
  M.clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  M.softmax = function (z) {
    const m = Math.max(...z), e = z.map(x => Math.exp(x - m)), s = M.sum(e);
    return e.map(x => x / s);
  };
  M.kl = function (p, q) { // KL(p || q)
    let s = 0;
    for (let i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]);
    return s;
  };

  // ---------- 回报：G_t = R_{t+1} + γ G_{t+1} ----------
  M.discountedReturns = function (rewards, gamma) {
    const G = new Array(rewards.length); let g = 0;
    for (let t = rewards.length - 1; t >= 0; t--) { g = rewards[t] + gamma * g; G[t] = g; }
    return G;
  };

  // ---------- 网格世界：确定性转移 + 价值迭代 ----------
  // grid = { w, h, walls:Set<idx>, terminals:Map<idx, reward>, stepReward }
  M.ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // 上 右 下 左
  M.gridStep = function (grid, s, a) {
    const x = s % grid.w, y = Math.floor(s / grid.w);
    const nx = x + M.ACTIONS[a][0], ny = y + M.ACTIONS[a][1];
    let s2 = ny * grid.w + nx;
    if (nx < 0 || ny < 0 || nx >= grid.w || ny >= grid.h || grid.walls.has(s2)) s2 = s; // 撞墙原地不动
    const r = grid.terminals.has(s2) ? grid.terminals.get(s2) : grid.stepReward;
    return { s2, r, done: grid.terminals.has(s2) };
  };
  // 对每个动作做一次 Bellman 备份：q(s,a) = r + γ V(s')（进入终点后不再有未来）
  M.bellmanBackups = function (grid, V, s, gamma) {
    return M.ACTIONS.map((_, a) => {
      const { s2, r, done } = M.gridStep(grid, s, a);
      return r + (done ? 0 : gamma * V[s2]);
    });
  };
  // 同步价值迭代一轮：V_{k+1}(s) = max_a [ r + γ V_k(s') ]
  M.valueIterationSweep = function (grid, V, gamma) {
    const n = grid.w * grid.h, V2 = new Array(n).fill(0), policy = new Array(n).fill(-1);
    let maxDelta = 0;
    for (let s = 0; s < n; s++) {
      if (grid.walls.has(s) || grid.terminals.has(s)) continue;
      const q = M.bellmanBackups(grid, V, s, gamma);
      let best = 0;
      for (let a = 1; a < q.length; a++) if (q[a] > q[best] + 1e-12) best = a;
      V2[s] = q[best]; policy[s] = best;
      maxDelta = Math.max(maxDelta, Math.abs(V2[s] - V[s]));
    }
    return { V: V2, policy, maxDelta };
  };

  // ---------- 广义优势估计 GAE ----------
  // values 长度 T+1，values[T] 是终止后的自举值（回合结束时为 0）
  M.gae = function (rewards, values, gamma, lambda) {
    const T = rewards.length, deltas = new Array(T), adv = new Array(T);
    for (let t = 0; t < T; t++) deltas[t] = rewards[t] + gamma * values[t + 1] - values[t];
    let a = 0;
    for (let t = T - 1; t >= 0; t--) { a = deltas[t] + gamma * lambda * a; adv[t] = a; }
    return { deltas, adv, returns: adv.map((x, t) => x + values[t]) };
  };

  // ---------- PPO 裁剪目标 ----------
  M.ppoClipObjective = (ratio, A, eps) => Math.min(ratio * A, M.clip(ratio, 1 - eps, 1 + eps) * A);
  // 对 ratio 的导数：一旦在"有利方向"越过 1±ε，这个样本就不再提供梯度
  M.ppoClipGradCoef = function (ratio, A, eps) {
    if (A > 0 && ratio > 1 + eps) return 0;
    if (A < 0 && ratio < 1 - eps) return 0;
    return A;
  };

  // ---------- 多臂老虎机上的 REINFORCE ----------
  // softmax 策略 π = softmax(θ)；∇_θ log π(a) = onehot(a) − π
  // opts: { means, noise, offset, lr, steps, useBaseline, baselineRate, seed }
  M.reinforceBandit = function (o) {
    const r = M.rng(o.seed), K = o.means.length;
    let theta = new Array(K).fill(0), b = null;
    const hist = [];
    const record = () => {
      const p = M.softmax(theta);
      hist.push({ p, expReward: M.dot(p, o.means) + o.offset });
    };
    record();
    for (let t = 0; t < o.steps; t++) {
      const p = M.softmax(theta), a = M.sampleCategorical(p, r);
      const reward = o.means[a] + o.offset + o.noise * M.randn(r);
      if (b === null) b = reward;
      const weight = o.useBaseline ? reward - b : reward; // 有/无基线
      theta = theta.map((th, k) => th + o.lr * weight * ((k === a ? 1 : 0) - p[k]));
      if (o.useBaseline) b += o.baselineRate * (reward - b); // 基线 = 奖励的滑动平均
      record();
    }
    return hist;
  };

  // ---------- PPO 在一个固定批次上做多轮更新 ----------
  // 策略：K 个动作的 softmax。批次中每个样本 (a_n, A_n) 都来自 π_old。
  // ratio_n = π_θ(a_n)/π_old(a_n)，∂ratio_n/∂θ_k = ratio_n (1[k=a_n] − π_θ(k))
  M.ppoBatchUpdate = function (oldLogits, batch, o) {
    const pOld = M.softmax(oldLogits);
    let theta = oldLogits.slice();
    const out = [];
    const snapshot = () => {
      const p = M.softmax(theta);
      const ratios = batch.map(s => p[s.a] / pOld[s.a]);
      const clipped = ratios.filter((x, n) => M.ppoClipGradCoef(x, batch[n].A, o.eps) === 0).length;
      out.push({ p, ratios, kl: M.kl(pOld, p), clippedFrac: clipped / batch.length });
    };
    snapshot();
    for (let e = 0; e < o.epochs; e++) {
      const p = M.softmax(theta), g = new Array(theta.length).fill(0);
      batch.forEach(s => {
        const ratio = p[s.a] / pOld[s.a];
        const c = o.clip ? M.ppoClipGradCoef(ratio, s.A, o.eps) : s.A;
        if (c === 0) return;
        for (let k = 0; k < theta.length; k++) g[k] += c * ratio * ((k === s.a ? 1 : 0) - p[k]);
      });
      theta = theta.map((th, k) => th + o.lr * g[k] / batch.length);
      snapshot();
    }
    return out;
  };
  // 从 π_old 采一个批次，并把优势做批内标准化（PPO 实现中的常见做法）
  M.sampleBanditBatch = function (logits, means, noise, n, seed) {
    const r = M.rng(seed), p = M.softmax(logits);
    const raw = [];
    for (let i = 0; i < n; i++) {
      const a = M.sampleCategorical(p, r);
      raw.push({ a, reward: means[a] + noise * M.randn(r) });
    }
    const rs = raw.map(s => s.reward), m = M.mean(rs), sd = M.std(rs) || 1;
    return raw.map(s => ({ a: s.a, reward: s.reward, A: (s.reward - m) / sd }));
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = M;
  else root.RLCore = M;
})(typeof window !== 'undefined' ? window : globalThis);
