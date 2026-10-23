/* ui.js — 控件、KaTeX、图注册、目录跟随、阅读进度、术语提示 */
(function () {
  'use strict';
  const RL = window.RL = window.RL || {};

  RL.$ = (sel, root) => (root || document).querySelector(sel);
  RL.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  RL.el = function (tag, attrs, children) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    });
    (children || []).forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return e;
  };
  let uid = 0;

  // ---------- 控件 ----------
  RL.controls = parent => parent.appendChild(RL.el('div', { class: 'controls' }));
  RL.slider = function (parent, o) {
    const id = 'rl-s' + (++uid);
    const input = RL.el('input', { type: 'range', id, min: o.min, max: o.max, step: o.step, value: o.value });
    const out = RL.el('output', { for: id });
    const fmt = o.fmt || (v => v);
    const wrap = RL.el('label', { class: 'slider', for: id }, [RL.el('span', { html: o.label }), input, out]);
    parent.appendChild(wrap);
    const sync = () => { out.textContent = fmt(+input.value); };
    input.addEventListener('input', () => { sync(); o.onInput(+input.value); });
    sync();
    return { input, get value() { return +input.value; }, set(v) { input.value = v; sync(); } };
  };
  RL.seg = function (parent, o) {
    const wrap = RL.el('div', { class: 'seg', role: 'group', 'aria-label': o.label });
    const btns = o.options.map(opt => {
      const b = RL.el('button', { type: 'button', 'aria-pressed': String(opt.value === o.value), html: opt.label });
      b.addEventListener('click', () => { api.set(opt.value); o.onChange(opt.value); });
      wrap.appendChild(b);
      return b;
    });
    parent.appendChild(wrap);
    const api = { value: o.value, set(v) { api.value = v; btns.forEach((b, i) => b.setAttribute('aria-pressed', String(o.options[i].value === v))); } };
    return api;
  };
  RL.button = function (parent, label, onClick, cls) {
    const b = RL.el('button', { type: 'button', class: 'btn ' + (cls || ''), html: label });
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  };
  RL.playButton = function (parent, loop) {
    const b = RL.button(parent, '', () => loop.toggle(), 'primary');
    const sync = p => { b.textContent = p ? '暂停' : '播放'; b.setAttribute('aria-pressed', String(p)); };
    loop.onchange = sync; sync(loop.playing);
    return b;
  };
  RL.stats = parent => parent.appendChild(RL.el('div', { class: 'stats', 'aria-live': 'polite' }));
  RL.stat = function (parent, label, cls) {
    const b = RL.el('b');
    parent.appendChild(RL.el('span', { class: 'stat ' + (cls || '') + (label ? '' : ' msg') }, label ? [label + ' ', b] : [b]));
    return { set(v) { b.textContent = v; }, el: b };
  };

  // ---------- 公式 ----------
  RL.math = function (el) {
    if (window.renderMathInElement) window.renderMathInElement(el, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false });
  };
  RL.tex = function (el, src, display) {
    if (window.katex) window.katex.render(src, el, { displayMode: !!display, throwOnError: false });
    else el.textContent = src;
  };

  // ---------- 图：滚动到附近才初始化 ----------
  const registry = {};
  RL.figure = (id, init) => { registry[id] = init; };
  RL.initFigures = function () {
    const eager = /[?&]eager\b/.test(location.search);
    const start = el => {
      if (el.dataset.ready) return;
      el.dataset.ready = '1';
      try { registry[el.dataset.fig](el); }
      catch (err) { el.dataset.error = err.message; console.error('[figure ' + el.dataset.fig + ']', err); RL.reportError(err); }
    };
    const els = RL.$$('[data-fig]').filter(el => {
      if (registry[el.dataset.fig]) return true;
      RL.reportError(new Error('未注册的图: ' + el.dataset.fig));
      return false;
    });
    if (eager || !('IntersectionObserver' in window)) { els.forEach(start); return; }
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); start(e.target); } }), { rootMargin: '600px 0px' });
    els.forEach(el => io.observe(el));
  };
  RL.reportError = function (err) {
    const root = document.documentElement;
    root.dataset.errors = ((root.dataset.errors || '') + ' | ' + (err && err.message)).slice(0, 2000);
  };
  window.addEventListener('error', e => {
    if (/ResizeObserver loop/.test(e.message || '')) return; // 浏览器的无害提示
    RL.reportError(e.error || new Error(e.message));
  });

  // ---------- 目录跟随 + 阅读进度 ----------
  RL.initToc = function () {
    const links = RL.$$('.toc a[href^="#"]');
    const targets = links.map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    const bar = RL.$('.progress');
    let ticking = false;
    const update = () => {
      ticking = false;
      const y = window.scrollY + window.innerHeight * 0.28;
      let cur = null;
      targets.forEach(t => { if (t.getBoundingClientRect().top + window.scrollY <= y) cur = t; });
      links.forEach(a => {
        const on = cur && a.getAttribute('href') === '#' + cur.id;
        a.classList.toggle('active', !!on);
        if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
      });
      // 主目录项：只要当前小节属于它就高亮
      RL.$$('.toc > ol > li').forEach(li => {
        const sub = cur && li.querySelector(`a[href="#${cur.id}"]`);
        li.querySelector(':scope > a').classList.toggle('active', !!sub);
      });
      const max = document.documentElement.scrollHeight - window.innerHeight;
      if (bar) bar.style.transform = `scaleX(${max > 0 ? window.scrollY / max : 0})`;
    };
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  };

  // ---------- 术语提示（悬停 / 键盘聚焦都能看到） ----------
  RL.initTerms = function () {
    const tip = RL.el('div', { class: 'tip', role: 'tooltip', id: 'rl-tip' });
    document.body.appendChild(tip);
    const show = el => {
      tip.textContent = el.dataset.tip;
      tip.classList.add('show');
      const r = el.getBoundingClientRect(), tw = Math.min(300, window.innerWidth - 24);
      tip.style.left = Math.max(12, Math.min(window.innerWidth - tw - 12, r.left + r.width / 2 - tw / 2)) + 'px';
      tip.style.top = (r.bottom + 8) + 'px';
      el.setAttribute('aria-describedby', 'rl-tip');
    };
    const hide = el => { tip.classList.remove('show'); el.removeAttribute('aria-describedby'); };
    RL.$$('.term[data-tip]').forEach(el => {
      el.tabIndex = 0;
      el.addEventListener('mouseenter', () => show(el)); el.addEventListener('focus', () => show(el));
      el.addEventListener('mouseleave', () => hide(el)); el.addEventListener('blur', () => hide(el));
    });
    window.addEventListener('scroll', () => tip.classList.remove('show'), { passive: true });
  };
})();
