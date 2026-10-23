/* main.js — 页面入口：渲染公式 → 初始化目录/术语 → 按需初始化各个图 */
(function () {
  'use strict';
  const RL = window.RL;
  const start = () => {
    // 开发用：?only=ch3 只显示某一章（无头浏览器截图总是从页顶开始）
    const only = (location.search.match(/[?&]only=([\w-]+)/) || [])[1];
    if (only) {
      document.querySelector('.hero').style.display = 'none';
      RL.$$('.chapter').forEach(s => { if (s.id !== only) s.remove(); });
      const from = (location.search.match(/[?&]from=([\w-]+)/) || [])[1], f = from && document.getElementById(from);
      if (f) while (f.previousElementSibling) f.previousElementSibling.remove();
    }
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
        ignoredClasses: ['no-math'],
        throwOnError: false
      });
    }
    RL.initToc();
    RL.initTerms();
    RL.initFigures();
    // 公式渲染会改变页面高度：带锚点打开时重新定位一次
    if (location.hash.length > 1) {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target) requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
    }
    document.documentElement.dataset.ready = '1';
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
