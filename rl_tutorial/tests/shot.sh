#!/usr/bin/env bash
# 用无头 Edge 给页面截图 / 导出 DOM，检查运行时错误。
# 用法：tests/shot.sh <锚点|-> <输出名> [宽] [高]
#   tests/shot.sh ch1 ch1 1400 2000   → tests/_shots/ch1.png
#   tests/shot.sh dom errors          → 打印 <html> 上记录的 data-errors
set -u
cd "$(dirname "$0")/.."
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
ROOT="$(pwd -W)"                      # D:/gmh/...
URL="file:///${ROOT}/index.html?eager&still"
mkdir -p tests/_shots
if [ "$1" = "dom" ]; then
  "$EDGE" --headless=new --disable-gpu --no-first-run --allow-file-access-from-files \
    --virtual-time-budget=8000 --dump-dom "$URL" 2>/dev/null > tests/_shots/dom.html
  node -e "
    const t=require('fs').readFileSync('tests/_shots/dom.html','utf8');
    const m=t.match(/<html[^>]*>/); console.log(m?m[0].slice(0,2500):'NO HTML TAG');
    const errs=[...t.matchAll(/data-error=\"([^\"]*)\"/g)].map(x=>x[1]); console.log('figure errors:', errs.length? errs : 'none');
    const ready=(t.match(/data-ready=\"1\"/g)||[]).length; console.log('figures ready:', ready);
    const kerr=[...t.matchAll(/class=\"katex-error\"[^>]*?title=\"([^\"]*)\"/g)].map(x=>x[1]);
    console.log('katex blocks:', (t.match(/class=\"katex\"/g)||[]).length, '| katex errors:', kerr.length); kerr.slice(0,8).forEach(e=>console.log('  -', e.slice(0,160)));
  "
  exit 0
fi
HASH=""; [ "$1" != "-" ] && HASH="&only=$1"   # 第一个参数：章节 id，只渲染这一章
"$EDGE" --headless=new --disable-gpu --no-first-run --hide-scrollbars --allow-file-access-from-files \
  --force-device-scale-factor=1 --window-size="${3:-1400},${4:-1800}" --virtual-time-budget=6000 \
  --screenshot="${ROOT}/tests/_shots/$2.png" "${URL}${HASH}" 2>/dev/null
ls -la "tests/_shots/$2.png"
