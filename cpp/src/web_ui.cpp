// 单页前端（HTML + CSS + JS）。独立 TU 是为了让 web.cpp 只管协议、这里只管界面。
//
// 切成多段 R"( )" 相邻拼接：MSVC 单个字符串字面量上限 65535 字节，超了是 C2026
// 且只截断不报错。每段不超过 ~8 KB，加内容时照样往里插新段即可。
//
// 与 C++ 侧的契约只有两条：/meta 给的几何与绘制口径（画布尺寸、ppm、骨架边表、
// 网格常数、kpt 门限），/stats 每帧一条 JSON（见 stats.cpp 的 json()）。
// 四个叠加层完全由这里用 <canvas> 重绘，C++ 只推未标注的原始画面。
namespace swim {

extern const char kIndexHtml[];

const char kIndexHtml[] =
R"HTML(<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>泳池实时分析</title><style>
*{box-sizing:border-box;margin:0}
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#30363d;--card:#161b22;--acc:#2f81f7}
body{background:var(--bg);color:var(--fg);font:13px/1.5 "Segoe UI",system-ui,sans-serif;
     height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:12px;padding:8px 14px;border-bottom:1px solid var(--line);
       background:var(--card);flex:none}
header h1{font-size:14px;font-weight:600;letter-spacing:.5px;white-space:nowrap}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:none;border:0;color:var(--dim);padding:4px 12px;cursor:pointer;font:inherit}
.seg button.on{background:var(--acc);color:#fff}
.grp{display:flex;align-items:center;gap:8px;padding:2px 10px;border:1px solid var(--line);
     border-radius:6px;white-space:nowrap}
.grp>em{font-style:normal;color:var(--dim);font-size:11px;letter-spacing:1px}
label{display:flex;align-items:center;gap:4px;color:var(--dim);cursor:pointer;user-select:none}
label:has(input:checked){color:var(--fg)}
input[type=checkbox]{accent-color:var(--acc);width:13px;height:13px}
.sp{flex:1}
#hud{color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}
#hud b{color:var(--fg);font-weight:600}
.btn{background:none;border:1px solid var(--line);color:var(--dim);border-radius:6px;
     padding:4px 10px;cursor:pointer;font:inherit}
.btn:hover{color:var(--fg);border-color:var(--acc)}
main{flex:1;display:grid;gap:8px;padding:8px;min-height:0}
main.m0{grid-template-columns:1fr;grid-template-rows:auto 1fr}
main.m1{grid-template-columns:1fr 300px;grid-template-rows:1.5fr 1fr}
main.m0 #fstage,main.m0 #person{display:none}
.stage{position:relative;background:#000;border:1px solid var(--line);border-radius:8px;
       overflow:hidden;min-height:0}
.stage img{width:100%;height:100%;object-fit:contain;display:block}
.stage canvas{position:absolute;inset:0;width:100%;height:100%}
#stage canvas{cursor:crosshair}
.tip{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
     color:var(--dim);pointer-events:none;text-align:center;padding:20px}
.tag{position:absolute;left:8px;top:6px;color:var(--dim);font-size:11px;letter-spacing:1px;
     pointer-events:none;text-shadow:0 1px 3px #000}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;
       overflow:auto;min-height:0}
.panel h2{font-size:11px;color:var(--dim);letter-spacing:1.5px;font-weight:600;
          padding-bottom:6px;border-bottom:1px solid var(--line);margin-bottom:8px}
.panel h2.sub{margin-top:12px}
.kv{display:grid;grid-template-columns:1fr auto;gap:2px 8px;font-variant-numeric:tabular-nums}
.kv span{color:var(--dim)}
.kv b{font-weight:600}
.big{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px}
.big div{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.big i{display:block;color:var(--dim);font-style:normal;font-size:11px}
.big b{font-size:19px;font-variant-numeric:tabular-nums}
.big b u{font-size:11px;color:var(--dim);margin-left:2px;text-decoration:none}
.lanes{margin-top:10px;display:grid;gap:3px}
.lane{display:grid;grid-template-columns:34px 1fr 20px;align-items:center;gap:6px;
      color:var(--dim);font-size:11px}
.lane u{height:7px;border-radius:4px;background:#21262d;display:block;text-decoration:none}
.lane u>i{display:block;height:100%;border-radius:4px;background:var(--acc)}
.lane b{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
/* 纯全景布局：概览横过来放顶上。画布是 5002x2102 的扁形，右侧栏会白白吃掉宽度，
   而上方那点高度画布本来也用不满。m1 下仍是右侧栏（那时高度才是紧的）。*/
main.m0 #global{order:-1;display:flex;align-items:center;gap:16px;padding:6px 12px;
                overflow:hidden}
main.m0 #global h2{border:0;padding:0;margin:0;flex:none}
main.m0 .big{margin:0;grid-template-columns:repeat(4,minmax(92px,1fr));flex:none}
main.m0 .kv{grid-template-columns:auto auto}
main.m0 .lanes{margin:0;grid-template-columns:repeat(8,1fr);flex:1;min-width:0}
main.m0 .lane{grid-template-columns:24px 1fr 14px;gap:4px}
main.m0 .sub{display:none}
#pid{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;
     vertical-align:-1px;background:#444}
.off{color:#f85149}
</style></head><body>
<header>
  <h1>泳池实时分析</h1>
  <div class="seg" id="modes">
    <button data-m="0" class="on">全景</button><button data-m="1">全景 + 个人</button>
  </div>
  <span class="grp"><em>全景</em>
    <label><input type="checkbox" data-g="c" data-k="kpts" checked>关键点</label>
    <label><input type="checkbox" data-g="c" data-k="det" checked>检测</label>
    <label><input type="checkbox" data-g="c" data-k="info" checked>分析</label>
    <label><input type="checkbox" data-g="c" data-k="grid" checked>网格</label></span>
  <span class="grp" id="fgrp"><em>跟随</em>
    <label><input type="checkbox" data-g="f" data-k="kpts">关键点</label>
    <label><input type="checkbox" data-g="f" data-k="det">检测</label>
    <label><input type="checkbox" data-g="f" data-k="info" checked>分析</label>
    <label><input type="checkbox" data-g="f" data-k="grid" checked>网格</label>
    <label title="没在跟谁时立刻跟上近期均速最快的人（与概览同一口径）；正在跟的人离场约 1.5 秒后换人">
      <input type="checkbox" data-g="f" data-k="auto" checked>自动接管</label></span>
  <span class="sp"></span>
  <span id="hud">连接中…</span>
  <button class="btn" id="fs">全屏</button>
</header>
<main id="app" class="m0">
  <section class="stage" id="stage">
    <img id="cv" alt=""><canvas id="co"></canvas><span class="tag">CANVAS</span>
  </section>
  <aside class="panel" id="global"></aside>
  <section class="stage" id="fstage">
    <img id="fv" alt=""><canvas id="fo"></canvas><span class="tag">FOLLOW</span>
    <div class="tip" id="ftip">在全景画面中点击一名运动员即可跟随</div>
  </section>
  <aside class="panel" id="person"></aside>
</main>
<script>
)HTML"
R"JS('use strict';
const $ = s => document.querySelector(s);
const app = $('#app'), hud = $('#hud');
const co = $('#co'), fo = $('#fo'), cv = $('#cv'), fv = $('#fv');
let meta = null, S = null, sel = -1, mode = 0;

// 两组叠加层开关：全景一组、跟随一组（同名不同默认值，见 header 里的 checked）。
// 绘制函数只收其中一组，于是「画哪个视图」与「开了什么」彻底解耦。
// det（检测框）与 info（文本标签）刻意分开：跟随视图要读数字但不要框挡住手臂。
// f.auto 不是叠加层，只是搭同一个收集器（省一套 DOM 绑定），绘制侧不看它。
const SW = {c: {}, f: {}};
for (const el of document.querySelectorAll('header input[type=checkbox]')) {
  const g = el.dataset.g, k = el.dataset.k;
  SW[g][k] = el.checked;
  el.onchange = () => { SW[g][k] = el.checked; dirty = true; };
}

// track 配色：与 C++ 的 id_rgb 同一个式子（Knuth 乘法散列）。
// 必须用 Math.imul —— 普通乘法在 id*2654435761 时已超出 2^53，低位会被舍掉。
const idColor = id => {
  const h = Math.imul(id, 2654435761) >>> 0;
  return `rgb(${h & 255},${h >> 8 & 255},${h >> 16 & 255})`;
};
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '—' : v.toFixed(d);

// 显示编号：界面一律「x 道 y 号」，道内号由服务端按占位分配（见 stats.cpp 的
// take_slot）。原始 track id 是选人、跟随、统计的唯一键 —— 它全程不变，而显示号会
// 随换道重排，拿它当键就会串；id 刻意不出现在界面上。
// 没有号只有两种情形：人在池岸（不属于任何泳道），或号已随离场回收 —— 后者只可能
// 出现在「跟随一个早已离场的人」这一档，故 alt 由调用方给。
const ptag = (ln, sl, alt = '池岸') => sl ? `${ln} 道 ${sl} 号` : alt;
/// 按原始 id 取本帧的显示号（概览的「最快」用，那人必然在场）。
const idtag = id => {
  const p = S.p.find(q => q.id === id);
  return p ? ptag(p.ln, p.sl) : '—';
};

// ── 画面区计算：<img object-fit:contain> 的实际显示矩形 ────────────────────
// 叠加层与画面必须共用同一个 letterbox，否则骨架整体偏移。
function fitRect(box, aw, ah) {
  const s = Math.min(box.clientWidth / aw, box.clientHeight / ah);
  const w = aw * s, h = ah * s;
  return {x: (box.clientWidth - w) / 2, y: (box.clientHeight - h) / 2, s};
}
function prep(cvs, box) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(box.clientWidth * dpr), h = Math.round(box.clientHeight * dpr);
  if (cvs.width !== w || cvs.height !== h) { cvs.width = w; cvs.height = h; }
  const g = cvs.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, box.clientWidth, box.clientHeight);
  return g;
}

// ── 米制标尺 ──────────────────────────────────────────────────────────────
// 口径全部来自 /meta，与 C++ 的 draw_grid 一致：横线即分道绳（pitch 2.5 m，8 条），
// 纵向零点在池端。lane>0 时只画那一道并沿道标米数（跟随视图），否则画全池。
// vx/vw 是可视范围（画布像素），画布坐标乘 s 即屏幕坐标。
//
// 跟随视图换配色与字号：水面本身偏青蓝，青色标注混在里面读不出来，所以改用
// 互补的橙黄并放大 —— 那一格是给人逐帧看泳姿的，米数要能一眼读到。
const LINE = 'rgba(0,220,220,.8)', HOT = '#ffb300', DIM = 'rgba(160,175,185,.22)';
function drawGrid(g, s, vx, vw, lane) {
  const G = meta.grid, m = meta.ppm;
  const tick = (lane ? 1 : G.pitch / G.subdiv) * m;   // 刻度间距（画布像素）
  if (tick * s < 3) return;                           // 太密就整体不画
  const k0 = Math.max(0, Math.floor(vx / tick)), k1 = Math.ceil((vx + vw) / tick);
  const r0 = lane ? lane - 1 : 0, r1 = lane ? lane : G.lanes;
  const yt = (G.margin + r0 * G.pitch) * m * s, yb = (G.margin + r1 * G.pitch) * m * s;
  const col = lane ? HOT : LINE, lw = lane ? 1.6 : 1;
  g.font = lane ? '700 19px system-ui' : '11px system-ui';
  g.textBaseline = 'top';
  g.lineJoin = 'round';
  // 文字一律「黑描边 + 亮填充」：压在泳道线或白色水花上都还读得清
  const label = (t, x, y) => {
    g.lineWidth = lane ? 4.5 : 3;
    g.strokeStyle = 'rgba(0,0,0,.8)';
    g.strokeText(t, x, y);
    g.fillStyle = col;
    g.fillText(t, x, y);
  };
  for (let r = r0; r <= r1; r++) {                    // 分道绳
    const y = Math.round((G.margin + r * G.pitch) * m * s) + .5;
    g.lineWidth = lw;
    g.strokeStyle = col;
    g.beginPath(); g.moveTo(vx * s, y); g.lineTo((vx + vw) * s, y); g.stroke();
  }
  for (let k = k0; k <= k1; k++) {                    // 纵向刻度，每 5 格标米数
    const major = k % 5 === 0;
    if (!major && tick * s < 6) continue;
    const x = Math.round(k * tick * s) + .5;
    g.lineWidth = lw;
    g.strokeStyle = major ? col : DIM;
    g.beginPath(); g.moveTo(x, yt); g.lineTo(x, yb); g.stroke();
    if (major && k) label(k * tick / m + ' m', x + 4, yt + 3);
  }
  if (lane) label(lane + ' 道', vx * s + 6, yb - 27);
}
)JS"
R"JS(
// ── 一个人的框 / 标签 / 骨架 ───────────────────────────────────────────────
// 骨架分色：手臂最亮（划水动作的主体），躯干次之，腿与面部压暗，
// 于是一眼看到的是划手而不是一团绿线。分类按关键点语义（COCO17）而非边序号，
// 换骨架表也不会错配。
const limbColor = (a, b) => {
  const hi = Math.max(a, b);
  return (a > 6 && a < 11) || (b > 6 && b < 11) ? '#ffd23f'
       : hi > 12 ? '#4fc3f7'
       : hi < 5  ? 'rgba(255,255,255,.5)'
       : '#7bed9f';
};

/// V = {s, tx, ty, thick, sw, fix?}：s 缩放、tx/ty 画布→屏幕平移、thick 线宽、
/// sw 该视图的开关组。fix 给出时标签钉死在这个画布坐标（跟随视图只画一个人，
/// 钉住远比跟着 bbox 上下抖好读）；不给则贴在框的上沿 —— 全景同一条道可能两人
/// 并列，标签必须各自跟着自己的框走。
function drawPerson(g, p, V) {
  const s = V.s, th = V.thick, sw = V.sw;
  const X = x => V.tx + x * s, Y = y => V.ty + y * s;
  const col = idColor(p.id), on = p.id === sel;
  const bx = X(p.b[0]), by = Y(p.b[1]);
  g.lineJoin = g.lineCap = 'round';
  if (sw.det) {
    g.beginPath();
    g.roundRect(bx, by, (p.b[2] - p.b[0]) * s, (p.b[3] - p.b[1]) * s, 4);
    g.strokeStyle = 'rgba(0,0,0,.5)';  g.lineWidth = th + 2; g.stroke();  // 暗描边
    g.strokeStyle = on ? '#fff' : col; g.lineWidth = th + (on ? 1.2 : 0); g.stroke();
  }
  if (sw.info) {
    // 半透明胶囊 + 色点：比纯色块压画面轻，水面上也读得清
    const txt = `${ptag(p.ln, p.sl)} · ${p.s} 划`
              + (p.v === null ? '' : ` · ${p.v.toFixed(2)} m/s`);
    const fh = V.fix ? 19 : Math.max(11, Math.min(16, 13 * Math.max(s * 3, .8)));
    g.font = `600 ${fh}px system-ui`;
    const ph = fh + 7, pw = g.measureText(txt).width + ph + 8;
    const lx = V.fix ? X(V.fix[0]) + 6 : bx;
    const ly = V.fix ? Y(V.fix[1]) + 6 : Math.max(by - ph - 3, 0);
    g.beginPath();
    g.roundRect(lx, ly, pw, ph, ph / 2);
    g.fillStyle = 'rgba(13,17,23,.8)'; g.fill();
    g.strokeStyle = on ? '#fff' : col; g.lineWidth = 1; g.stroke();
    g.beginPath();
    g.arc(lx + ph / 2, ly + ph / 2, fh * .28, 0, 6.2832);
    g.fillStyle = col; g.fill();
    g.fillStyle = '#fff';
    g.textBaseline = 'middle';
    g.fillText(txt, lx + ph - 1, ly + ph / 2 + .5);
  }
  if (!sw.kpts) return;
  const K = p.k, thr = meta.kpt_thr;
  // 按颜色归拢成几条 Path2D，再「暗底 + 亮线」各描一遍：描两遍的成本远小于
  // 逐边 beginPath，而暗底让亮线在白色水花上也不糊。
  const paths = {};
  for (const [a, b] of meta.skel) {
    if (K[a * 3 + 2] < thr || K[b * 3 + 2] < thr) continue;
    const c = limbColor(a, b);
    const q = paths[c] || (paths[c] = new Path2D());
    q.moveTo(X(K[a * 3]), Y(K[a * 3 + 1]));
    q.lineTo(X(K[b * 3]), Y(K[b * 3 + 1]));
  }
  for (const c in paths) {
    g.strokeStyle = 'rgba(0,0,0,.45)'; g.lineWidth = th + 2.2; g.stroke(paths[c]);
    g.strokeStyle = c;                 g.lineWidth = th;       g.stroke(paths[c]);
  }
  const r = Math.max(1.6, th * 1.3);
  for (let i = 0; i < meta.nk; i++) {
    if (K[i * 3 + 2] < thr) continue;
    const wrist = i === 9 || i === 10;                // 手腕点大一号：划水看的就是它
    g.beginPath();
    g.arc(X(K[i * 3]), Y(K[i * 3 + 1]), r * (wrist ? 1.7 : 1), 0, 6.2832);
    g.fillStyle = wrist ? '#ffd23f' : '#fff';
    g.fill();
  }
}
)JS"
R"JS(
// ── 每帧重绘（rAF 驱动，与 SSE 的到达节奏解耦）────────────────────────────
let dirty = true;
function render() {
  requestAnimationFrame(render);
  if (!meta || !S || !dirty) return;
  dirty = false;

  const box = $('#stage'), r = fitRect(box, meta.w, meta.h), g = prep(co, box);
  g.save();
  g.translate(r.x, r.y);
  if (SW.c.grid) drawGrid(g, r.s, 0, meta.w, 0);
  const V = {s: r.s, tx: 0, ty: 0, thick: 1.4, sw: SW.c};
  for (const p of S.p) {
    g.globalAlpha = p.g ? .55 : 1;             // ghost 占位框画淡一点（web 侧才区分）
    drawPerson(g, p, V);
  }
  g.globalAlpha = 1;
  g.restore();

  if (mode === 1) {
    autoFollow();
    const fbox = $('#fstage'), rc = S.sel && S.sel.rect;
    // 提示层按「有没有跟随矩形」显示，而不是按有没有选人 —— 选中的人离场后
    // 跟随流停在最后一帧，不给提示的话看起来像画面卡住了。自动接管开着时不说
    // 「点一名运动员」（它自己会挑），而是照实说在等什么：清场、还是刚入场的人
    // 还没观察满 3 秒（服务端的排名门限，见 stats.cpp 的 kRecentMinSec）。
    $('#ftip').style.display = rc ? 'none' : '';
    // 离场者的号取自 S.sel（服务端保留他最后的道与号），不查 S.p —— 他已经不在里面。
    const who = S.sel ? ptag(S.sel.lane, S.sel.slot, '跟随目标') + '已离场，' : '';
    $('#ftip').textContent = !SW.f.auto
      ? (who ? who + '等待重新出现' : '在全景画面中点击一名运动员即可跟随')
      : who + (S.all.hotid >= 0 ? '正在接管近期最快的人'
             : S.all.now > 0    ? '正在挑选跟随目标…'
                                : '全场暂无人在场，有人入场即自动跟随');
    const fg = prep(fo, fbox);
    if (rc) {
      const fr = fitRect(fbox, rc[2], rc[3]);
      fg.save();
      fg.translate(fr.x - rc[0] * fr.s, fr.y - rc[1] * fr.s);
      if (SW.f.grid) drawGrid(fg, fr.s, rc[0], rc[2], S.sel.lane);
      const FV = {s: fr.s, tx: 0, ty: 0, thick: 2.2, sw: SW.f, fix: [rc[0], rc[1]]};
      for (const p of S.p)
        if (p.id === sel) drawPerson(fg, p, FV);
      fg.restore();
    }
  }
  paintPanels();
}

// ── 自动接管：让跟随视图在「有人可跟」时永不空着 ──────────────────────────
// 三种入场都走同一段：进入本模式时还没选人、被跟的人离场、清场后又有人下水。
// 差别只在宽限 —— 已经在跟的人丢了要等 AUTO_GRACE_SEC（短暂丢检很常见，服务端
// 会用 ghost 顶几帧，一丢就跳会让画面乱蹦）；而「本来就没在跟谁」没有可丢的东西，
// 立刻接管。全场无人时 hotid 为 -1，什么都不做、也不清计时：等到有人入场并观察
// 满 3 秒，那时 S.t - lostAt 早已超过宽限，于是自动接管，正是清场后想要的行为。
// 计时用 S.t（服务端的流内秒数）而不是墙钟：暂停或掉帧时两者会分叉。
// 判据是概览里那个「近 N 秒最快」（all.hotid，服务端算的均速）—— 前端只有逐帧
// 瞬时速度，自己反推不出来，而瞬时会每帧换人。
// 副作用：开着它就点不成「谁都不跟」（点空水面会立刻被接管回来），要空着就取消勾选。
const AUTO_GRACE_SEC = 1.5;
let lostAt = -1;                               // <0 = 未在计时（S.t 可能正好是 0）
function autoFollow() {
  if (!SW.f.auto || (S.sel && S.sel.rect)) { lostAt = -1; return; }
  if (sel >= 0 && lostAt < 0) { lostAt = S.t; return; }  // 目标刚丢：宽限从此刻起算
  const hot = S.all.hotid;                               // 全场无人在场时为 -1
  if (hot < 0 || hot === sel) return;                    // 无人可跟 / 已经在跟他
  if (sel < 0 || S.t - lostAt >= AUTO_GRACE_SEC) select(hot);
}

// ── 统计面板 ──────────────────────────────────────────────────────────────
// 口径刻意只报「此刻在场者」的均值：逐帧人数之和、累计 track 数、全场里程合计
// 这类量会被 ID 切换灌水，对教练没有意义。
const row = (k, v) => `<span>${k}</span><b>${v}</b>`;
const cell = (k, v, u) => `<div><i>${k}</i><b>${v}${u ? `<u>${u}</u>` : ''}</b></div>`;
function paintPanels() {
  const a = S.all;
  hud.innerHTML = `帧 <b>${S.i}</b> · <b>${num(S.fps, 1)}</b> fps · ` +
                  `${Math.floor(S.t / 60)}:${String(Math.floor(S.t % 60)).padStart(2, '0')}`;
  $('#global').innerHTML =
    `<h2>全场概览</h2><div class="big">
       ${cell('在场人数', a.now)}
       ${cell('平均速度', num(a.vavg), 'm/s')}
       ${cell('平均划频', num(a.spm, 1), 'spm')}
       ${cell('每划距离', num(a.dps), 'm')}
     </div>
     <div class="kv">
       ${row(`近 ${a.hotsec} 秒最快`,
             (a.hotid >= 0 ? idtag(a.hotid) + ' · ' : '') + num(a.hotv) + ' m/s')}
       ${row('画布', meta.w + ' x ' + meta.h + ' px')}
       ${row('标定', meta.ppm + ' px/m · ' + meta.grid.lanes + ' 道')}
     </div>
     <div class="lanes"><h2 class="sub">分道占用</h2>` +
    a.lanes.map((n, i) => `<div class="lane"><span>${i + 1} 道</span>
       <u><i style="width:${Math.min(100, n * 25)}%"></i></u><b>${n}</b></div>`).join('') +
    '</div>';

  if (mode !== 1) return;
  const s = S.sel;
  // 色点仍按原始 id 取（与全景里他的框同色，换道也不变），但标题只写显示号。
  // 标题已含道号，所以不再单列「所在泳道」那一行。
  $('#person').innerHTML = !s
    ? '<h2>个人统计</h2><div class="kv"><span>未选中运动员</span><b></b></div>'
    : `<h2><i id="pid" style="background:${idColor(s.id)}"></i>${ptag(s.lane, s.slot, '跟随目标')}
         ${s.rect ? '' : '<span class="off">（已离场）</span>'}</h2>
       <div class="big">
         ${cell('划水次数', s.strokes)}
         ${cell('当前速度', num(s.v), 'm/s')}
         ${cell('平均速度', num(s.vavg), 'm/s')}
         ${cell('划频', num(s.spm, 1), 'spm')}
       </div>
       <div class="kv">
         ${row('在场时长', num(s.sec, 1) + ' s')}
         ${row('累计里程', num(s.dist, 1) + ' m')}
         ${row('峰值速度', num(s.vmax) + ' m/s')}
         ${row('每划距离', num(s.dps) + ' m')}
         ${row('划水指数', num(s.si))}
       </div>`;
}
)JS"
R"JS(
// ── 交互 ──────────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  app.className = 'm' + m;
  $('#fgrp').style.display = m === 1 ? '' : 'none';
  for (const b of $('#modes').children) b.classList.toggle('on', +b.dataset.m === m);
  // 跟随流只在模式 1 下拉：没人看时 C++ 侧连裁切与编码都跳过。
  // 停流用 removeAttribute 而不是 src=''：后者会被解析成页面 URL，既触发一次
  // onerror（重连逻辑会把流拉回来），又让 <img> 去下载整张 HTML。
  if (m === 1) stream(fv, 'follow.mjpg');
  else { fv.onerror = null; fv.removeAttribute('src'); }
  dirty = true;
}
$('#modes').onclick = e => { if (e.target.dataset.m) setMode(+e.target.dataset.m); };
$('#fs').onclick = () => document.fullscreenElement ? document.exitFullscreen()
                                                    : document.documentElement.requestFullscreen();
addEventListener('resize', () => { dirty = true; });

/// 点击选人：命中框内则选中（重叠时取面积最小的那个，通常是想点的人）。
co.onclick = e => {
  if (!meta || !S) return;
  const box = $('#stage'), r = fitRect(box, meta.w, meta.h), b = box.getBoundingClientRect();
  const x = (e.clientX - b.left - r.x) / r.s, y = (e.clientY - b.top - r.y) / r.s;
  let hit = -1, best = Infinity;
  for (const p of S.p) {
    if (x < p.b[0] || x > p.b[2] || y < p.b[1] || y > p.b[3]) continue;
    const area = (p.b[2] - p.b[0]) * (p.b[3] - p.b[1]);
    if (area < best) { best = area; hit = p.id; }
  }
  select(hit);
};
function select(id) {
  sel = id;
  lostAt = -1;                                 // 手工点人即重置自动接管的宽限计时
  fetch('select?id=' + id);
  if (id >= 0 && mode !== 1) setMode(1);
  dirty = true;
}

// ── 连接 ──────────────────────────────────────────────────────────────────
// MJPEG 断了（服务退出、网络抖动）就退避重连；SSE 由浏览器自己重连。
function stream(img, url) {
  img.onerror = () => setTimeout(() => { if (img.src) img.src = url + '?' + Date.now(); }, 1500);
  img.src = url + '?' + Date.now();
}
setMode(0);
fetch('meta').then(r => r.json()).then(m => {
  meta = m;
  document.title = `泳池实时分析 ${m.w}x${m.h}`;
  stream(cv, 'canvas.mjpg');
  const es = new EventSource('stats');
  es.onmessage = e => { S = JSON.parse(e.data); dirty = true; };
  es.onerror = () => { hud.textContent = '连接中断，重试中…'; };
  render();
}).catch(() => { hud.textContent = '无法连接服务'; });
</script></body></html>
)JS";

}  // namespace swim
