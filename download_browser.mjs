#!/usr/bin/env node
/**
 * 雀魂牌谱「本地浏览器」下载器（不依赖任何第三方中转服务）
 *
 * 原理（产品化自 mortal-paipu-analyzer 的 browser_probe.py）：
 *   1. 启动你电脑上的 Edge/Chrome，使用工作区内的持久化浏览器配置目录，
 *      首次用雀魂手机 App 扫码登录一次（也可账密登录），登录态长期保存；
 *   2. 逐个打开 https://game.maj-soul.com/1/?paipu=<UUID>；
 *      客户端登录后会自动发 .lq.Lobby.fetchGameRecord（见客户端 checkPaiPu）；
 *   3. 通过 Chrome DevTools Protocol 拦截该 WebSocket 响应帧，
 *      逐字节保存 ResGameRecord 的 head / data（data_url 则 HTTP 直取 + gunzip）；
 *   4. 调用 convert_paipu.py（vendored tensoul 转换子集）生成 tenhou.net/6 JSON，
 *      之后可直接 python tenhou2mjai.py paipu_json paipu_mjai。
 *
 * 用法：
 *   node download_browser.mjs login                       # 首次：打开浏览器扫码登录
 *   node download_browser.mjs <UUID...>                   # 下载（未登录会提示扫码）
 *   node download_browser.mjs --uuids-file uuids_20260101_20260831.txt
 *   node download_browser.mjs <UUID> --keep-open          # 完成后保留浏览器窗口（排错）
 *   node download_browser.mjs <UUID> --no-convert         # 只抓原始字节，不转 JSON
 *
 * 可选：
 *   --browser <exe路径>   指定 Chrome/Edge（默认自动查找，也可用环境变量 CHROME_PATH）
 *   --port <端口>         CDP 调试端口（默认自动选择）
 *   --raw-dir <目录>      原始字节目录（默认 paipu_raw）
 *   --out-dir <目录>      tenhou JSON 输出目录（默认 paipu_json）
 */

import { spawn, spawnSync } from 'node:child_process';
import { gunzipSync } from 'node:zlib';
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import net from 'node:net';
import protobuf from 'protobufjs';
import WebSocket from 'ws';

const ROOT_DIR = dirname(fileURLToPath(import.meta.url));
const PROFILE_DIR = join(ROOT_DIR, '.browser-profile');
const CACHE_DIR = join(ROOT_DIR, '.cache');
const HOME_URL = 'https://game.maj-soul.com/1/';
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
  + '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);
const DBG = process.env.DEBUG_FRAMES
  ? (...a) => appendFileSync(join(CACHE_DIR, 'sniffer_debug.log'), `${new Date().toISOString()} ${a.join(' ')}\n`)
  : () => {};

// ---------------------------------------------------------------- 协议定义
async function httpText(url) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(url, { signal: ctrl.signal, headers: { 'User-Agent': UA } });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
    return await res.text();
  } finally {
    clearTimeout(timer);
  }
}

async function loadProtoTypes() {
  mkdirSync(CACHE_DIR, { recursive: true });
  const { version: versionRaw } = JSON.parse(await httpText('https://game.maj-soul.com/1/version.json'));
  const resVersion = JSON.parse(await httpText(`https://game.maj-soul.com/1/resversion${versionRaw}.json`));
  const prefix = resVersion.res['res/proto/liqi.json'].prefix;
  const cacheFile = join(CACHE_DIR, `liqi.${prefix.replace(/[\\/]/g, '_')}.json`);
  if (!existsSync(cacheFile)) {
    log('下载协议定义 liqi.json (' + prefix + ')');
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 60000);
    const res = await fetch(`https://game.maj-soul.com/1/${prefix}/res/proto/liqi.json`, {
      signal: ctrl.signal, headers: { 'User-Agent': UA },
    });
    clearTimeout(timer);
    const buf = Buffer.from(await res.arrayBuffer());
    writeFileSync(cacheFile, buf);
  }
  const root = await protobuf.load(cacheFile);
  return {
    root,
    Wrapper: root.lookupType('lq.Wrapper'),
    ResGameRecord: root.lookupType('lq.ResGameRecord'),
  };
}

// ---------------------------------------------------------------- UUID 解析
const RE_DATE_UUID = /^\d{6}-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const RE_STD_UUID = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const RE_LOOSE = /^\d{6}-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4,12}){2,}$/;
const isUuid = (s) => RE_DATE_UUID.test(s) || RE_STD_UUID.test(s) || RE_LOOSE.test(s);
function extractUuid(token) {
  let t = String(token ?? '').trim();
  const m = t.match(/[?&]paipu=([^&#\s]+)/);
  if (m) t = decodeURIComponent(m[1]);
  return t.split('_')[0];
}
function readUuidsFromFile(file) {
  const raw = readFileSync(file, 'utf-8');
  const out = [], seen = new Set();
  const add = (v) => {
    const u = extractUuid(v);
    if (u && isUuid(u) && !seen.has(u.toLowerCase())) { seen.add(u.toLowerCase()); out.push(u); }
  };
  try {
    const walk = (x) => {
      if (typeof x === 'string') add(x);
      else if (Array.isArray(x)) x.forEach(walk);
      else if (x && typeof x === 'object') Object.values(x).forEach(walk);
    };
    walk(JSON.parse(raw));
  } catch { /* 纯文本 */ }
  raw.split(/\r?\n/).map((l) => l.replace(/#.*$/, '')).join('\n')
    .split(/[\s,;，；"'`[\](){}<>|]+/).forEach(add);
  return out;
}

// ---------------------------------------------------------------- 浏览器 / CDP
function findBrowser() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  for (const p of candidates) if (existsSync(p)) return p;
  throw new Error('找不到 Chrome/Edge，请用 --browser 指定浏览器 exe 路径，或设置环境变量 CHROME_PATH。');
}

async function pickPort(preferred) {
  if (preferred) return Number(preferred);
  return await new Promise((resolvePort) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const p = srv.address().port;
      srv.close(() => resolvePort(p));
    });
  });
}

async function waitCdpVersion(port, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return await res.json();
    } catch (e) { last = e; }
    await sleep(300);
  }
  throw new Error(`浏览器 DevTools 端口 ${port} 未就绪（${last?.message ?? 'timeout'}）`);
}

/** 最小 CDP 客户端：send + 事件监听 */
class Cdp {
  constructor(wsUrl, label) {
    this.ws = new WebSocket(wsUrl, { handshakeTimeout: 10000 });
    this.nextId = 1;
    this.pending = new Map();
    this.handlers = new Map();
    this.label = label;
  }
  async open() {
    this.ws.on('message', (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch { return; }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(`${this.label} CDP ${msg.error.message}`)) : resolve(msg.result);
      } else if (msg.method) {
        if (msg.method.startsWith('Network.webSocket')) DBG(this.label, msg.method, 'rid=' + msg.params?.requestId);
        for (const h of this.handlers.get(msg.method) ?? []) {
          try { h(msg.params); } catch (e) { DBG('handler error', msg.method, e?.stack || String(e)); /* 监听处理不能炸主流程 */ }
        }
      }
    });
    await new Promise((res, rej) => {
      this.ws.once('open', res);
      this.ws.once('error', rej);
    });
  }
  send(method, params = {}, timeoutMs = 30000) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`CDP 超时: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve: (r) => { clearTimeout(t); resolve(r); }, reject: (e) => { clearTimeout(t); reject(e); } });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  on(method, handler) {
    if (!this.handlers.has(method)) this.handlers.set(method, new Set());
    this.handlers.get(method).add(handler);
  }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

class BrowserSession {
  constructor(exePath, port) {
    this.exePath = exePath;
    this.port = port;
    this.proc = null;
    this.browserCdp = null;
    this.tabCdp = null;
  }

  async start() {
    mkdirSync(PROFILE_DIR, { recursive: true });
    this.proc = spawn(this.exePath, [
      `--remote-debugging-port=${this.port}`,
      `--user-data-dir=${PROFILE_DIR}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-popup-blocking',
      'about:blank',
    ], { stdio: 'ignore', detached: false });
    this.proc.on('exit', (code) => {
      if (code !== 0 && !this.closed) log('浏览器进程提前退出 code =', code);
    });

    const version = await waitCdpVersion(this.port);
    this.browserCdp = new Cdp(version.webSocketDebuggerUrl, 'browser');
    await this.browserCdp.open();
    const { targetId } = await this.browserCdp.send('Target.createTarget', { url: 'about:blank' });
    await sleep(500);
    const list = await (await fetch(`http://127.0.0.1:${this.port}/json/list`)).json();
    const tab = list.find((t) => t.id === targetId) || list.find((t) => t.type === 'page');
    if (!tab) throw new Error('未找到可调试的浏览器标签页');
    this.tabCdp = new Cdp(tab.webSocketDebuggerUrl, 'tab');
    await this.tabCdp.open();
    await this.tabCdp.send('Network.enable');
    await this.tabCdp.send('Page.enable');
    await this.tabCdp.send('Runtime.enable');
  }

  async navigate(url) {
    await this.tabCdp.send('Page.navigate', { url }, 30000);
  }

  close() {
    this.closed = true;
    this.tabCdp?.close();
    this.browserCdp?.close();
    if (this.proc && !this.proc.killed) {
      try { this.proc.kill(); } catch { /* ignore */ }
    }
  }
}

// ---------------------------------------------------------------- lq 帧解析
const LOGIN_RPC = new Set(['.lq.Lobby.oauth2Login', '.lq.Lobby.login', '.lq.Lobby.emailLogin']);

/** 在线路层挂 WS 帧监听；返回事件注册函数
 * 注意：雀魂协议【响应帧】的 Wrapper.name 为空，只能用 (WS请求id, 帧idx)
 * 与发送帧关联后，按发送帧的方法名解码响应。 */
function attachFrameSniffer(tabCdp, types) {
  const sentByKey = new Map();    // `${requestId}:${idx}` -> { name, uuid? }
  const onRecord = new Map();     // uuid -> resolve(bytes)
  const loginWaiters = new Set();
  const otherRecordResolves = []; // 关联到 fetchGameRecord 但 uuid 解码失败的响应兜底

  const decodeFrame = (payloadData) => {
    let buf;
    try { buf = Buffer.from(payloadData, 'base64'); } catch { return null; }
    if (buf.length < 3 || (buf[0] !== 2 && buf[0] !== 3)) return null;
    try {
      return { tag: buf[0], idx: buf.readUInt16LE(1), wrap: types.Wrapper.decode(buf.subarray(3)) };
    } catch { return null; }
  };

  tabCdp.on('Network.webSocketFrameSent', ({ requestId, response }) => {
    const f = decodeFrame(response?.payloadData);
    if (!f || f.tag !== 2 || !f.wrap.name) return;
    DBG('SENT', requestId, f.idx, f.wrap.name);
    const key = `${requestId}:${f.idx}`;
    const entry = { name: f.wrap.name, uuid: null };
    if (f.wrap.name === '.lq.Lobby.fetchGameRecord') {
      try {
        entry.uuid = types.root.lookupType('lq.ReqGameRecord').decode(f.wrap.data).game_uuid;
      } catch { /* ignore */ }
    }
    sentByKey.set(key, entry);
  });

  tabCdp.on('Network.webSocketFrameReceived', ({ requestId, response }) => {
    const f = decodeFrame(response?.payloadData);
    if (!f || f.tag !== 3) return;
    const key = `${requestId}:${f.idx}`;
    const sent = sentByKey.get(key);
    if (!sent) { DBG('RECV orphan', requestId, f.idx, 'name=' + f.wrap.name); return; } // 没有对应发送帧（通常是启用监听前的遗留帧），忽略
    sentByKey.delete(key); // 一次性关联，避免心跳等帧无限堆积
    DBG('RECV match', sent.name);

    if (sent.name === '.lq.Lobby.fetchGameRecord') {
      const bytes = Buffer.from(f.wrap.data);
      if (sent.uuid && onRecord.has(sent.uuid)) {
        const resolve = onRecord.get(sent.uuid);
        onRecord.delete(sent.uuid);
        resolve(bytes);
      } else {
        otherRecordResolves.push(bytes);
      }
      return;
    }

    if (LOGIN_RPC.has(sent.name)) {
      try {
        const [, , service, method] = sent.name.split('.');
        const methodDef = types.root.lookup(`lq.${service}`).methods[method];
        const Res = types.root.lookupType(`lq.${methodDef.responseType}`);
        const res = Res.decode(f.wrap.data);
        if (!res.error?.code) for (const w of loginWaiters) w(res);
      } catch { /* ignore */ }
    }
  });

  return {
    waitForRecord(uuid, timeoutMs) {
      return new Promise((resolve, reject) => {
        let done = false;
        let sweep = null;
        const finishOk = (bytes) => {
          if (done) return;
          done = true;
          clearTimeout(t);
          clearInterval(sweep);
          onRecord.delete(uuid);
          resolve(bytes);
        };
        const finishErr = (e) => {
          if (done) return;
          done = true;
          clearTimeout(t);
          clearInterval(sweep);
          onRecord.delete(uuid);
          reject(e);
        };
        const t = setTimeout(() => {
          finishErr(new Error('等待 fetchGameRecord 响应超时（牌谱可能不可见，或登录已失效）'));
        }, timeoutMs);
        onRecord.set(uuid, finishOk);
        // 兜底：若发送帧错过了关联，按响应里的 head.uuid 匹配
        sweep = setInterval(() => {
          while (otherRecordResolves.length) {
            const bytes = otherRecordResolves.shift();
            try {
              const res = types.ResGameRecord.decode(bytes);
              if (res.head?.uuid === uuid) { finishOk(bytes); return; }
            } catch { /* ignore */ }
          }
        }, 300);
      });
    },
    waitLogin(timeoutMs) {
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => {
          loginWaiters.delete(waiter);
          reject(new Error('等待登录超时'));
        }, timeoutMs);
        const waiter = (res) => { clearTimeout(t); loginWaiters.delete(waiter); resolve(res); };
        loginWaiters.add(waiter);
      });
    },
  };
}

// ResGameRecord 字段号：error=1(消息) head=3(消息) data=4(bytes) data_url=5(string)
function readVarint(buf, p) {
  let v = 0n, shift = 0n;
  while (true) {
    const b = buf[p++];
    v |= BigInt(b & 0x7f) << shift;
    if (!(b & 0x80)) return [Number(v), p];
    shift += 7n;
  }
}
function extractResFields(buf) {
  const out = {};
  let p = 0;
  while (p < buf.length) {
    let tag; [tag, p] = readVarint(buf, p);
    const field = tag >> 3, wt = tag & 7;
    if (wt === 2) {
      let len; [len, p] = readVarint(buf, p);
      out[field] = buf.subarray(p, p + len);
      p += len;
    } else if (wt === 0) { [, p] = readVarint(buf, p); }
    else if (wt === 5) { p += 4; }
    else if (wt === 1) { p += 8; }
    else break;
  }
  return out;
}

async function fetchDataUrl(url) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60000);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      headers: { 'User-Agent': UA, Referer: HOME_URL },
    });
    if (!res.ok) throw new Error(`data_url HTTP ${res.status}`);
    let buf = Buffer.from(await res.arrayBuffer());
    if (buf.length >= 2 && buf[0] === 0x1f && buf[1] === 0x8b) buf = gunzipSync(buf);
    return buf;
  } finally {
    clearTimeout(timer);
  }
}

function atomicWrite(file, buf) {
  const tmp = `${file}.tmp`;
  writeFileSync(tmp, buf);
  renameSync(tmp, file);
}

// ---------------------------------------------------------------- 登录
async function ensureLogin(session, sniffer) {
  log('打开雀魂主页…');
  await session.navigate(HOME_URL);
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const promptText = '如果浏览器里还没登录，请在弹出的浏览器窗口中用【雀魂手机 App 扫码】（或账密）登录；\n'
    + '登录成功后脚本会自动继续。也可以登录完成后回到这里按回车手动继续……\n';
  const autoLogin = sniffer.waitLogin(300_000).then(() => 'auto');
  const manual = new Promise((resManual) => {
    rl.question(promptText, () => resManual('manual'));
  });
  try {
    const via = await Promise.race([autoLogin, manual]);
    await sleep(1500);
    log(via === 'auto' ? '检测到登录成功' : '已继续（手动确认）');
  } finally {
    rl.close();
  }
}

// ---------------------------------------------------------------- 参数 / 主流程
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    uuids: [], file: null, rawDir: join(ROOT_DIR, 'paipu_raw'), outDir: join(ROOT_DIR, 'paipu_json'),
    browser: process.env.CHROME_PATH || null, port: null, timeoutSec: 120,
    convert: true, keepOpen: false, loginOnly: false,
  };
  const takes = {
    '--uuids-file': (v) => { opts.file = v; },
    '--raw-dir': (v) => { opts.rawDir = resolve(v); },
    '--out-dir': (v) => { opts.outDir = resolve(v); },
    '--browser': (v) => { opts.browser = v; },
    '--port': (v) => { opts.port = Number(v); },
    '--timeout-sec': (v) => { opts.timeoutSec = Number(v); },
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === 'login') { opts.loginOnly = true; continue; }
    if (a === '--no-convert') { opts.convert = false; continue; }
    if (a === '--keep-open') { opts.keepOpen = true; continue; }
    if (takes[a]) { const v = args[++i]; if (v === undefined) throw new Error(`${a} 缺少取值`); takes[a](v); }
    else opts.uuids.push(a);
  }
  return opts;
}

async function main() {
  const flags = parseArgs();
  const types = await loadProtoTypes();
  log('协议定义就绪');

  const browser = flags.browser || findBrowser();
  const port = await pickPort(flags.port);
  log('浏览器:', browser);
  log('CDP 端口:', port, ' 配置目录:', PROFILE_DIR);
  const session = new BrowserSession(browser, port);
  await session.start();
  const sniffer = attachFrameSniffer(session.tabCdp, types);

  try {
    await ensureLogin(session, sniffer);
    if (flags.loginOnly) {
      log('登录态已保存在本地配置目录，以后下载无需再次扫码（除非登录过期/被挤下线）。');
      return;
    }

    const inputs = [...flags.uuids];
    if (flags.file) inputs.push(...readUuidsFromFile(flags.file));
    const uuids = [], seen = new Set();
    for (const raw of inputs) {
      const u = extractUuid(raw);
      if (!u || !isUuid(u)) throw new Error(`无法识别的 UUID: ${raw}`);
      if (!seen.has(u.toLowerCase())) { seen.add(u.toLowerCase()); uuids.push(u); }
    }
    if (!uuids.length) throw new Error('没有指定 UUID：node download_browser.mjs <UUID...> 或 --uuids-file 文件');

    mkdirSync(flags.rawDir, { recursive: true });
    mkdirSync(flags.outDir, { recursive: true });

    let done = 0, skipped = 0, usedRaw = 0;
    const failed = [];
    for (let i = 0; i < uuids.length; i++) {
      const uuid = uuids[i];
      const headFile = join(flags.rawDir, `${uuid}.head.bin`);
      const dataFile = join(flags.rawDir, `${uuid}.data.bin`);
      const jsonFile = join(flags.outDir, `paipu-${uuid}.json`);
      if (existsSync(jsonFile)) { skipped++; log(`(${i + 1}/${uuids.length}) 已存在，跳过: ${jsonFile}`); continue; }
      if (existsSync(headFile) && existsSync(dataFile)) { usedRaw++; log(`(${i + 1}/${uuids.length}) 已有原始字节，直接转换: ${uuid}`); continue; }

      log(`(${i + 1}/${uuids.length}) 打开牌谱页 ${uuid}`);
      await session.navigate(`${HOME_URL}?paipu=${encodeURIComponent(uuid)}`);
      try {
        const resBytes = await sniffer.waitForRecord(uuid, flags.timeoutSec * 1000);
        const res = types.ResGameRecord.decode(resBytes);
        if (res.error?.code) throw new Error(`fetchGameRecord code=${res.error.code} ${res.error.message ?? ''}`);
        const fields = extractResFields(resBytes);
        const head = fields[3];
        if (!head) throw new Error('响应缺少 head 字段');
        let data;
        if (fields[4]?.length) {
          data = fields[4];
        } else if (fields[5]?.length) {
          const url = fields[5].toString('utf-8');
          log('  牌谱走 data_url，HTTP 直取…');
          data = await fetchDataUrl(url);
        } else {
          throw new Error('响应中没有牌谱数据（data / data_url 均为空）');
        }
        atomicWrite(headFile, head);
        atomicWrite(dataFile, data);
        done++;
        log(`  已抓取原始字节（head ${head.length}B / data ${data.length}B）`);
      } catch (err) {
        failed.push({ uuid, error: err.message });
        log(`  × 失败: ${err.message}`);
      }
      await sleep(1500);
    }

    log('───────────────');
    log(`抓取 ${done}，原始字节复用 ${usedRaw}，已存在跳过 ${skipped}，失败 ${failed.length}`);

    if (flags.convert) {
      log('转换 tenhou.net/6 JSON…');
      const py = spawnSync('python', [join(ROOT_DIR, 'convert_paipu.py'), flags.rawDir, flags.outDir], {
        cwd: ROOT_DIR, stdio: 'inherit',
      });
      if (py.status !== 0) process.exitCode = 1;
      else log('完成。转 MJAI：python tenhou2mjai.py', flags.outDir, 'paipu_mjai');
    }
    if (failed.length) {
      for (const f of failed) log('  失败:', f.uuid, f.error);
      process.exitCode = 1;
    }
  } finally {
    if (flags.keepOpen) {
      log('--keep-open：浏览器窗口保留，按 Ctrl+C 结束脚本（不影响登录态）。');
      await new Promise(() => {});
    } else {
      session.close();
    }
  }
}

main().catch((err) => {
  console.error('致命错误:', err.message);
  process.exit(1);
});
