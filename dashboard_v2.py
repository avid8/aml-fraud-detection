import asyncio
import json
import random
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from rules import RuleEngine, AccountType
from risk_engine import RiskEngine, RiskLevel

logger = logging.getLogger(__name__)
app = FastAPI(title="AML Dashboard v2")

store = {
    "total_tx": 0, "blocked": 0,
    "risk_dist": {"low": 0, "medium": 0, "high": 0, "critical": 0},
    "recent_tx": [], "alerts": [],
}
connected_clients: list[WebSocket] = []


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    await ws.send_json({"type": "init", "data": store})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def broadcast(data: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


@app.get("/api/tx/{tx_id}")
def get_tx_detail(tx_id: str):
    for tx in reversed(store["recent_tx"]):
        if tx["id"] == tx_id:
            return tx
    return {"error": "not found"}


async def simulate():
    rule_engine = RuleEngine()
    risk_engine = RiskEngine()
    counter = 0
    ACCOUNTS = [f"ACC-{i:03d}" for i in range(1, 31)]
    CARDS    = [f"603799{random.randint(1000000000,9999999999)}" for _ in range(20)]

    while True:
        await asyncio.sleep(1.8)
        counter += 1
        is_fraud  = random.random() < 0.18
        acc_type  = random.choice(list(AccountType))
        amount    = random.randint(400_000_000, 900_000_000) if is_fraud else random.randint(500_000, 150_000_000)
        is_night  = random.random() < 0.35
        account   = random.choice(ACCOUNTS)
        card      = random.choice(CARDS)

        tx = {
            "transaction_id":    f"TX-{counter:05d}",
            "account_number":    account,
            "amount_rial":       amount,
            "is_night":          is_night,
            "card_pan":          card,
            "ip_address":        f"185.{random.randint(1,255)}.{random.randint(1,255)}.1",
            "ip_confidence":     0.3 if random.random() < 0.2 else 1.0,
            "device_fp_hash":    f"fp-{random.randint(1,15):02d}",
            "national_code":     "1000000001",
            "mobile_normalized": "09121111111",
        }
        features = {
            "acc_total_amount_24h":    random.randint(500_000_000, 3_000_000_000) if is_fraud else random.randint(0, 300_000_000),
            "acc_small_tx_count_24h":  random.randint(3, 9) if is_fraud else random.randint(0, 2),
            "ip_unique_cards_1h":      random.randint(3, 10) if is_fraud else random.randint(1, 2),
            "dev_unique_cards_1h":     random.randint(3, 7) if is_fraud else 1,
            "acc_fail_ratio_1h":       random.uniform(0.4, 0.95) if is_fraud else random.uniform(0, 0.15),
            "card_unique_accounts_24h":random.randint(3, 7) if is_fraud else 1,
        }

        rule_result = rule_engine.evaluate(tx, features, acc_type)
        decision    = risk_engine.decide(rule_result)

        store["total_tx"] += 1
        store["risk_dist"][decision.risk_level] += 1
        if decision.should_block:
            store["blocked"] += 1

        record = {
            "id":           tx["transaction_id"],
            "account":      account,
            "card":         card[:6] + "xxxxxx" + card[-4:],
            "amount":       amount,
            "amount_fmt":   f"{amount:,}",
            "risk_score":   decision.risk_score,
            "risk_level":   decision.risk_level,
            "blocked":      decision.should_block,
            "is_night":     is_night,
            "ip":           tx["ip_address"],
            "ip_confidence":tx["ip_confidence"],
            "acc_type":     acc_type.value,
            "alerts":       decision.top_alerts,
            "alert_count":  len(decision.top_alerts),
            "rule_score":   decision.rule_score,
            "ml_score":     decision.ml_score,
            "time":         datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "ts":           datetime.now(timezone.utc).isoformat(),
        }
        store["recent_tx"].append(record)
        if len(store["recent_tx"]) > 200:
            store["recent_tx"].pop(0)

        for a in decision.top_alerts:
            store["alerts"].append({**a, "tx_id": record["id"], "time": record["time"]})
        if len(store["alerts"]) > 300:
            store["alerts"] = store["alerts"][-300:]

        await broadcast({"type": "tx",    "data": record})
        await broadcast({"type": "stats", "data": {
            "total_tx":  store["total_tx"],
            "blocked":   store["blocked"],
            "risk_dist": store["risk_dist"],
            "block_rate": round(store["blocked"] / max(store["total_tx"], 1) * 100, 1),
        }})


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulate())


HTML = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AML Shield — سیستم تشخیص تقلب</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root {
  --bg:       #080B14;
  --surface:  #0D1120;
  --surface2: #131829;
  --border:   #1E2A45;
  --purple:   #7C3AED;
  --purple2:  #9F67FF;
  --green:    #10B981;
  --yellow:   #F59E0B;
  --orange:   #F97316;
  --red:      #EF4444;
  --text:     #E2E8F0;
  --muted:    #64748B;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Inter', sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}

/* ── header ── */
header {
  background:var(--surface);
  border-bottom:1px solid var(--border);
  padding:0 28px;
  height:56px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(12px);
}
.logo {
  display:flex;align-items:center;gap:10px;
  font-family:var(--mono);font-size:14px;font-weight:600;
  color:var(--purple2);letter-spacing:0.05em;
}
.logo-icon {
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--purple),#4F46E5);
  display:flex;align-items:center;justify-content:center;
  font-size:16px;
}
.live-badge {
  display:flex;align-items:center;gap:6px;
  font-size:11px;color:var(--green);
  font-family:var(--mono);
  background:rgba(16,185,129,0.08);
  border:1px solid rgba(16,185,129,0.2);
  padding:4px 10px;border-radius:20px;
}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(0.8)}}

/* ── metrics ── */
.metrics {
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--border);
  border-bottom:1px solid var(--border);
}
.metric {
  background:var(--surface);
  padding:20px 24px;
  position:relative;overflow:hidden;
}
.metric::before {
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
}
.metric.m-total::before{background:var(--purple)}
.metric.m-blocked::before{background:var(--red)}
.metric.m-rate::before{background:var(--orange)}
.metric.m-high::before{background:var(--yellow)}
.metric-label{font-size:11px;color:var(--muted);margin-bottom:6px;letter-spacing:0.06em;text-transform:uppercase}
.metric-value{font-family:var(--mono);font-size:36px;font-weight:600;line-height:1}
.metric-sub{font-size:11px;color:var(--muted);margin-top:4px}
.m-total .metric-value{color:var(--purple2)}
.m-blocked .metric-value{color:var(--red)}
.m-rate .metric-value{color:var(--orange)}
.m-high .metric-value{color:var(--yellow)}

/* ── layout ── */
.body-grid {
  display:grid;
  grid-template-columns:320px 1fr 300px;
  gap:0;
  height:calc(100vh - 56px - 77px);
}

/* ── radar panel ── */
.radar-panel {
  background:var(--surface);
  border-left:1px solid var(--border);
  padding:20px;
  display:flex;flex-direction:column;gap:16px;
}
.panel-title {
  font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.08em;
  display:flex;align-items:center;gap:6px;
}
.panel-title::before{content:'';width:3px;height:12px;background:var(--purple);border-radius:2px}
canvas#radarCanvas {
  width:100%!important;
  aspect-ratio:1;
}

/* ── main table ── */
.table-panel {
  background:var(--bg);
  border-right:1px solid var(--border);
  border-left:1px solid var(--border);
  overflow:hidden;
  display:flex;flex-direction:column;
}
.table-header {
  padding:16px 20px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);
}
.table-scroll{overflow-y:auto;flex:1}
table{width:100%;border-collapse:collapse}
thead th {
  font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.07em;
  padding:10px 16px;
  border-bottom:1px solid var(--border);
  background:var(--surface);
  text-align:right;
  position:sticky;top:0;z-index:2;
  font-weight:500;
}
tbody tr {
  border-bottom:1px solid rgba(30,42,69,0.6);
  cursor:pointer;
  transition:background 0.15s;
}
tbody tr:hover{background:var(--surface2)}
tbody tr.flagged{background:rgba(239,68,68,0.04)}
tbody tr.flagged:hover{background:rgba(239,68,68,0.08)}
td {
  padding:11px 16px;
  font-size:12px;
  font-family:var(--mono);
}
td.text{font-family:var(--sans)}

/* ── badges ── */
.badge {
  font-size:10px;padding:2px 8px;border-radius:3px;
  font-weight:500;font-family:var(--mono);
  display:inline-block;letter-spacing:0.03em;
}
.b-low     {background:rgba(16,185,129,0.12);color:var(--green);border:1px solid rgba(16,185,129,0.2)}
.b-medium  {background:rgba(245,158,11,0.12);color:var(--yellow);border:1px solid rgba(245,158,11,0.2)}
.b-high    {background:rgba(249,115,22,0.12);color:var(--orange);border:1px solid rgba(249,115,22,0.2)}
.b-critical{background:rgba(239,68,68,0.15);color:var(--red);border:1px solid rgba(239,68,68,0.25)}
.b-blocked {background:rgba(239,68,68,0.12);color:var(--red);border:1px solid rgba(239,68,68,0.2)}
.b-ok      {background:rgba(16,185,129,0.08);color:var(--green);border:1px solid rgba(16,185,129,0.15)}

/* ── score bar ── */
.score-bar{width:60px;height:4px;background:var(--border);border-radius:2px;display:inline-block;vertical-align:middle;margin-right:6px}
.score-fill{height:100%;border-radius:2px;transition:width 0.3s}

/* ── alerts panel ── */
.alerts-panel {
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.alerts-scroll{overflow-y:auto;flex:1}
.alert-item {
  padding:12px 16px;
  border-bottom:1px solid rgba(30,42,69,0.5);
  cursor:pointer;transition:background 0.15s;
}
.alert-item:hover{background:var(--surface2)}
.alert-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.alert-code{font-family:var(--mono);font-size:11px;font-weight:600}
.alert-time{font-family:var(--mono);font-size:10px;color:var(--muted)}
.alert-msg{font-size:11px;color:#94A3B8;line-height:1.4}
.sev-high   .alert-code{color:var(--red)}
.sev-medium .alert-code{color:var(--yellow)}
.sev-low    .alert-code{color:var(--green)}

/* ── modal ── */
.modal-overlay {
  position:fixed;inset:0;background:rgba(0,0,0,0.75);
  backdrop-filter:blur(4px);
  z-index:200;display:none;
  align-items:center;justify-content:center;
}
.modal-overlay.open{display:flex}
.modal {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:12px;
  width:580px;max-width:95vw;
  max-height:85vh;overflow-y:auto;
  box-shadow:0 24px 64px rgba(0,0,0,0.6);
}
.modal-header {
  padding:20px 24px 16px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.modal-title{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--purple2)}
.modal-close {
  width:28px;height:28px;border-radius:6px;
  background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:14px;
  transition:all 0.15s;
}
.modal-close:hover{background:var(--border);color:var(--text)}
.modal-body{padding:20px 24px}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.info-item{background:var(--surface2);border-radius:8px;padding:12px 14px;border:1px solid var(--border)}
.info-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px}
.info-value{font-family:var(--mono);font-size:13px;font-weight:500}
.section-title{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.section-title::before{content:'';width:3px;height:10px;background:var(--purple);border-radius:2px}
.alert-detail{background:var(--surface2);border-radius:8px;padding:12px 14px;border-right:3px solid;margin-bottom:8px}
.alert-detail.sev-high{border-color:var(--red)}
.alert-detail.sev-medium{border-color:var(--yellow)}
.alert-detail.sev-low{border-color:var(--green)}
.ad-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.ad-code{font-family:var(--mono);font-size:11px;font-weight:600}
.ad-msg{font-size:12px;color:#94A3B8}
.score-section{margin-bottom:20px}
.score-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px}
.score-name{width:100px;color:var(--muted)}
.score-track{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.score-prog{height:100%;border-radius:3px;transition:width 0.5s}
.score-val{width:40px;font-family:var(--mono);font-size:11px;text-align:left}

/* ── chart bottom ── */
.bottom-bar {
  border-top:1px solid var(--border);
  background:var(--surface);
  padding:16px 24px;
  display:flex;gap:24px;
  align-items:center;
}
.dist-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;margin-left:16px}
.dist-item{display:flex;align-items:center;gap:6px;font-size:11px}
.dist-dot{width:8px;height:8px;border-radius:50%}
.dist-count{font-family:var(--mono);font-weight:600}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🛡</div>
    AML SHIELD
  </div>
  <div class="live-badge"><div class="pulse"></div>LIVE MONITORING</div>
</header>

<div class="metrics">
  <div class="metric m-total">
    <div class="metric-label">تراکنش‌های پردازش‌شده</div>
    <div class="metric-value" id="m-total">0</div>
    <div class="metric-sub">از ابتدای session</div>
  </div>
  <div class="metric m-blocked">
    <div class="metric-label">مسدود شده</div>
    <div class="metric-value" id="m-blocked">0</div>
    <div class="metric-sub" id="m-blocked-sub">— از کل</div>
  </div>
  <div class="metric m-rate">
    <div class="metric-label">نرخ تقلب</div>
    <div class="metric-value" id="m-rate">0%</div>
    <div class="metric-sub">میانگین rolling</div>
  </div>
  <div class="metric m-high">
    <div class="metric-label">HIGH + CRITICAL</div>
    <div class="metric-value" id="m-high">0</div>
    <div class="metric-sub">نیاز به بررسی</div>
  </div>
</div>

<div class="body-grid">

  <!-- alerts -->
  <div class="alerts-panel">
    <div class="table-header">
      <div class="panel-title">هشدارها</div>
      <span id="alert-count" style="font-family:var(--mono);font-size:11px;color:var(--muted)">0</span>
    </div>
    <div class="alerts-scroll" id="alerts-list"></div>
  </div>

  <!-- table -->
  <div class="table-panel">
    <div class="table-header">
      <div class="panel-title">تراکنش‌های اخیر</div>
      <span style="font-size:11px;color:var(--muted)">کلیک برای جزئیات</span>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>شناسه</th>
            <th>حساب</th>
            <th>مبلغ (ریال)</th>
            <th>Risk Score</th>
            <th>سطح</th>
            <th>وضعیت</th>
            <th>زمان</th>
          </tr>
        </thead>
        <tbody id="tx-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- radar -->
  <div class="radar-panel">
    <div class="panel-title">توزیع ریسک</div>
    <canvas id="riskChart"></canvas>
    <div style="margin-top:auto">
      <div class="panel-title" style="margin-bottom:10px">آخرین ۵۰ تراکنش</div>
      <canvas id="timelineChart" height="80"></canvas>
    </div>
  </div>

</div>

<div class="bottom-bar">
  <span class="dist-label">توزیع:</span>
  <div class="dist-item"><div class="dist-dot" style="background:var(--green)"></div><span class="dist-count" id="d-low">0</span><span style="color:var(--muted);font-size:10px"> کم</span></div>
  <div class="dist-item"><div class="dist-dot" style="background:var(--yellow)"></div><span class="dist-count" id="d-med">0</span><span style="color:var(--muted);font-size:10px"> متوسط</span></div>
  <div class="dist-item"><div class="dist-dot" style="background:var(--orange)"></div><span class="dist-count" id="d-high">0</span><span style="color:var(--muted);font-size:10px"> بالا</span></div>
  <div class="dist-item"><div class="dist-dot" style="background:var(--red)"></div><span class="dist-count" id="d-crit">0</span><span style="color:var(--muted);font-size:10px"> بحرانی</span></div>
</div>

<!-- Modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title" id="modal-title">TX-00000</div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
// ── charts ──
const riskChart = new Chart(document.getElementById('riskChart'), {
  type: 'doughnut',
  data: {
    labels: ['کم','متوسط','بالا','بحرانی'],
    datasets:[{data:[0,0,0,0],backgroundColor:['#10B981','#F59E0B','#F97316','#EF4444'],borderWidth:0,hoverOffset:4}]
  },
  options:{
    responsive:true,maintainAspectRatio:true,
    plugins:{legend:{position:'bottom',labels:{color:'#64748B',font:{size:10},padding:8}}}
  }
});

const timelineData = {labels:[], datasets:[{data:[],borderColor:'#7C3AED',backgroundColor:'rgba(124,58,237,0.1)',borderWidth:1.5,pointRadius:3,pointBackgroundColor:[],fill:true,tension:0.4}]};
const timelineChart = new Chart(document.getElementById('timelineChart'),{
  type:'line', data:timelineData,
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{display:false},y:{display:false,min:0,max:1}},
    animation:{duration:300}}
});

// ── risk colors ──
const RC = {low:'#10B981',medium:'#F59E0B',high:'#F97316',critical:'#EF4444'};
const RL = {low:'کم',medium:'متوسط',high:'بالا',critical:'بحرانی'};
const SC = {high:'#EF4444',medium:'#F59E0B',low:'#10B981'};

// ── modal ──
function openModal(tx) {
  document.getElementById('modal-title').textContent = tx.id;
  const blocked = tx.blocked;
  const scoreColor = tx.risk_score > 0.6 ? '#EF4444' : tx.risk_score > 0.3 ? '#F59E0B' : '#10B981';

  let alertsHTML = '';
  if (tx.alerts && tx.alerts.length > 0) {
    alertsHTML = `<div class="section-title" style="margin-bottom:10px">دلایل هشدار</div>`;
    tx.alerts.forEach(a => {
      alertsHTML += `
        <div class="alert-detail sev-${a.severity}">
          <div class="ad-top">
            <span class="ad-code" style="color:${SC[a.severity]}">${a.code}</span>
            <span class="badge b-${a.severity}">${a.severity.toUpperCase()}</span>
          </div>
          <div class="ad-msg">${a.message}</div>
        </div>`;
    });
  } else {
    alertsHTML = `<div style="padding:16px;text-align:center;color:var(--muted);font-size:12px">هیچ هشداری ثبت نشده</div>`;
  }

  document.getElementById('modal-body').innerHTML = `
    <div class="info-grid">
      <div class="info-item">
        <div class="info-label">شناسه تراکنش</div>
        <div class="info-value" style="color:var(--purple2)">${tx.id}</div>
      </div>
      <div class="info-item">
        <div class="info-label">وضعیت</div>
        <div class="info-value">${blocked ? '<span style="color:#EF4444">🔴 مسدود شده</span>' : '<span style="color:#10B981">🟢 تأیید شده</span>'}</div>
      </div>
      <div class="info-item">
        <div class="info-label">حساب</div>
        <div class="info-value">${tx.account}</div>
      </div>
      <div class="info-item">
        <div class="info-label">نوع حساب</div>
        <div class="info-value">${tx.acc_type || '—'}</div>
      </div>
      <div class="info-item">
        <div class="info-label">شماره کارت</div>
        <div class="info-value">${tx.card}</div>
      </div>
      <div class="info-item">
        <div class="info-label">مبلغ (ریال)</div>
        <div class="info-value" style="color:${scoreColor}">${tx.amount_fmt}</div>
      </div>
      <div class="info-item">
        <div class="info-label">آدرس IP</div>
        <div class="info-value">${tx.ip}</div>
      </div>
      <div class="info-item">
        <div class="info-label">اعتماد IP</div>
        <div class="info-value" style="color:${tx.ip_confidence < 1 ? '#F59E0B' : '#10B981'}">${tx.ip_confidence < 1 ? 'NAT (پایین)' : 'عادی'}</div>
      </div>
      <div class="info-item">
        <div class="info-label">زمان</div>
        <div class="info-value">${tx.time} ${tx.is_night ? '🌙' : '☀️'}</div>
      </div>
      <div class="info-item">
        <div class="info-label">سطح ریسک</div>
        <div class="info-value"><span class="badge b-${tx.risk_level}">${RL[tx.risk_level]}</span></div>
      </div>
    </div>

    <div class="score-section">
      <div class="section-title" style="margin-bottom:12px">تحلیل ریسک</div>
      <div class="score-row">
        <span class="score-name">Rule Score</span>
        <div class="score-track"><div class="score-prog" style="width:${(tx.rule_score||0)*100}%;background:#7C3AED"></div></div>
        <span class="score-val" style="color:#7C3AED">${((tx.rule_score||0)*100).toFixed(0)}%</span>
      </div>
      <div class="score-row">
        <span class="score-name">ML Score</span>
        <div class="score-track"><div class="score-prog" style="width:${(tx.ml_score||0)*100}%;background:#06B6D4"></div></div>
        <span class="score-val" style="color:#06B6D4">${((tx.ml_score||0)*100).toFixed(0)}%</span>
      </div>
      <div class="score-row">
        <span class="score-name">Final Score</span>
        <div class="score-track"><div class="score-prog" style="width:${tx.risk_score*100}%;background:${scoreColor}"></div></div>
        <span class="score-val" style="color:${scoreColor}">${(tx.risk_score*100).toFixed(0)}%</span>
      </div>
    </div>

    ${alertsHTML}
  `;
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}
document.getElementById('modal').addEventListener('click', e => {
  if (e.target === document.getElementById('modal')) closeModal();
});
document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });

// ── WebSocket ──
const ws = new WebSocket(`ws://${location.host}/ws`);
let alertCount = 0;

ws.onmessage = e => {
  const msg = JSON.parse(e.data);

  if (msg.type === 'init') {
    const s = msg.data;
    s.recent_tx.slice(-20).forEach(addTxRow);
    s.alerts.slice(-30).forEach(addAlertRow);
    updateStats({
      total_tx: s.total_tx, blocked: s.blocked,
      risk_dist: s.risk_dist,
      block_rate: Math.round(s.blocked/Math.max(s.total_tx,1)*100*10)/10
    });
    return;
  }

  if (msg.type === 'tx') {
    addTxRow(msg.data);
    updateTimeline(msg.data.risk_score, msg.data.risk_level);
  }

  if (msg.type === 'stats') updateStats(msg.data);

  if (msg.type === 'alert' && msg.data) addAlertRow(msg.data);
};

function addTxRow(tx) {
  const tbody = document.getElementById('tx-tbody');
  const tr = document.createElement('tr');
  if (tx.blocked) tr.classList.add('flagged');
  const sc = tx.risk_score;
  const fillColor = RC[tx.risk_level];
  tr.innerHTML = `
    <td style="color:var(--purple2)">${tx.id}</td>
    <td class="text">${tx.account}</td>
    <td>${tx.amount_fmt}</td>
    <td>
      <div style="display:flex;align-items:center;gap:6px">
        <div class="score-bar"><div class="score-fill" style="width:${sc*100}%;background:${fillColor}"></div></div>
        <span style="color:${fillColor};font-size:11px">${(sc*100).toFixed(0)}%</span>
      </div>
    </td>
    <td><span class="badge b-${tx.risk_level}">${RL[tx.risk_level]}</span></td>
    <td><span class="badge ${tx.blocked ? 'b-blocked':'b-ok'}">${tx.blocked?'مسدود':'تأیید'}</span></td>
    <td style="color:var(--muted)">${tx.time}</td>`;
  tr.addEventListener('click', () => openModal(tx));
  tbody.insertBefore(tr, tbody.firstChild);
  while (tbody.children.length > 50) tbody.lastChild.remove();
}

function addAlertRow(a) {
  if (!a) return;
  alertCount++;
  document.getElementById('alert-count').textContent = alertCount;
  const list = document.getElementById('alerts-list');
  const div = document.createElement('div');
  div.className = `alert-item sev-${a.severity}`;
  div.innerHTML = `
    <div class="alert-top">
      <span class="alert-code">${a.code} <span style="font-size:10px;color:var(--muted);font-weight:400">${a.tx_id||''}</span></span>
      <span class="alert-time">${a.time}</span>
    </div>
    <div class="alert-msg">${a.message}</div>`;
  list.insertBefore(div, list.firstChild);
  while (list.children.length > 60) list.lastChild.remove();
}

function updateStats(s) {
  document.getElementById('m-total').textContent   = s.total_tx.toLocaleString();
  document.getElementById('m-blocked').textContent = s.blocked.toLocaleString();
  document.getElementById('m-rate').textContent    = s.block_rate + '%';
  const high = (s.risk_dist.high||0) + (s.risk_dist.critical||0);
  document.getElementById('m-high').textContent    = high.toLocaleString();
  document.getElementById('m-blocked-sub').textContent = `${s.block_rate}% از کل`;
  document.getElementById('d-low').textContent  = (s.risk_dist.low||0).toLocaleString();
  document.getElementById('d-med').textContent  = (s.risk_dist.medium||0).toLocaleString();
  document.getElementById('d-high').textContent = (s.risk_dist.high||0).toLocaleString();
  document.getElementById('d-crit').textContent = (s.risk_dist.critical||0).toLocaleString();
  riskChart.data.datasets[0].data = [
    s.risk_dist.low||0, s.risk_dist.medium||0,
    s.risk_dist.high||0, s.risk_dist.critical||0
  ];
  riskChart.update();
}

function updateTimeline(score, level) {
  const d = timelineData;
  d.labels.push('');
  d.datasets[0].data.push(score);
  d.datasets[0].pointBackgroundColor.push(RC[level]);
  if (d.labels.length > 50) {
    d.labels.shift(); d.datasets[0].data.shift();
    d.datasets[0].pointBackgroundColor.shift();
  }
  timelineChart.update();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
