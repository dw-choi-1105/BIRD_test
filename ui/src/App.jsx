import { useState, useEffect, useRef } from "react";

const W = 620, H = 420;
const START = { x: 90, y: 340 };
const END   = { x: 530, y: 80 };

const DIR = { x: END.x - START.x, y: END.y - START.y };
const LEN = Math.sqrt(DIR.x**2 + DIR.y**2);
const PERP = { x: -DIR.y / LEN, y: DIR.x / LEN };
const CURVE = 110;

function truePath(t) {
  return {
    x: START.x + DIR.x * t + Math.sin(Math.PI * t) * CURVE * PERP.x,
    y: START.y + DIR.y * t + Math.sin(Math.PI * t) * CURVE * PERP.y,
  };
}

function trueVelocity(t) {
  const c = Math.PI * Math.cos(Math.PI * t) * CURVE;
  return {
    vx: DIR.x + c * PERP.x,
    vy: DIR.y + c * PERP.y,
  };
}

function simulateEuler(nfe) {
  const pts = [{ ...START }];
  let cx = START.x, cy = START.y;
  const dt = 1 / nfe;
  for (let i = 0; i < nfe; i++) {
    const v = trueVelocity(i / nfe);
    cx += v.vx * dt;
    cy += v.vy * dt;
    pts.push({ x: cx, y: cy });
  }
  return pts;
}

function groundTruth() {
  return Array.from({ length: 201 }, (_, i) => truePath(i / 200));
}

function toPath(pts) {
  return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

const CONFIGS = [
  { nfe: 50, label: "NFE = 50", color: "#4ade80", quality: "최상" },
  { nfe: 20, label: "NFE = 20", color: "#facc15", quality: "양호" },
  { nfe:  8, label: "NFE = 8",  color: "#fb923c", quality: "저하" },
  { nfe:  4, label: "NFE = 4",  color: "#f87171", quality: "심각" },
];

function endpointError(pts) {
  const last = pts[pts.length - 1];
  return Math.sqrt((last.x - END.x) ** 2 + (last.y - END.y) ** 2);
}

export default function App() {
  const [active, setActive]    = useState(null);
  const [prog,   setProgState] = useState(1);
  const rafRef  = useRef(null);

  const trajs = useRef(CONFIGS.map(c => simulateEuler(c.nfe)));
  const gt    = useRef(groundTruth());

  const play = (idx) => {
    if (active === idx) { setActive(null); return; }
    setActive(idx);
    cancelAnimationFrame(rafRef.current);
    const t0 = performance.now(), dur = 1200;
    setProgState(0);
    const go = (now) => {
      const p = Math.min((now - t0) / dur, 1);
      setProgState(p);
      if (p < 1) rafRef.current = requestAnimationFrame(go);
    };
    rafRef.current = requestAnimationFrame(go);
  };

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  return (
    <div style={{
      background: "#07080f", minHeight: "100vh",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "'JetBrains Mono', monospace", padding: "24px 12px",
    }}>
      <div style={{ marginBottom: 18, textAlign: "center" }}>
        <div style={{ color: "#e2e8f0", fontSize: 18, fontWeight: 700, letterSpacing: "0.05em", marginBottom: 3 }}>
          Rectified Flow — NFE vs Trajectory
        </div>
        <div style={{ color: "#334155", fontSize: 11, letterSpacing: "0.12em" }}>
          SD3.5 · Euler Discretization Error
        </div>
      </div>

      <div style={{
        background: "#0a0c18", borderRadius: 18,
        border: "1px solid #1a1f3a",
        boxShadow: "0 0 80px #0008", overflow: "hidden",
      }}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          {[...Array(8)].map((_, i) => (
            <line key={`v${i}`} x1={90+i*66} y1={40} x2={90+i*66} y2={390} stroke="#0e1128" strokeWidth={1}/>
          ))}
          {[...Array(6)].map((_, i) => (
            <line key={`h${i}`} x1={40} y1={60+i*64} x2={590} y2={60+i*64} stroke="#0e1128" strokeWidth={1}/>
          ))}

          <path d={toPath(gt.current)} fill="none" stroke="#3b82f6" strokeWidth={6} opacity={0.07}/>
          <path d={toPath(gt.current)} fill="none" stroke="#3b82f6" strokeWidth={2}
            strokeDasharray="8,5" opacity={0.55}/>

          {active === null && trajs.current.map((pts, i) => (
            <path key={i} d={toPath(pts)} fill="none"
              stroke={CONFIGS[i].color} strokeWidth={1.5} opacity={0.3}/>
          ))}

          {active !== null && (() => {
            const cfg = CONFIGS[active];
            const pts = trajs.current[active];
            const cut = Math.max(2, Math.floor(prog * pts.length));
            const vis = pts.slice(0, cut);

            return (
              <g>
                {vis.slice(1).map((p, pi) => {
                  const ti = Math.round(((pi + 1) / pts.length) * (gt.current.length - 1));
                  const gp = gt.current[ti];
                  const err = Math.sqrt((p.x-gp.x)**2+(p.y-gp.y)**2);
                  if (err < 6) return null;
                  return <line key={pi} x1={gp.x} y1={gp.y} x2={p.x} y2={p.y}
                    stroke={cfg.color} strokeWidth={1} strokeDasharray="3,2" opacity={0.5}/>;
                })}
                <path d={toPath(vis)} fill="none" stroke={cfg.color} strokeWidth={10} opacity={0.1}/>
                <path d={toPath(vis)} fill="none" stroke={cfg.color} strokeWidth={3}
                  strokeLinecap="round" strokeLinejoin="round"/>
                {vis.map((p, pi) => pi === 0 ? null :
                  <circle key={pi} cx={p.x} cy={p.y}
                    r={pi === vis.length-1 ? 7 : 3.5}
                    fill={cfg.color} opacity={0.85}/>
                )}
                {prog > 0.95 && (() => {
                  const last = pts[pts.length-1];
                  const err  = endpointError(pts).toFixed(1);
                  return (
                    <g>
                      <line x1={END.x} y1={END.y} x2={last.x} y2={last.y}
                        stroke={cfg.color} strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}/>
                      <rect x={last.x+10} y={last.y-14} width={88} height={20}
                        rx={5} fill="#0a0c18" stroke={cfg.color} strokeWidth={1}/>
                      <text x={last.x+54} y={last.y+1} fill={cfg.color} fontSize={10}
                        textAnchor="middle">err = {err}px</text>
                    </g>
                  );
                })()}
              </g>
            );
          })()}

          <circle cx={START.x} cy={START.y} r={9} fill="#1e3a8a" stroke="#3b82f6" strokeWidth={2.5}/>
          <text x={START.x+14} y={START.y+5} fill="#60a5fa" fontSize={11}>x_T (noise)</text>

          <circle cx={END.x} cy={END.y} r={9} fill="#14532d" stroke="#4ade80" strokeWidth={2.5}/>
          <text x={END.x-14} y={END.y-14} fill="#4ade80" fontSize={11} textAnchor="end">x_0 (image)</text>
          <line x1={END.x-14} y1={END.y} x2={END.x+14} y2={END.y} stroke="#4ade80" strokeWidth={1} opacity={0.4}/>
          <line x1={END.x} y1={END.y-14} x2={END.x} y2={END.y+14} stroke="#4ade80" strokeWidth={1} opacity={0.4}/>
        </svg>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap", justifyContent: "center" }}>
        {CONFIGS.map((cfg, idx) => {
          const err = endpointError(trajs.current[idx]).toFixed(1);
          return (
            <button key={cfg.nfe} onClick={() => play(idx)} style={{
              background: active===idx ? `${cfg.color}18` : "#0a0c18",
              border: `1.5px solid ${active===idx ? cfg.color : "#1a1f3a"}`,
              borderRadius: 12, padding: "10px 18px", cursor: "pointer",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
              transition: "all 0.15s", minWidth: 112,
              boxShadow: active===idx ? `0 0 20px ${cfg.color}28` : "none",
            }}>
              <span style={{ color: cfg.color, fontSize: 13, fontWeight: 700 }}>{cfg.label}</span>
              <span style={{ background: `${cfg.color}20`, color: cfg.color, fontSize: 10, borderRadius: 4, padding: "2px 8px" }}>
                {cfg.quality}
              </span>
              <span style={{ color: "#475569", fontSize: 10 }}>Δ = {err}px</span>
            </button>
          );
        })}
      </div>

      <div style={{ display:"flex", alignItems:"center", gap:8, marginTop:14 }}>
        <svg width={38} height={10}>
          <line x1={0} y1={5} x2={38} y2={5} stroke="#3b82f6" strokeWidth={2} strokeDasharray="6,4"/>
        </svg>
        <span style={{ color:"#3b82f6", fontSize:11, opacity:0.7 }}>ground truth — p(t) 해석해</span>
      </div>

      <div style={{
        marginTop:14, maxWidth:580, width:"100%",
        background:"#0a0c18", border:"1px solid #1a1f3a",
        borderRadius:12, padding:"13px 18px",
        color:"#475569", fontSize:11, lineHeight:1.8,
      }}>
        <span style={{color:"#94a3b8", fontWeight:700}}>핵심:</span> velocity = dp/dt를 해석적으로 정의 →
        NFE=50은 거의 정확히 x_0 도달, NFE=4는 큰 Δt로 곡률을 놓쳐 크게 이탈.<br/>
        점선 = 각 step의 ground truth 대비 편차 · 버튼 하단 숫자 = 최종 도달 오차(px)
      </div>
    </div>
  );
}
