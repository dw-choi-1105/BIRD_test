import { useState, useEffect, useRef } from "react";

const W = 620, H = 420;
const START = { x: 90, y: 340 };
const END = { x: 530, y: 70 };

function lerp(a, b, t) { return a + (b - a) * t; }

function velocityField(_x, _y, t) {
  const bx = END.x - START.x;
  const by = END.y - START.y;
  const perp = { x: -by, y: bx };
  const len = Math.sqrt(perp.x ** 2 + perp.y ** 2);
  const pn = { x: perp.x / len, y: perp.y / len };
  const curve = Math.sin(t * Math.PI) * 0.45;
  return {
    vx: bx + pn.x * curve * Math.sqrt(bx*bx+by*by),
    vy: by + pn.y * curve * Math.sqrt(bx*bx+by*by),
  };
}

function simulateEuler(nfe) {
  const pts = [{ ...START }];
  let cx = START.x, cy = START.y;
  const dt = 1 / nfe;
  for (let i = 0; i < nfe; i++) {
    const t = i / nfe;
    const v = velocityField(cx, cy, t);
    cx += v.vx * dt;
    cy += v.vy * dt;
    pts.push({ x: cx, y: cy });
  }
  return pts;
}

function simulateTruth() { return simulateEuler(500); }

function pointsToPath(pts) {
  if (pts.length < 2) return "";
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

function clipPath(pts, prog) {
  const count = Math.max(2, Math.floor(prog * pts.length));
  return pts.slice(0, count);
}

const CONFIGS = [
  { nfe: 50, label: "NFE = 50", color: "#4ade80", quality: "최상",  qColor: "#4ade80",  desc: "거의 완벽" },
  { nfe: 20, label: "NFE = 20", color: "#facc15", quality: "양호",  qColor: "#facc15",  desc: "미세 편차" },
  { nfe: 8,  label: "NFE = 8",  color: "#fb923c", quality: "저하",  qColor: "#fb923c",  desc: "오차 누적" },
  { nfe: 4,  label: "NFE = 4",  color: "#f87171", quality: "심각",  qColor: "#f87171",  desc: "크게 이탈" },
];

export default function App() {
  const [active, setActive] = useState(null);
  const [progs, setProgs] = useState({});
  const rafRef = useRef({});

  const allTraj = useRef(CONFIGS.map(c => simulateEuler(c.nfe)));
  const truth = useRef(simulateTruth());

  const play = (idx) => {
    if (active === idx) { setActive(null); return; }
    setActive(idx);
    const key = CONFIGS[idx].nfe;
    cancelAnimationFrame(rafRef.current[key]);
    const t0 = performance.now();
    const dur = 1100;
    const go = (now) => {
      const p = Math.min((now - t0) / dur, 1);
      setProgs(prev => ({ ...prev, [key]: p }));
      if (p < 1) rafRef.current[key] = requestAnimationFrame(go);
    };
    setProgs(prev => ({ ...prev, [key]: 0 }));
    rafRef.current[key] = requestAnimationFrame(go);
  };

  useEffect(() => () => Object.values(rafRef.current).forEach(cancelAnimationFrame), []);

  const errors = CONFIGS.map(cfg => {
    const pts = allTraj.current[cfg.nfe === 50 ? 0 : cfg.nfe === 20 ? 1 : cfg.nfe === 8 ? 2 : 3];
    const last = pts[pts.length - 1];
    return Math.sqrt((last.x - END.x) ** 2 + (last.y - END.y) ** 2).toFixed(1);
  });

  return (
    <div style={{
      background: "#07080f",
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'JetBrains Mono', 'Courier New', monospace",
      padding: "24px 12px",
    }}>
      <div style={{ marginBottom: 18, textAlign: "center" }}>
        <div style={{ color: "#e2e8f0", fontSize: 18, fontWeight: 700, letterSpacing: "0.05em", marginBottom: 3 }}>
          Rectified Flow — NFE vs Trajectory
        </div>
        <div style={{ color: "#334155", fontSize: 11, letterSpacing: "0.12em" }}>
          SD3.5 · Euler Discretization Error · Click to animate
        </div>
      </div>

      <div style={{
        background: "#0a0c18",
        borderRadius: 18,
        border: "1px solid #1a1f3a",
        boxShadow: "0 0 80px #0008",
        overflow: "hidden",
      }}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          {[...Array(8)].map((_, i) => (
            <line key={`gv${i}`} x1={90 + i * 64} y1={40} x2={90 + i * 64} y2={380}
              stroke="#10132a" strokeWidth={1} />
          ))}
          {[...Array(6)].map((_, i) => (
            <line key={`gh${i}`} x1={40} y1={70 + i * 62} x2={580} y2={70 + i * 62}
              stroke="#10132a" strokeWidth={1} />
          ))}

          <path d={pointsToPath(truth.current)} fill="none"
            stroke="#3b82f6" strokeWidth={2.5} strokeDasharray="8,5" opacity={0.55} />

          {active === null && CONFIGS.map((cfg, idx) => (
            <path key={cfg.nfe}
              d={pointsToPath(allTraj.current[idx])}
              fill="none" stroke={cfg.color} strokeWidth={1.5} opacity={0.35} />
          ))}

          {active !== null && (() => {
            const cfg = CONFIGS[active];
            const pts = allTraj.current[active];
            const prog = progs[cfg.nfe] ?? 1;
            const vis = clipPath(pts, prog);
            return (
              <g>
                {pts.slice(1).map((p, pi) => {
                  if ((pi + 1) / pts.length > prog) return null;
                  const ti = Math.floor(((pi + 1) / pts.length) * truth.current.length);
                  const tp = truth.current[Math.min(ti, truth.current.length - 1)];
                  const err = Math.sqrt((p.x - tp.x) ** 2 + (p.y - tp.y) ** 2);
                  if (err < 5) return null;
                  return (
                    <line key={pi}
                      x1={tp.x} y1={tp.y} x2={p.x} y2={p.y}
                      stroke={cfg.color} strokeWidth={1}
                      strokeDasharray="3,2" opacity={0.4} />
                  );
                })}
                <path d={pointsToPath(vis)} fill="none"
                  stroke={cfg.color} strokeWidth={3} opacity={1}
                  strokeLinecap="round" strokeLinejoin="round" />
                {pts.map((p, pi) => {
                  if (pi / pts.length > prog || pi === 0) return null;
                  return <circle key={pi} cx={p.x} cy={p.y} r={pi === pts.length - 1 ? 6 : 4}
                    fill={cfg.color} opacity={0.9} />;
                })}
                <path d={pointsToPath(vis)} fill="none"
                  stroke={cfg.color} strokeWidth={8} opacity={0.12}
                  strokeLinecap="round" />
              </g>
            );
          })()}

          <circle cx={START.x} cy={START.y} r={9} fill="#1e3a8a" stroke="#3b82f6" strokeWidth={2.5} />
          <text x={START.x + 14} y={START.y + 5} fill="#60a5fa" fontSize={11}>x_T (noise)</text>

          <circle cx={END.x} cy={END.y} r={9} fill="#14532d" stroke="#4ade80" strokeWidth={2.5} />
          <text x={END.x - 80} y={END.y + 5} fill="#4ade80" fontSize={11}>x_0 (image)</text>

          <line x1={END.x - 16} y1={END.y} x2={END.x + 16} y2={END.y} stroke="#4ade80" strokeWidth={1} opacity={0.4} />
          <line x1={END.x} y1={END.y - 16} x2={END.x} y2={END.y + 16} stroke="#4ade80" strokeWidth={1} opacity={0.4} />

          {active !== null && (() => {
            const prog = progs[CONFIGS[active].nfe] ?? 1;
            if (prog < 0.95) return null;
            const pts = allTraj.current[active];
            const last = pts[pts.length - 1];
            const cfg = CONFIGS[active];
            const err = parseFloat(errors[active]);
            return (
              <g>
                <rect x={last.x + 10} y={last.y - 16} width={80} height={22}
                  rx={5} fill="#0a0c18" stroke={cfg.color} strokeWidth={1} opacity={0.95} />
                <text x={last.x + 50} y={last.y - 1} fill={cfg.color} fontSize={10}
                  textAnchor="middle">err={err.toFixed(0)}px</text>
              </g>
            );
          })()}
        </svg>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap", justifyContent: "center" }}>
        {CONFIGS.map((cfg, idx) => (
          <button key={cfg.nfe} onClick={() => play(idx)} style={{
            background: active === idx ? `${cfg.color}1a` : "#0a0c18",
            border: `1.5px solid ${active === idx ? cfg.color : "#1a1f3a"}`,
            borderRadius: 12, padding: "10px 16px", cursor: "pointer",
            display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
            transition: "all 0.15s", minWidth: 108,
            boxShadow: active === idx ? `0 0 18px ${cfg.color}30` : "none",
          }}>
            <span style={{ color: cfg.color, fontSize: 13, fontWeight: 700 }}>{cfg.label}</span>
            <span style={{ background: `${cfg.color}20`, color: cfg.color, fontSize: 10, borderRadius: 4, padding: "2px 8px" }}>
              {cfg.quality}
            </span>
            <span style={{ color: "#334155", fontSize: 10 }}>Δerr ≈ {errors[idx]}px</span>
          </button>
        ))}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 14 }}>
        <svg width={38} height={10}>
          <line x1={0} y1={5} x2={38} y2={5} stroke="#3b82f6" strokeWidth={2} strokeDasharray="6,4" />
        </svg>
        <span style={{ color: "#3b82f6", fontSize: 11, opacity: 0.7 }}>
          ground truth (NFE=500, 이상 경로)
        </span>
      </div>

      <div style={{
        marginTop: 14, maxWidth: 580, width: "100%",
        background: "#0a0c18", border: "1px solid #1a1f3a",
        borderRadius: 12, padding: "13px 18px",
        color: "#475569", fontSize: 11, lineHeight: 1.8,
      }}>
        <span style={{ color: "#94a3b8", fontWeight: 700 }}>버튼 클릭</span> → trajectory 애니메이션 재생 · 짧은 점선 = 이상 경로 대비 편차<br />
        Velocity field에 곡률이 있을 때, <span style={{ color: "#fb923c" }}>큰 Δt</span>는 방향을 잘못 적분해 최종 x₀에서 멀어짐
      </div>
    </div>
  );
}
