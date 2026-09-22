import { loadFont } from "@remotion/google-fonts/Inter";
import {
  AbsoluteFill,
  Composition,
  Easing,
  Img,
  Interactive,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { seasons, teams } from "./data";

const { fontFamily } = loadFont("normal", {
  weights: ["400", "600", "800"],
  subsets: ["latin"],
});

// Paleta del proyecto (CLAUDE.md)
const BG = "#0f1115";
const CARD = "#151924";
const FG = "#EDEDED";
const GRID = "#2a2f3a";
const ACCENT = "#2d6cdf";
const GREEN = "#06d6a0";
const RED = "#d84a4a";
const MUTED = "#8a93a6";

const SCENE1 = 270; // 9 s
const SCENE2 = 360; // 12 s

export const MyComposition = () => {
  return (
    <Composition
      id="UnderCenter"
      component={UnderCenter}
      durationInFrames={SCENE1 + SCENE2}
      fps={30}
      width={1080}
      height={1920}
    />
  );
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

export const UnderCenter: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: BG, fontFamily, color: FG }}>
      <Sequence name="Temporadas" durationInFrames={SCENE1} layout="absolute-fill">
        <SeasonsScene />
      </Sequence>
      <Sequence name="Equipos" from={SCENE1} durationInFrames={SCENE2} layout="absolute-fill">
        <TeamsScene />
      </Sequence>
      <Interactive.Div
        name="Fuente"
        style={{
          position: "absolute",
          left: 60,
          bottom: 64,
          fontSize: 22,
          color: "#5d6577",
        }}
      >
        Fuente: nflverse-data · NFL 2026 · datos hasta sem. 2
      </Interactive.Div>
      <Interactive.Div
        name="Firma"
        style={{
          position: "absolute",
          right: 60,
          bottom: 58,
          fontSize: 30,
          fontStyle: "italic",
          color: "#888888",
        }}
      >
        @CuartayDato
      </Interactive.Div>
    </AbsoluteFill>
  );
};

/* ---------------- Escena 1: evolución por temporada ---------------- */

const CHART_TOP = 560;
const CHART_H = 900;
const MAX_PCT = 50;

const SeasonsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  return (
    <AbsoluteFill
      style={{
        opacity: interpolate(frame, [durationInFrames - 20, durationInFrames], [1, 0], clamp),
      }}
    >
      <Interactive.Div
        name="Titulo"
        style={{
          position: "absolute",
          top: 150,
          left: 60,
          right: 60,
          textAlign: "center",
          fontSize: 96,
          fontWeight: 800,
          lineHeight: 1.05,
          opacity: interpolate(frame, [0, 20], [0, 1], clamp),
          translate: interpolate(frame, [0, 25], ["0px 40px", "0px 0px"], { ...clamp, easing: ease }),
        }}
      >
        LA NFL SE PONE BAJO CENTRO
      </Interactive.Div>
      <Interactive.Div
        name="Subtitulo"
        style={{
          position: "absolute",
          top: 380,
          left: 80,
          right: 80,
          textAlign: "center",
          fontSize: 40,
          color: MUTED,
          opacity: interpolate(frame, [12, 35], [0, 1], clamp),
        }}
      >
        % de jugadas sin shotgun en la semana 2
      </Interactive.Div>

      <div
        style={{
          position: "absolute",
          left: 60,
          right: 60,
          top: CHART_TOP,
          height: CHART_H,
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          borderBottom: `2px solid ${GRID}`,
        }}
      >
        {seasons.map((s, i) => {
          const highlight = s.year === 2026;
          const start = 40 + i * 22 + (highlight ? 20 : 0);
          const p = interpolate(frame, [start, start + 35], [0, 1], { ...clamp, easing: ease });
          return (
            <div
              key={s.year}
              style={{ width: 168, display: "flex", flexDirection: "column", alignItems: "center" }}
            >
              <div
                style={{
                  fontSize: 48,
                  fontWeight: 800,
                  marginBottom: 14,
                  opacity: p,
                  color: highlight ? GREEN : FG,
                }}
              >
                {(s.pct * p).toFixed(1)}%
              </div>
              <div
                style={{
                  width: "100%",
                  height: (s.pct / MAX_PCT) * CHART_H * p,
                  backgroundColor: highlight ? GREEN : ACCENT,
                  borderRadius: "10px 10px 0 0",
                  boxShadow: highlight ? `0 0 ${60 * p}px ${GREEN}66` : "none",
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "center",
                  paddingBottom: 18,
                  fontSize: 24,
                  fontWeight: 600,
                  color: highlight ? "#0b2a22" : "#dce6ff",
                }}
              >
                <span style={{ opacity: interpolate(frame, [start + 25, start + 40], [0, 1], clamp) }}>
                  EPA {s.epa >= 0 ? "+" : ""}
                  {s.epa.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <div
        style={{
          position: "absolute",
          left: 60,
          right: 60,
          top: CHART_TOP + CHART_H + 18,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        {seasons.map((s) => (
          <div
            key={s.year}
            style={{
              width: 168,
              textAlign: "center",
              fontSize: 40,
              fontWeight: 600,
              color: s.year === 2026 ? GREEN : MUTED,
            }}
          >
            {s.year}
          </div>
        ))}
      </div>

      <Interactive.Div
        name="Callout 2026"
        style={{
          position: "absolute",
          top: 1600,
          left: 60,
          right: 60,
          textAlign: "center",
          fontSize: 52,
          fontWeight: 800,
          color: GREEN,
          opacity: interpolate(frame, [175, 195], [0, 1], clamp),
          scale: interpolate(frame, [175, 200], [0.85, 1], {
            ...clamp,
            easing: ease,
            output: "perceptual-scale",
          }),
        }}
      >
        +13 pp respecto a 2025
      </Interactive.Div>
    </AbsoluteFill>
  );
};

/* ---------------- Escena 2: cambio por equipo ---------------- */

const ZERO_X = 380; // posición del 0 dentro del panel
const PX_PER_PP = 14;
const ROW_H = 48;
const BAR_H = 32;
const LOGO = 40;

const TeamsScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ opacity: interpolate(frame, [0, 15], [0, 1], clamp) }}>
      <Interactive.Div
        name="Titulo equipos"
        style={{
          position: "absolute",
          top: 110,
          left: 60,
          right: 60,
          fontSize: 64,
          fontWeight: 800,
          lineHeight: 1.05,
          translate: interpolate(frame, [0, 25], ["0px 30px", "0px 0px"], { ...clamp, easing: ease }),
        }}
      >
        Quién ha cambiado y cuánto
      </Interactive.Div>
      <Interactive.Div
        name="Subtitulo equipos"
        style={{
          position: "absolute",
          top: 210,
          left: 60,
          right: 60,
          fontSize: 34,
          color: MUTED,
          opacity: interpolate(frame, [10, 30], [0, 1], clamp),
        }}
      >
        % bajo centro 2026 · cambio en pp vs su 2025 completo
      </Interactive.Div>

      <div
        style={{
          position: "absolute",
          top: 300,
          left: 30,
          right: 30,
          height: teams.length * ROW_H + 30,
          backgroundColor: CARD,
          borderRadius: 18,
        }}
      >
        <div
          style={{
            position: "absolute",
            left: ZERO_X,
            top: 10,
            bottom: 10,
            width: 2,
            backgroundColor: "#5a6275",
          }}
        />
        {teams.map((t, i) => (
          <TeamRow key={t.team} index={i} team={t.team} pct={t.pct} delta={t.delta} />
        ))}
      </div>
    </AbsoluteFill>
  );
};

const TeamRow: React.FC<{ index: number; team: string; pct: number; delta: number }> = ({
  index,
  team,
  pct,
  delta,
}) => {
  const frame = useCurrentFrame();
  const start = 30 + index * 5;
  const p = interpolate(frame, [start, start + 30], [0, 1], { ...clamp, easing: ease });
  const positive = delta >= 0;
  const color = positive ? GREEN : RED;
  const w = Math.abs(delta) * PX_PER_PP * p;
  const top = 15 + index * ROW_H + (ROW_H - BAR_H) / 2;

  return (
    <div style={{ position: "absolute", left: 0, right: 0, top, height: BAR_H }}>
      <div
        style={{
          position: "absolute",
          left: positive ? ZERO_X : ZERO_X - w,
          width: w,
          height: BAR_H,
          backgroundColor: color,
          borderRadius: positive ? "0 6px 6px 0" : "6px 0 0 6px",
        }}
      />
      <Img
        src={staticFile(`logos/${team}.png`)}
        style={{
          position: "absolute",
          left: positive ? ZERO_X - LOGO - 10 : ZERO_X + 10,
          top: (BAR_H - LOGO) / 2,
          width: LOGO,
          height: LOGO,
          objectFit: "contain",
          opacity: interpolate(frame, [start - 5, start + 8], [0, 1], clamp),
          scale: interpolate(frame, [start - 5, start + 12], [0.5, 1], { ...clamp, easing: ease }),
        }}
      />
      <div
        style={{
          position: "absolute",
          left: positive ? ZERO_X + w + 14 : ZERO_X + LOGO + 22,
          top: 0,
          height: BAR_H,
          display: "flex",
          alignItems: "center",
          whiteSpace: "nowrap",
          fontSize: 26,
          fontWeight: 600,
          color,
          opacity: interpolate(frame, [start + 12, start + 28], [0, 1], clamp),
        }}
      >
        {Math.round(pct * p)}% ({positive ? "+" : "−"}
        {(Math.abs(delta) * p).toFixed(1)} pp)
      </div>
    </div>
  );
};
