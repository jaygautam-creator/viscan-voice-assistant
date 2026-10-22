// FDI odontogram (permanent dentition) rendered as SVG.
// marks: { "48": { state: "present"|"uncertain"|"absent"|"mentioned"|"missing", tip: "...", cited: bool } }

const WIDTH = { 1: 19, 2: 18, 3: 21, 4: 22, 5: 22, 6: 28, 7: 27, 8: 25 };
const GAP = 3;
const MID = 10;
const H = 34;

function sideTeeth(quadrant, reverse) {
  const pos = [1, 2, 3, 4, 5, 6, 7, 8];
  return (reverse ? pos.reverse() : pos).map((p) => `${quadrant}${p}`);
}

const ROWS = [
  { teeth: [...sideTeeth(1, true), ...sideTeeth(2, false)], y: 22, label: "UPPER", labelAbove: true },
  { teeth: [...sideTeeth(4, true), ...sideTeeth(3, false)], y: 86, label: "LOWER", labelAbove: false },
];

function toothPath(x, y, w, h, upper, pos) {
  // A subtle anatomy hint: cusps for posterior teeth, an incisal line for anteriors.
  const cy = upper ? y + h - 7 : y + 7;
  if (pos >= 6) return `M${x + 5} ${cy} q${w / 4 - 2} ${upper ? -4 : 4} ${w / 2 - 5} 0 q${w / 4 - 2} ${upper ? 4 : -4} ${w / 2 - 5} 0`;
  if (pos >= 4) return `M${x + 5} ${cy} q${w / 2 - 5} ${upper ? -5 : 5} ${w - 10} 0`;
  return `M${x + 5} ${cy} L${x + w - 5} ${cy}`;
}

export function renderOdontogram(el, { marks = {}, onClick = null, caption = "" } = {}) {
  const totalSide = [1, 2, 3, 4, 5, 6, 7, 8].reduce((a, p) => a + WIDTH[p], 0) + 7 * GAP;
  const W = totalSide * 2 + MID + 20;
  const VH = 142;
  let x0 = 10;
  let svg = `<svg viewBox="0 0 ${W} ${VH}" role="img" aria-label="Dental chart (FDI numbering)">`;
  svg += `<line class="midline" x1="${W / 2}" y1="6" x2="${W / 2}" y2="${VH - 6}"/>`;
  svg += `<text class="arch-label" x="10" y="${VH / 2 + 3}">R</text><text class="arch-label" x="${W - 16}" y="${VH / 2 + 3}">L</text>`;
  for (const row of ROWS) {
    let x = x0;
    row.teeth.forEach((code, i) => {
      const pos = Number(code[1]);
      const w = WIDTH[pos];
      const m = marks[code];
      const cls = ["tooth", m?.state || "", m?.cited ? "cited" : "", onClick ? "clickable" : ""].join(" ");
      const upper = row.labelAbove;
      const ty = upper ? row.y - 6 : row.y + H + 12;
      svg += `<g class="${cls}" data-fdi="${code}" tabindex="${onClick ? 0 : -1}">`;
      svg += `<title>${code}${m?.tip ? " · " + m.tip.replace(/</g, "&lt;") : ""}</title>`;
      svg += `<rect x="${x}" y="${row.y}" width="${w}" height="${H}" rx="${pos >= 6 ? 7 : 9}"/>`;
      svg += `<path d="${toothPath(x, row.y, w, H, upper, pos)}"/>`;
      if (m?.state === "missing") svg += `<path d="M${x + 5} ${row.y + 6} L${x + w - 5} ${row.y + H - 6} M${x + w - 5} ${row.y + 6} L${x + 5} ${row.y + H - 6}" style="stroke: var(--muted)"/>`;
      svg += `<text x="${x + w / 2}" y="${ty}">${code}</text></g>`;
      x += w + (i === 7 ? MID : GAP);
    });
  }
  svg += "</svg>";
  el.innerHTML = `<div class="odo">${svg}
    <div class="legend">
      <span><i class="swatch" style="background:var(--bad-soft);border-color:var(--bad)"></i>Finding</span>
      <span><i class="swatch" style="background:var(--warn-soft);border-color:var(--warn);border-style:dashed"></i>Uncertain</span>
      <span><i class="swatch" style="background:var(--ok-soft);border-color:var(--ok)"></i>Normal / negative</span>
      <span><i class="swatch" style="background:var(--brand-soft);border-color:var(--brand)"></i>Mentioned</span>
      ${caption ? `<span style="margin-left:auto">${caption}</span>` : ""}
    </div></div>`;
  if (onClick) {
    el.querySelectorAll(".tooth").forEach((g) => {
      const go = () => onClick(g.dataset.fdi);
      g.addEventListener("click", go);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    });
  }
}
