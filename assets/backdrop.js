// The homepage hero's animated backdrop: two streams of noisy points drifting across the page, each
// with a Whittaker smoother fitted to it, frame by frame. The smoothing parameter swings slowly from
// small (the curve chases the noise) to large (it flattens the signal), and the band around the
// curve widens as the fit misses. It's faint behind the logo and tagline, paused out of view, and
// still with reduced motion.
(() => {
  "use strict";

  const hero = document.querySelector(".gd-hero");
  if (!hero) return;
  const canvas = document.createElement("canvas");
  canvas.id = "wkh-backdrop";
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  hero.prepend(canvas);

  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const root = document.documentElement;
  const isDark = () => root.classList.contains("quarto-dark") || document.body.classList.contains("quarto-dark");

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const smooth = (lo, hi, x) => {
    const k = clamp((x - lo) / (hi - lo), 0, 1);
    return k * k * (3 - 2 * k);
  };

  // The canvas's size (in CSS pixels); the logo and tagline's box in it, which is kept clear; and
  // the page's text over it (the lead, the calls to action and the sidebar), kept nearly so
  let W = 0;
  let H = 0;
  let dpr = 1;
  const focus = { left: 0, right: 0, top: 0, bottom: 0 };
  let texts = [];
  const TEXTS = ["#quarto-document-content > p:first-of-type", ".wkh-cta", "#quarto-margin-sidebar"];

  // How much of a thing to draw at a point: less near the logo and tagline, and over text
  const gap = (r, x, y) => Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
  const clearing = (x, y, over = 0.3) => {
    let k = 0.1 + 0.9 * smooth(0, 120, gap(focus, x, y));
    for (const r of texts) k = Math.min(k, over + (1 - over) * smooth(0, 110, gap(r, x, y)));
    return k;
  };

  function layout() {
    const box = hero.getBoundingClientRect();
    const nav = document.getElementById("quarto-header");
    const navHeight = nav ? nav.offsetHeight : 0;
    const heroTop = box.top + window.scrollY;
    // From under the navbar to below the hero (where it fades out), across the page
    const top = navHeight - heroTop;
    const width = root.clientWidth;
    const height = Math.round(heroTop - navHeight + box.height + Math.min(340, window.innerHeight * 0.4));
    canvas.style.left = `${-box.left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // (The logo shown, for the theme, and the tagline)
    const parts = Array.from(hero.querySelectorAll(".gd-hero-logo, .gd-hero-tagline"))
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0);
    if (parts.length > 0) {
      focus.left = Math.min(...parts.map((r) => r.left));
      focus.right = Math.max(...parts.map((r) => r.right));
      focus.top = Math.min(...parts.map((r) => r.top)) - box.top - top;
      focus.bottom = Math.max(...parts.map((r) => r.bottom)) - box.top - top;
    } else {
      focus.left = width * 0.3;
      focus.right = width * 0.7;
      focus.top = height * 0.2;
      focus.bottom = height * 0.6;
    }
    texts = TEXTS.map((selector) => document.querySelector(selector))
      .filter((el) => el !== null)
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0)
      .map((r) => ({ left: r.left, right: r.right, top: r.top - box.top - top, bottom: r.bottom - box.top - top }));

    const ratio = Math.min(2, window.devicePixelRatio || 1);
    if (width !== W || height !== H || ratio !== dpr) {
      W = width;
      H = height;
      dpr = ratio;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
  }

  /* ── The Whittaker smoother ────────────────────────────────────────────────────────────────── */

  // The fit z minimizing |y - z|² + λ|D z|², with D the second differences: the solution of the
  // pentadiagonal system (I + λ D'D) z = y, by its banded Cholesky factor
  function whittaker(y, lambda) {
    const n = y.length;
    const l0 = new Float64Array(n);
    const l1 = new Float64Array(n);
    const l2 = new Float64Array(n);
    const last = n - 3;
    const ok = (k) => k >= 0 && k <= last;
    for (let i = 0; i < n; i++) {
      // (The band of D'D at row i: D's rows k = i - 2, i - 1, i have 1, -2, 1 at columns k, k+1, k+2)
      const a0 = 1 + lambda * ((ok(i) ? 1 : 0) + (ok(i - 1) ? 4 : 0) + (ok(i - 2) ? 1 : 0));
      const a1 = lambda * ((ok(i - 1) ? -2 : 0) + (ok(i - 2) ? -2 : 0));
      const a2 = lambda * (ok(i - 2) ? 1 : 0);
      l2[i] = i >= 2 ? a2 / l0[i - 2] : 0;
      l1[i] = i >= 1 ? (a1 - l2[i] * (i >= 2 ? l1[i - 1] : 0)) / l0[i - 1] : 0;
      l0[i] = Math.sqrt(a0 - l1[i] * l1[i] - l2[i] * l2[i]);
    }
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      z[i] = (y[i] - (i >= 1 ? l1[i] * z[i - 1] : 0) - (i >= 2 ? l2[i] * z[i - 2] : 0)) / l0[i];
    }
    for (let i = n - 1; i >= 0; i--) {
      z[i] = (z[i] - (i + 1 < n ? l1[i + 1] * z[i + 1] : 0) - (i + 2 < n ? l2[i + 2] * z[i + 2] : 0)) / l0[i];
    }
    return z;
  }

  /* ── The streams of points ─────────────────────────────────────────────────────────────────── */

  // The space between points, and how fast they drift (in pixels a second); how long the smoothing
  // parameter takes to swing from small to large and back (in seconds), and how far it swings
  const DX = 13;
  const SPEED = 16;
  const CYCLE = 18;
  const LOG_LAMBDA = [-0.6, 4.2];

  // Each stream: its height on the canvas (as a share of it), its own wave and noise, and where its
  // smoothing parameter starts in the cycle (opposite to the other's)
  const STREAMS = [
    { at: 0.3, seed: 11, phase: 0, waves: [[180, 0.6, 0.2], [71, 0.3, 1.7], [33, 0.14, 0.5]] },
    { at: 0.72, seed: 37, phase: 0.5, waves: [[150, 0.55, 2.1], [83, 0.32, 0.3], [41, 0.18, 2.9]] },
  ];

  // A point's noise: a standard normal, the same for the same point every frame
  const hash = (j, k) => {
    const s = Math.sin(j * 127.1 + k * 311.7) * 43758.5453;
    return s - Math.floor(s);
  };
  const noise = (j, seed) => {
    const u = Math.max(1e-6, hash(j, seed));
    const v = hash(j, seed + 0.5);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  function draw(t) {
    const dark = isDark();
    const pts = dark ? "208,181,226" : "136,104,164";
    const rose = dark ? "233,117,140" : "184,64,95";
    const amp = clamp(H * 0.11, 26, 70);
    const sd = amp * 0.33;
    const shift = t * SPEED;
    const first = Math.floor(shift / DX) - 2;
    const n = Math.ceil(W / DX) + 5;

    for (const stream of STREAMS) {
      const base = H * stream.at;
      const xs = new Float64Array(n);
      const ys = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const j = first + i;
        const u = j * DX;
        let signal = 0;
        for (const [period, size, offset] of stream.waves) signal += size * Math.sin(u / period + offset);
        xs[i] = u - shift;
        ys[i] = base + amp * signal + sd * noise(j, stream.seed);
      }

      // (The smoothing parameter's swing: slow at its ends, where the fit is at its worst and best)
      const swing = 0.5 - 0.5 * Math.cos(2 * Math.PI * (t / CYCLE + stream.phase));
      const lambda = 10 ** (LOG_LAMBDA[0] + (LOG_LAMBDA[1] - LOG_LAMBDA[0]) * swing);
      const z = whittaker(ys, lambda);
      let rss = 0;
      for (let i = 0; i < n; i++) rss += (ys[i] - z[i]) ** 2;
      const band = 1.8 * Math.max(sd * 0.4, Math.sqrt(rss / n));

      // The band, a strip at a time (so that each part is as faint as where it is)
      for (let i = 0; i + 1 < n; i++) {
        const a = (dark ? 0.1 : 0.07) * clearing((xs[i] + xs[i + 1]) / 2, (z[i] + z[i + 1]) / 2, 0.2);
        if (a < 0.003) continue;
        ctx.fillStyle = `rgba(${rose},${a})`;
        ctx.beginPath();
        ctx.moveTo(xs[i], z[i] - band);
        ctx.lineTo(xs[i + 1], z[i + 1] - band);
        ctx.lineTo(xs[i + 1], z[i + 1] + band);
        ctx.lineTo(xs[i], z[i] + band);
        ctx.closePath();
        ctx.fill();
      }

      // The points
      for (let i = 0; i < n; i++) {
        const a = (dark ? 0.5 : 0.42) * clearing(xs[i], ys[i], 0.18);
        if (a < 0.01) continue;
        ctx.fillStyle = `rgba(${pts},${a})`;
        ctx.beginPath();
        ctx.arc(xs[i], ys[i], 2.6, 0, Math.PI * 2);
        ctx.fill();
      }

      // The curve, a segment at a time
      ctx.lineWidth = 2.4;
      ctx.lineCap = "round";
      for (let i = 0; i + 1 < n; i++) {
        const a = (dark ? 0.7 : 0.55) * clearing((xs[i] + xs[i + 1]) / 2, (z[i] + z[i + 1]) / 2, 0.2);
        if (a < 0.01) continue;
        ctx.strokeStyle = `rgba(${rose},${a})`;
        ctx.beginPath();
        ctx.moveTo(xs[i], z[i]);
        ctx.lineTo(xs[i + 1], z[i + 1]);
        ctx.stroke();
      }
    }
  }

  /* ── Running it ────────────────────────────────────────────────────────────────────────────── */

  let t = 0;
  let last = 0;
  let frame = 0;
  let inView = true;
  let ready = false;

  function paint() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    draw(t);
  }

  const running = () => inView && !motion.matches && !document.hidden;

  function tick(now) {
    frame = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    t += dt;
    paint();
    if (running()) frame = requestAnimationFrame(tick);
  }

  function start() {
    if (!ready) return;
    if (running()) {
      if (frame === 0) {
        last = 0;
        frame = requestAnimationFrame(tick);
      }
    } else {
      // (Still: one frame, as it is)
      if (frame !== 0) cancelAnimationFrame(frame);
      frame = 0;
      paint();
    }
  }

  let pending = 0;
  function refit() {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      layout();
      ready = true;
      canvas.classList.add("wkh-on");
      start();
    });
  }

  // (Started partway through the cycle, with one stream well fitted and the other not)
  t = CYCLE * 0.3;
  refit();
  if ("ResizeObserver" in window) new ResizeObserver(refit).observe(hero);
  window.addEventListener("resize", refit);
  motion.addEventListener("change", start);
  document.addEventListener("visibilitychange", start);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      inView = entries[entries.length - 1].isIntersecting;
      start();
    }).observe(canvas);
  }
  // (A theme's toggle: redrawn at once, if it's still)
  new MutationObserver(() => {
    if (!running()) paint();
  }).observe(root, { attributes: true, attributeFilter: ["class", "data-bs-theme"] });
})();
