// The homepage's showcase: the stats that count up, the stepper that builds a model one step at a
// time, the switch between the four ways of drawing a band, the marquee of smooths, and
// the copy button
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // Copy text, and show that it's copied on the button (its icon, for a moment)
  function copyText(button, text) {
    const done = () => {
      const icon = button.querySelector(".bi");
      button.classList.add("wkh-copied");
      if (icon) icon.className = "bi bi-check2";
      setTimeout(() => {
        button.classList.remove("wkh-copied");
        if (icon) icon.className = "bi bi-clipboard";
      }, 1400);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => {});
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        done();
      } catch (_) {}
      area.remove();
    }
  }

  function setUpCopyButtons() {
    for (const button of document.querySelectorAll("[data-copy]")) {
      button.addEventListener("click", () => copyText(button, button.dataset.copy));
    }
  }

  // The lines of a code cell's code, as elements, each with the line break after it (so that a line
  // can be folded away; the code's text, which the copy button copies, is as it was)
  function codeLines(cell) {
    const lines = Array.from(cell.querySelectorAll("pre code > span"));
    for (const line of lines) {
      const after = line.nextSibling;
      if (after && after.nodeType === Node.TEXT_NODE && after.data === "\n") line.appendChild(after);
    }
    return lines;
  }

  // Fold the runs of lines that aren't new (but the line before and after each new one) into a
  // line that says how many there are, which unfolds them when clicked
  function foldUnchanged(lines, added) {
    const keep = lines.map((_, k) => added[k] || added[k - 1] || added[k + 1]);
    let k = 0;
    while (k < lines.length) {
      if (keep[k]) {
        k++;
        continue;
      }
      let end = k;
      while (end < lines.length && !keep[end]) end++;
      const run = lines.slice(k, end);
      if (run.length >= 3) {
        const [mark, ...rest] = run;
        mark.classList.add("wkh-fold-mark");
        mark.dataset.fold = `${run.length} unchanged lines`;
        mark.tabIndex = 0;
        mark.setAttribute("role", "button");
        mark.setAttribute("aria-label", `Show ${run.length} unchanged lines`);
        for (const line of rest) line.classList.add("wkh-folded");
        const unfold = () => {
          mark.classList.remove("wkh-fold-mark");
          mark.removeAttribute("role");
          mark.removeAttribute("tabindex");
          mark.removeAttribute("aria-label");
          for (const line of rest) line.classList.remove("wkh-folded");
        };
        mark.addEventListener("click", unfold);
        mark.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            unfold();
          }
        });
      }
      k = end;
    }
  }

  // Which lines of `next` aren't in `prev` (by the longest common subsequence of the two)
  function newLines(prev, next) {
    const m = prev.length;
    const n = next.length;
    const lcs = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        lcs[i][j] = prev[i] === next[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
      }
    }
    const added = new Array(n).fill(true);
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (prev[i] === next[j]) {
        added[j] = false;
        i++;
        j++;
      } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
        i++;
      } else {
        j++;
      }
    }
    return added;
  }

  // Panels, one shown at a time, chosen by tabs: numbered ones over the stepper (which marks each
  // step's new lines of code, shows the step's output before its code, and plays through the steps
  // while it's in view), and chips over the switch
  function setUpPanels(container, { panel, name, stepper, label }) {
    const panels = Array.from(container.querySelectorAll(`:scope > .${panel}`));
    if (panels.length === 0) return;

    const tabs = document.createElement("div");
    tabs.className = stepper ? "wkh-tabs" : "wkh-chips";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", label);

    const viewport = document.createElement("div");
    viewport.className = "wkh-panels";
    const stack = document.createElement("div");
    stack.className = "wkh-stack";
    viewport.appendChild(stack);

    const id = container.id || (stepper ? "wkh-stepper" : "wkh-switch");
    let previous = [];

    const buttons = panels.map((step, index) => {
      // The panel's name (its first paragraph) goes on its tab
      const nameEl = step.querySelector(`.${name}`);
      const title = nameEl ? nameEl.textContent.trim() : `Step ${index + 1}`;
      if (nameEl) {
        const para = nameEl.closest("p");
        (para && para.parentElement === step ? para : nameEl).remove();
      }

      const button = document.createElement("button");
      button.type = "button";
      button.id = `${id}-tab-${index}`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `${id}-panel-${index}`);
      if (stepper) {
        button.className = "wkh-tab";
        button.innerHTML = `<span class="wkh-tab-num">${index + 1}</span><span class="wkh-tab-name"></span><span class="wkh-tab-progress" aria-hidden="true"></span>`;
        button.querySelector(".wkh-tab-name").textContent = title;
      } else {
        button.className = "wkh-chip";
        button.textContent = title;
      }
      tabs.appendChild(button);

      step.id = `${id}-panel-${index}`;
      step.setAttribute("role", "tabpanel");
      step.setAttribute("aria-labelledby", button.id);
      step.classList.add("wkh-panel");

      // The code's lines, with those new to this step marked
      const cell = step.querySelector(".cell");
      if (stepper && cell) {
        const lines = codeLines(cell);
        const text = lines.map((line) => line.textContent.replace(/\s+$/, ""));
        const added = newLines(previous, text);
        let count = 0;
        lines.forEach((line, k) => {
          if (added[k] && text[k] !== "" && previous.length > 0) {
            line.classList.add("wkh-new");
            count++;
          }
        });
        if (previous.length > 0) foldUnchanged(lines, added);
        previous = text;
        const code = cell.querySelector(".cell-code");
        if (code && count > 0) {
          const badge = document.createElement("span");
          badge.className = "wkh-new-count";
          badge.textContent = `${count} new line${count === 1 ? "" : "s"}`;
          code.appendChild(badge);
        }
        // The output first, then the code that made it
        const output = cell.querySelector(".cell-output-display");
        if (code && output) cell.insertBefore(output, code);
      }

      stack.appendChild(step);
      return button;
    });

    container.prepend(tabs);
    container.appendChild(viewport);

    let active = -1;
    let observer = null;

    function fitHeight() {
      if (active < 0) return;
      viewport.style.height = `${panels[active].offsetHeight}px`;
    }

    function scrollToNew(step) {
      const pre = step.querySelector(".cell-code pre");
      const first = step.querySelector(".wkh-new");
      if (!pre || !first) return;
      const top = first.offsetTop - pre.offsetTop - 12;
      pre.scrollTo({ top: Math.max(0, top), behavior: reduceMotion.matches ? "auto" : "smooth" });
    }

    function show(index, focus = false) {
      if (index === active) return;
      active = index;
      panels.forEach((step, k) => {
        const on = k === index;
        step.classList.toggle("wkh-active", on);
        step.inert = !on;
        step.setAttribute("aria-hidden", on ? "false" : "true");
        buttons[k].setAttribute("aria-selected", on ? "true" : "false");
        buttons[k].tabIndex = on ? 0 : -1;
        if (stepper) buttons[k].classList.toggle("wkh-done", k < index);
      });
      if (focus) buttons[index].focus();
      if (observer) {
        observer.disconnect();
        observer.observe(panels[index]);
      }
      fitHeight();
      if (stepper) scrollToNew(panels[index]);
    }

    // Autoplay (for the stepper): through the steps while it's in view, until the reader takes over
    const DWELL = 5200;
    let timer = null;
    let playing = stepper && !reduceMotion.matches;
    let visible = false;

    function stopPlaying() {
      playing = false;
      container.classList.remove("wkh-playing");
      clearTimeout(timer);
      timer = null;
    }

    function schedule() {
      clearTimeout(timer);
      timer = null;
      if (!playing || !visible) {
        container.classList.remove("wkh-playing");
        return;
      }
      if (active >= panels.length - 1) {
        stopPlaying();
        return;
      }
      // (Restarting the tab's progress bar)
      container.classList.remove("wkh-playing");
      void container.offsetWidth;
      container.style.setProperty("--wkh-dwell", `${DWELL}ms`);
      container.classList.add("wkh-playing");
      timer = setTimeout(() => {
        show(active + 1);
        schedule();
      }, DWELL);
    }

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        stopPlaying();
        show(index);
      });
    });

    tabs.addEventListener("keydown", (event) => {
      const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      let next = null;
      if (event.key in keys) next = (active + keys[event.key] + panels.length) % panels.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = panels.length - 1;
      if (next === null) return;
      event.preventDefault();
      stopPlaying();
      show(next, true);
    });

    // Working with the output (or its code) stops the autoplay too
    viewport.addEventListener("pointerdown", stopPlaying);
    viewport.addEventListener("focusin", stopPlaying);

    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(fitHeight);
    }
    window.addEventListener("resize", fitHeight);

    container.classList.add("wkh-ready");
    show(0);

    if (stepper && "IntersectionObserver" in window) {
      new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            visible = entry.isIntersecting;
            schedule();
          }
        },
        // (In view: across the middle of the window, however tall the stepper is)
        { rootMargin: "-35% 0px -35% 0px" },
      ).observe(container);
    }
  }

  /* ── The stats, counting up the first time they're seen ─────────────────────────────────────── */

  function setUpStats(stats) {
    const numbers = Array.from(stats.querySelectorAll("[data-count]"));
    const format = (n) => Math.round(n).toLocaleString("en-US");
    if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
    for (const el of numbers) el.textContent = `0${el.dataset.suffix || ""}`;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        const begin = performance.now();
        const DURATION = 1500;
        const step = (now) => {
          const k = Math.min(1, (now - begin) / DURATION);
          const eased = 1 - Math.pow(1 - k, 3);
          for (const el of numbers) {
            el.textContent = `${format(Number(el.dataset.count) * eased)}${el.dataset.suffix || ""}`;
          }
          if (k < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      },
      { rootMargin: "0px 0px -15% 0px" },
    );
    observer.observe(stats);
  }

  /* ── The marquee of smooths ─────────────────────────────────────────────────────────────────── */

  function setUpMarquee(marquee) {
    for (const track of marquee.querySelectorAll(".wkh-marquee-track")) {
      // (A second copy of the tiles after the first, so the row loops without a seam)
      const copy = Array.from(track.children).map((tile) => {
        const clone = tile.cloneNode(true);
        clone.setAttribute("aria-hidden", "true");
        clone.tabIndex = -1;
        return clone;
      });
      track.append(...copy);
      const seconds = Math.max(40, track.scrollWidth / 2 / 38);
      track.style.setProperty("--wkh-marquee-time", `${seconds}s`);
    }
    marquee.classList.add("wkh-ready");
  }

  function init() {
    setUpCopyButtons();
    document.querySelectorAll(".wkh-stats").forEach(setUpStats);
    document.querySelectorAll(".wkh-marquee").forEach(setUpMarquee);
    document.querySelectorAll(".wkh-stepper").forEach((stepper) =>
      setUpPanels(stepper, {
        panel: "wkh-step",
        name: "wkh-step-name",
        stepper: true,
        label: "The steps of a model",
      }),
    );
    document.querySelectorAll(".wkh-switch").forEach((switcher) =>
      setUpPanels(switcher, {
        panel: "wkh-option",
        name: "wkh-option-name",
        stepper: false,
        label: "Four ways to draw a band",
      }),
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
