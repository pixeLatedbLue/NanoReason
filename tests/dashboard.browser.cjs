/* Browser regression checks for the local dashboard.
 * NODE_PATH must contain Playwright; no test dependency is shipped to the browser.
 * node tests/dashboard.browser.cjs [--live] [--url=http://127.0.0.1:8000]
 * --live sends one short request to the already-running model.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
const url =
  process.argv.find((a) => a.startsWith("--url="))?.slice(6) ||
  "http://127.0.0.1:8000";
const out = path.resolve("artifacts/slm/dashboard");
fs.mkdirSync(out, { recursive: true });
const report = {
  url,
  checkedAt: new Date().toISOString(),
  checks: [],
  live: null,
};
const check = (name) => {
  report.checks.push(name);
  console.log("PASS", name);
};
const settle = async (p) => {
  await p.evaluate(() => document.fonts.ready);
  await p.waitForFunction(
    () =>
      !window.gsap ||
      !gsap.globalTimeline.getChildren().some((t) => t.isActive()),
  );
};
(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      colorScheme: "light",
      permissions: ["clipboard-read", "clipboard-write"],
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(
      () =>
        !document.querySelector("#runs-refresh").disabled &&
        !document
          .querySelector("#engine-badge")
          .textContent.includes("Connecting"),
    );
    await settle(page);
    assert.equal(await page.title(), "NanoReason · Solve");
    assert.equal(await page.locator("#response-empty").isVisible(), true);
    await page.screenshot({
      path: path.join(out, "solve-desktop.png"),
      fullPage: true,
    });
    check("Initial empty state, self-hosted assets, and motion completion");

    await page.locator("#tab-solve").focus();
    await page.keyboard.press("ArrowDown");
    assert.equal(
      await page.locator("#tab-verify").getAttribute("aria-selected"),
      "true",
    );
    assert.equal(
      await page
        .locator("#tab-verify")
        .evaluate((e) => e === document.activeElement),
      true,
    );
    await page.click("#verify-btn");
    await page.locator("#verify-analysis .ans__value").waitFor();
    assert.equal(await page.locator("#verify-analysis .row--bad").count(), 1);
    assert.equal(
      await page.locator("#verify-analysis .ans__value").textContent(),
      "213",
    );
    assert.match(
      await page.locator("#verify-analysis").textContent(),
      /Does not match gold/,
    );
    await settle(page);
    await page.screenshot({
      path: path.join(out, "verify-desktop.png"),
      fullPage: true,
    });
    check(
      "Keyboard tabs and real arithmetic verification identify the deliberate mistake",
    );

    await page.fill("#verify-gold", "");
    assert.equal(await page.locator("#verify-analysis .ans__value").count(), 0);
    await page.click("#verify-btn");
    await page.locator("#verify-analysis .ans__value").waitFor();
    assert.match(
      await page.locator("#verify-analysis").textContent(),
      /correctness is unscored/,
    );
    check(
      "Input edits invalidate stale verification and missing reference is explicit",
    );

    await page.route("**/api/analyze", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: '{"detail":"Test service unavailable"}',
      }),
    );
    await page.click("#verify-btn");
    await page.waitForFunction(
      () => document.querySelector("#verify-status").textContent === "failed",
    );
    assert.match(
      await page.locator("#verify-analysis").textContent(),
      /Verification unavailable/,
    );
    await page.unroute("**/api/analyze");
    await page.click("#verify-sample");
    await page.click("#verify-btn");
    await page.locator("#verify-analysis .ans__value").waitFor();
    check("Verification HTTP failure is recoverable");

    await page.click("#tab-runs");
    await page.waitForFunction(() => location.hash === "#runs");
    const runs = (await (await page.request.get(url + "/api/runs")).json())
      .runs;
    const chartExpected = runs
      .flatMap((r) =>
        Object.values(r.tasks || {}).map(
          (s) => (s.accuracy * 100).toFixed(1) + "%",
        ),
      )
      .sort();
    assert.deepEqual(
      (await page.locator(".chart-value").allTextContents()).sort(),
      chartExpected,
    );
    await page.click("#cmp-btn");
    await page.locator("#compare-host table").waitFor();
    await settle(page);
    await page.screenshot({
      path: path.join(out, "runs-desktop.png"),
      fullPage: true,
    });
    const baseline = await page.inputValue("#cmp-baseline"),
      candidate = await page.inputValue("#cmp-candidate");
    const compare = await (
      await page.request.get(
        url + "/api/compare?baseline=" + baseline + "&candidate=" + candidate,
      )
    ).json();
    for (const row of compare.rows) {
      if (row.paired && !row.paired.significant)
        assert.match(
          await page.locator("#compare-host").textContent(),
          /difference has not been established/,
        );
    }
    check(
      "Charts match saved API measurements and paired interpretation is honest",
    );

    await page.click(".run-records summary");
    const downloadEvent = page.waitForEvent("download");
    await page.locator(".run-record a").first().click();
    const download = await downloadEvent;
    assert.equal(await download.failure(), null);
    assert.ok(JSON.parse(fs.readFileSync(await download.path(), "utf8")));
    check("Source artifact JSON downloads");

    await page.selectOption("#cmp-candidate", baseline);
    await page.click("#cmp-btn");
    assert.equal(await page.locator("#compare-host table").count(), 0);
    assert.match(
      await page.locator("#toasts").textContent(),
      /Choose different runs/,
    );
    await page.selectOption("#cmp-candidate", candidate);
    await page.click("#cmp-btn");
    await page.locator("#compare-host table").waitFor();

    await page.click("#theme-toggle");
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await settle(page);
    await page.screenshot({
      path: path.join(out, "runs-dark.png"),
      fullPage: true,
    });
    await page.reload();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    assert.equal(await page.locator("#panel-runs").isVisible(), true);
    await page.click("#theme-toggle");
    check("Theme preference and tab deep link survive reload");

    for (const width of [1440, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ["solve", "verify", "runs"]) {
        await page.click("#tab-" + tab);
        await settle(page);
        assert.ok(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
          "Horizontal overflow on " + tab + " at " + width,
        );
        if (width === 390)
          await page.screenshot({
            path: path.join(out, tab + "-mobile.png"),
            fullPage: true,
          });
      }
    }
    check("All three screens fit 1440, 1024, 768, 390 and 320 px widths");

    await page.route("**/api/runs", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: '{"runs":[]}',
      }),
    );
    await page.click("#runs-refresh");
    await page.waitForFunction(() =>
      document
        .querySelector("#runs-host")
        .textContent.includes("No evaluation"),
    );
    assert.equal(await page.locator("#cmp-btn").isDisabled(), true);
    await page.unroute("**/api/runs");
    await page.route("**/api/runs", (route) =>
      route.fulfill({ status: 500, body: "Test error" }),
    );
    await page.click("#runs-refresh");
    await page.waitForFunction(() =>
      document
        .querySelector("#runs-host")
        .textContent.includes("Could not load"),
    );
    assert.equal(await page.locator(".chart-value").count(), 0);
    await page.unroute("**/api/runs");
    await page.click("#runs-refresh");
    await page.locator(".chart-value").first().waitFor();
    check("Empty results and network failures are distinct; refresh recovers");

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.click("#tab-solve");
    await page.locator(".ex").first().click();
    await page.route("**/api/solve/stream?*", (route) =>
      route.fulfill({
        contentType: "text/event-stream",
        body: 'data: {"type":"token","text":"Partial test output"}\n\ndata: {"type":"error","message":"Test generation failure"}\n\ndata: {"type":"done","failed":true,"completion":"Partial test output","analysis":null}\n\n',
      }),
    );
    await page.click("#solve-btn");
    await page.waitForFunction(() =>
      document
        .querySelector("#solve-status")
        .textContent.includes("partial output"),
    );
    assert.equal(await page.locator("#solve-btn").isDisabled(), false);
    assert.equal(await page.locator("#copy-response").isDisabled(), false);
    await page.click("#copy-response");
    assert.equal(
      await page.evaluate(() => navigator.clipboard.readText()),
      "Partial test output",
    );
    await page.unroute("**/api/solve/stream?*");
    await page.route("**/api/solve/stream?*", (route) =>
      route.fulfill({
        contentType: "text/event-stream",
        body: 'data: {"type":"token","text":"Interrupted test"}\n\n',
      }),
    );
    await page.click("#solve-btn");
    await page.waitForFunction(
      () =>
        document.querySelector("#solve-status").textContent === "interrupted",
    );
    await page.unroute("**/api/solve/stream?*");
    check(
      "Failed and interrupted streams retain partial output without claiming success; copy works",
    );

    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.click("#tab-verify");
    assert.equal(
      await page.evaluate(
        () => matchMedia("(prefers-reduced-motion: reduce)").matches,
      ),
      true,
    );
    assert.equal(
      await page
        .locator("#panel-verify")
        .evaluate((e) => getComputedStyle(e).opacity),
      "1",
    );
    check("Reduced-motion preference disables reveal animation");

    if (process.argv.includes("--live")) {
      await page.click("#tab-solve");
      const health = await (await page.request.get(url + "/api/health")).json();
      assert.equal(
        health.engine.is_demo,
        false,
        "Live smoke must use a real model",
      );
      await page.fill(
        "#solve-input",
        "A shop has 20 apples and receives 15 more. How many apples does it have now?",
      );
      await page.fill("#solve-gold", "35");
      await page.fill("#solve-tokens", "128");
      await page.locator("#solve-input").press("Control+Enter");
      await page.waitForFunction(
        () =>
          !document.querySelector("#solve-btn").disabled &&
          document
            .querySelector("#solve-status")
            .textContent.includes("Completed"),
        null,
        { timeout: 120000 },
      );
      assert.ok(
        (await page.locator("#solve-output").textContent()).trim().length > 0,
      );
      report.live = {
        engine: health.engine,
        status: await page.locator("#solve-status").textContent(),
        answer: await page.locator("#solve-analysis .ans__value").textContent(),
        completion: await page.locator("#solve-output").textContent(),
      };
      await page.screenshot({
        path: path.join(out, "solve-live.png"),
        fullPage: true,
      });
      check(
        "One real model inference through the keyboard shortcut, streaming UI and final analysis",
      );
    }
    assert.deepEqual(errors, []);
    check("No uncaught browser JavaScript errors");
    fs.writeFileSync(
      path.join(out, "browser-checks.json"),
      JSON.stringify(report, null, 2),
    );
    await context.close();
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
