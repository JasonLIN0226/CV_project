const { chromium } = require('/Users/xintongyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

(async () => {
  const output = path.resolve('results/literature_diagnosis/visual-casebook-20261022');
  const screenshots = path.resolve('build/literature-diagnosis-20261022/visual-casebook-render');
  fs.mkdirSync(screenshots, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport: {width: 1360, height: 1000}, deviceScaleFactor: 1});
  const errors = [];
  const remote = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('request', r => { if (/^https?:/.test(r.url())) remote.push(r.url()); });
  await page.goto(pathToFileURL(path.join(output, 'index.html')).href, {waitUntil: 'load'});
  const cards = JSON.parse(fs.readFileSync(path.join(output, 'cases.json'), 'utf8'));
  if (await page.locator('.case').count() !== 3) throw new Error('Expected exactly 3 cases');
  const checks = [];
  for (const [kind, c] of Object.entries(cards)) {
    const root = page.locator(`#${kind}`);
    if (await root.locator('tbody tr').count() !== 4) throw new Error('Missing displayed numeric conditions');
    for (const t of [c.cue_time, c.query, c.query+1, c.query+3, c.query+6]) {
      await root.locator(`button[data-time="${t}"]`).click();
      await root.locator('.actual').evaluate(img => img.decode());
      await root.locator('.sibling').evaluate(img => img.decode());
      const state = await root.locator('.player').evaluate(p => ({time: Number(p.dataset.time),
        width: p.querySelector('.actual').naturalWidth, same: p.querySelector('.actual').src === p.querySelector('.sibling').src,
        stage: p.querySelector('.stage').textContent}));
      if (state.time !== t || state.width !== 96) throw new Error('Frame jump did not resolve native RGB');
      if (state.same !== (t === c.query)) throw new Error('Paired cue/query/future relationship wrong');
      checks.push({case: kind, time: t, ...state});
    }
    await root.locator(`button[data-time="${c.cue_time}"]`).click();
    await root.screenshot({path: path.join(screenshots, `${kind}-desktop.png`)});
  }
  const first = page.locator('#content_harm');
  await first.locator('.play').click();
  await page.waitForTimeout(260);
  const playAdvanced = await first.locator('.player').evaluate(p => Number(p.dataset.time)) !== cards.content_harm.cue_time;
  await first.locator('.play').click();
  if (!playAdvanced) throw new Error('Playback did not advance');
  await page.setViewportSize({width: 390, height: 844});
  await page.evaluate(() => scrollTo(0, 0));
  const noHorizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
  if (!noHorizontalOverflow) throw new Error('Mobile body overflows viewport');
  await page.screenshot({path: path.join(screenshots, 'mobile-intro.png')});
  await first.locator(`button[data-time="${cards.content_harm.cue_time}"]`).click();
  await first.screenshot({path: path.join(screenshots, 'content_harm-mobile.png')});
  if (errors.length || remote.length) throw new Error(JSON.stringify({errors, remote}));
  fs.writeFileSync(path.join(output, 'render_audit.json'), JSON.stringify({status: 'passed',
    browser: await browser.version(), javascript_errors: errors, remote_requests: remote,
    frame_jump_checks: checks, playback_advanced: playAdvanced, mobile_width: 390,
    mobile_body_no_horizontal_overflow: noHorizontalOverflow, displayed_numeric_rows: 12,
    screenshots_directory: path.relative(process.cwd(), screenshots)}, null, 2)+'\n');
  console.log(JSON.stringify({status: 'passed', frame_jump_checks: checks.length, playback_advanced: playAdvanced,
    javascript_errors: errors.length, remote_requests: remote.length, mobile_body_no_horizontal_overflow: noHorizontalOverflow}));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
