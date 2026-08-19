const fs = require("fs");
const path = require("path");

const playwrightModule = process.argv[4] || process.env.PLAYWRIGHT_MODULE || "playwright";
const { chromium } = require(playwrightModule);

async function main() {
  const baseUrl = process.argv[2] || "http://127.0.0.1:8765";
  const outputDir = path.resolve(process.argv[3] || "runs/ui_checks");
  fs.mkdirSync(outputDir, { recursive: true });
  const executablePath = process.argv[5] || process.env.CHROME_PATH;
  if (!executablePath) throw new Error("CHROME_PATH is required");

  const browser = await chromium.launch({ executablePath, headless: true });
  const contextA = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const contextB = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const pageA = await contextA.newPage();
  const pageB = await contextB.newPage();
  const errors = [];
  for (const page of [pageA, pageB]) {
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(String(error)));
  }

  await pageA.goto(baseUrl, { waitUntil: "networkidle" });
  await pageB.goto(baseUrl, { waitUntil: "networkidle" });
  const queueA = pageA.locator(".queue-item");
  const queueB = pageB.locator(".queue-item");
  await queueA.first().waitFor({ state: "visible" });
  await queueB.first().waitFor({ state: "visible" });
  if ((await queueA.count()) !== 1 || (await queueB.count()) !== 1) {
    throw new Error("verification project must contain exactly one image");
  }
  if (errors.length) throw new Error("browser console errors on load: " + errors.join(" | "));
  await queueA.first().click();
  await pageA.locator("#annotationCanvas").waitFor({ state: "visible" });

  await queueB.first().click();
  await pageB.locator("#toast.show.error").waitFor({ state: "visible" });
  const lockMessage = await pageB.locator("#toast").innerText();
  if (!lockMessage.includes("另一位队友")) {
    throw new Error(`second client did not receive lock conflict: ${lockMessage}`);
  }
  // The intentionally provoked HTTP 409 is logged by Chromium as a resource error.
  errors.length = 0;

  const canvas = pageA.locator("#annotationCanvas");
  const box = await canvas.boundingBox();
  if (!box || box.width < 100 || box.height < 100) throw new Error("annotation canvas is not visible");
  if ((await pageA.locator("#boxCount").innerText()) === "0") {
    await pageA.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.2);
    await pageA.mouse.down();
    await pageA.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.6, { steps: 8 });
    await pageA.mouse.up();
  }
  if ((await pageA.locator("#boxCount").innerText()) !== "1") {
    throw new Error("drawing did not create one annotation box");
  }
  await pageA.locator("#saveButton").click();
  await pageA.locator("#doneCount").waitFor({ state: "visible" });
  await pageA.waitForFunction(() => document.querySelector("#doneCount").textContent === "1");
  await pageA.waitForFunction(() => document.querySelector("#toast").className === "toast");
  await pageA.screenshot({ path: path.join(outputDir, "dataset_studio_1440x900.png"), fullPage: false });

  await pageA.setViewportSize({ width: 1280, height: 720 });
  await pageA.screenshot({ path: path.join(outputDir, "dataset_studio_1280x720.png"), fullPage: false });
  await pageA.setViewportSize({ width: 390, height: 844 });
  await pageA.screenshot({ path: path.join(outputDir, "dataset_studio_390x844.png"), fullPage: true });

  await pageA.setViewportSize({ width: 1440, height: 900 });
  await pageA.locator('[data-view="trainingView"]').click();
  await pageA.locator("#trainingView.active").waitFor({ state: "visible" });
  const baseModelCount = await pageA.locator("#baseModel option").count();
  if (baseModelCount < 1) throw new Error("no offline YOLO base model is listed");
  await pageA.screenshot({ path: path.join(outputDir, "dataset_studio_training.png"), fullPage: false });

  if (errors.length) throw new Error("browser console errors: " + errors.join(" | "));
  console.log(
    JSON.stringify(
      {
        ok: true,
        lockMessage,
        baseModelCount,
        screenshots: fs.readdirSync(outputDir).sort(),
      },
      null,
      2
    )
  );
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
