/**
 * Path-traced frames drawn off screen.
 *
 * The tracer is loaded on demand. One frame is accumulated a tile at a time until it holds the
 * sample count asked for, and the count is only whole between passes.
 */

// The vendored build, loaded the first time a frame is traced rather than with the pack.
const TRACER_MODULE = "../vendor/pathtracer/index.module.js";

// Tiles traced between yields to the browser, so a long frame leaves the tab answerable.
const TILES_PER_YIELD = 24;

let pending = null;

/**
 * Load the path tracer.
 *
 * @returns {Promise<object>} The tracer module's exports.
 */
export function loadPathTracer() {
    if (!pending) pending = import(TRACER_MODULE);
    return pending;
}

/**
 * Build a tracer set up to draw frames nobody is watching.
 *
 * @param {THREE.WebGLRenderer} renderer - The renderer to trace with.
 * @param {object} settings - `{bounces, transmissiveBounces, filterGlossy, tiles, textureSize,
 *   multipleImportanceSampling}`, each optional.
 * @returns {Promise<object>} The tracer.
 */
export async function createPathTracer(renderer, settings = {}) {
    const { WebGLPathTracer } = await loadPathTracer();
    const tracer = new WebGLPathTracer(renderer);

    // Everything that eases a traced picture into an interactive preview is turned off, since
    // each of them would otherwise land in a captured frame: a wall-clock delay before the
    // first sample, an opacity fade over the half-finished result, and a sample floor below
    // which the preview shows something else.
    tracer.renderDelay = 0;
    tracer.fadeDuration = 0;
    tracer.minSamples = 0;
    tracer.dynamicLowRes = false;
    tracer.renderScale = 1;
    tracer.synchronizeRenderSize = true;
    tracer.renderToCanvas = true;

    // A frame short of its samples stays black rather than falling back to an ordinary draw,
    // which would be captured and read as a traced picture.
    tracer.rasterizeScene = false;

    // The same frame traced twice comes out the same both times.
    tracer.stableNoise = true;

    tracer.bounces = Math.max(1, Math.floor(settings.bounces ?? 5));
    tracer.transmissiveBounces = Math.max(0, Math.floor(settings.transmissiveBounces ?? 10));
    tracer.filterGlossyFactor = Math.max(0, Number(settings.filterGlossy) || 0);
    tracer.multipleImportanceSampling = settings.multipleImportanceSampling !== false;

    const tiles = Math.max(1, Math.floor(settings.tiles ?? 3));
    tracer.tiles.set(tiles, tiles);

    const texture = Math.max(16, Math.floor(settings.textureSize ?? 1024));
    tracer.textureSize.set(texture, texture);

    return tracer;
}

/**
 * Hand the tracer the pose to trace, and clear what it had accumulated.
 *
 * @param {object} tracer - The tracer.
 * @param {THREE.Scene} scene - The scene at this moment.
 * @param {THREE.Camera} camera - The camera at this moment.
 * @param {boolean} rebuild - Whether the geometry moved and its hierarchy needs rebuilding.
 * @returns {void}
 */
export function prepareFrame(tracer, scene, camera, rebuild) {
    if (rebuild) {
        tracer.setScene(scene, camera);
        return;
    }
    tracer.setCamera(camera);
}

/**
 * Accumulate one frame.
 *
 * @param {object} tracer - The tracer, already given a pose.
 * @param {number} samples - Samples per pixel to reach.
 * @param {number} patience - Milliseconds the frame is given.
 * @param {Function} [onSample] - Called with the samples standing, as they accumulate.
 * @returns {Promise<number>} How many tiles were traced.
 *
 * @throws {Error} The frame did not reach its sample count within its patience.
 */
export async function traceFrame(tracer, samples, patience, onSample) {
    const target = Math.max(1, Math.floor(samples));
    const deadline = performance.now() + Math.max(1000, patience);
    let drawn = 0;

    // A count is fractional part way through a pass, and stopping there leaves the tiles
    // holding different numbers of samples, so the frame runs on to the end of its pass.
    while (tracer.samples < target || !Number.isInteger(tracer.samples)) {
        tracer.renderSample();
        if (performance.now() > deadline) {
            throw new Error(
                `Path tracing reached ${tracer.samples} of ${target} samples per pixel before `
                + `the ${Math.round(patience / 1000)}s timeout ran out. Raise timeout, or `
                + `lower samples, bounces, width and height.`
            );
        }
        // No sample lands while the shader compiles, so the loop waits that out on the
        // browser's own clock rather than spinning.
        if (tracer.isCompiling) {
            await new Promise(requestAnimationFrame);
            continue;
        }
        drawn += 1;
        if (drawn % TILES_PER_YIELD === 0) {
            onSample?.(tracer.samples);
            await new Promise(requestAnimationFrame);
        }
    }
    return drawn;
}
