/**
 * random.js - deterministic pseudo-random helpers.
 *
 * Keep stochastic models reproducible without relying on the global RNG.
 * @module random
 */

/**
 * Create a deterministic Mulberry32 pseudo-random generator.
 * @param {number} seed
 * @returns {() => number} Function returning values in [0, 1)
 */
export function createSeededRandom(seed = 42) {
    let s = Number.isFinite(Number(seed)) ? Number(seed) | 0 : 42;
    return function random() {
        s = (s + 0x6d2b79f5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Pick an integer in [0, max).
 * @param {() => number} rng
 * @param {number} max
 * @returns {number}
 */
export function randomInt(rng, max) {
    return Math.floor(rng() * max);
}

/**
 * Return a shuffled copy using Fisher-Yates.
 * @param {Array} values
 * @param {() => number} rng
 * @returns {Array}
 */
export function shuffleCopy(values, rng) {
    const result = [...values];
    for (let i = result.length - 1; i > 0; i--) {
        const j = randomInt(rng, i + 1);
        [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
}
