/**
 * The three arithmetic rules Python follows and JavaScript does not.
 *
 * `roundHalfEven` for Python's `round()`, `truncate` for its `int()`, and `floorMod` for its
 * `%`. Every function takes and answers a plain number.
 */

/**
 * Round half to even, the rule Python's `round()` follows.
 *
 * 2.5 answers 2 and 3.5 answers 4, where `Math.round` answers 3 and 4.
 *
 * @param {number} value - Value to round.
 * @returns {number} The value rounded to a whole number.
 */
export function roundHalfEven(value) {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Drop the fraction toward zero, the rule Python's `int()` follows.
 *
 * It differs from a floor below zero: `int(-0.5)` is 0 and `Math.floor(-0.5)` is -1.
 *
 * @param {number} value - Value to truncate.
 * @returns {number} The whole part, with the sign kept.
 */
export function truncate(value) {
  return Math.trunc(value);
}

/**
 * The remainder of a division, with the sign of the divisor, as Python's and numpy's `%` give it.
 *
 * @param {number} value - Value to reduce.
 * @param {number} modulus - Positive modulus.
 * @returns {number} The remainder, 0 up to the modulus.
 */
export function floorMod(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}
