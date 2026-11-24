// main.ts — mend, a command line tool that applies AI-authored edits to files.
//
// mend understands four patch notations — unified diffs, SEARCH/REPLACE blocks,
// V4A patches and JSON tool calls — as pinned down by the SPEC.md that ships with
// the kit. Each edit is resolved against the file it names at the first of three
// matching tiers that finds it, and a patch changes every file it touches or
// none of them. Node 22+, standard library only.
//
// Usage: node main.ts [options] [PATCH]
//
// When MAKEFUN_ENV_URL is set (a grading trial), control passes to runner.ts at
// the bottom of this file; runner.ts belongs to the kit and stays untouched.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const KNOWN_FORMATS: readonly string[] = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** A patch that cannot be parsed; the message is the whole stdout line (exit 2). */
class ParseError {
  readonly text: string;
  constructor(text: string) {
    this.text = text;
  }
}

/** One edit that could not be applied: a failure code plus the path it blames. */
class ApplyError {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const isBlankLine = (s: string): boolean => s.trim() === '';

/** Split patch text into lines, tolerating CRLF input by dropping a trailing CR. */
function splitPatchText(text: string): string[] {
  if (text === '') return [];
  const rows = text.split('\n');
  if (text.endsWith('\n')) rows.pop();
  return rows.map((r) => (r.endsWith('\r') ? r.slice(0, -1) : r));
}

/** A path mend may touch: relative, forward slashes, no empty/dot/dotdot segments. */
function pathOk(p: string): boolean {
  if (p === '' || p.startsWith('/') || p.includes('\\')) return false;
  return p.split('/').every((seg) => seg !== '' && seg !== '.' && seg !== '..');
}

/**
 * The working tree as a patch leaves it. `pending` holds every path an edit has
 * decided (string content, or null for deleted); any other path is read from
 * disk once and cached in `diskCache`.
 */
class Overlay {
  private readonly diskCache = new Map<string, string | null>();
  readonly pending = new Map<string, string | null>();

  onDisk(p: string): string | null {
    if (!this.diskCache.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.diskCache.set(p, v);
    }
    return this.diskCache.get(p)!;
  }

  current(p: string): string | null {
    return this.pending.has(p) ? this.pending.get(p)! : this.onDisk(p);
  }
}

type ApplyFn = (tree: Overlay) => void;
/** A rename an edit performed, for the `R old -> new` report line (SPEC.md §2). */
type Rename = { from: string; to: string };

/** Guess a patch's notation from its lines; null when nothing recognisable. */
function sniffFormat(rows: string[]): string | null {
  const first = rows.findIndex((r) => !isBlankLine(r));
  if (first < 0) return null;
  if (rows[first]!.trim() === '*** Begin Patch') return 'v4a';
  const lead = rows[first]!.trimStart().charAt(0);
  if (lead === '[' || lead === '{') return 'json';
  if (rows.some((r) => r.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  const looksUnified = rows.some(
    (r, i) => r.startsWith('diff --git ') || (r.startsWith('--- ') && (rows[i + 1] ?? '').startsWith('+++ ')),
  );
  return looksUnified ? 'unified' : null;
}

/* ------------------------------------------------------------------ */
/* Files as an array of lines plus their terminator (SPEC.md §3)       */
/* ------------------------------------------------------------------ */

interface LineDoc {
  rows: string[];
  /** The line terminator the file uses: '\r\n' or '\n'. */
  eol: string;
  /** Whether the file's last line is terminated. */
  terminated: boolean;
}

function docFromText(text: string): LineDoc {
  if (text === '') return { rows: [], eol: '\n', terminated: false };
  const nl = text.indexOf('\n');
  const eol = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const rows = text.split(eol);
  const terminated = text.endsWith(eol);
  if (terminated) rows.pop();
  return { rows, eol, terminated };
}

function textFromDoc(doc: LineDoc): string {
  if (doc.rows.length === 0) return '';
  return doc.rows.join(doc.eol) + (doc.terminated ? doc.eol : '');
}

function indentOf(s: string): string {
  const m = /^[ \t]*/.exec(s);
  return m === null ? '' : m[0];
}

/* ------------------------------------------------------------------ */
/* Finding a block of lines at three tiers (SPEC.md §5)                */
/* ------------------------------------------------------------------ */

/** A line as a tier sees it: exact, ignoring trailing blanks, ignoring both. */
function foldLine(s: string, tier: number): string {
  if (tier === 0) return s;
  const noTrail = s.replace(/[ \t]+$/, '');
  return tier === 1 ? noTrail : noTrail.replace(/^[ \t]+/, '');
}

function blockMatchesAt(rows: string[], probe: string[], tier: number, at: number): boolean {
  for (let k = 0; k < probe.length; k++) {
    if (foldLine(rows[at + k]!, tier) !== foldLine(probe[k]!, tier)) return false;
  }
  return true;
}

/** Every 0-based start where `probe` matches `rows` at `tier`, at or after `from`. */
function findMatches(rows: string[], probe: string[], tier: number, from: number): number[] {
  const hits: number[] = [];
  if (probe.length === 0) return hits;
  for (let i = from; i + probe.length <= rows.length; i++) {
    if (blockMatchesAt(rows, probe, tier, i)) hits.push(i);
  }
  return hits;
}

/** The first tier with a match at or after `from`, with its positions, or null. */
function locateBlock(
  rows: string[],
  probe: string[],
  strict: boolean,
  from: number,
): { tier: number; hits: number[] } | null {
  for (const tier of strict ? [0] : [0, 1, 2]) {
    const hits = findMatches(rows, probe, tier, from);
    if (hits.length > 0) return { tier, hits };
  }
  return null;
}

/**
 * Replace the `probe` rows at `at` with `fresh`. `freshFrom[i]` is the probe
 * index fresh[i] copies when it is a context row (the file's own matched line
 * is written for those), or null for a genuinely new row. At tier 2 the new
 * rows are re-indented: the probe's first-row indent is swapped for the file's.
 */
function splice(
  rows: string[],
  at: number,
  probe: string[],
  fresh: string[],
  freshFrom: (number | null)[],
  tier: number,
): string[] {
  const reindent = tier === 2 && probe.length > 0;
  const fileIndent = reindent ? indentOf(rows[at]!) : '';
  const probeIndent = reindent ? indentOf(probe[0]!) : '';
  const out = rows.slice(0, at);
  for (let i = 0; i < fresh.length; i++) {
    const src = freshFrom[i];
    if (src !== null && src !== undefined) {
      out.push(rows[at + src]!);
      continue;
    }
    const row = fresh[i]!;
    if (reindent && row !== '' && row.trim() !== '' && row.startsWith(probeIndent)) {
      out.push(fileIndent + row.slice(probeIndent.length));
    } else {
      out.push(row);
    }
  }
  out.push(...rows.slice(at + probe.length));
  return out;
}

/* ------------------------------------------------------------------ */
/* JSON tool calls (SPEC.md §9)                                        */
/* ------------------------------------------------------------------ */

/** The current content of a path an edit needs to exist (unsafe/missing guards). */
function requireExisting(tree: Overlay, p: string): string {
  if (!pathOk(p)) throw new ApplyError('unsafe', p);
  const t = tree.current(p);
  if (t === null) throw new ApplyError('missing', p);
  return t;
}

/** Turn one decoded JSON value into an edit, or return null when it is not one. */
function jsonCommandToFn(v: unknown): ApplyFn | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;

  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!pathOk(p)) throw new ApplyError('unsafe', p);
      if (tree.current(p) !== null) throw new ApplyError('exists', p);
      tree.pending.set(p, text);
    };
  }

  if (
    o.command === 'str_replace' &&
    typeof o.old_str === 'string' &&
    o.old_str !== '' &&
    (o.new_str === undefined || typeof o.new_str === 'string')
  ) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = requireExisting(tree, p);
      // Work in LF space when the file is CRLF, then convert the result back.
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const toLf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const hay = toLf(before);
      const needle = toLf(oldStr);
      let count = 0;
      for (let k = hay.indexOf(needle); k >= 0; k = hay.indexOf(needle, k + 1)) count += 1;
      if (count === 0) throw new ApplyError('no-match', p);
      if (count > 1) throw new ApplyError('ambiguous', p);
      const at = hay.indexOf(needle);
      const joined = hay.slice(0, at) + toLf(newStr) + hay.slice(at + needle.length);
      tree.pending.set(p, crlf ? joined.replaceAll('\n', '\r\n') : joined);
    };
  }

  if (
    o.command === 'insert' &&
    typeof o.insert_line === 'number' &&
    Number.isInteger(o.insert_line) &&
    typeof o.new_str === 'string'
  ) {
    const line = o.insert_line;
    const add = splitPatchText(o.new_str);
    return (tree) => {
      const doc = docFromText(requireExisting(tree, p));
      if (line < 0 || line > doc.rows.length) throw new ApplyError('no-match', p);
      const rows = [...doc.rows.slice(0, line), ...add, ...doc.rows.slice(line)];
      tree.pending.set(p, textFromDoc({ rows, eol: doc.eol, terminated: doc.terminated }));
    };
  }
  return null;
}

/** Parse a JSON patch: one array of edits, or one edit object per non-blank line. */
function parseJsonPatch(text: string, rows: string[]): ApplyFn[] {
  const first = rows.findIndex((r) => !isBlankLine(r));
  if (first < 0) throw new ParseError('error: parse 1');
  if (rows[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new ParseError(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new ParseError(`error: parse ${first + 1}`);
    return v.map((x) => {
      const fn = jsonCommandToFn(x);
      if (fn === null) throw new ParseError(`error: parse ${first + 1}`);
      return fn;
    });
  }
  const fns: ApplyFn[] = [];
  rows.forEach((r, i) => {
    if (isBlankLine(r)) return;
    let fn: ApplyFn | null = null;
    try {
      fn = jsonCommandToFn(JSON.parse(r));
    } catch {
      fn = null;
    }
    if (fn === null) throw new ParseError(`error: parse ${i + 1}`);
    fns.push(fn);
  });
  return fns;
}

/* ------------------------------------------------------------------ */
/* Unified diffs (SPEC.md §6)                                          */
/* ------------------------------------------------------------------ */

interface HunkRec {
  /** The @@ header's old-side start line (1-based). */
  start: number;
  oldRows: string[];
  newRows: string[];
  /** For each new row: the old-row index it copies (context), or null (addition). */
  newFrom: (number | null)[];
  /** A `\` marker follows the hunk's last context or added row. */
  eolMark: boolean;
  /** A `\` marker follows the hunk's last added row. */
  eolMarkPlus: boolean;
}

interface SectionRec {
  diffFrom: string | undefined;
  diffTo: string | undefined;
  pairFrom: string | undefined;
  pairTo: string | undefined;
  renameFrom: string | undefined;
  renameTo: string | undefined;
  isCreate: boolean;
  isDelete: boolean;
  hunks: HunkRec[];
}

/** The path named by a `--- `/`+++ ` line: up to a tab, trimmed, a/ or b/ stripped. */
function headerPath(raw: string): string {
  let p = raw.slice(4).split('\t')[0]!.trim();
  if (p.startsWith('a/') || p.startsWith('b/')) p = p.slice(2);
  return p;
}

/** Read the hunk whose header is rows[h]; returns the hunk and the next unread index. */
function readHunk(rows: string[], h: number): { hunk: HunkRec; next: number } {
  const m = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(rows[h]!);
  if (m === null) throw new ParseError(`error: parse ${h + 1}`);
  const body: string[] = [];
  let j = h + 1;
  while (j < rows.length) {
    const r = rows[j]!;
    if (r.startsWith('diff --git ')) break;
    if (r.startsWith('--- ') && (rows[j + 1] ?? '').startsWith('+++ ')) break;
    if (r === '' || r[0] === ' ' || r[0] === '-' || r[0] === '+' || r[0] === '\\') {
      body.push(r);
      j++;
    } else break;
  }
  // Empty rows at the end of a body are not part of the hunk.
  while (body.length > 0) {
    const last = body[body.length - 1]!;
    if (last === '' || last === ' ') body.pop();
    else break;
  }
  const oldRows: string[] = [];
  const newRows: string[] = [];
  const newFrom: (number | null)[] = [];
  for (const r of body) {
    if (r === '' || r[0] === ' ') {
      const t = r === '' ? '' : r.slice(1);
      oldRows.push(t);
      newRows.push(t);
      newFrom.push(oldRows.length - 1);
    } else if (r[0] === '-') {
      oldRows.push(r.slice(1));
    } else if (r[0] === '+') {
      newRows.push(r.slice(1));
      newFrom.push(null);
    }
    // `\` rows matter only for the eolMark flags below.
  }
  let lastNew = -1;
  let lastPlus = -1;
  body.forEach((r, k) => {
    if (r === '' || r[0] === ' ' || r[0] === '+') lastNew = k;
    if (r[0] === '+') lastPlus = k;
  });
  const marked = (idx: number): boolean =>
    idx >= 0 && idx + 1 < body.length && body[idx + 1]![0] === '\\';
  return {
    hunk: {
      start: Number(m[1]),
      oldRows,
      newRows,
      newFrom,
      eolMark: marked(lastNew),
      eolMarkPlus: marked(lastPlus),
    },
    next: j,
  };
}

/** Run one section's hunks over a file's text, in order; ApplyError when one fails. */
function applyHunks(text: string, hunks: HunkRec[], strict: boolean, path: string): string {
  let doc = docFromText(text);
  let drift = 0;
  for (const hk of hunks) {
    if (hk.oldRows.length === 0) {
      // Pure insertion: the header position shifted by earlier hunks, clamped.
      const at = Math.min(Math.max(hk.start + drift, 0), doc.rows.length);
      const rows = [...doc.rows.slice(0, at), ...hk.newRows, ...doc.rows.slice(at)];
      let terminated = doc.terminated;
      if (at === doc.rows.length && hk.newRows.length > 0) terminated = !hk.eolMark;
      doc = { rows, eol: doc.eol, terminated };
      drift += hk.newRows.length;
      continue;
    }
    const found = locateBlock(doc.rows, hk.oldRows, strict, 0);
    if (found === null) throw new ApplyError('no-match', path);
    // Of the occurrences at the winning tier, the one nearest the header position.
    const target = hk.start - 1 + drift;
    let best = found.hits[0]!;
    for (const p of found.hits) {
      if (Math.abs(p - target) < Math.abs(best - target)) best = p;
    }
    const rows = splice(doc.rows, best, hk.oldRows, hk.newRows, hk.newFrom, found.tier);
    let terminated = doc.terminated;
    if (best + hk.oldRows.length === doc.rows.length) {
      terminated = hk.newRows.length > 0 ? !hk.eolMark : rows.length > 0;
    }
    doc = { rows, eol: doc.eol, terminated };
    drift += hk.newRows.length - hk.oldRows.length;
  }
  return textFromDoc(doc);
}

/** Which paths a section acts on: ---/+++ pair first, then rename from/to, then diff --git. */
function sectionTargets(s: SectionRec): { from: string | null; to: string | null } {
  const pick = (pair: string | undefined, rename: string | undefined, diff: string | undefined): string | null => {
    const v = pair !== undefined ? pair : rename !== undefined ? rename : diff;
    return v === undefined || v === '/dev/null' ? null : v;
  };
  return { from: pick(s.pairFrom, s.renameFrom, s.diffFrom), to: pick(s.pairTo, s.renameTo, s.diffTo) };
}

function sectionToFn(s: SectionRec, strict: boolean, renames: Rename[]): ApplyFn {
  const { from, to } = sectionTargets(s);
  const create = from === null || s.isCreate;
  const remove = !create && (to === null || s.isDelete);
  const move = !create && !remove && from !== null && to !== null && from !== to;
  return (tree) => {
    if (from !== null && !pathOk(from)) throw new ApplyError('unsafe', from);
    if (to !== null && !pathOk(to)) throw new ApplyError('unsafe', to);
    if (create) {
      const np = to!;
      if (tree.current(np) !== null) throw new ApplyError('exists', np);
      // A created file is the hunks' added rows, in order; its final break is
      // set by the last hunk that adds anything, via that hunk's `\` marker.
      const made: string[] = [];
      let terminated = true;
      let anyAdded = false;
      for (const hk of s.hunks) {
        let hunkAdded = false;
        hk.newRows.forEach((row, i) => {
          if (hk.newFrom[i] === null) {
            made.push(row);
            hunkAdded = true;
          }
        });
        if (hunkAdded) {
          anyAdded = true;
          terminated = !hk.eolMarkPlus;
        }
      }
      tree.pending.set(np, !anyAdded ? '' : made.join('\n') + (terminated ? '\n' : ''));
      return;
    }
    if (remove) {
      const op = from!;
      if (tree.current(op) === null) throw new ApplyError('missing', op);
      tree.pending.set(op, null);
      return;
    }
    if (move) {
      const op = from!;
      const np = to!;
      const cur = tree.current(op);
      if (cur === null) throw new ApplyError('missing', op);
      if (tree.current(np) !== null) throw new ApplyError('exists', np);
      tree.pending.set(np, applyHunks(cur, s.hunks, strict, op));
      tree.pending.set(op, null);
      renames.push({ from: op, to: np });
      return;
    }
    const p = from!;
    const cur = tree.current(p);
    if (cur === null) throw new ApplyError('missing', p);
    tree.pending.set(p, applyHunks(cur, s.hunks, strict, p));
  };
}

function parseUnifiedPatch(rows: string[], strict: boolean, renames: Rename[]): ApplyFn[] {
  const fns: ApplyFn[] = [];
  const blankSection = (): SectionRec => ({
    diffFrom: undefined,
    diffTo: undefined,
    pairFrom: undefined,
    pairTo: undefined,
    renameFrom: undefined,
    renameTo: undefined,
    isCreate: false,
    isDelete: false,
    hunks: [],
  });
  let sec: SectionRec | null = null;
  let inHeader = false;
  const flush = (): void => {
    if (sec !== null) fns.push(sectionToFn(sec, strict, renames));
    sec = null;
    inHeader = false;
  };
  let i = 0;
  while (i < rows.length) {
    const r = rows[i]!;
    if (r.startsWith('diff --git ')) {
      const m = /^diff --git a\/(\S+) b\/(\S+)$/.exec(r);
      if (m === null) throw new ParseError(`error: parse ${i + 1}`);
      flush();
      sec = { ...blankSection(), diffFrom: m[1]!, diffTo: m[2]! };
      inHeader = true;
      i++;
      continue;
    }
    if (r.startsWith('--- ') && (rows[i + 1] ?? '').startsWith('+++ ')) {
      if (sec !== null && inHeader) {
        sec.pairFrom = headerPath(r);
        sec.pairTo = headerPath(rows[i + 1]!);
      } else {
        flush();
        sec = { ...blankSection(), pairFrom: headerPath(r), pairTo: headerPath(rows[i + 1]!) };
        inHeader = true;
      }
      i += 2;
      continue;
    }
    if (r.startsWith('@@')) {
      if (sec === null) throw new ParseError(`error: parse ${i + 1}`);
      const { hunk, next } = readHunk(rows, i);
      sec.hunks.push(hunk);
      inHeader = false;
      i = next;
      continue;
    }
    if (sec !== null && inHeader) {
      if (r.startsWith('--- ')) throw new ParseError(`error: parse ${i + 2}`);
      if (r.startsWith('+++ ')) {
        sec.pairTo = headerPath(r);
        i++;
        continue;
      }
      if (r.startsWith('rename from ')) {
        sec.renameFrom = r.slice('rename from '.length).trim();
        i++;
        continue;
      }
      if (r.startsWith('rename to ')) {
        sec.renameTo = r.slice('rename to '.length).trim();
        i++;
        continue;
      }
      if (r.startsWith('new file mode')) {
        sec.isCreate = true;
        i++;
        continue;
      }
      if (r.startsWith('deleted file mode')) {
        sec.isDelete = true;
        i++;
        continue;
      }
      // `index`, mode and similarity lines, and any other header line, are ignored.
      i++;
      continue;
    }
    // Outside hunk bodies, lines that start no section are ignored.
    i++;
  }
  flush();
  return fns;
}

/* ------------------------------------------------------------------ */
/* SEARCH/REPLACE blocks (SPEC.md §7)                                  */
/* ------------------------------------------------------------------ */

function srToFn(path: string, oldRows: string[], newRows: string[], strict: boolean): ApplyFn {
  return (tree) => {
    if (!pathOk(path)) throw new ApplyError('unsafe', path);
    if (oldRows.length === 0) {
      // An empty SEARCH side creates the file, terminated with a line break.
      if (tree.current(path) !== null) throw new ApplyError('exists', path);
      tree.pending.set(path, newRows.length === 0 ? '' : newRows.join('\n') + '\n');
      return;
    }
    const cur = tree.current(path);
    if (cur === null) throw new ApplyError('missing', path);
    const doc = docFromText(cur);
    const found = locateBlock(doc.rows, oldRows, strict, 0);
    if (found === null) throw new ApplyError('no-match', path);
    if (found.hits.length > 1) throw new ApplyError('ambiguous', path);
    const rows = splice(doc.rows, found.hits[0]!, oldRows, newRows, newRows.map(() => null), found.tier);
    tree.pending.set(path, textFromDoc({ rows, eol: doc.eol, terminated: doc.terminated }));
  };
}

function parseSearchReplacePatch(rows: string[], strict: boolean): ApplyFn[] {
  const fns: ApplyFn[] = [];
  let prevEnd = 0;
  let prevPath: string | null = null;
  let i = 0;
  while (i < rows.length) {
    if (rows[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchLine = i + 1;
    // The path is the nearest non-blank, non-fence line above the block without
    // crossing the previous block; a block with none reuses the previous path,
    // and the first block with none is a parse error at its SEARCH line.
    let path: string | null = null;
    for (let j = i - 1; j >= prevEnd; j--) {
      const t = rows[j]!.trim();
      if (t === '' || t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new ParseError(`error: parse ${searchLine}`);
      path = prevPath;
    }
    let k = i + 1;
    while (k < rows.length && rows[k]!.trimEnd() !== '=======') k++;
    if (k >= rows.length) throw new ParseError(`error: parse ${searchLine}`);
    let m = k + 1;
    while (m < rows.length && rows[m]!.trimEnd() !== '>>>>>>> REPLACE') m++;
    if (m >= rows.length) throw new ParseError(`error: parse ${searchLine}`);
    fns.push(srToFn(path, rows.slice(i + 1, k), rows.slice(k + 1, m), strict));
    prevPath = path;
    prevEnd = m + 1;
    i = m + 1;
  }
  return fns;
}

/* ------------------------------------------------------------------ */
/* V4A patches (SPEC.md §8)                                            */
/* ------------------------------------------------------------------ */

interface ChunkRec {
  anchor: string | null;
  eof: boolean;
  oldRows: string[];
  newRows: string[];
  newFrom: (number | null)[];
}

/** One file section's content lines (with 0-based indexes), trailing blanks dropped. */
function readSectionLines(
  rows: string[],
  start: number,
  end: number,
): { content: [string, number][]; next: number } {
  const content: [string, number][] = [];
  let i = start;
  while (i < end) {
    const r = rows[i]!;
    if (r.startsWith('*** Add File: ') || r.startsWith('*** Delete File: ') || r.startsWith('*** Update File: ')) break;
    content.push([r, i]);
    i++;
  }
  while (content.length > 0 && content[content.length - 1]![0].trim() === '') content.pop();
  return { content, next: i };
}

/** Split a chunk body into its old side, new side and context map. */
function chunkSides(body: [string, number][]): { oldRows: string[]; newRows: string[]; newFrom: (number | null)[] } {
  const oldRows: string[] = [];
  const newRows: string[] = [];
  const newFrom: (number | null)[] = [];
  for (const [r] of body) {
    if (r === '' || r[0] === ' ') {
      const t = r === '' ? '' : r.slice(1);
      oldRows.push(t);
      newRows.push(t);
      newFrom.push(oldRows.length - 1);
    } else if (r[0] === '-') {
      oldRows.push(r.slice(1));
    } else if (r[0] === '+') {
      newRows.push(r.slice(1));
      newFrom.push(null);
    }
  }
  return { oldRows, newRows, newFrom };
}

function v4aUpdateToFn(
  path: string,
  updateLine: number,
  content: [string, number][],
  strict: boolean,
  renames: Rename[],
): ApplyFn {
  let moveTo: string | null = null;
  const chunks: ChunkRec[] = [];
  let open: { anchor: string | null; eof: boolean; body: [string, number][] } | null = null;
  let started = false;
  const closeChunk = (): void => {
    if (open === null) return;
    const s = chunkSides(open.body);
    chunks.push({ anchor: open.anchor, eof: open.eof, oldRows: s.oldRows, newRows: s.newRows, newFrom: s.newFrom });
    open = null;
  };
  content.forEach(([ln, idx], k) => {
    if (k === 0 && ln.startsWith('*** Move to: ')) {
      moveTo = ln.slice('*** Move to: '.length).trim();
      return;
    }
    if (ln.startsWith('@@')) {
      closeChunk();
      const anchor = ln.slice(2).trim();
      open = { anchor: anchor === '' ? null : anchor, eof: false, body: [] };
      started = true;
      return;
    }
    if (ln === '*** End of File') {
      if (open === null) throw new ParseError(`error: parse ${idx + 1}`);
      open.eof = true;
      closeChunk();
      return;
    }
    if (ln === '' || ln[0] === ' ' || ln[0] === '-' || ln[0] === '+') {
      if (open === null) {
        // A body line before any @@ opens an implicit first chunk — but only
        // before anything else; after a closed chunk it is a parse error.
        if (chunks.length === 0 && !started) {
          open = { anchor: null, eof: false, body: [] };
          started = true;
        } else {
          throw new ParseError(`error: parse ${idx + 1}`);
        }
      }
      open.body.push([ln, idx]);
      return;
    }
    throw new ParseError(`error: parse ${idx + 1}`);
  });
  closeChunk();
  if (chunks.length === 0 && moveTo === null) throw new ParseError(`error: parse ${updateLine}`);
  return (tree) => {
    if (!pathOk(path)) throw new ApplyError('unsafe', path);
    if (moveTo !== null && !pathOk(moveTo)) throw new ApplyError('unsafe', moveTo);
    const curText = tree.current(path);
    if (curText === null) throw new ApplyError('missing', path);
    if (moveTo !== null && tree.current(moveTo) !== null) throw new ApplyError('exists', moveTo);
    let doc = docFromText(curText);
    let cursor = 0;
    for (const ch of chunks) {
      if (ch.anchor !== null) {
        // The anchor is located first; the chunk body is searched after it.
        const f = locateBlock(doc.rows, [ch.anchor], strict, cursor);
        if (f === null) throw new ApplyError('no-match', path);
        cursor = f.hits[0]! + 1;
      }
      if (ch.oldRows.length === 0) {
        const at = ch.eof ? doc.rows.length : cursor;
        const rows = [...doc.rows.slice(0, at), ...ch.newRows, ...doc.rows.slice(at)];
        doc = { rows, eol: doc.eol, terminated: doc.terminated };
        cursor = at + ch.newRows.length;
        continue;
      }
      const f = locateBlock(doc.rows, ch.oldRows, strict, cursor);
      if (f === null) throw new ApplyError('no-match', path);
      // An end-of-file chunk only matches where its old side ends the file.
      const hits = ch.eof ? f.hits.filter((p) => p + ch.oldRows.length === doc.rows.length) : f.hits;
      if (hits.length === 0) throw new ApplyError('no-match', path);
      const rows = splice(doc.rows, hits[0]!, ch.oldRows, ch.newRows, ch.newFrom, f.tier);
      doc = { rows, eol: doc.eol, terminated: doc.terminated };
      cursor = hits[0]! + ch.newRows.length;
    }
    const out = textFromDoc(doc);
    if (moveTo !== null) {
      tree.pending.set(moveTo, out);
      tree.pending.set(path, null);
      renames.push({ from: path, to: moveTo });
    } else {
      tree.pending.set(path, out);
    }
  };
}

function parseV4aPatch(rows: string[], strict: boolean, renames: Rename[]): ApplyFn[] {
  const first = rows.findIndex((r) => !isBlankLine(r));
  if (first < 0) throw new ParseError('error: parse 1');
  if (rows[first]!.trim() !== '*** Begin Patch') throw new ParseError(`error: parse ${first + 1}`);
  let last = -1;
  for (let i = rows.length - 1; i >= 0; i--) {
    if (!isBlankLine(rows[i]!)) {
      last = i;
      break;
    }
  }
  if (rows[last]!.trim() !== '*** End Patch') throw new ParseError(`error: parse ${rows.length + 1}`);
  const fns: ApplyFn[] = [];
  let i = first + 1;
  while (i < last) {
    const r = rows[i]!;
    if (isBlankLine(r)) {
      i++;
      continue;
    }
    if (r.startsWith('*** Add File: ')) {
      const path = r.slice('*** Add File: '.length).trim();
      const { content, next } = readSectionLines(rows, i + 1, last);
      for (const [ln, idx] of content) {
        if (!(ln.startsWith('+') || ln === '')) throw new ParseError(`error: parse ${idx + 1}`);
      }
      const made = content.map(([ln]) => (ln.startsWith('+') ? ln.slice(1) : ''));
      fns.push((tree) => {
        if (!pathOk(path)) throw new ApplyError('unsafe', path);
        if (tree.current(path) !== null) throw new ApplyError('exists', path);
        tree.pending.set(path, made.length === 0 ? '' : made.join('\n') + '\n');
      });
      i = next;
      continue;
    }
    if (r.startsWith('*** Delete File: ')) {
      const path = r.slice('*** Delete File: '.length).trim();
      const { content, next } = readSectionLines(rows, i + 1, last);
      for (const [ln, idx] of content) {
        if (ln !== '') throw new ParseError(`error: parse ${idx + 1}`);
      }
      fns.push((tree) => {
        if (!pathOk(path)) throw new ApplyError('unsafe', path);
        if (tree.current(path) === null) throw new ApplyError('missing', path);
        tree.pending.set(path, null);
      });
      i = next;
      continue;
    }
    if (r.startsWith('*** Update File: ')) {
      const path = r.slice('*** Update File: '.length).trim();
      const { content, next } = readSectionLines(rows, i + 1, last);
      fns.push(v4aUpdateToFn(path, i + 1, content, strict, renames));
      i = next;
      continue;
    }
    throw new ParseError(`error: parse ${i + 1}`);
  }
  return fns;
}

/* ------------------------------------------------------------------ */
/* Invocation, reporting and exit codes (SPEC.md §1, §2 and §10)       */
/* ------------------------------------------------------------------ */

const byBytes = (a: string, b: string): number => Buffer.compare(Buffer.from(a), Buffer.from(b));

function execute(argv: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let checkOnly = false;
  let strict = false;
  const positional: string[] = [];
  let flagsDone = false;
  for (const a of argv) {
    if (!flagsDone && a === '--') {
      flagsDone = true;
    } else if (!flagsDone && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && KNOWN_FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !checkOnly) checkOnly = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else {
      positional.push(a);
    }
  }
  if (positional.length > 1) return { code: 2, out: ['error: usage'] };
  const file = positional.length === 0 || positional[0] === '-' ? null : positional[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const rows = splitPatchText(text);
  const kind = format === null || format === 'auto' ? sniffFormat(rows) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  const renames: Rename[] = [];
  let fns: ApplyFn[];
  try {
    if (kind === 'json') fns = parseJsonPatch(text, rows);
    else if (kind === 'unified') fns = parseUnifiedPatch(rows, strict, renames);
    else if (kind === 'search-replace') fns = parseSearchReplacePatch(rows, strict);
    else fns = parseV4aPatch(rows, strict, renames);
    if (fns.length === 0) throw new ParseError('error: parse 1');
  } catch (e) {
    if (e instanceof ParseError) return { code: 2, out: [e.text] };
    throw e;
  }
  // Apply every edit against the overlay; an edit that fails is rolled back
  // and reported, and any failure means nothing is written at all.
  const tree = new Overlay();
  const failures: ApplyError[] = [];
  for (const fn of fns) {
    const snapshot = new Map(tree.pending);
    try {
      fn(tree);
    } catch (e) {
      if (!(e instanceof ApplyError)) throw e;
      tree.pending.clear();
      for (const [k, v] of snapshot) tree.pending.set(k, v);
      failures.push(e);
    }
  }
  if (failures.length > 0) {
    return {
      code: 1,
      out: [...failures.map((f) => `error: ${f.code} ${f.path}`), `failed: ${failures.length} of ${fns.length} edits`],
    };
  }
  const report: { key: string; line: string }[] = [];
  const covered = new Set<string>();
  for (const rn of renames) {
    if (
      tree.onDisk(rn.from) !== null &&
      tree.current(rn.from) === null &&
      tree.onDisk(rn.to) === null &&
      tree.current(rn.to) !== null
    ) {
      report.push({ key: rn.to, line: `R ${rn.from} -> ${rn.to}` });
      covered.add(rn.from);
      covered.add(rn.to);
    }
  }
  const writeOut = (p: string): void => {
    const now = tree.pending.get(p);
    if (now === undefined) return;
    if (now === null) rmSync(p, { force: true });
    else {
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, now);
    }
  };
  for (const p of [...tree.pending.keys()].sort(byBytes)) {
    if (covered.has(p)) continue;
    const was = tree.onDisk(p);
    const now = tree.pending.get(p)!;
    if (was === now) continue;
    report.push({ key: p, line: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
    if (!checkOnly) writeOut(p);
  }
  // Paths named by an R line were skipped above; write them here.
  for (const p of [...covered].sort(byBytes)) {
    if (!checkOnly) writeOut(p);
  }
  report.sort((a, b) => byBytes(a.key, b.key));
  return { code: 0, out: [...report.map((r) => r.line), `ok: ${report.length} ${report.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = execute(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
