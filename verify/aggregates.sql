-- Recompute every published retrieval aggregate from the per-question numbers.
--
-- eval/run_eval.py scores each question, then takes the mean over questions to
-- get the "overall" and "by_type" blocks that RESULTS.md and the README table
-- are printed from. The mean is the step nothing checked: the per-question
-- values and the aggregate come out of the same loop, so an error in the
-- averaging would be invisible. This redoes the averaging in SQLite, reading
-- the same JSON with SQLite's own parser, and joins eval/qa_set.jsonl for the
-- question types rather than guessing them from the id prefix.
--
-- Emits one FAIL line per disagreement and one CHECKED line with the number of
-- comparisons made. verify/verify.sh requires zero FAIL and a non-zero count.
--
-- Run: sqlite3 -init verify/aggregates.sql :memory: ""

.mode list
.headers off

-- The published values are rounded to 4 decimals by run_eval.py's mean(), so
-- half of the last printed digit is the tightest agreement that can be asked
-- for. Anything looser would let a real error through.
CREATE TEMP TABLE tol(v REAL);
INSERT INTO tol VALUES (5e-5);

CREATE TEMP TABLE files(path TEXT);
INSERT INTO files VALUES
    ('eval/results/eval_latest.json'),
    ('eval/results/eval_retrieval_only.json'),
    ('eval/results/eval_cap800_superseded.json');

CREATE TEMP TABLE doc AS
    SELECT path, CAST(readfile(path) AS TEXT) AS j FROM files;

-- qa_set.jsonl is line-delimited JSON, which SQLite has no importer for, so
-- split it on newlines first. json_extract then reads each line on its own.
CREATE TEMP TABLE qa AS
WITH RECURSIVE lines(line, rest) AS (
    SELECT '', CAST(readfile('eval/qa_set.jsonl') AS TEXT)
    UNION ALL
    SELECT CASE WHEN instr(rest, char(10)) > 0
                THEN substr(rest, 1, instr(rest, char(10)) - 1) ELSE rest END,
           CASE WHEN instr(rest, char(10)) > 0
                THEN substr(rest, instr(rest, char(10)) + 1) ELSE '' END
    FROM lines WHERE rest <> ''
)
SELECT json_extract(line, '$.id')   AS id,
       json_extract(line, '$.type') AS type
FROM lines WHERE line <> '';

-- One row per (file, strategy, k, question, metric).
CREATE TEMP TABLE pq AS
SELECT d.path AS path, s.key AS strategy, k.key AS kk,
       q.key AS qid, m.key AS metric, m.value AS val
FROM doc d,
     json_each(d.j, '$.retrieval')          s,
     json_each(s.value, '$.at_k')           k,
     json_each(k.value, '$.per_question')   q,
     json_each(q.value)                     m;

-- What the file says the aggregates are.
CREATE TEMP TABLE published AS
SELECT d.path AS path, s.key AS strategy, k.key AS kk,
       'overall' AS subset, m.key AS metric, m.value AS val
FROM doc d,
     json_each(d.j, '$.retrieval')  s,
     json_each(s.value, '$.at_k')   k,
     json_each(k.value, '$.overall') m
UNION ALL
SELECT d.path, s.key, k.key, t.key, m.key, m.value
FROM doc d,
     json_each(d.j, '$.retrieval')  s,
     json_each(s.value, '$.at_k')   k,
     json_each(k.value, '$.by_type') t,
     json_each(t.value)              m;

-- What they are, averaged here.
CREATE TEMP TABLE recomputed AS
SELECT path, strategy, kk, 'overall' AS subset, metric,
       avg(val) AS val, count(*) AS n
FROM pq GROUP BY path, strategy, kk, metric
UNION ALL
SELECT p.path, p.strategy, p.kk, qa.type, p.metric,
       avg(p.val), count(*)
FROM pq p JOIN qa ON qa.id = p.qid
GROUP BY p.path, p.strategy, p.kk, qa.type, p.metric;

.separator "|"
SELECT 'FAIL', r.path, r.strategy, 'k=' || r.kk, r.subset, r.metric,
       printf('recomputed %.6f over %d questions', r.val, r.n),
       printf('published %.6f', p.val),
       printf('delta %.2e', abs(r.val - p.val))
FROM recomputed r
JOIN published p USING (path, strategy, kk, subset, metric)
WHERE abs(r.val - p.val) > (SELECT v FROM tol);

-- An aggregate present in the file with no per-question data behind it would
-- otherwise vanish from the join instead of failing.
SELECT 'FAIL', p.path, p.strategy, 'k=' || p.kk, p.subset, p.metric,
       'no per-question rows to recompute this from', '', ''
FROM published p
LEFT JOIN recomputed r USING (path, strategy, kk, subset, metric)
WHERE r.val IS NULL;

SELECT 'MAXDELTA', printf('%.3e', max(abs(r.val - p.val)))
FROM recomputed r JOIN published p USING (path, strategy, kk, subset, metric);

SELECT 'CHECKED', count(*)
FROM recomputed r JOIN published p USING (path, strategy, kk, subset, metric);
