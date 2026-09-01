# What the published retrieval numbers are worth at n = 33.
#
# The README reports 90.9% hit rate and 69.7% full recall for hybrid retrieval
# against 84.9% and 63.6% for BM25, and says the fusion is what buys the gap.
# Every one of those is a proportion over 33 questions, and the repository
# reports them as point estimates with no interval anywhere. Nothing said how
# wide they are, and nothing tested whether the gap between two of them is
# larger than the noise.
#
# This does three things, in base R with no packages:
#
#   deterministic  the point estimates, recomputed from the per-question
#                  outcomes, which must match the published values
#   interval       exact Clopper-Pearson and Wilson 95% intervals on each
#                  headline proportion
#   comparison     an exact paired test of hybrid against BM25, which is the
#                  claim the README's retrieval section is built on
#
# Run: Rscript verify/inference.R [root]

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

TOL <- 5e-5   # the published values are rounded to four decimals
Z <- qnorm(0.975)

fail <- 0
problem <- function(...) {
    cat("  FAIL", sprintf(...), "\n")
    fail <<- fail + 1
}

# --- reading the json without a package -----------------------------------

# Locate the value of a key that opens an object, searching forward from `from`
# so that nested keys are found inside the block already selected. Matching the
# key together with its opening brace is what stops "hybrid" the config value
# from being mistaken for "hybrid" the strategy.
key_at <- function(txt, key, from) {
    pat <- paste0('"', key, '"[[:space:]]*:[[:space:]]*\\{')
    m <- regexpr(pat, substr(txt, from, nchar(txt)))
    if (m < 0) stop(sprintf("key %s not found after position %d", key, from))
    from + m - 1 + attr(m, "match.length") - 1   # index of the '{'
}

# The substring of one json object, brace counted. Nothing inside the blocks
# read here is a string containing a brace, and the entry count is asserted
# afterwards, so a mis-parse cannot pass quietly.
object_at <- function(txt, open) {
    depth <- 0
    chars <- strsplit(substr(txt, open, nchar(txt)), "")[[1]]
    for (i in seq_along(chars)) {
        if (chars[i] == "{") depth <- depth + 1
        else if (chars[i] == "}") {
            depth <- depth - 1
            if (depth == 0) return(substr(txt, open, open + i - 1))
        }
    }
    stop("unterminated object")
}

# per-question values of one metric, for one strategy at one k
per_question <- function(txt, strategy, k, metric) {
    p <- key_at(txt, strategy, 1)
    p <- key_at(txt, k, p)
    p <- key_at(txt, "per_question", p)
    blk <- object_at(txt, p)
    entries <- regmatches(blk, gregexpr('"[a-z][0-9]+"[[:space:]]*:[[:space:]]*\\{[^}]*\\}', blk))[[1]]
    ids <- sub('^"([^"]+)".*', "\\1", entries)
    vals <- sapply(entries, function(e) {
        m <- regmatches(e, regexpr(paste0('"', metric, '"[[:space:]]*:[[:space:]]*[-0-9.eE]+'), e))
        if (length(m) == 0) stop(sprintf("no %s in a per_question entry", metric))
        as.numeric(sub('.*:[[:space:]]*', "", m))
    })
    stats::setNames(as.numeric(vals), ids)
}

published <- function(txt, strategy, k, subset, metric) {
    p <- key_at(txt, strategy, 1)
    p <- key_at(txt, k, p)
    p <- if (subset == "overall") key_at(txt, "overall", p) else {
        q <- key_at(txt, "by_type", p); key_at(txt, subset, q)
    }
    blk <- object_at(txt, p)
    m <- regmatches(blk, regexpr(paste0('"', metric, '"[[:space:]]*:[[:space:]]*[-0-9.eE]+'), blk))
    as.numeric(sub('.*:[[:space:]]*', "", m))
}

txt <- paste(readLines(file.path(root, "eval/results/eval_latest.json"), warn = FALSE),
             collapse = "\n")
qa <- readLines(file.path(root, "eval/qa_set.jsonl"), warn = FALSE)
qa <- qa[nzchar(trimws(qa))]
qa_id <- sub('.*"id"[[:space:]]*:[[:space:]]*"([^"]+)".*', "\\1", qa)
qa_type <- sub('.*"type"[[:space:]]*:[[:space:]]*"([^"]+)".*', "\\1", qa)
types <- stats::setNames(qa_type, qa_id)

K <- "6"
hyb_full <- per_question(txt, "hybrid", K, "full_recall")
hyb_hit  <- per_question(txt, "hybrid", K, "hit_rate")
bm_full  <- per_question(txt, "bm25", K, "full_recall")
bm_hit   <- per_question(txt, "bm25", K, "hit_rate")

if (length(hyb_full) != 33) problem("read %d per-question outcomes, expected 33", length(hyb_full))
if (!identical(sort(names(hyb_full)), sort(names(bm_full))))
    problem("hybrid and bm25 do not cover the same questions")
if (!all(names(hyb_full) %in% names(types)))
    problem("a scored question is not in eval/qa_set.jsonl")
if (!all(sort(unique(c(hyb_full, hyb_hit, bm_full, bm_hit))) %in% c(0, 1)))
    problem("hit rate and full recall are meant to be 0/1 per question")

multi <- names(hyb_full)[types[names(hyb_full)] == "multi_hop"]

# --- point estimates -------------------------------------------------------

cat("point estimates, recomputed from the per-question outcomes\n")
checks <- list(
    list("hybrid hit rate",             mean(hyb_hit),          published(txt, "hybrid", K, "overall", "hit_rate")),
    list("hybrid full recall",          mean(hyb_full),         published(txt, "hybrid", K, "overall", "full_recall")),
    list("bm25 hit rate",               mean(bm_hit),           published(txt, "bm25", K, "overall", "hit_rate")),
    list("bm25 full recall",            mean(bm_full),          published(txt, "bm25", K, "overall", "full_recall")),
    list("hybrid multi-hop full recall", mean(hyb_full[multi]), published(txt, "hybrid", K, "multi_hop", "full_recall")),
    list("hybrid multi-hop hit rate",   mean(hyb_hit[multi]),   published(txt, "hybrid", K, "multi_hop", "hit_rate"))
)
for (c in checks) {
    d <- abs(c[[2]] - c[[3]])
    if (d > TOL) problem("%s: recomputed %.6f, published %.6f", c[[1]], c[[2]], c[[3]])
    cat(sprintf("  %-28s %.4f  published %.4f  |d| %.1e\n", c[[1]], c[[2]], c[[3]], d))
}

# --- intervals -------------------------------------------------------------

wilson <- function(x, n) {
    p <- x / n
    d <- 1 + Z^2 / n
    ctr <- (p + Z^2 / (2 * n)) / d
    half <- Z / d * sqrt(p * (1 - p) / n + Z^2 / (4 * n^2))
    c(ctr - half, ctr + half)
}

cat("\n95% intervals on the headline proportions, n is small and the README says so\n")
rows <- list(
    list("hybrid hit rate",              sum(hyb_hit),        length(hyb_hit)),
    list("hybrid full recall",           sum(hyb_full),       length(hyb_full)),
    list("hybrid multi-hop full recall", sum(hyb_full[multi]), length(multi))
)
rowsfile <- readLines(file.path(root, "eval/results/rows_latest.jsonl"), warn = FALSE)
rowsfile <- rowsfile[nzchar(trimws(rowsfile))]
r_type <- sub('.*"type"[[:space:]]*:[[:space:]]*"([^"]+)".*', "\\1", rowsfile)
r_abst <- grepl('"abstained"[[:space:]]*:[[:space:]]*true', rowsfile)
rows[[length(rows) + 1]] <- list("correct abstention", sum(r_abst[r_type == "unanswerable"]),
                                 sum(r_type == "unanswerable"))
for (r in rows) {
    x <- r[[2]]; n <- r[[3]]
    ex <- stats::binom.test(x, n)$conf.int
    wi <- wilson(x, n)
    cat(sprintf("  %-28s %2d/%-2d = %.3f  exact [%.3f, %.3f]  Wilson [%.3f, %.3f]\n",
                r[[1]], x, n, x / n, ex[1], ex[2], wi[1], wi[2]))
}

# --- hybrid against bm25 ---------------------------------------------------

cat("\nhybrid against BM25 on the same 33 questions, exact paired test\n")
for (nm in c("full recall", "hit rate")) {
    h <- if (nm == "full recall") hyb_full else hyb_hit
    b <- if (nm == "full recall") bm_full else bm_hit
    b10 <- sum(h > b)   # hybrid found it, BM25 did not
    b01 <- sum(h < b)
    if (b10 + b01 == 0) {
        cat(sprintf("  %-12s no discordant question, the two are identical here\n", nm))
        next
    }
    p <- stats::binom.test(b10, b10 + b01, 0.5)$p.value
    cat(sprintf("  %-12s hybrid alone on %d question(s), BM25 alone on %d, gap %+.3f, exact p = %.3f%s\n",
                nm, b10, b01, mean(h) - mean(b), p,
                if (p < 0.05) "" else "  (not separable at this n)"))
}

if (fail > 0) {
    cat(sprintf("\n  %d failure(s)\n", fail))
    quit(status = 1)
}
quit(status = 0)
