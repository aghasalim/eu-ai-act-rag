//! How wide are the published retrieval numbers, and how much of that width is
//! the resampling itself?
//!
//! README.md reports hybrid retrieval at 90.9% hit rate and 69.7% full recall
//! over 33 questions, and reports them as bare point estimates. The repository
//! contains no interval anywhere, so a reader has no way to tell a real gap
//! from a coin flip. eval/run_eval.py could not add one cheaply: it holds an
//! embedding model and a Chroma index open, so resampling inside it is not free.
//! Read out of the results file the outcomes are 33 zeros and ones, and a
//! million resamples of them costs nothing.
//!
//! This does three things:
//!
//!   1. a 1,000,000 draw paired bootstrap of hybrid full recall and of the
//!      hybrid minus BM25 difference
//!   2. 40 independent 25,000 draw runs, whose spread is the Monte Carlo error
//!      on the interval endpoints, so the interval is not itself noise
//!   3. checks that the published point estimate is where the bootstrap says it
//!      is, which fails if the results file and the published number part company
//!
//! Run: cd verify/bootstrap && cargo run --release --quiet -- <repo root>

use std::env;
use std::fs;
use std::process::exit;

const REFERENCE_DRAWS: usize = 1_000_000;
const SMALL_DRAWS: usize = 25_000;
const REPLICATES: usize = 40;
const SEED: u64 = 0x4555_4149_4143_54; // "EUAIACT" in ascii, so the run is reproducible
const TOL: f64 = 5e-5; // the published values are rounded to four decimals

/// xorshift64*. Not cryptographic and not trying to be: it has to be uniform,
/// fast and reproducible, so that a failure here can be re-run and looked at.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ----------------------------------------------------------------- json

/// Index of the `{` that opens the object stored under `key`, searching from
/// `from`. Requiring the brace is what keeps `"hybrid"` the config value from
/// being mistaken for `"hybrid"` the strategy.
fn key_object(s: &str, key: &str, from: usize) -> usize {
    let needle = format!("\"{}\"", key);
    let b = s.as_bytes();
    let mut at = from;
    while let Some(off) = s[at..].find(&needle) {
        let mut i = at + off + needle.len();
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b':' {
            i += 1;
            while i < b.len() && (b[i] as char).is_whitespace() {
                i += 1;
            }
            if i < b.len() && b[i] == b'{' {
                return i;
            }
        }
        at = at + off + needle.len();
    }
    panic!("no object stored under key {:?}", key);
}

/// The balanced object beginning at `open`. Nothing read here contains a string
/// with a brace in it, and the number of entries recovered is asserted by the
/// caller, so a mis-parse cannot pass quietly.
fn balanced(s: &str, open: usize) -> &str {
    let b = s.as_bytes();
    let mut depth = 0usize;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &s[open..=i];
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("unterminated object");
}

fn number_after(s: &str, key: &str) -> f64 {
    let needle = format!("\"{}\"", key);
    let start = s.find(&needle).unwrap_or_else(|| panic!("no {:?} here", key)) + needle.len();
    let rest = &s[start..];
    let colon = rest.find(':').expect("no colon after key") + 1;
    let tail = &rest[colon..];
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' || c.is_whitespace()))
        .unwrap_or(tail.len());
    tail[..end].trim().parse().expect("not a number")
}

/// (question id, metric value) for one strategy at one k.
fn per_question(json: &str, strategy: &str, k: &str, metric: &str) -> Vec<(String, f64)> {
    let mut p = key_object(json, strategy, 0);
    p = key_object(json, k, p);
    p = key_object(json, "per_question", p);
    let blk = balanced(json, p);
    let b = blk.as_bytes();
    let mut out = Vec::new();
    let mut i = 1;
    while i < b.len() {
        if b[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j] != b'"' {
                j += 1;
            }
            let name = &blk[start..j];
            let mut m = j + 1;
            while m < b.len() && (b[m] == b':' || (b[m] as char).is_whitespace()) {
                m += 1;
            }
            if m < b.len() && b[m] == b'{' {
                let entry = balanced(blk, m);
                out.push((name.to_string(), number_after(entry, metric)));
                i = m + entry.len();
                continue;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

fn published(json: &str, strategy: &str, k: &str, metric: &str) -> f64 {
    let mut p = key_object(json, strategy, 0);
    p = key_object(json, k, p);
    p = key_object(json, "overall", p);
    number_after(balanced(json, p), metric)
}

// ------------------------------------------------------------ bootstrap

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

/// One paired bootstrap. Questions are the resampling unit and both strategies
/// are read on the same draw, which is what makes the difference paired.
fn bootstrap(hyb: &[f64], bm: &[f64], draws: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
    let n = hyb.len();
    let mut a = Vec::with_capacity(draws);
    let mut d = Vec::with_capacity(draws);
    for _ in 0..draws {
        let (mut sh, mut sd) = (0.0, 0.0);
        for _ in 0..n {
            let i = rng.below(n);
            sh += hyb[i];
            sd += hyb[i] - bm[i];
        }
        a.push(sh / n as f64);
        d.push(sd / n as f64);
    }
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (a, d)
}

fn sd(xs: &[f64]) -> f64 {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let path = format!("{}/eval/results/eval_latest.json", root);
    let json = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  cannot read {}: {}", path, e);
            exit(2);
        }
    };

    let k = "6";
    let hyb_pairs = per_question(&json, "hybrid", k, "full_recall");
    let bm_pairs = per_question(&json, "bm25", k, "full_recall");
    let mut failures = 0;

    if hyb_pairs.len() != 33 {
        eprintln!("  read {} per-question outcomes, expected 33", hyb_pairs.len());
        failures += 1;
    }
    if hyb_pairs.iter().map(|(i, _)| i).ne(bm_pairs.iter().map(|(i, _)| i)) {
        eprintln!("  hybrid and bm25 are not scored on the same questions in the same order");
        exit(1);
    }
    let hyb: Vec<f64> = hyb_pairs.iter().map(|(_, v)| *v).collect();
    let bm: Vec<f64> = bm_pairs.iter().map(|(_, v)| *v).collect();
    if hyb.iter().chain(bm.iter()).any(|v| *v != 0.0 && *v != 1.0) {
        eprintln!("  full recall is meant to be 0 or 1 for each question");
        failures += 1;
    }

    let n = hyb.len() as f64;
    let point = hyb.iter().sum::<f64>() / n;
    let want = published(&json, "hybrid", k, "full_recall");
    if (point - want).abs() > TOL {
        eprintln!(
            "  hybrid full recall: per-question mean {:.6}, published {:.6}",
            point, want
        );
        failures += 1;
    }

    let mut rng = Rng::new(SEED);
    let (a, d) = bootstrap(&hyb, &bm, REFERENCE_DRAWS, &mut rng);
    let (lo, hi) = (percentile(&a, 0.025), percentile(&a, 0.975));
    let (dlo, dhi) = (percentile(&d, 0.025), percentile(&d, 0.975));
    let boot_mean = a.iter().sum::<f64>() / a.len() as f64;
    let mc_se = sd(&a) / (REFERENCE_DRAWS as f64).sqrt();

    println!(
        "  hybrid full recall {:.4} over {} questions, published {:.4}",
        point, hyb.len(), want
    );
    println!(
        "  {} draw bootstrap: mean {:.5}, 95% [{:.4}, {:.4}], width {:.4}",
        REFERENCE_DRAWS, boot_mean, lo, hi, hi - lo
    );

    // The bootstrap mean estimates the point estimate, so the two agreeing to
    // within a few Monte Carlo standard errors is a real check on both.
    if (boot_mean - point).abs() > 5.0 * mc_se {
        eprintln!(
            "  bootstrap mean {:.6} is {:.1} Monte Carlo se from the point estimate {:.6}",
            boot_mean,
            (boot_mean - point).abs() / mc_se,
            point
        );
        failures += 1;
    }
    if want < lo || want > hi {
        eprintln!("  the published {:.4} is outside its own bootstrap interval", want);
        failures += 1;
    }

    println!(
        "  hybrid minus BM25 {:+.4}, 95% [{:+.4}, {:+.4}], excludes no difference: {}",
        hyb.iter().zip(&bm).map(|(x, y)| x - y).sum::<f64>() / n,
        dlo,
        dhi,
        if dlo <= 0.0 && dhi >= 0.0 { "no" } else { "yes" }
    );

    // How much of the interval above is the resampling rather than the data.
    let mut los = Vec::with_capacity(REPLICATES);
    let mut his = Vec::with_capacity(REPLICATES);
    for r in 0..REPLICATES {
        let mut rr = Rng::new(SEED.wrapping_add(1 + r as u64));
        let (aa, _) = bootstrap(&hyb, &bm, SMALL_DRAWS, &mut rr);
        los.push(percentile(&aa, 0.025));
        his.push(percentile(&aa, 0.975));
    }
    // A proportion over 33 questions can only land on multiples of 1/33, so the
    // percentile endpoints sit on a lattice. Counting how many distinct values
    // 40 independent runs produced says more than a standard deviation does.
    let distinct = |v: &Vec<f64>| {
        let mut u = v.clone();
        u.sort_by(|a, b| a.partial_cmp(b).unwrap());
        u.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        u.len()
    };
    println!(
        "  Monte Carlo error, {} runs of {} draws: sd {:.2e} low / {:.2e} high, \
         {} and {} distinct endpoint value(s)",
        REPLICATES,
        SMALL_DRAWS,
        sd(&los),
        sd(&his),
        distinct(&los),
        distinct(&his)
    );

    if failures > 0 {
        eprintln!("  {} failure(s)", failures);
        exit(1);
    }
}
