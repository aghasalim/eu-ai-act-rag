/* The retrieval metric kernel, recomputed in C from the unit lists.
 *
 * eval/metrics.py turns two lists of provision ids, the gold ones and the ones
 * retrieved in rank order, into six numbers per question. Those six numbers are
 * what every table in RESULTS.md and the README is averaged from, and until now
 * the only implementation of them was the Python that produced them. A wrong
 * discount in the nDCG, or an MRR that scored the last hit instead of the
 * first, would have produced numbers that looked plausible and that nothing
 * would have contradicted.
 *
 * This reads eval/results/rows_*.jsonl, which carry both the inputs
 * (gold_units, retrieved_units, cited_units) and the outputs the Python
 * computed from them, and recomputes every output. Fields are found by key
 * name, so a field added or reordered upstream cannot shift what is read.
 *
 * Exits non-zero on the first disagreement past the tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL      1e-12
#define MAXUNITS 64
#define IDLEN    32
#define NMETRIC  6

static const char *METRIC[NMETRIC] = {
    "hit_rate", "recall", "full_recall", "precision", "mrr", "ndcg"
};

typedef struct {
    char id[IDLEN];
    char gold[MAXUNITS][IDLEN];
    char ret[MAXUNITS][IDLEN];
    char cit[MAXUNITS][IDLEN];
    int  n_gold, n_ret, n_cit;
    double metric[NMETRIC];
    int  has_metric;          /* the retrieval block is {} for out-of-scope questions */
    double cite_valid;
    int  has_cite_valid;      /* null when the answer cited nothing */
} Row;

/* --- a JSON walker just large enough for these files -------------------- */

/* Skip one JSON value starting at p, honouring string escapes so a brace or a
 * bracket inside the legal text of a context cannot be mistaken for structure. */
static const char *skip_value(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '"') {
        p++;
        while (*p && *p != '"') p += (*p == '\\' && p[1]) ? 2 : 1;
        return *p ? p + 1 : p;
    }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        for (; *p; p++) {
            if (*p == '"') {
                p++;
                while (*p && *p != '"') p += (*p == '\\' && p[1]) ? 2 : 1;
                if (!*p) return p;
                continue;
            }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') { depth--; if (depth == 0) return p + 1; }
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

/* Value of a top-level key of the object starting at obj, or NULL. Only depth
 * one is searched, so "retrieval" inside a nested object is not confused with
 * the row's own. */
static const char *member(const char *obj, const char *key)
{
    const char *p = obj;
    while (*p && *p != '{') p++;
    if (!*p) return NULL;
    p++;
    for (;;) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == '}' || !*p) return NULL;
        if (*p != '"') return NULL;
        const char *ks = ++p;
        while (*p && *p != '"') p += (*p == '\\' && p[1]) ? 2 : 1;
        size_t klen = (size_t)(p - ks);
        p++;
        while (*p == ' ' || *p == ':') p++;
        if (klen == strlen(key) && strncmp(ks, key, klen) == 0) return p;
        p = skip_value(p);
    }
}

/* Copy an array of short strings into out, returning how many were read. */
static int string_array(const char *p, char out[][IDLEN], int cap)
{
    int n = 0;
    if (!p) return 0;
    while (*p && *p != '[') p++;
    if (!*p) return 0;
    p++;
    for (;;) {
        while (*p == ' ' || *p == ',' || *p == '\n') p++;
        if (*p == ']' || !*p) return n;
        if (*p != '"') return n;
        const char *s = ++p;
        while (*p && *p != '"') p++;
        size_t len = (size_t)(p - s);
        if (len >= IDLEN) len = IDLEN - 1;
        if (n < cap) { memcpy(out[n], s, len); out[n][len] = '\0'; n++; }
        p++;
    }
}

static int is_null(const char *p)
{
    while (*p == ' ') p++;
    return strncmp(p, "null", 4) == 0;
}

/* --- the metrics ------------------------------------------------------- */

static int contains(char list[][IDLEN], int n, const char *u)
{
    for (int i = 0; i < n; i++)
        if (strcmp(list[i], u) == 0) return 1;
    return 0;
}

static void score(const Row *r, int k, double out[NMETRIC])
{
    int overlap = 0;
    for (int i = 0; i < r->n_gold; i++)
        if (contains((char (*)[IDLEN])r->ret, r->n_ret, r->gold[i])) overlap++;

    out[0] = overlap > 0 ? 1.0 : 0.0;                              /* hit_rate */
    out[1] = (double)overlap / r->n_gold;                          /* recall */
    out[2] = overlap == r->n_gold ? 1.0 : 0.0;                     /* full_recall */
    out[3] = r->n_ret ? (double)overlap / r->n_ret : 0.0;          /* precision */

    out[4] = 0.0;                                                  /* mrr */
    for (int i = 0; i < r->n_ret; i++)
        if (contains((char (*)[IDLEN])r->gold, r->n_gold, r->ret[i])) {
            out[4] = 1.0 / (i + 1);
            break;
        }

    double dcg = 0.0, idcg = 0.0;                                  /* ndcg */
    for (int i = 0; i < r->n_ret; i++)
        if (contains((char (*)[IDLEN])r->gold, r->n_gold, r->ret[i]))
            dcg += 1.0 / log2((double)i + 2.0);
    int ideal = r->n_gold < k ? r->n_gold : k;
    for (int i = 0; i < ideal; i++)
        idcg += 1.0 / log2((double)i + 2.0);
    out[5] = idcg > 0.0 ? dcg / idcg : 0.0;
}

static double citation_validity(const Row *r)
{
    int ok = 0;
    for (int i = 0; i < r->n_cit; i++)
        if (contains((char (*)[IDLEN])r->ret, r->n_ret, r->cit[i])) ok++;
    return (double)ok / r->n_cit;
}

/* --- driver ------------------------------------------------------------ */

static char *slurp_line(FILE *f, char **buf, size_t *cap)
{
    size_t len = 0;
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (len + 2 > *cap) {
            *cap = *cap ? *cap * 2 : 65536;
            *buf = realloc(*buf, *cap);
            if (!*buf) { perror("realloc"); exit(2); }
        }
        (*buf)[len++] = (char)c;
    }
    if (c == EOF && len == 0) return NULL;
    (*buf)[len] = '\0';
    return *buf;
}

static double worst = 0.0;
static long compared = 0;

static int check_file(const char *root, const char *rel, int k)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", root, rel);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    char *buf = NULL;
    size_t cap = 0;
    char *line;
    int bad = 0, rows = 0;

    while ((line = slurp_line(f, &buf, &cap))) {
        if (!*line) continue;
        rows++;
        Row r;
        memset(&r, 0, sizeof r);

        const char *v = member(line, "id");
        if (!v) { fprintf(stderr, "%s row %d: no id\n", path, rows); bad++; continue; }
        char idbuf[1][IDLEN];
        const char *s = v + 1;
        size_t len = strcspn(s, "\"");
        if (len >= IDLEN) len = IDLEN - 1;
        memcpy(idbuf[0], s, len); idbuf[0][len] = '\0';
        strcpy(r.id, idbuf[0]);

        r.n_gold = string_array(member(line, "gold_units"), r.gold, MAXUNITS);
        r.n_ret  = string_array(member(line, "retrieved_units"), r.ret, MAXUNITS);
        r.n_cit  = string_array(member(line, "cited_units"), r.cit, MAXUNITS);

        const char *cv = member(line, "citation_validity");
        if (cv && !is_null(cv)) { r.has_cite_valid = 1; r.cite_valid = atof(cv); }

        const char *ret = member(line, "retrieval");
        if (ret) {
            int found = 0;
            for (int m = 0; m < NMETRIC; m++) {
                const char *mv = member(ret, METRIC[m]);
                if (mv) { r.metric[m] = atof(mv); found++; }
            }
            r.has_metric = (found == NMETRIC);
            if (found && found != NMETRIC) {
                fprintf(stderr, "%s %s: retrieval block has %d of %d metrics\n",
                        path, r.id, found, NMETRIC);
                bad++;
            }
        }

        if (r.n_gold > 0) {
            if (!r.has_metric) {
                fprintf(stderr, "%s %s: answerable but no retrieval metrics\n",
                        path, r.id);
                bad++;
            } else {
                double got[NMETRIC];
                score(&r, k, got);
                for (int m = 0; m < NMETRIC; m++) {
                    double d = fabs(got[m] - r.metric[m]);
                    if (d > worst) worst = d;
                    compared++;
                    if (d > TOL) {
                        fprintf(stderr, "%s %s %s: C %.15f, file %.15f, |d| %.1e\n",
                                path, r.id, METRIC[m], got[m], r.metric[m], d);
                        bad++;
                    }
                }
            }
        } else if (r.has_metric) {
            fprintf(stderr, "%s %s: no gold units but a retrieval block\n", path, r.id);
            bad++;
        }

        if (r.n_cit > 0) {
            if (!r.has_cite_valid) {
                fprintf(stderr, "%s %s: cited %d units, citation_validity is null\n",
                        path, r.id, r.n_cit);
                bad++;
            } else {
                double got = citation_validity(&r);
                double d = fabs(got - r.cite_valid);
                if (d > worst) worst = d;
                compared++;
                if (d > TOL) {
                    fprintf(stderr, "%s %s citation_validity: C %.15f, file %.15f\n",
                            path, r.id, got, r.cite_valid);
                    bad++;
                }
            }
        } else if (r.has_cite_valid) {
            fprintf(stderr, "%s %s: cited nothing but citation_validity is not null\n",
                    path, r.id);
            bad++;
        }
    }
    free(buf);
    fclose(f);

    if (rows == 0) { fprintf(stderr, "%s: no rows\n", path); return 1; }
    printf("  %-44s %3d rows, %s\n", rel, rows,
           bad ? "DISAGREES" : "every metric reproduced");
    return bad;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    /* k is the top_k the rows were produced at; nDCG's ideal ranking depends on
     * it. Every committed rows file was written at the repository default. */
    const int k = 6;
    int bad = 0;
    bad += check_file(root, "eval/results/rows_latest.jsonl", k);
    bad += check_file(root, "eval/results/rows_cap800_superseded.jsonl", k);
    bad += check_file(root, "eval/results/rows_gptoss120b.jsonl", k);

    if (bad) {
        printf("  %d disagreement(s)\n", bad);
        return 1;
    }
    printf("  %ld values recomputed from the unit lists, worst |d| %.1e\n",
           compared, worst);
    return 0;
}
