// Structural validation of every committed evaluation artefact, plus an
// independent recomputation of the answer-quality summary.
//
// Every number in README.md and RESULTS.md is read out of one of six files:
// the corpus, the question set, three eval json files and three rows files.
// Nothing checked that those files are well formed. A truncated write, a
// duplicated question id, a probability that came out at 1.4 because of a
// division by the wrong denominator, or a question set that had drifted out of
// step with the corpus would all have been invisible until a reader noticed the
// table looked wrong.
//
// This walks all of them and, separately, recomputes the summary block of
// eval_latest.json from eval/results/rows_latest.jsonl, which is the raw
// per-question record the summary is averaged from.
//
// Run: cd verify/gocheck && go run . -root ..
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// The published summary is rounded to four decimals, so half of the last
// printed digit is as close as an independent mean can be required to land.
const tol = 5e-5

type problems []string

func (p *problems) add(format string, a ...any) { *p = append(*p, fmt.Sprintf(format, a...)) }

// ---------------------------------------------------------------- reading

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) != "" {
			out = append(out, sc.Text())
		}
	}
	return out, sc.Err()
}

// readJSONL decodes a line-delimited json file. DisallowUnknownFields is not
// wanted here, but a duplicate key within one object is: json.Decoder keeps the
// last one silently, so the keys are counted by hand.
func readJSONL(path string) ([]map[string]any, error) {
	lines, err := readLines(path)
	if err != nil {
		return nil, err
	}
	out := make([]map[string]any, 0, len(lines))
	for i, l := range lines {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			return nil, fmt.Errorf("line %d: %w", i+1, err)
		}
		if dup := duplicateKeys(l); len(dup) > 0 {
			return nil, fmt.Errorf("line %d: duplicate key(s) %v", i+1, dup)
		}
		out = append(out, m)
	}
	return out, nil
}

// duplicateKeys reports top-level keys that appear more than once in one json
// object. encoding/json silently keeps the last, so a file where a field was
// appended twice with different values would decode without complaint.
func duplicateKeys(s string) []string {
	dec := json.NewDecoder(strings.NewReader(s))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil
	}
	seen, dup := map[string]bool{}, []string{}
	for dec.More() {
		k, err := dec.Token()
		if err != nil {
			return nil
		}
		name, _ := k.(string)
		if seen[name] {
			dup = append(dup, name)
		}
		seen[name] = true
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return nil
		}
	}
	sort.Strings(dup)
	return dup
}

func readJSON(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if dup := duplicateKeys(string(b)); len(dup) > 0 {
		return nil, fmt.Errorf("duplicate top-level key(s) %v", dup)
	}
	return m, nil
}

// ---------------------------------------------------------------- helpers

func num(v any) (float64, bool) {
	f, ok := v.(float64)
	return f, ok
}

func str(m map[string]any, k string) string {
	s, _ := m[k].(string)
	return s
}

func strList(v any) []string {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, x := range raw {
		s, _ := x.(string)
		out = append(out, s)
	}
	return out
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func agrees(got, want float64) bool { return math.Abs(got-want) <= tol }

// ---------------------------------------------------------------- corpus

// The README states the corpus shape in prose: "113 articles + 180 recitals +
// 13 annexes -> 464 chunks". Those four numbers are the only description a
// reader gets of what the system searches, and they were typed by hand.
func checkCorpus(root string, p *problems) map[string]bool {
	path := filepath.Join(root, "data", "processed", "chunks.jsonl")
	rows, err := readJSONL(path)
	if err != nil {
		p.add("chunks.jsonl: %v", err)
		return nil
	}
	units := map[string]bool{}
	unitsByKind := map[string]map[string]bool{}
	chunkIDs := map[string]bool{}
	partsOf := map[string]int{}
	for i, c := range rows {
		id, unit, kind := str(c, "chunk_id"), str(c, "unit_id"), str(c, "kind")
		if id == "" || unit == "" || kind == "" {
			p.add("chunks.jsonl line %d: missing chunk_id, unit_id or kind", i+1)
			continue
		}
		if chunkIDs[id] {
			p.add("chunks.jsonl: duplicate chunk_id %q", id)
		}
		chunkIDs[id] = true
		if strings.TrimSpace(str(c, "body")) == "" {
			p.add("chunks.jsonl: %s has an empty body", id)
		}
		units[unit] = true
		if unitsByKind[kind] == nil {
			unitsByKind[kind] = map[string]bool{}
		}
		unitsByKind[kind][unit] = true

		np, okN := num(c["n_parts"])
		pt, okP := num(c["part"])
		if !okN || !okP || pt < 1 || np < 1 || pt > np {
			p.add("chunks.jsonl: %s has part %v of %v", id, c["part"], c["n_parts"])
			continue
		}
		if prev, seen := partsOf[unit]; seen && prev != int(np) {
			p.add("chunks.jsonl: %s claims %d parts, a sibling claims %d", unit, int(np), prev)
		}
		partsOf[unit] = int(np)
		if tk, ok := num(c["n_tokens"]); !ok || tk <= 0 {
			p.add("chunks.jsonl: %s has n_tokens %v", id, c["n_tokens"])
		}
	}

	want := map[string]int{"article": 113, "recital": 180, "annex": 13}
	for kind, n := range want {
		if got := len(unitsByKind[kind]); got != n {
			p.add("corpus: %d %s units, the README says %d", got, kind, n)
		}
	}
	if len(rows) != 464 {
		p.add("corpus: %d chunks, the README says 464", len(rows))
	}
	if len(unitsByKind) != len(want) {
		p.add("corpus: kinds present are %v, expected exactly article/recital/annex",
			sortedKeys(unitsByKind))
	}
	// Article numbering is contiguous, so a lost article shows up as a gap
	// rather than only as a count that happens to be right for another reason.
	for i := 1; i <= 113; i++ {
		if !units[fmt.Sprintf("art_%d", i)] {
			p.add("corpus: art_%d is missing", i)
		}
	}
	fmt.Printf("  corpus            %d chunks over %d units (%d articles, %d recitals, %d annexes)\n",
		len(rows), len(units), len(unitsByKind["article"]),
		len(unitsByKind["recital"]), len(unitsByKind["annex"]))
	return units
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------- qa set

func checkQA(root string, corpus map[string]bool, p *problems) []map[string]any {
	path := filepath.Join(root, "eval", "qa_set.jsonl")
	qa, err := readJSONL(path)
	if err != nil {
		p.add("qa_set.jsonl: %v", err)
		return nil
	}
	seen := map[string]bool{}
	byType := map[string]int{}
	answerable := 0
	for i, q := range qa {
		id := str(q, "id")
		if id == "" {
			p.add("qa_set.jsonl line %d: no id", i+1)
			continue
		}
		if seen[id] {
			p.add("qa_set.jsonl: duplicate id %q", id)
		}
		seen[id] = true
		t := str(q, "type")
		byType[t]++
		if t != "single_hop" && t != "multi_hop" && t != "unanswerable" {
			p.add("qa_set.jsonl: %s has type %q", id, t)
		}
		gold := strList(q["gold_units"])
		if len(gold) > 0 {
			answerable++
		}
		if (t == "unanswerable") != (len(gold) == 0) {
			p.add("qa_set.jsonl: %s is %q but has %d gold units", id, t, len(gold))
		}
		// A gold label naming a provision the corpus does not contain makes
		// that question unanswerable by construction, and silently drags every
		// recall number down.
		for _, u := range gold {
			if corpus != nil && !corpus[u] {
				p.add("qa_set.jsonl: %s cites %q, which is not in the corpus", id, u)
			}
		}
		if strings.TrimSpace(str(q, "question")) == "" {
			p.add("qa_set.jsonl: %s has an empty question", id)
		}
	}
	if len(qa) != 45 {
		p.add("qa set: %d questions, the README says 45", len(qa))
	}
	if answerable != 33 {
		p.add("qa set: %d answerable questions, the README's tables average over 33", answerable)
	}
	if byType["unanswerable"] != 12 {
		p.add("qa set: %d out-of-scope questions, the README says 12", byType["unanswerable"])
	}
	fmt.Printf("  qa set            %d questions (%d single-hop, %d multi-hop, %d out-of-scope)\n",
		len(qa), byType["single_hop"], byType["multi_hop"], byType["unanswerable"])
	return qa
}

// ---------------------------------------------------------------- eval json

// checkEval walks the retrieval block of one eval json: every rate must be a
// real number in [0, 1], every strategy must cover the same k values, and every
// at_k block must carry one per-question entry per answerable question.
func checkEval(root, name string, nAnswerable int, p *problems) map[string]any {
	path := filepath.Join(root, "eval", "results", name)
	d, err := readJSON(path)
	if err != nil {
		p.add("%s: %v", name, err)
		return nil
	}
	ret, ok := d["retrieval"].(map[string]any)
	if !ok {
		p.add("%s: no retrieval block", name)
		return d
	}
	var ksSeen []string
	for _, mode := range sortedKeys(ret) {
		md, _ := ret[mode].(map[string]any)
		if lat, ok := num(md["latency_s_per_query"]); !ok || lat <= 0 || math.IsInf(lat, 0) {
			p.add("%s: %s latency_s_per_query is %v", name, mode, md["latency_s_per_query"])
		}
		atk, _ := md["at_k"].(map[string]any)
		ks := sortedKeys(atk)
		if ksSeen == nil {
			ksSeen = ks
		} else if strings.Join(ks, ",") != strings.Join(ksSeen, ",") {
			p.add("%s: %s covers k=%v, another strategy covers k=%v", name, mode, ks, ksSeen)
		}
		for _, k := range ks {
			blk, _ := atk[k].(map[string]any)
			pq, _ := blk["per_question"].(map[string]any)
			if len(pq) != nAnswerable {
				p.add("%s: %s@%s has %d per-question entries, the qa set has %d answerable",
					name, mode, k, len(pq), nAnswerable)
			}
			for _, where := range []string{"overall", "per_question"} {
				checkRates(fmt.Sprintf("%s %s@%s %s", name, mode, k, where), blk[where], p)
			}
			checkRates(fmt.Sprintf("%s %s@%s by_type", name, mode, k), blk["by_type"], p)
		}
	}
	if abl, ok := d["ablation_recital_weight"].(map[string]any); ok {
		checkRates(name+" ablation", abl, p)
	}
	fmt.Printf("  %-17s %d strategies, k=%s, per-question entries present\n",
		strings.TrimSuffix(name, ".json"), len(ret), strings.Join(ksSeen, "/"))
	return d
}

// checkRates walks a nested block and requires every number under it to be a
// finite value in [0, 1]. NaN and Inf cannot survive json.Unmarshal, so a
// number that escaped a bad division arrives here as a value out of range or
// as a parse error one level up.
func checkRates(where string, v any, p *problems) {
	switch t := v.(type) {
	case map[string]any:
		for _, k := range sortedKeys(t) {
			checkRates(where+"."+k, t[k], p)
		}
	case float64:
		if math.IsNaN(t) || math.IsInf(t, 0) || t < 0 || t > 1 {
			p.add("%s = %v, not a rate in [0,1]", where, t)
		}
	case nil:
	}
}

// ---------------------------------------------------------------- summary

// recomputeSummary rebuilds the answer-quality block of eval_latest.json from
// the per-question rows, which is the file the README's answers table is
// ultimately printed from. Every rule here is stated independently of
// eval/run_eval.py: an answerable question is one whose type is not
// "unanswerable", faithfulness is averaged over answerable questions only, and
// a refusal on an answerable question is a false abstention.
func recomputeSummary(root string, p *problems) {
	rows, err := readJSONL(filepath.Join(root, "eval", "results", "rows_latest.jsonl"))
	if err != nil {
		p.add("rows_latest.jsonl: %v", err)
		return
	}
	d, err := readJSON(filepath.Join(root, "eval", "results", "eval_latest.json"))
	if err != nil {
		return
	}
	s, ok := d["summary"].(map[string]any)
	if !ok {
		p.add("eval_latest.json: no summary block")
		return
	}

	var strict, lenient, faith, cite, abstainA, abstainU, answeredU, lat []float64
	seen := map[string]bool{}
	for i, r := range rows {
		id := str(r, "id")
		if id == "" || seen[id] {
			p.add("rows_latest.jsonl line %d: missing or duplicate id %q", i+1, id)
			continue
		}
		seen[id] = true
		t := str(r, "type")
		grade := str(r, "grade")
		abst, isBool := r["abstained"].(bool)
		if !isBool {
			p.add("rows_latest.jsonl: %s has abstained=%v", id, r["abstained"])
			continue
		}
		b := func(x bool) float64 {
			if x {
				return 1
			}
			return 0
		}
		if v, ok := num(r["latency_s"]); ok {
			lat = append(lat, v)
		} else {
			p.add("rows_latest.jsonl: %s has latency_s=%v", id, r["latency_s"])
		}
		if v, ok := num(r["citation_validity"]); ok {
			cite = append(cite, v)
		} else if r["citation_validity"] != nil {
			p.add("rows_latest.jsonl: %s has citation_validity=%v", id, r["citation_validity"])
		}
		if t == "unanswerable" {
			abstainU = append(abstainU, b(abst))
			answeredU = append(answeredU, b(!abst))
			continue
		}
		abstainA = append(abstainA, b(abst))
		if v, ok := num(r["faithfulness"]); ok {
			faith = append(faith, v)
		}
		switch grade {
		case "correct", "partial", "incorrect":
			strict = append(strict, b(grade == "correct"))
			lenient = append(lenient, b(grade == "correct" || grade == "partial"))
		default:
			p.add("rows_latest.jsonl: %s has grade %q", id, grade)
		}
	}

	if n, ok := num(s["n"]); !ok || int(n) != len(rows) {
		p.add("summary n is %v, rows_latest.jsonl has %d rows", s["n"], len(rows))
	}
	checks := []struct {
		key string
		got float64
	}{
		{"answer_accuracy_strict", mean(strict)},
		{"answer_accuracy_lenient", mean(lenient)},
		{"faithfulness", mean(faith)},
		{"citation_validity", mean(cite)},
		{"correct_abstention_rate", mean(abstainU)},
		{"false_abstention_rate", mean(abstainA)},
		{"hallucination_rate_unanswerable", mean(answeredU)},
		{"mean_latency_s", mean(lat)},
	}
	worst, latDelta := 0.0, 0.0
	for _, c := range checks {
		want, ok := num(s[c.key])
		if !ok {
			p.add("summary has no %s", c.key)
			continue
		}
		d := math.Abs(c.got - want)
		// mean_latency_s is a duration in seconds, not a rate, so it is not
		// comparable with the others and is reported on its own.
		lim := tol
		if c.key == "mean_latency_s" {
			lim = 5e-3
			latDelta = d
		} else if d > worst {
			worst = d
		}
		if d > lim {
			p.add("summary %s: recomputed %.6f, published %.6f, |d| %.1e",
				c.key, c.got, want, d)
		}
	}
	fmt.Printf("  summary           7 published rates recomputed from %d rows, worst |d| %.1e"+
		" (mean latency |d| %.1e s)\n", len(rows), worst, latDelta)

	// The rows are stored twice, inside eval_latest.json and as the checkpoint
	// file. They are meant to be the same rows, with the context passages
	// dropped from the copy in the json.
	gen, _ := d["generation"].(map[string]any)
	inner, _ := gen["rows"].([]any)
	if len(inner) != len(rows) {
		p.add("eval_latest.json carries %d rows, rows_latest.jsonl has %d", len(inner), len(rows))
		return
	}
	drift := 0
	for i, raw := range inner {
		a, _ := raw.(map[string]any)
		bRow := rows[i]
		if str(a, "id") != str(bRow, "id") {
			p.add("row %d: eval_latest.json has %q, rows_latest.jsonl has %q",
				i, str(a, "id"), str(bRow, "id"))
			drift++
			continue
		}
		for _, k := range sortedKeys(bRow) {
			if k == "contexts" {
				continue
			}
			x, _ := json.Marshal(a[k])
			y, _ := json.Marshal(bRow[k])
			if string(x) != string(y) {
				p.add("row %s field %q differs between eval_latest.json and rows_latest.jsonl",
					str(a, "id"), k)
				drift++
			}
		}
	}
	if drift == 0 {
		fmt.Printf("  cross-file        %d rows identical in eval_latest.json and rows_latest.jsonl\n",
			len(rows))
	}
}

// ------------------------------------------------- retrieval, run against run

// The same retrieval runs twice over the same questions: once in the sweep that
// fills the retrieval block, and once inside the generation pass, whose result
// is stored per row. RESULTS.md prints its tables from the first and its
// failure taxonomy from the second, and nothing had ever put the two side by
// side. They must at least agree on which provisions were found, because that
// is what both sets of numbers mean.
//
// Rank order is reported rather than required: generation rows are checkpointed
// and reused across runs while the sweep is redone every time, so a question
// whose top-k ties can be ordered differently in the two passes without either
// being wrong. The set is not allowed to differ.
func checkRetrievalConsistency(root string, p *problems) {
	d, err := readJSON(filepath.Join(root, "eval", "results", "eval_latest.json"))
	if err != nil {
		return
	}
	rows, err := readJSONL(filepath.Join(root, "eval", "results", "rows_latest.jsonl"))
	if err != nil {
		return
	}
	cfg, _ := d["config"].(map[string]any)
	mode := str(cfg, "gen_mode")
	kf, _ := num(cfg["top_k"])
	k := strconv.Itoa(int(kf))

	ret, _ := d["retrieval"].(map[string]any)
	md, _ := ret[mode].(map[string]any)
	atk, _ := md["at_k"].(map[string]any)
	kb, _ := atk[k].(map[string]any)
	sweep, ok := kb["per_question"].(map[string]any)
	if !ok {
		p.add("eval_latest.json: no per_question block for %s at k=%s", mode, k)
		return
	}

	setMetrics := []string{"hit_rate", "recall", "full_recall", "precision"}
	var reordered []string
	compared, scored := 0, 0
	for _, r := range rows {
		if len(strList(r["gold_units"])) == 0 {
			continue
		}
		id := str(r, "id")
		gen, _ := r["retrieval"].(map[string]any)
		sw, ok := sweep[id].(map[string]any)
		if !ok {
			p.add("%s is scored in rows_latest.jsonl but not in the %s sweep at k=%s", id, mode, k)
			continue
		}
		scored++
		for _, m := range setMetrics {
			a, okA := num(gen[m])
			b, okB := num(sw[m])
			if !okA || !okB {
				p.add("%s: %s missing from one of the two passes", id, m)
				continue
			}
			compared++
			if !agrees(a, b) {
				p.add("%s %s: generation pass %.6f, retrieval sweep %.6f, so the two passes "+
					"retrieved different provisions", id, m, a, b)
			}
		}
		for _, m := range []string{"mrr", "ndcg"} {
			a, okA := num(gen[m])
			b, okB := num(sw[m])
			if okA && okB && !agrees(a, b) {
				reordered = append(reordered, id)
				break
			}
		}
	}
	note := "identical rank order too"
	if len(reordered) > 0 {
		note = fmt.Sprintf("rank order differs on %d (%s)",
			len(reordered), strings.Join(reordered, " "))
	}
	fmt.Printf("  cross-run         %d questions, %d set-level values identical in both retrieval passes, %s\n",
		scored, compared, note)
}

// ---------------------------------------------------------------- main

func main() {
	root := flag.String("root", "../..", "repository root")
	flag.Parse()

	var p problems
	corpus := checkCorpus(*root, &p)
	qa := checkQA(*root, corpus, &p)
	answerable := 0
	for _, q := range qa {
		if len(strList(q["gold_units"])) > 0 {
			answerable++
		}
	}
	for _, name := range []string{
		"eval_latest.json", "eval_retrieval_only.json", "eval_cap800_superseded.json",
	} {
		checkEval(*root, name, answerable, &p)
	}
	for _, name := range []string{
		"rows_latest.jsonl", "rows_cap800_superseded.jsonl", "rows_gptoss120b.jsonl",
	} {
		path := filepath.Join(*root, "eval", "results", name)
		rows, err := readJSONL(path)
		if err != nil {
			p.add("%s: %v", name, err)
			continue
		}
		fmt.Printf("  %-17s %d rows, well formed\n", strings.TrimSuffix(name, ".jsonl"), len(rows))
	}
	recomputeSummary(*root, &p)
	checkRetrievalConsistency(*root, &p)

	if len(p) > 0 {
		fmt.Printf("\n  %d problem(s):\n", len(p))
		for _, s := range p {
			fmt.Println("    " + s)
		}
		os.Exit(1)
	}
}
