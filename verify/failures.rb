# Re-derive the failure taxonomy from the raw rows.
#
# "Where it fails" is the section of RESULTS.md that the whole project argues
# from: six of the twelve non-ok outcomes are retrieval failures, so effort
# belongs in retrieval rather than in prompting. That claim rests entirely on
# eval/run_eval.py's classify_failure, a chain of branches where the ordering is
# the whole content. If a refusal were tested before a retrieval miss, every
# question that both missed and refused would move category, the totals would
# still sum to 45, and nothing would look wrong.
#
# This assigns every row a cause from the definitions rather than from that
# function, and requires the same label on all 45 rows, the same counter, and
# the same grouped totals that the README prints in prose.
#
# Run: ruby verify/failures.rb [root]

require 'json'

root = ARGV[0] || '.'
rows = File.readlines(File.join(root, 'eval/results/rows_latest.jsonl'))
           .reject { |l| l.strip.empty? }.map { |l| JSON.parse(l) }
summary = JSON.parse(File.read(File.join(root, 'eval/results/eval_latest.json')))['summary']

# The earliest cause in the chain wins, because a wrong answer produced from
# passages that never contained the provision is not a generation problem.
def cause(r)
  if r['type'] == 'unanswerable'
    return r['abstained'] ? 'ok' : 'hallucination_no_abstain'
  end
  return 'retrieval_miss' if r['retrieval']['hit_rate'].zero?
  return 'false_abstention' if r['abstained']
  if r['type'] == 'multi_hop' && r['retrieval']['full_recall'].zero?
    return 'partial_retrieval'
  end
  return 'generation_error' if r['grade'] == 'incorrect'
  return 'incomplete_answer' if r['grade'] == 'partial'

  'ok'
end

problems = []

rows.each do |r|
  got = cause(r)
  problems << "#{r['id']}: derived #{got}, the file says #{r['failure']}" if got != r['failure']
end

counts = Hash.new(0)
rows.each { |r| counts[cause(r)] += 1 }
published = summary['failure_modes']

(counts.keys | published.keys).sort.each do |k|
  next if counts[k] == published.fetch(k, 0)

  problems << "failure_modes[#{k}]: derived #{counts[k]}, published #{published.fetch(k, 0)}"
end

# The groupings the README states in prose, which exist nowhere in the json.
groups = {
  'non-ok outcomes' => [45 - counts['ok'], 12],
  'retrieval failures' => [counts['retrieval_miss'] + counts['partial_retrieval'], 6],
  'complete misses' => [counts['retrieval_miss'], 3],
  'partial retrievals' => [counts['partial_retrieval'], 3],
  'refusals of answerable questions' => [counts['false_abstention'], 4],
  'generation faults' => [counts['generation_error'] + counts['incomplete_answer'], 2]
}
groups.each do |name, (got, want)|
  problems << "#{name}: derived #{got}, the README says #{want}" if got != want
end

# The per-type generation table in RESULTS.md, averaged here rather than read.
TOL = 5e-5
%w[single_hop multi_hop unanswerable].each do |t|
  sub = rows.select { |r| r['type'] == t }
  want = summary['by_type'][t]
  problems << "by_type[#{t}].n: derived #{sub.size}, published #{want['n']}" if sub.size != want['n']

  acc = sub.count { |r| r['grade'] == 'correct' }.to_f / sub.size
  if (acc - want['accuracy_strict']).abs > TOL
    problems << format('by_type[%s].accuracy_strict: derived %.6f, published %.6f',
                       t, acc, want['accuracy_strict'])
  end

  faith = sub.map { |r| r['faithfulness'] }.compact
  if faith.empty?
    problems << "by_type[#{t}].faithfulness: no scored claims, published #{want['faithfulness']}" unless want['faithfulness'].nil?
  else
    m = faith.sum / faith.size
    if want['faithfulness'].nil? || (m - want['faithfulness']).abs > TOL
      problems << format('by_type[%s].faithfulness: derived %.6f, published %s',
                         t, m, want['faithfulness'].inspect)
    end
  end
end

if problems.empty?
  puts format('  %d rows relabelled from the definitions, all %d causes agree',
              rows.size, counts.values.sum)
  puts format('  %s', counts.sort.map { |k, v| "#{k}=#{v}" }.join(' '))
  puts format('  grouped totals the README states in prose: %s',
              groups.map { |n, (g, _)| "#{n} #{g}" }.join(', '))
  exit 0
end

problems.each { |p| warn "  #{p}" }
exit 1
