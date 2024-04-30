from factscore.factscorer import FactScorer

fs = FactScorer(openai_key="...")

# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
out = fs.get_score(topics, generations, gamma=10)
print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response