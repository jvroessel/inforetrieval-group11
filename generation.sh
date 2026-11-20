#!/bin/bash -e
#SBATCH --account=csedui00041
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=31G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gen-%j.out
#SBATCH --error=logs/gen-%j.err

DATASET=ecommerce      # ecommerce or wiki1
METHOD=search_r1_raac          # single or search_r1
LIMIT=""      # "" for a full run
RUN=run1               # run1 for the full run

BASE=/vol/csedu-nobackup/course/I00041_informationretrieval/users/$USER
SHARED=/vol/csedu-nobackup/course/I00041_informationretrieval/shared/trqa
cd $BASE/total-recall-qa
source .venv/bin/activate
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=not-used
export PYTHONUNBUFFERED=1

if [ "$DATASET" = ecommerce ]; then
  NAME=synthetic_ecommerce; DIR=E-Commerce; SHORT=ecommerce
  INDEX=$BASE/total-recall-qa/ecom_data/indices/ecommerce/spladepp
  CORPUS=$SHARED/data/ecommerce/corpus_ecommerce_rewritten.jsonl
else
  NAME=wikidata; DIR=Wiki1; SHORT=wiki1
  INDEX=$BASE/indices/wiki_partial/spladepp_index
  CORPUS=$SHARED/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl
fi
ln -sfn $INDEX corpus_datasets/corpus/spladepp_index

if [ "$METHOD" = single ]; then
  ARGS="--generation_method single_retrieval --model Qwen/Qwen2.5-7B-Instruct --retrieval_topk 50"
elif [ "$METHOD" = search_r1 ]; then
  ARGS="--generation_method deep_research --deep_research_model search_r1 --retrieval_topk 3 --retrieval_eval_ks 3 10 100 1000"
elif [ "$METHOD" = search_r1_raac ]; then
  ARGS="--generation_method deep_research --deep_research_model search_r1_raac --retrieval_topk 3 --retrieval_eval_ks 3 10 100 1000"
fi

nvidia-smi
python c5_task_evaluation/run_evalution.py \
  --pipeline generation \
  --dataset $NAME --subset test \
  --dataset_file dataset/TRQA/$DIR/test/queries_${SHORT}_test.jsonl \
  --qrel_file dataset/TRQA/$DIR/test/trec_qrels_${SHORT}_test.txt \
  --retriever spladepp --index_dir corpus_datasets/corpus --corpus_path $CORPUS \
  $ARGS \
  --devices 0 \
  --output_dir $BASE/results/${DATASET}_test/${METHOD}_spladepp_$RUN \
  --run $RUN $LIMIT
nvidia-smi