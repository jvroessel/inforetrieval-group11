import json
import re

import torch
import transformers

from utils.general_utils import passages2string
from c5_task_evaluation.prompts.prompt_templetes import PROMPT_SEARCHR1
from c5_task_evaluation.methods.retrieval_augmented_models import SearchR1_Model


# --- Prompts from the paper's appendix (system messages, used as plain text) ---
# DECISION: Figure 13 describes four signals; this setup computes doc_novelty, and the two
# similarity signals only with an embedder. Adapt the text to the signals you actually use.
CONTROLLER_PROMPT = """
You are a trajectory decision maker for an iterative deep research agent. After each search iteration you receive productivity signals, signal history, and action history. Select one action to guide the agent's next step.
== Inputs ==
- Current iteration number
- Current signals: doc_novelty, consec_query_sim, aspect_coverage, orig_query_sim
- Recent signal history
- Recent action history
== Signal Definitions ==
Signals are listed in priority order. When signals conflict, follow priority.
1. doc_novelty [PRIMARY] (value in [0, 1]):
Fraction of retrieved documents not seen before.
High -> fresh information being discovered. Low -> recycling old documents.
This is the most important signal. Decisions should be driven primarily by doc_novelty and its trend over recent iterations.
2. consec_query_sim [SUPPORTIVE] (value in [0, 1]):
Cosine similarity between the current and previous query embeddings.
High -> near-identical queries (repetition). Low -> diverse queries.
Use this to confirm or reinforce patterns observed in doc_novelty. High consec_query_sim alongside dropping doc_novelty is a strong stagnation signal; on its own it is less
decisive.
3. aspect_coverage [CONFIRMATORY] (structured summary):
Tracks how many information-need aspects of the original query have been addressed.
Reports: covered/total aspects, partial/total, not_covered/total, a list of critical gaps (aspects with no evidence), and a list of minor gaps (aspects with partial evidence).
Use this to modulate stop/intervene decisions already indicated by doc_novelty. All aspects (partially or fully) covered confirms stop; not_covered aspects favor intervene
over stop. When doc_novelty is moderate-to-high, ignore this signal.
4. orig_query_sim [GUARDRAIL] (value in [0, 1]):
Cosine similarity between the current query embedding and the original user question.
High -> on-topic. Low -> drifted from the original question.
This signal does not drive continue/intervene/stop decisions. Its sole role is as a safety check: only act on it when it drops very low, indicating the agent is generating
unrelated subqueries and retrieving irrelevant documents.
== Action Definitions ==
"continue" — The trajectory is productive; no intervention needed.
Choose when:
- [Primary] doc_novelty is moderate-to-high, indicating fresh documents are still being found.
- [Supportive] consec_query_sim is low-to-moderate, confirming query diversity.
- [Confirmatory] aspect_coverage does not affect this decision. When doc_novelty is healthy, continue regardless of coverage status.
- [Guardrail] orig_query_sim is not alarmingly low (the agent has not drifted into irrelevant territory).
"intervene" — The trajectory is stale but recoverable. An external nudge (e.g., critical thinking) can redirect the search.
Choose when:
- [Primary] doc_novelty has dropped noticeably and sustained a decline over recent iterations. This is the main trigger for intervention.
- [Supportive] consec_query_sim is high, reinforcing the stagnation pattern (query loop).
- [Confirmatory] Not_covered aspects alongside near-zero doc_novelty favors intervene over stop. Key aspects of the question remain unaddressed, so stopping would
produce an incomplete result.
- [Guardrail] A very low orig_query_sim alone can also trigger intervention if the agent has drifted far off-topic, even when doc_novelty appears adequate (novel but
irrelevant documents).
- If a prior "intervene" was issued, enough iterations have passed to assess its effect.
"stop" — The trajectory is exhausted. Further iterations are unlikely to yield new information.
Choose when:
- [Primary] doc_novelty has been near zero over many consecutive iterations. This is the essential condition for stopping.
- [Supportive] consec_query_sim has been persistently high over the same stretch, confirming the agent is stuck.
- [Confirmatory] All aspects (partially or fully) covered alongside near-zero doc_novelty strengthens the case for stopping. The agent has addressed every facet and is out of
new information.
- Multiple well-spaced interventions have failed to recover doc_novelty.
== Decision Guidelines ==
1. Default to "continue" when the trajectory is healthy. Expect natural signal fluctuation.
2. Weight signals by iteration stage: early iterations naturally have higher novelty and lower query similarity, so favor exploration and prefer "continue" unless doc_novelty is
severely low. Later iterations tend to converge, making the same novelty drop more significant.
3. Respect the agent's self-correction ability. After issuing "intervene" or before resorting to "stop", wait several iterations — the agent may recover on its own. Avoid
back-to-back interventions or premature stops.
4. Reserve "stop" for thoroughly demonstrated exhaustion: doc_novelty near zero over many iterations, multiple interventions attempted, each given time, none recovered
doc_novelty. Full aspect coverage confirms stop; uncovered aspects favor intervene over stop. Do not escalate to "stop" without a sufficient observation window after the last
intervention.
== Output Format ==
Respond with ONLY a raw JSON object — no markdown, no code blocks.
Keys:
- "reasoning": Brief explanation referencing current signals, recent history, and action history.
- "action": One of "continue", "intervene", "stop".
Example: {"reasoning": "your reasoning here", "action": "your selected action here"}
"""
RETHINK_PROMPT = """
You are a critical rethinking agent for an iterative deep-research agent.
The research agent answers complex questions by repeatedly generating sub-queries, retrieving documents, and reasoning over the results.
A trajectory tracker monitors productivity signals (document novelty, query similarity) after each iteration. When those signals indicate the trajectory has stagnated — retrieving the same documents and repeating similar queries — the tracker triggers you to redirect the search.
Your task is to:
1. Critically assess why the current search trajectory is stuck.
2. Generate a creative, fundamentally different search query to break out of the loop.
== Inputs ==
- Original question: The user's top-level research question. Your new query must remain relevant to this question.
- Last N turns: The most recent search iterations, each containing:
  - Think: The agent's reasoning about what to search next.
  - Search query: The sub-query the agent issued to the retriever.
== Output Format ==
You MUST respond with ONLY a valid JSON object (no markdown code blocks, no ```json tags).
The JSON object must have exactly these two keys:
1. "reasoning":
  - Explain why the search trajectory is stuck
  - Identify which query patterns are being repeated and why they fail to retrieve new relevant documents
  - Suggest a fundamentally different angle or perspective to explore
2. "search_query" (a new search query):
  - A creative, fundamentally different search query to redirect the trajectory
  - Must target an unexplored aspect of the original question and use substantially different terminology
  - Must be concise, keyword-based, and specific
Example structure:
{
  "reasoning": "your reasoning here",
  "search_query": "your new search query here"
}
CRITICAL: Return ONLY the raw JSON object. Do NOT wrap it in markdown code blocks.
"""

VALID_ACTIONS = ("continue", "intervene", "stop")


def parse_json(text):
    """Take the first {...} block out of an LLM answer and parse it; None if that fails."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class RAACController:
    def __init__(self, args):
        self.args = args
        model_name = getattr(args, "raac_model", "Qwen/Qwen2.5-7B-Instruct")
        quant = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,   # RTX 2080 Ti: no native bf16
        )
        # DECISION: where the controller lives. "auto" spreads it over the visible GPUs;
        # with a third GPU you could use {"": 2} to put it on GPU 2 only.
        self.tok = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            device_map=getattr(args, "raac_device_map", "auto"),
        )
        self.model.eval()
        self.history_len = getattr(args, "raac_history_len", 5)   # DECISION: how many past turns to show

        # Optional embedder for diversity / similarity (DECISION: which model, or none = N only).
        self.embedder = None
        emb_name = getattr(args, "raac_embedder", None)
        if emb_name:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(emb_name, device="cuda:0")

    # --- per-question state -------------------------------------------------
    def reset(self, question):
        self.question = question
        self.seen_ids = set()
        self.prev_query_emb = None
        self.question_emb = self._embed(question) if self.embedder else None
        self.signal_history = []
        self.action_history = []

    # --- signals ------------------------------------------------------------
    def _embed(self, text):
        return self.embedder.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    def add_seen(self, docs):
        """Mark documents as seen without computing signals (e.g. documents from an intervention)."""
        self.seen_ids.update(d["id"] for d in docs)

    def observe(self, query, docs):
        """Signals for one iteration. docs = the documents given to the agent this step."""
        ids = [d["id"] for d in docs]
        if ids:
            novelty = sum(1 for i in ids if i not in self.seen_ids) / len(ids)
        else:
            novelty = None   # DECISION: a step without a search query or results
        self.seen_ids.update(ids)
        signals = {"doc_novelty": novelty}

        if self.embedder and query:
            q = self._embed(query)
            signals["consec_query_sim"] = (
                None if self.prev_query_emb is None else float(q @ self.prev_query_emb)
            )   # the paper's diversity is 1 - this value
            signals["orig_query_sim"] = float(q @ self.question_emb)
            self.prev_query_emb = q

        signals = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in signals.items()}
        self.signal_history.append(signals)
        return signals

    # --- LLM calls ----------------------------------------------------------
    @torch.no_grad()
    def _chat(self, system, user, max_new_tokens=300):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,   # DECISION: greedy or sampled
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def decide(self, iteration, signals):
        """Returns (action, reasoning, raw_output)."""
        if not CONTROLLER_PROMPT:
            action, reasoning, raw = "continue", "no controller prompt (dry run)", ""
        else:
            # signal_history already ends with the current signals; show the ones before it
            past_signals = self.signal_history[:-1][-self.history_len:]
            user = (
                f"Current iteration: {iteration}\n"
                f"Current signals: {json.dumps(signals)}\n"
                f"Recent signal history (oldest first): {json.dumps(past_signals)}\n"
                f"Recent action history (oldest first): {json.dumps(self.action_history[-self.history_len:])}"
            )
            raw = self._chat(CONTROLLER_PROMPT, user)
            parsed = parse_json(raw) or {}
            action = str(parsed.get("action", "")).strip().lower()
            reasoning = parsed.get("reasoning", "")
            if action not in VALID_ACTIONS:
                action = "continue"   # DECISION: fallback when the answer can't be read
        self.action_history.append(action)
        return action, reasoning, raw

    def rethink(self, recent_turns):
        """Returns (new_query or None, reasoning, raw_output)."""
        if not RETHINK_PROMPT:
            return None, "", ""
        turns = "\n".join(
            f"Turn {i}:\n  Think: {t['think']}\n  Search query: {t['search_query']}"
            for i, t in enumerate(recent_turns, 1)
        )
        user = f"Original question: {self.question}\n\nLast {len(recent_turns)} turns:\n{turns}"
        raw = self._chat(RETHINK_PROMPT, user)
        parsed = parse_json(raw) or {}
        query = parsed.get("search_query")
        return (query.strip() if isinstance(query, str) and query.strip() else None), parsed.get("reasoning", ""), raw


class SearchR1_RAAC_Model(SearchR1_Model):
    """The TRQA SearchR1_Model loop, with a RAAC decision after every retrieval step."""

    # DECISION: how an intervention appears to Search-R1. This writes it in Search-R1's own
    # tag format, as if it were an extra step of the agent itself.
    intervention_template = "\n\n<think>{reasoning}</think>\n<search>{query}</search>\n\n<information>{results}</information>\n\n"

    def __init__(self, device, args):
        super().__init__(device, args)   # loads Search-R1 + SPLADE++ exactly like the baseline
        self.raac = RAACController(args)

    def _recent_turns(self, reasoning_path):
        turns = [s for s in reasoning_path if "search_query" in s][-self.raac.history_len:]
        return [{"think": s.get("think", ""), "search_query": s.get("search_query")} for s in turns]

    def inference(self, question, generation_temp=0.7, qid=None):
        input_prompt = PROMPT_SEARCHR1.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        self.raac.reset(question)

        reasoning_path = []
        for step_idx in range(self.args.max_iter):   # DECISION: does an intervention count as an iteration?
            output_text, output_ = self.generator.generate(
                messages, self.generator.rar_stopping_criteria, temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break   # Search-R1 answered by itself

            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs, ranked_list = self._search_for_step(tmp_query, qid=qid)
                search_results = passages2string(search_docs)
            else:
                search_docs, ranked_list, search_results = [], None, ""
            step_entry = {"think": self.get_think(output_text), "search_query": tmp_query, "docs": search_docs}
            if ranked_list is not None:
                step_entry["ranked_list"] = ranked_list

            # --- RAAC ---
            signals = self.raac.observe(tmp_query, search_docs)
            action, why, _raw = self.raac.decide(step_idx + 1, signals)
            step_entry["raac"] = {"signals": signals, "action": action, "reasoning": why}
            reasoning_path.append(step_entry)

            input_prompt += self.curr_step_template.format(output_text=output_text, search_results=search_results)

            if action == "intervene":
                new_query, crit, _raw = self.raac.rethink(self._recent_turns(reasoning_path))
                if new_query:
                    docs2, ranked2 = self._search_for_step(new_query, qid=qid)
                    self.raac.add_seen(docs2)
                    entry2 = {"think": crit, "search_query": new_query, "docs": docs2,
                              "raac": {"type": "intervention"}}
                    if ranked2 is not None:
                        entry2["ranked_list"] = ranked2   # so the intervention counts in DRA recall
                    reasoning_path.append(entry2)
                    input_prompt += self.intervention_template.format(
                        reasoning=crit, query=new_query, results=passages2string(docs2))

            elif action == "stop":
                # DECISION: how the answer is forced. Here Search-R1 continues its own history,
                # but generation stops at </answer> instead of </search>.
                messages = [{"role": "user", "content": input_prompt}]
                output_text, output_ = self.generator.generate(
                    messages, self.generator.rar_answer_stopping_criteria, temperature=generation_temp
                )
                break

            messages = [{"role": "user", "content": input_prompt}]

        one_step_think = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path.append({"think": one_step_think, "prediction": prediction})
        return reasoning_path, prediction