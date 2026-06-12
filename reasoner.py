import argparse
import asyncio
import json
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import torch
from prompts import SYSTEM_PROMPT_ARBITER,FEW_SHOT_EXAMPLES_ARBITER,SYSTEM_PROMPT_GROUNDED_FALLBACK,FEW_SHOT_EXAMPLES_GROUNDED_FALLBACK

class LLMArbiter:


    def __init__(self, sample_data: Dict[str, Any], llm_client: AsyncOpenAI):
        self.sample_data = sample_data
        self.llm_client = llm_client

    def _textualize_path(self, path: List[Any]) -> str:

        if not path or not isinstance(path, list):
            return str(path)

        flat_path = [path[0][0]]
        for h, r, t in path:
            flat_path.append(r)
            flat_path.append(t)
        return " - ".join(flat_path)

    def _format_decision_brief(self, portfolios: List[Dict[str, Any]]) -> str:
        brief_parts = ["Here is the Decision Brief for your review.",f"\n[Question]\n{self.sample_data['question']}\n"]
        for i, portfolio in enumerate(portfolios):
            brief_parts.append("=" * 40)
            brief_parts.append(f"[Candidate Answer: {portfolio['candidate_answer']}]\n")
            brief_parts.append("[Supporting Evidence]\n")

            evidence_locker = portfolio.get("primary_evidence", {}).get("evidence_locker", {})
            if not evidence_locker:
                brief_parts.append("NO_EVIDENCE_FOUND.\n")
                continue

            for j, (constraint_id, path_data) in enumerate(evidence_locker.items()):
                if path_data == "NO_EVIDENCE_FOUND":
                    continue
                path_text = self._textualize_path(path_data)
                brief_parts.append(f"{j + 1}. {path_text}\n")

        brief_parts.append("=" * 40)
        return "\n".join(brief_parts)

    async def _make_llm_call(self, system_prompt: str, user_prompt: str, few_shot_examples: List[dict],
                             max_retries: int = 3) -> Dict[str, Any]:

        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_examples,
            {"role": "user", "content": user_prompt}
        ]
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model="",
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content

                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    return json.loads(response_content[json_start:json_end])


            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return {"error": "API call failed after multiple retries."}



    async def run(self):
        portfolios = self.sample_data.get("portfolios", [])

        concrete_candidates = [p for p in portfolios if p.get("candidate_answer") != "UNCERTAIN"]

        if not concrete_candidates:
            return await self._run_grounded_fallback()
        else:
            decision_brief = self._format_decision_brief(concrete_candidates)
            response = await self._make_llm_call(SYSTEM_PROMPT_ARBITER, decision_brief, FEW_SHOT_EXAMPLES_ARBITER)
            return {"mode": "evidence_based", "result": response}





    async def _run_grounded_fallback(self, top_k_triples=200):

        scored_triples = self.sample_data.get("scored_triples", [])
        sorted_triples = sorted(scored_triples, key=lambda x: x[3], reverse=True)
        top_triples = sorted_triples[:top_k_triples]
        triples_text = "\n".join([f"- ({h}, {r}, {t})" for h, r, t, s in top_triples])
        user_prompt = f"""Please perform an expert analysis based on the following information.

    [Question]
    {self.sample_data['question']}

    [Knowledge Graph Triples]
    {triples_text}
    """
        response = await self._make_llm_call(
            SYSTEM_PROMPT_GROUNDED_FALLBACK,
            user_prompt,
            FEW_SHOT_EXAMPLES_GROUNDED_FALLBACK
        )
        return {"mode": "grounded_fallback", "result": response}
async def process_sample(sample_data: Dict[str, Any],
                         llm_client: AsyncOpenAI,
                         semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    async with semaphore:
        sample_id = sample_data.get('id')
        try:
            arbiter = LLMArbiter(sample_data, llm_client)
            arbitration_result = await arbiter.run()

            answer_to_score_map = {
                p['candidate_answer']: p['cumulative_log_prob']
                for p in sample_data.get('portfolios', [])
            }

            if arbitration_result["mode"] == "evidence_based":
                ranked_candidates = arbitration_result["result"].get("ranking", [])
                for item in ranked_candidates:
                    candidate_name = item.get("candidate")
                    item["score"] = answer_to_score_map.get(candidate_name, -float('inf'))
            elif arbitration_result["mode"] == "grounded_fallback":
                arbitration_result["result"]["score"] = 0.0

            final_output = {
                "id": sample_data.get("id"),
                "question": sample_data.get("question"),
                "answer_entities": sample_data.get("answer_entities", []),
                "arbitration": arbitration_result
            }
            return final_output

        except Exception as e:

            return {
                "id": sample_data.get("id"),
                "question": sample_data.get("question"),
                "answer_entities": sample_data.get("answer_entities", []),
                "arbitration": {"error": str(e)}
            }



async def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p','--portfolios_path', type=str, default="")
    parser.add_argument('--features_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=100)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    llm_client = AsyncOpenAI(api_key="",base_url="")



    with open(args.portfolios_path, 'r', encoding='utf-8') as f:
        portfolios_data = {item['id']: item for item in [json.loads(line) for line in f]}

    features_data = torch.load(args.features_path, map_location='cpu')

    all_sample_data = {}
    for sample_id, features in features_data.items():
        if sample_id in portfolios_data:
            combined_data = features.copy()
            combined_data.update(portfolios_data[sample_id])
            all_sample_data[sample_id] = combined_data


    if args.output_path is None:
        dir_name = os.path.dirname(args.portfolios_path)
        name, _ = os.path.splitext(os.path.basename(args.portfolios_path))
        args.output_path = os.path.join(dir_name, f"{name.replace('_evidence_portfolios', '')}_final_answers.jsonl")

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    all_samples_as_list = list(all_sample_data.values())
    items_to_process = [item for item in all_samples_as_list if item['id'] not in processed_ids]
    if args.limit:
        items_to_process = items_to_process[:args.limit]

    if not items_to_process:
        return

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = []
    for sample_data in items_to_process:
        tasks.append(process_sample(sample_data, llm_client, semaphore))

    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Arbitrating Answers"):
            result_with_original_data = await future
            if result_with_original_data:
                f_out.write(json.dumps(result_with_original_data) + '\n')
                f_out.flush()



if __name__ == '__main__':

    asyncio.run(main())

