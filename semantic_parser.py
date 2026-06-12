
import argparse
import asyncio
import json
import os
from typing import Optional

import torch
import networkx as nx
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from entity_grounder import ground_missing_entities_async
from prompts import SYSTEM_PROMPT_PARSER,FEW_SHOT_EXAMPLES_PARSER,SYSTEM_PROMPT_COREFERENCE_CHECK,FEW_SHOT_EXAMPLES_COREFERENCE_CHECK,SYSTEM_PROMPT_CHAINED_REPLAN,FEW_SHOT_EXAMPLES_CHAINED_REPLAN

async def verify_coreference_async(client, question, coref_group, model_name=""):

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_COREFERENCE_CHECK},
            *FEW_SHOT_EXAMPLES_COREFERENCE_CHECK,
            {
                "role": "user",
                "content": f'''Question: "{question}"\nPotential Co-reference Group: {json.dumps(coref_group)}'''
            }
        ]
        response = await client.chat.completions.create(
            model=model_name, messages=messages, temperature=0, response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("is_coreferent", False)
    except Exception as e:
        return False


def _check_connectivity(fragments: list, topic_entities: list, target_variable: str) -> bool:

    if not fragments:
        return target_variable in topic_entities

    g = nx.Graph()
    all_nodes_in_fragments = set()
    for frag in fragments:
        s = frag["subject"]
        o = frag["object"]
        g.add_node(s)
        g.add_node(o)
        g.add_edge(s, o)
        all_nodes_in_fragments.add(s)
        all_nodes_in_fragments.add(o)

    if not nx.is_connected(g):
        return False

    critical_nodes = set(topic_entities + [target_variable])
    if not critical_nodes.issubset(all_nodes_in_fragments):
        return False

    return True


async def get_logical_fragments_async(client, question, topic_entities, model_name="", max_retries=5):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PARSER},
        *FEW_SHOT_EXAMPLES_PARSER,
        {
            "role": "user",
            "content": f'Question: "{question}"\nTopic Entities: {json.dumps(topic_entities)}'
        }
    ]
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content

            if not response_content or not response_content.strip():
                continue

            try:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    clean_json_str = response_content[json_start:json_end]
                    data = json.loads(clean_json_str)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                await asyncio.sleep(wait_time)

    return None


async def _get_logical_fragments_chained_replan_async(
        client: AsyncOpenAI,
        question: str,
        topic_entities: list,
        disconnected_fragments: list,
        model_name: str = "",
        max_retries: int = 3
) -> Optional[dict]:

    fragments_str = json.dumps(disconnected_fragments, indent=2)
    user_prompt = f"""Question: "{question}"
Topic Entities: {json.dumps(topic_entities)}
Disconnected Fragments: {fragments_str}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CHAINED_REPLAN},
        *FEW_SHOT_EXAMPLES_CHAINED_REPLAN,
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
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

    return None

async def parse_question_orchestrator_async(client, question, topic_entities, semaphore, max_replan_attempts=5):

    async with semaphore:
        for attempt in range(max_replan_attempts):


            parsed_plan = await get_logical_fragments_async(client, question, topic_entities)

            if not parsed_plan or "target_variable" not in parsed_plan or not parsed_plan.get("logical_fragments") or "co_reference_groups" not in parsed_plan:
                continue

            fragments = parsed_plan["logical_fragments"]
            target_variable = parsed_plan["target_variable"]


            coref_groups = parsed_plan.get("co_reference_groups", [])
            if coref_groups:
                verified_groups = []
                for group in coref_groups:
                    is_verified = await verify_coreference_async(client, question, group)
                    if is_verified:
                        verified_groups.append(group)

                coref_groups = verified_groups


            all_used_entities = set()
            for frag in fragments:
                all_used_entities.add(frag.get("subject"))
                all_used_entities.add(frag.get("object"))

            missing_entities = set(topic_entities) - all_used_entities

            cleaned_fragments = fragments
            if missing_entities:
                grounding_result = await ground_missing_entities_async(client, question, topic_entities,
                                                                       list(all_used_entities))

                if grounding_result["status"] == "failed":
                    continue

                grounding_map = grounding_result["grounding_map"]

                temp_cleaned_fragments = []
                for frag in fragments:
                    s = grounding_map.get(frag["subject"], frag["subject"])
                    o = grounding_map.get(frag["object"], frag["object"])
                    temp_cleaned_fragments.append({"subject": s, "relation": frag["relation"], "object": o})
                cleaned_fragments = temp_cleaned_fragments


            is_connected = _check_connectivity(cleaned_fragments, topic_entities, target_variable)

            if not is_connected:
                re_planned_plan = await _get_logical_fragments_chained_replan_async(
                    client,
                    question,
                    topic_entities,
                    cleaned_fragments
                )

                if not re_planned_plan or not re_planned_plan.get("logical_fragments"):
                    continue

                re_planned_plan["co_reference_groups"] = coref_groups
                is_replan_connected = _check_connectivity(
                    fragments=re_planned_plan["logical_fragments"],
                    topic_entities=topic_entities,
                    target_variable=re_planned_plan["target_variable"]
                )

                if is_replan_connected:

                    cleaned_fragments = re_planned_plan["logical_fragments"]
                    target_variable = re_planned_plan["target_variable"]
                    is_connected = True
                else:
                    continue

            if is_connected:
                final_plan = {
                    "target_variable": target_variable,
                    "logical_fragments": cleaned_fragments,
                    "co_reference_groups": coref_groups
                }
                return {"status": "success", "plan": final_plan}


        return {"status": "failed", "plan": None}


async def process_sample_async(
    sample_id: str,
    sample_info: dict,
    cleaned_entities_map: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> dict:

    question = sample_info['question']
    topic_entities = cleaned_entities_map.get(sample_id)

    if not topic_entities:
        return {"id": sample_id, "status": "skipped_no_entities"}

    result = await parse_question_orchestrator_async(client, question, topic_entities, semaphore)


    if result and result["status"] == "success":
        final_topic_entities = cleaned_entities_map.get(sample_id, [])
        output_entry = {
            "id": sample_id,
            "question": question,
            "topic_entities": final_topic_entities,
            "answer_entities": sample_info.get('a_entity', []),
            **result["plan"]
        }
        return {"id": sample_id, "status": "success", "data": output_entry}
    else:
        return {"id": sample_id, "status": "failed_planning"}
async def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--input_path', type=str, default="")
    parser.add_argument('-c','--cleaned_entities_path', type=str, default="")
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=100)
    args = parser.parse_args()


    client = AsyncOpenAI(api_key="",base_url="")
    input_data = torch.load(args.input_path, map_location='cpu')
    try:
        with open(args.cleaned_entities_path, 'r', encoding='utf-8') as f:
            cleaned_entities_list = json.load(f)

        cleaned_entities_map = {item['id']: item['q_entity'] for item in cleaned_entities_list}
    except FileNotFoundError:
        return
    except (json.JSONDecodeError, KeyError) as e:
        return
    if args.output_path is None:
        dir_name = os.path.dirname(args.input_path)
        name, _ = os.path.splitext(os.path.basename(args.input_path))
        args.output_path = os.path.join(dir_name, f"{name}_semantic_plans.jsonl")

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue


    items_to_process = []
    for sample_id, sample_info in input_data.items():
        if sample_id not in processed_ids:
            items_to_process.append((sample_id, sample_info))

    if args.limit:
        items_to_process = items_to_process[:args.limit]

    if not items_to_process:
        return


    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for sample_id, sample_info in items_to_process:
        task = process_sample_async(sample_id, sample_info, cleaned_entities_map, client, semaphore)
        tasks.append(task)


    results = await tqdm.gather(*tasks)


    success_count = 0
    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        for result in results:
            if result and result["status"] == "success":
                f_out.write(json.dumps(result["data"]) + '\n')
                success_count += 1
            elif result and result["status"] == "failed_planning":
                with open("failed_samples.log", "a") as f_err:
                    f_err.write(f"{result['id']}\n")


if __name__ == '__main__':
     asyncio.run(main())
