
import asyncio
import json
from tqdm.asyncio import tqdm
from prompts import GROUNDING_SYSTEM_PROMPT,GROUNDING_FEW_SHOTS




async def _find_alias_for_entity_async(client, question, canonical_entity, generated_entities, semaphore,
                                       max_retries=3):
    async with semaphore:
        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            *GROUNDING_FEW_SHOTS,
            {
                "role": "user",
                "content": f'''Question: "{question}"
Canonical KG Entity: "{canonical_entity}"
Actually Used Entities: {json.dumps(generated_entities)}'''
            }
        ]
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="",
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                data = json.loads(response.choices[0].message.content)
                if "best_match" in data:
                    return canonical_entity, data["best_match"]
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return canonical_entity, None


async def ground_missing_entities_async(client, question, canonical_entities, generated_entities, concurrency_limit=5):

    canonical_set = set(canonical_entities)
    generated_set = set(generated_entities)
    missing_entities = list(canonical_set - generated_set)

    if not missing_entities:
        return {"status": "success", "grounding_map": {}}


    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = [
        _find_alias_for_entity_async(client, question, missing_entity, generated_entities, semaphore)
        for missing_entity in missing_entities
    ]

    results = await tqdm.gather(*tasks, desc="Grounding Entities")

    grounding_map = {}
    all_grounded = True
    for canonical_entity, matched_alias in results:
        if matched_alias is not None and matched_alias in generated_set:

            grounding_map[matched_alias] = canonical_entity
        else:
            all_grounded = False
            break

    if all_grounded:
        return {"status": "success", "grounding_map": grounding_map}
    else:
        return {"status": "failed", "reason": f"Could not ground critical entity: {canonical_entity}"}






