import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from src.model.text_encoders import GTELargeEN
from openai import AsyncOpenAI
import asyncio
import numpy as np
from tqdm.asyncio import tqdm
import torch
from dataclasses import dataclass
from typing import List, Tuple, Set ,Any
import heapq
from sentence_transformers import util
from prompts import SYSTEM_PROMPT_PATH_RERANKER,FEW_SHOT_EXAMPLES_PATH_RERANKER,ALIGNMENT_SYSTEM_PROMPT,ALIGNMENT_FEW_SHOTS


MAX_ALIGNMENT_ATTEMPTS = 3
CONFIDENCE_MAP = {"High": 0.9, "Medium": 0.5, "Low": 0.2, None: 0.1, "UNCERTAIN": 0.1}
CONFIDENCE_LEVEL_PRIORITY = {"High": 4, "Medium": 3, "Low": 2, None: 1}


call = 0

class DSU:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root2] = root1


@dataclass
class BeamState:
    state_id: int
    score: float
    path: List[Tuple[str, str, str]]
    endpoint: str
    nodes_in_path: Set[str]
    path_quality_score: float
    heuristic_score: float
    parent_state: Any = None
    score_structural: float = 0.0
    score_coherence: float = 0.0
    h_score_raw: float = 0.0
    score_holistic: float = 0.0

class ConstraintVerifier:


    def __init__(self, query_plan, subgraph_data, text_encoder, llm_client=None):
        self.beam_width = 10
        self.query_plan = query_plan
        self.subgraph_data = subgraph_data
        self.text_encoder = text_encoder
        self.llm_client = llm_client
        self.subgraph = [(h, r, t) for h, r, t, s in self.subgraph_data.get('scored_triples', [])]
        original_triples = subgraph_data.get('scored_triples', [])
        normalized_triples_map = {}
        for h, r, t, s in original_triples:

            if (h, r, t) not in normalized_triples_map or s > normalized_triples_map[(h, r, t)]:
                normalized_triples_map[(h, r, t)] = s
            r_inv = f"{r}.inv"
            if (t, r_inv, h) not in normalized_triples_map or s > normalized_triples_map[(t, r_inv, h)]:
                normalized_triples_map[(t, r_inv, h)] = s

        self.subgraph_with_inv = [(h, r, t, s) for (h, r, t), s in normalized_triples_map.items()]

        self.adjacency_list = defaultdict(list)
        for h, r, t, s in self.subgraph_with_inv:
            self.adjacency_list[h].append((h, r, t))
        self.target_variable = query_plan.get('target_variable')
        if self.target_variable and not self.target_variable.startswith('?'):
            self.target_variable = f"?{self.target_variable.replace(' ', '_')}"

        all_constraints = query_plan.get('constraint_indicators', [])
        co_ref_groups = query_plan.get('co_reference_groups', [])

        all_variables = self._get_all_variables(all_constraints)
        dsu = self._build_dsu(all_variables, co_ref_groups)

        self.execution_plan = self._rewrite_constraints(all_constraints, dsu)


        self.scored_triples_map = { (h, r, t): s for h, r, t, s in self.subgraph_data.get('scored_triples', []) }
        self.dsrg = self.subgraph_data.get('dsrg', {})
        self.indicator_embeddings = self.subgraph_data.get('indicator_embeddings', {})
        self.variable_target_vectors = self.subgraph_data.get('variable_target_vectors', {})
        self.entity_embeddings = self.subgraph_data.get('entity_embeddings', {})
        self.contextual_type_vectors = self.subgraph_data.get('contextual_type_vectors', {})
        self.shortest_path_distances = self.subgraph_data.get('shortest_path_distances', {})

    def _get_all_variables(self, constraints):
        all_variables = set()
        for constraint in constraints:
            path = constraint.get("indicator_sub_path", "")
            found_vars = self._extract_variables_from_path(path)
            all_variables.update(found_vars)
        return all_variables

    def _build_dsu(self, variables, groups):
        dsu = DSU(variables)
        for group in groups:
            norm_group = [f"?{var.replace('?', '')}" for var in group]
            if len(norm_group) > 1:
                first_var = norm_group[0]
                for other_var in norm_group[1:]:
                    if first_var in variables and other_var in variables:
                        dsu.union(first_var, other_var)
        return dsu

    def _rewrite_constraints(self, constraints, dsu):

        rewritten_constraints = []
        for constraint in constraints:
            new_constraint = constraint.copy()
            path_str = new_constraint.get("indicator_sub_path", "")

            vars_in_path = self._extract_variables_from_path(path_str)
            for var in vars_in_path:
                representative = dsu.find(var)
                if var != representative:
                    path_str = re.sub(r'\b' + re.escape(var) + r'\b', representative, path_str)

            new_constraint["indicator_sub_path"] = path_str
            if new_constraint["start_entity"].startswith("?"):
                new_constraint["start_entity"] = dsu.find(new_constraint["start_entity"])

            rewritten_constraints.append(new_constraint)
        return rewritten_constraints


    async def run(self):

        hypothesis_beam = [{
            "bindings": {},
            "evidence_locker": {},
            "cumulative_log_prob": 0.0
        }]


        for i, constraint in enumerate(self.execution_plan):
            constraint_id = constraint.get('constraint_id')

            tasks_by_input_context = defaultdict(list)
            required_vars = self._get_input_vars_for_constraint(constraint)

            for hypo in hypothesis_beam:
                context_bindings = {
                    var: hypo['bindings'][var]
                    for var in required_vars if var in hypo['bindings']
                }

                context_key = frozenset(
                    (var, frozenset(binding.items())) for var, binding in context_bindings.items()
                )
                tasks_by_input_context[context_key].append(hypo)

            async_tasks = []
            unique_contexts = list(tasks_by_input_context.keys())

            for context_key in unique_contexts:
                task_bindings = {var: dict(binding_frozenset) for var, binding_frozenset in context_key}
                task = self._execute_constraint_for_context(constraint, task_bindings)
                async_tasks.append(task)

            all_results_list = await asyncio.gather(*async_tasks)
            context_to_results_map = {unique_contexts[i]: res_list for i, res_list in enumerate(all_results_list)}

            next_hypothesis_pool = []
            for context_key, parent_hypotheses in tasks_by_input_context.items():
                execution_results = context_to_results_map.get(context_key, [])

                best_result_for_binding = {}
                for result in execution_results:
                    new_bindings = result.get("new_bindings", {})
                    binding_key = frozenset(
                        (var, val.get("value")) for var, val in new_bindings.items()
                    )
                    if (binding_key not in best_result_for_binding or
                            result["evidence_log_prob"] > best_result_for_binding[binding_key]["evidence_log_prob"]):
                        best_result_for_binding[binding_key] = result

                unique_results = list(best_result_for_binding.values())

                for parent_hypo in parent_hypotheses:
                    for result in unique_results:
                        new_hypothesis = self._create_new_hypothesis(parent_hypo, constraint_id, result)
                        next_hypothesis_pool.append(new_hypothesis)

            if not next_hypothesis_pool:
                hypothesis_beam = []
                break

            next_hypothesis_pool.sort(key=lambda h: h['cumulative_log_prob'], reverse=True)

            if i == 0:

                concrete_hypotheses = []
                uncertain_hypotheses = []
                for hypo in next_hypothesis_pool:
                    target_binding = hypo['bindings'].get(self.target_variable)
                    if target_binding and target_binding.get("value") == "UNCERTAIN":
                        uncertain_hypotheses.append(hypo)
                    else:
                        concrete_hypotheses.append(hypo)
                meaningful_hypotheses = []
                for concrete in concrete_hypotheses:
                    target_binding = concrete['bindings'].get(self.target_variable)
                    if not target_binding.get("value").startswith("m.") and not target_binding.get("value").startswith("g."):
                        meaningful_hypotheses.append(concrete)

                if concrete_hypotheses:
                    hypothesis_beam = meaningful_hypotheses
                elif uncertain_hypotheses:
                    hypothesis_beam = uncertain_hypotheses[:1]
                else:
                    hypothesis_beam = []

            else:
                hypothesis_beam = next_hypothesis_pool[:self.beam_width]


        return self._assemble_portfolios(hypothesis_beam)

    def _get_input_vars_for_constraint(self, constraint: dict) -> set:
        path_str = constraint.get("indicator_sub_path", "")
        return set(self._extract_variables_from_path(path_str))

    def _create_new_hypothesis(self, parent_hypothesis, constraint_id, execution_result):
        new_bindings = execution_result.get("new_bindings", {})

        extended_bindings = {**parent_hypothesis['bindings'], **new_bindings}
        extended_evidence = {**parent_hypothesis['evidence_locker'],
                             constraint_id: execution_result.get("evidence_path")}

        result_log_prob = execution_result.get("evidence_log_prob", np.log(0.1))
        new_cumulative_log_prob = parent_hypothesis['cumulative_log_prob'] + result_log_prob

        return {
            "bindings": extended_bindings,
            "evidence_locker": extended_evidence,
            "cumulative_log_prob": new_cumulative_log_prob
        }


    async def _execute_find_task(self, constraint: dict, existing_bindings: dict) -> List[dict]:


        constraint_id = constraint.get('constraint_id', '')
        if constraint_id.startswith('MC'):
            top_k = 50
        else:
            top_k = 10

        logical_hops = len(constraint["indicator_sub_path"].split(' -> ')) // 2

        top_path_states = await self._run_beam_search(
            constraint=constraint,
            existing_bindings=existing_bindings,
            task_type="FIND",
            beam_width=10,
            max_depth= logical_hops + 1,
            top_k_results=top_k
        )

        if not top_path_states:
            return [{"status": "failed", "new_bindings": {},"evidence_log_prob": np.log(0.1), "evidence_path": "NO_EVIDENCE_FOUND"}]

        alignment_tasks = []
        for path_state in top_path_states:
            task = self._align_variables_with_llm(
                path_state=path_state,
                constraint=constraint,
                existing_bindings=existing_bindings
            )
            alignment_tasks.append(task)

        all_alignment_results = []
        if alignment_tasks:
            all_alignment_results = await asyncio.gather(*alignment_tasks)


        final_execution_results = []
        for result_list in all_alignment_results:
            final_execution_results.extend(result_list)

        return final_execution_results

    async def _execute_verify_task(self, constraint: dict, existing_bindings: dict, verification_info: dict) -> List[
        dict]:

        logical_hops = len(constraint["indicator_sub_path"].split(' -> ')) // 2
        top_path_states = await self._run_beam_search(
            constraint=constraint,
            existing_bindings=existing_bindings,
            task_type="VERIFY",
            verification_target_value=verification_info.get("target_value"),
            beam_width=10,
            max_depth= logical_hops + 1,
            top_k_results=10
        )

        if not top_path_states:
            return [{"status": "failed", "new_bindings": {}, "evidence_log_prob": np.log(0.1), "evidence_path": "NO_EVIDENCE_FOUND"}]

        if len(top_path_states) == 1:
            best_path_state = top_path_states[0]
        else:
            try:
                reranked_paths = await self._rerank_paths_with_llm(top_path_states, constraint)
                best_path_state = reranked_paths[0]
            except (ValueError, IndexError):
                best_path_state = top_path_states[0]

        alignment_results_list = await self._align_variables_with_llm(
            path_state=best_path_state,
            constraint=constraint,
            existing_bindings=existing_bindings
        )
        return alignment_results_list

    async def _make_llm_call(self,
                             system_prompt: str,
                             user_prompt: str,
                             few_shot_examples: List[dict],
                             model_name: str = "",
                             max_retries: int = 3,
                             initial_delay: int = 2) -> dict:

        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_examples,
            {"role": "user", "content": user_prompt}
        ]
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                response_content = response.choices[0].message.content

                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    return json.loads(json_str)

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = initial_delay ** attempt
                    await asyncio.sleep(delay)

        return {"error": f"API call failed after {max_retries} retries."}
    async def _rerank_paths_with_llm(self, path_states: List[BeamState], constraint: dict) -> List[BeamState]:

        if len(path_states) <= 1:
            return path_states

        question = self.query_plan['question']
        indicator = constraint['indicator_sub_path']

        candidate_paths_text_list = []
        for i, state in enumerate(path_states):
            path_str = self._format_path_for_llm(state.path)
            candidate_paths_text_list.append(f"{i + 1}. {path_str}")

        candidate_paths_text = "\n".join(candidate_paths_text_list)

        user_prompt = f"""CONTEXT:
    - Overall Question: "{question}"
    - Logical Requirement (Indicator): "{indicator}"

    Candidate Evidence Paths:
    {candidate_paths_text}

    Please re-rank these paths from 1 to {len(path_states)} based on the criteria provided."""

        llm_response = await self._make_llm_call(
            SYSTEM_PROMPT_PATH_RERANKER,
            user_prompt,
            FEW_SHOT_EXAMPLES_PATH_RERANKER
        )

        try:
            if not llm_response or "ranking" not in llm_response:
                raise ValueError("")

            reranked_indices = [item['path_index'] - 1 for item in llm_response['ranking']]

            if len(reranked_indices) != len(path_states) or len(set(reranked_indices)) != len(path_states):
                raise ValueError("")

            reranked_states = [path_states[i] for i in reranked_indices if 0 <= i < len(path_states)]

            if len(reranked_states) != len(path_states):
                raise ValueError("")

            return reranked_states

        except (ValueError, TypeError, KeyError, IndexError) as e:
            return path_states


    async def _execute_constraint_for_context(self, constraint: dict, context_bindings: dict) -> List[dict]:
        task_type = "FIND"
        verification_info = {}
        last_entity = constraint["indicator_sub_path"].split(' -> ')[- 1]
        if not last_entity.startswith('?'):
            task_type = "VERIFY"
            verification_info = {"target_value": last_entity}


        if task_type == "FIND":
            return await self._execute_find_task(constraint, context_bindings)
        else:
            return await self._execute_verify_task(constraint, context_bindings, verification_info)

    def _get_output_vars(self, constraint, existing_bindings):
        path = constraint.get("indicator_sub_path", "")
        start_entity = constraint.get("start_entity")

        all_vars_in_path = self._extract_variables_from_path(path)

        output_vars = all_vars_in_path - {start_entity}


        final_new_vars = output_vars - set(existing_bindings.keys())

        return list(final_new_vars)


    def _assemble_portfolios(self, final_solution_table: List[dict]) -> List[dict]:

        if not self.target_variable or not final_solution_table:
            return []

        portfolios_by_answer = defaultdict(list)
        for solution in final_solution_table:
            if self.target_variable in solution["bindings"]:
                answer = solution["bindings"][self.target_variable]["value"]
                portfolios_by_answer[answer].append(solution)

        final_portfolios = []

        for answer, solutions in portfolios_by_answer.items():

            solutions.sort(key=lambda s: s['cumulative_log_prob'], reverse=True)

            best_solution = solutions[0]

            portfolio = {
                "candidate_answer": answer,
                "cumulative_log_prob": best_solution["cumulative_log_prob"],
                "reasoning_paths_count": len(solutions),
                "primary_evidence": {
                    "full_solution_bindings": best_solution["bindings"],
                    "evidence_locker": best_solution["evidence_locker"]
                }
            }
            final_portfolios.append(portfolio)

        final_portfolios.sort(key=lambda p: p['cumulative_log_prob'], reverse=True)

        return final_portfolios


    def _get_neighbors(self, node: str) -> List[Tuple[str, str, str]]:
        return self.adjacency_list.get(node, [])

    def _is_path_complete(self, state: BeamState, constraint: dict, task_type: str,
                          verification_target_value: str = None) -> bool:

        if task_type == "VERIFY":
            endpoint = state.path[-1][2]
            return endpoint == verification_target_value

        else:
            expected_hops = (len(constraint["indicator_sub_path"].split(' -> ')) // 2)
            current_hops = len(state.path)
            return current_hops >= expected_hops and not state.path[-1][2].startswith("m.")

    def _textualize_path(self, path: List[Tuple[str, str, str]]) -> str:
        if not path: return ""
        text_parts = [path[0][0]]
        for h, r, t in path:
            text_parts.append(r)
            text_parts.append(t)
        return " ".join(text_parts)

    def _prefix(self, path: List[Any], for_comparison: bool = False) -> Any:
        if not path or not isinstance(path, list):
            return str(path)

        flat_path = [path[0][0]]
        for h, r, t in path:
            flat_path.append(r)
            flat_path.append(t)

        if for_comparison:
            return tuple(flat_path)

    def _textualize_path_for_print(self, path: List[Tuple[str, str, str]]) -> str:
        if not path: return ""
        text_parts = [path[0][0]]
        for h, r, t in path:
            text_parts.append(r)
            text_parts.append(t)
        return " - ".join(text_parts)

    def _format_path_for_llm(self, path: List[Tuple[str, str, str]]) -> str:
        if not path: return ""
        text_parts = [path[0][0]]
        for h, r, t in path:
            text_parts.append(r)
            text_parts.append(t)
        return " - ".join(text_parts)


    def _normalize_scores(self, states: List[BeamState], score_attribute: str):

        scores = [getattr(state, score_attribute, 0.0) for state in states]
        min_score, max_score = min(scores), max(scores)


        if max_score == min_score:
            for state in states:
                setattr(state, f"{score_attribute}_norm", 0.5)
            return

        for state in states:
            score = getattr(state, score_attribute, 0.0)
            norm_score = (score - min_score) / (max_score - min_score)
            setattr(state, f"{score_attribute}_norm", norm_score)

    async def _rank_and_score_candidates(self,
                                         candidate_states: List[BeamState],
                                         constraint: dict,
                                         existing_bindings: dict,
                                         task_type: str,
                                         verification_target_value: str = None) -> List[BeamState]:

        if not candidate_states:
            return []


        paths_to_embed_texts = [self._textualize_path(state.path) for state in candidate_states]


        if paths_to_embed_texts:
            path_embeddings = self.text_encoder.embed(paths_to_embed_texts)
        else:
            path_embeddings = []
        embedding_map = {state.state_id: emb for state, emb in zip(candidate_states, path_embeddings)}

        for state in candidate_states:
            state.score_structural = self._get_structural_score(state.path[-1])


            path_embedding = embedding_map.get(state.state_id)
            state.score_holistic = self._get_holistic_alignment_score(
                path_embedding=path_embedding,
                constraint=constraint
            )

        candidates_ranked_by_struct = sorted(candidate_states, key=lambda s: s.score_structural, reverse=True)
        candidates_ranked_by_holistic = sorted(candidate_states, key=lambda s: s.score_holistic, reverse=True)

        log_prob_g_map = self._fuse_rankings([candidates_ranked_by_struct])
        log_prob_h_map = self._fuse_rankings([candidates_ranked_by_holistic])

        for state in candidate_states:
            h_score = state.score_holistic

            log_prob_g = log_prob_g_map.get(state.state_id, -np.log(len(candidate_states)))
            log_prob_h = log_prob_h_map.get(state.state_id, -np.log(len(candidate_states)))
            log_prob_s = log_prob_g
            final_log_prob_step = self._apply_intermediate_node_bonus(log_prob_s, state.path[-1], existing_bindings,
                                                                      constraint)
            parent_g_score = state.parent_state.path_quality_score
            new_path_quality_score = (parent_g_score  + final_log_prob_step)
            new_heuristic_score = log_prob_h
            if task_type == "FIND":
                total_score = new_path_quality_score + new_heuristic_score
            else:
                total_score = 0.1*new_path_quality_score + 0.9*new_heuristic_score


            state.path_quality_score = new_path_quality_score
            state.heuristic_score = new_heuristic_score
            state.score = total_score

        return sorted(candidate_states, key=lambda s: s.score, reverse=True)

    def _get_structural_score(self, edge: Tuple[str, str, str]) -> float:

        return self.scored_triples_map.get(edge, 0.0)

    def _get_coherence_score(self, path: List[Tuple[str, str, str]]) -> float:
        if len(path) < 2:
            return 0.5

        rel_prev = path[-2][1]
        rel_curr = path[-1][1]

        dsrg_edge_info = self.dsrg.get((rel_prev, rel_curr), {})

        weight = dsrg_edge_info.get('weight', 0.0)
        confidence = dsrg_edge_info.get('confidence', 0.0)

        return weight * confidence

    def _get_holistic_alignment_score(self, path_embedding: torch.Tensor, constraint: dict) -> float:
        if path_embedding is None:
            return 0.0

        constraint_id = constraint["constraint_id"]
        indicator_embedding = self.indicator_embeddings.get(constraint_id)

        if indicator_embedding is None:
            return 0.0

        return util.cos_sim(path_embedding, indicator_embedding).item()

    def _get_goal_oriented_alignment_score(self, path: List[tuple], constraint: dict) -> float:
        if not path:
            return 0.0

        current_endpoint = path[-1][2]
        endpoint_type_info = self.contextual_type_vectors.get(current_endpoint)
        if not endpoint_type_info:
            return 0.0
        endpoint_type_vector = endpoint_type_info['vector']

        indicator_path_str = constraint["indicator_sub_path"]
        indicator_endpoint_var = indicator_path_str.split(' -> ')[-1]

        target_vector = self.variable_target_vectors.get(indicator_endpoint_var)
        if target_vector is None:
            return 0.0

        return util.cos_sim(endpoint_type_vector, target_vector).item()

    def _calculate_verify_heuristic(self, path: List[tuple], verification_target_value: str) -> float:
        if not path: return 0.0

        current_endpoint = path[-1][2]

        emb_curr = self.entity_embeddings.get(current_endpoint)
        emb_target = self.entity_embeddings.get(verification_target_value)

        embedding_similarity = 0.0
        if emb_curr is not None and emb_target is not None:
            embedding_similarity = util.cos_sim(emb_curr, emb_target).item()

        distance = self.shortest_path_distances.get(current_endpoint, {}).get(verification_target_value, float('inf'))
        distance_penalty = 1 / (1 + distance)

        return embedding_similarity * distance_penalty

    def _fuse_rankings(self, ranked_lists: List[List[BeamState]], k: int = 60) -> dict:


        rrf_scores = defaultdict(float)
        state_map = {}

        for ranked_list in ranked_lists:
            for rank, state in enumerate(ranked_list):
                state_id = state.state_id
                rrf_scores[state_id] += 1 / (k + rank + 1)
                if state_id not in state_map:
                    state_map[state_id] = state


        sorted_by_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        log_prob_map = {state_id: -np.log(rank + 1) for rank, (state_id, score) in enumerate(sorted_by_rrf)}

        return log_prob_map


    def _apply_intermediate_node_bonus(self,
                                       log_prob_step: float,
                                       new_edge: tuple,
                                       existing_bindings: dict,
                                       constraint: dict) -> float:
        bonus_weight = 1.0
        confidence_map = {"High": 0.9, "Medium": 0.5, "Low": 0.1}

        path_entities = constraint["indicator_sub_path"].split(' -> ')[::2]
        intermediate_vars = {entity for entity in path_entities[1:-1] if entity.startswith('?')}

        if not intermediate_vars:
            return log_prob_step

        edge_entities = {new_edge[0], new_edge[2]}

        for var in intermediate_vars:
            if var in existing_bindings:
                binding = existing_bindings[var]
                bound_value = binding.get("value")

                if bound_value != "UNCERTAIN" and bound_value in edge_entities:
                    confidence_level = binding.get("confidence_level", "Low")
                    confidence_score = confidence_map.get(confidence_level, 0.1)

                    if confidence_score > 0:
                        return log_prob_step / (1 + bonus_weight * confidence_score)

        return log_prob_step


    async def _align_variables_with_llm(self,
                                        path_state: BeamState,
                                        constraint: dict,
                                        existing_bindings: dict) -> List[dict]:

        best_path_state = path_state
        best_path_triples = best_path_state.path
        path_text = self._format_path_for_llm(best_path_triples)

        all_entities_in_path = []
        seen_entities = set()
        for h, r, t in best_path_triples:
            if h not in seen_entities:
                all_entities_in_path.append(h)
                seen_entities.add(h)
            if t not in seen_entities:
                all_entities_in_path.append(t)
                seen_entities.add(t)

        variables_to_align = self._determine_variables_to_align(constraint, existing_bindings)

        if not variables_to_align:
            return [{"status": "passed", "new_bindings": {}, "evidence_log_prob": np.log(1.0), "evidence_path": best_path_triples}]

        attempt_results = []
        successful_bindings = None

        for attempt in range(MAX_ALIGNMENT_ATTEMPTS):
            llm_response = await self._make_llm_alignment_call(self.query_plan['question'], path_text,all_entities_in_path,
                                                               variables_to_align)
            if not llm_response:
                attempt_results.append(None)
                continue

            validated_bindings = self._validate_and_tag_hallucinations(llm_response, best_path_triples, variables_to_align)

            is_perfect_success = "hallucination" not in [b.get('status') for b in validated_bindings.values()]

            if is_perfect_success:
                successful_bindings = validated_bindings
                break
            else:
                attempt_results.append(validated_bindings)

        if successful_bindings is not None:
            final_bindings_normalized = successful_bindings
            for var, binding in successful_bindings.items():
                value = "UNCERTAIN" if binding.get("entity") is None else binding.get("entity")
                final_bindings_normalized[var] = {
                    "value": value,
                    "confidence_level": binding.get("confidence_level")
                }
        else:
            final_bindings_normalized = self._consolidate_best_bindings(attempt_results, variables_to_align)

        new_bindings_for_solution = self._update_bindings(final_bindings_normalized, existing_bindings)
        path_quality_log_prob = best_path_state.score
        alignment_log_prob = self._calculate_alignment_confidence(new_bindings_for_solution)

        return [{
            "status": "passed",
            "new_bindings": new_bindings_for_solution,
            "evidence_log_prob": path_quality_log_prob + alignment_log_prob,
            "evidence_path": best_path_triples
        }]

    def _extract_variables_from_path(self, path_str: str) -> set:

        if not path_str:
            return set()

        parts = path_str.split(' -> ')
        variables = set()
        for i in range(0, len(parts), 2):
            if parts[i].startswith('?'):
                variables.add(parts[i])

        return variables
    def _determine_variables_to_align(self, constraint: dict, existing_bindings: dict) -> List[str]:

        path = constraint.get("indicator_sub_path", "")
        start_entity = constraint.get("start_entity")

        all_vars_in_path = self._extract_variables_from_path(path)
        potential_targets = all_vars_in_path - {start_entity}

        variables_to_align = []
        for var in potential_targets:
            if var not in existing_bindings:
                variables_to_align.append(var)
            else:
                binding = existing_bindings[var]
                if binding.get("value") == "UNCERTAIN" or binding.get("confidence_level") in ["Medium", "Low"]:
                    variables_to_align.append(var)

        return variables_to_align

    async def _make_llm_alignment_call(self, question: str, path_text: str, candidate_entities: List[str],variables_to_align: List[str],
                                       max_retries=3) -> dict:
        messages = [
            {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT},
            *ALIGNMENT_FEW_SHOTS,
            {
                "role": "user",
                "content": f'Question: "{question}"\nPath Text: "{path_text}"\nEntity Choices: {json.dumps(candidate_entities)}\nVariables to Align: {json.dumps(variables_to_align)}'
            }
        ]
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model="",
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return {}

    def _validate_and_tag_hallucinations(self, llm_response: dict, path_triples: List[Tuple[str, str, str]],
                                         variables_to_align: List[str]) -> dict:
        validated_bindings = {}
        all_entities_in_path = set()
        for h, r, t in path_triples:
            all_entities_in_path.add(h)
            all_entities_in_path.add(t)

        for var in variables_to_align:
            binding = llm_response.get(var)
            if not isinstance(binding, dict) or "entity" not in binding:
                validated_bindings[var] = {"status": "hallucination", "reason": "Invalid format"}
                continue

            entity = binding.get("entity")
            if entity is not None and entity not in all_entities_in_path:
                validated_bindings[var] = {"status": "hallucination", "reason": f"Entity '{entity}' not in path"}
            else:
                validated_bindings[var] = {
                    "status": "valid",
                    "entity": entity,
                    "confidence_level": binding.get("confidence_level")
                }
        return validated_bindings

    def _consolidate_best_bindings(self, attempt_results: List[dict], variables_to_align: List[str]) -> dict:
        best_bindings = {}
        for var in variables_to_align:
            best_outcome = None
            best_priority = -1

            for attempt_result in attempt_results:
                if not attempt_result or var not in attempt_result: continue

                outcome = attempt_result[var]
                priority = -1
                if outcome.get("status") == "valid":
                    priority = CONFIDENCE_LEVEL_PRIORITY.get(outcome.get("confidence_level"), 0)

                if priority > best_priority:
                    best_priority = priority
                    best_outcome = outcome

            if best_outcome and best_outcome.get("status") == "valid":
                if best_outcome.get("entity") is None:
                    best_bindings[var] = {"value": "UNCERTAIN", "confidence_level": None}
                else:
                    best_bindings[var] = {"value": best_outcome["entity"],
                                          "confidence_level": best_outcome["confidence_level"]}
            else:
                best_bindings[var] = {"value": "UNCERTAIN", "confidence_level": "Low"}

        return best_bindings

    def _update_bindings(self, final_bindings: dict, existing_bindings: dict) -> dict:
        new_bindings_for_solution = {}
        for var, final_binding in final_bindings.items():
            if var not in existing_bindings:
                new_bindings_for_solution[var] = final_binding
            else:
                old_binding = existing_bindings[var]
                old_level = "UNCERTAIN" if old_binding.get("value") == "UNCERTAIN" else old_binding.get(
                    "confidence_level")
                new_level = final_binding.get("confidence_level")

                old_priority = CONFIDENCE_LEVEL_PRIORITY.get(old_level, 0)
                new_priority = CONFIDENCE_LEVEL_PRIORITY.get(new_level, 0)

                if new_priority > old_priority:
                    new_bindings_for_solution[var] = final_binding

        return new_bindings_for_solution

    def _calculate_alignment_confidence(self, new_bindings: dict) -> float:
        total_log_prob = 0.0
        for var, binding in new_bindings.items():
            level = "Low"
            if binding.get("value") == "UNCERTAIN":
                level = None
            elif "confidence_level" in binding:
                level = binding.get("confidence_level")

            total_log_prob += np.log(CONFIDENCE_MAP.get(level, 0.1))

        return total_log_prob


    async def _run_beam_search(self,
                               constraint: dict,
                               existing_bindings: dict,
                               beam_width: int,
                               max_depth: int,
                               top_k_results: int,
                               task_type: str,
                               verification_target_value: str = None) -> List[BeamState]:

        start_entity = constraint.get("start_entity")
        if start_entity.startswith("?"):
            start_node = existing_bindings.get(start_entity, {}).get("value")
            if start_node is None or start_node == "UNCERTAIN":
                return []
        else:
            start_node = start_entity

        tie_breaker = itertools.count()
        state_id_counter = itertools.count()
        initial_state = BeamState(
            state_id=next(state_id_counter),
            score=0.0,
            path=[],
            endpoint=start_node,
            nodes_in_path={start_node},
            path_quality_score=0.0,
            heuristic_score=0.0
        )


        beam = [(initial_state.score, next(tie_breaker), initial_state)]
        completed_paths_heap = []

        for depth in range(max_depth):
            if not beam: break


            nodes_to_expand = defaultdict(list)
            for _, _, state in beam:
                nodes_to_expand[state.endpoint].append(state)

            next_beam_candidates = []
            for current_node, parent_states in nodes_to_expand.items():
                for h, r, t in self._get_neighbors(current_node):
                    next_node = t

                    for parent_state in parent_states:
                        if next_node in parent_state.nodes_in_path:
                            continue

                        new_state = BeamState(
                            state_id=next(state_id_counter),
                            score=0.0,
                            path=parent_state.path + [(h, r, t)],
                            endpoint=next_node,
                            nodes_in_path=parent_state.nodes_in_path.union({next_node}),
                            path_quality_score=0.0,
                            heuristic_score=0.0,
                            parent_state=parent_state
                        )
                        next_beam_candidates.append(new_state)

            if not next_beam_candidates: break

            scored_and_ranked_candidates = await self._rank_and_score_candidates(
                candidate_states=next_beam_candidates,
                constraint=constraint,
                existing_bindings=existing_bindings,
                task_type=task_type,
                verification_target_value=verification_target_value
            )

            explorers_for_next_beam = []

            for state in scored_and_ranked_candidates:
                is_complete = self._is_path_complete(state, constraint, task_type, verification_target_value)

                if is_complete:
                    heapq.heappush(completed_paths_heap, (state.score, next(tie_breaker), state))

                if not (task_type == "VERIFY" and is_complete):
                    explorers_for_next_beam.append(state)

            if not explorers_for_next_beam:
                break

            top_k_for_next_beam = heapq.nlargest(beam_width, explorers_for_next_beam, key=lambda s: s.score)

            beam = []
            for state in top_k_for_next_beam:
                heapq.heappush(beam, (state.score, next(tie_breaker), state))
        all_final_candidates = []
        if completed_paths_heap:
            all_final_candidates.extend([item[2] for item in completed_paths_heap])
        else:
            if task_type == "FIND":
                all_final_candidates.extend([item[2] for item in beam])
        if not all_final_candidates:
            return []


        all_final_candidates.sort(key=lambda s: len(s.path), reverse=True)

        maximal_paths = []
        for state in all_final_candidates:
            is_prefix = False
            path_tuple = tuple(self._prefix(state.path, for_comparison=True))

            for max_state in maximal_paths:
                max_path_tuple = tuple(self._prefix(max_state.path, for_comparison=True))
                if len(path_tuple) < len(max_path_tuple) and max_path_tuple[:len(path_tuple)] == path_tuple:
                    is_prefix = True
                    break

            if not is_prefix:
                maximal_paths.append(state)


        maximal_paths.sort(key=lambda s: s.score, reverse=True)

        return maximal_paths[:top_k_results]
async def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-q','--query_plans_path', type=str, default="")
    parser.add_argument('-g','--features_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = GTELargeEN(device=device)
    llm_client = AsyncOpenAI(api_key="",base_url="")

    with open(args.query_plans_path, 'r', encoding='utf-8') as f:
        query_plans = {item['id']: item for item in [json.loads(line) for line in f]}

    features_data = torch.load(args.features_path, map_location='cpu')

    if args.output_path is None:
        dir_name = os.path.dirname(args.query_plans_path)
        name, _ = os.path.splitext(os.path.basename(args.query_plans_path))
        args.output_path = os.path.join(dir_name, f"{name.replace('_query_plans', '')}_evidence_portfolios.jsonl")

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    items_to_process = list(query_plans.values())
    if args.limit:
        items_to_process = items_to_process[:args.limit]

    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        pbar = tqdm(items_to_process, desc="Verifying Query Plans")
        for plan in pbar:
            sample_id = plan['id']
            if sample_id in processed_ids:
                continue

            pbar.set_description(f"Processing {sample_id}")

            subgraph_features = features_data.get(sample_id)
            if not subgraph_features:
                continue

            verifier = ConstraintVerifier(
                query_plan=plan,
                subgraph_data=subgraph_features,
                text_encoder=text_encoder,
                llm_client=llm_client
            )
            portfolios = await verifier.run()

            if portfolios:
                output_entry = {
                    "id": sample_id,
                    "question": plan['question'],
                    "topic_entities": plan['topic_entities'],
                    "answer_entities": plan.get('answer_entities', []),
                    "portfolios": portfolios
                }
                f_out.write(json.dumps(output_entry) + '\n')
                f_out.flush()

if __name__ == '__main__':
    asyncio.run(main())

