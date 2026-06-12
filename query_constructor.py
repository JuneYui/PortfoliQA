import os
import argparse
from tqdm import tqdm
import json
import networkx as nx
from collections import defaultdict
import torch
from src.model.text_encoders import GTELargeEN
from sentence_transformers import util


class QueryConstructor:

    def __init__(self, parsed_plan, topic_entities, subgraph_triples, text_encoder):
        self.parsed_plan = parsed_plan
        self.logical_fragments = parsed_plan["logical_fragments"]
        self.target_variable_name = parsed_plan["target_variable"]
        self.topic_entities = set(topic_entities)
        self.subgraph = [(h, r, t) for h, r, t, s in subgraph_triples]
        self.text_encoder = text_encoder
        self.constants = set()
        self.variables = set()
        self.variable_map = {}
        self.local_fan_outs = {}
        self._precompute_local_fan_outs()
        self._assign_roles()

    def run(self):
        try:
            blueprint = self._build_and_infer_clqg()
            if not blueprint:
                return None

            optimized_blueprint = self._optimize_and_rewrite(blueprint)
            if not optimized_blueprint:
                return None

            final_plan = self._generate_final_json(optimized_blueprint)
            return final_plan

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


    def _assign_roles(self):
        all_entities_in_fragments = set()
        for frag in self.logical_fragments:
            all_entities_in_fragments.add(frag["subject"])
            all_entities_in_fragments.add(frag["object"])

        self.constants = self.topic_entities

        for entity in all_entities_in_fragments:
            if entity not in self.constants:
                var_name = f"?{entity.replace(' ', '_')}"
                self.variables.add(var_name)
                self.variable_map[entity] = var_name


        for frag in self.logical_fragments:
            if frag["subject"] in self.variable_map:
                frag["subject"] = self.variable_map[frag["subject"]]
            if frag["object"] in self.variable_map:
                frag["object"] = self.variable_map[frag["object"]]


        if self.target_variable_name in self.variable_map:
            self.target_variable_name = self.variable_map[self.target_variable_name]

    def _precompute_local_fan_outs(self):

        self.local_fan_outs = {}
        relation_subjects = defaultdict(set)
        relation_objects = defaultdict(set)

        for h, r, t in self.subgraph:
            relation_subjects[r].add(h)
            relation_objects[r].add(t)

        for r in relation_subjects:
            num_subjects = len(relation_subjects[r])
            num_objects = len(relation_objects[r])
            self.local_fan_outs[r] = num_objects / num_subjects if num_subjects > 0 else 1.0


    def _build_and_infer_clqg(self):

        g_entity_undirected = nx.Graph()
        fragment_map = {f"F{i}": frag for i, frag in enumerate(self.logical_fragments)}

        for fid, frag in fragment_map.items():
            s = self.variable_map.get(frag["subject"], frag["subject"])
            o = self.variable_map.get(frag["object"], frag["object"])
            if s is not None and o is not None:
                g_entity_undirected.add_edge(s, o, fragment_id=fid)

        blueprint = {"main_chains": [], "side_constraints": []}
        resolved_fragments = set()

        defined_variables = {const: "CONSTANT" for const in self.constants}


        all_starts = list(self.constants)
        processed_starts = set()

        for i, start_node in enumerate(all_starts):
            if start_node in processed_starts: continue


            target_var = self.target_variable_name
            if start_node in g_entity_undirected and nx.has_path(g_entity_undirected, start_node, target_var):

                paths = list(nx.all_shortest_paths(g_entity_undirected, start_node, target_var))
                paths.sort()
                main_entity_path = paths[0]

                chain_id = f"MC{len(blueprint['main_chains']) + 1}"
                chain_obj = {"chain_id": chain_id, "fragments": []}


                for j in range(len(main_entity_path) - 1):
                    u, v = main_entity_path[j], main_entity_path[j + 1]
                    fid = g_entity_undirected.get_edge_data(u, v)["fragment_id"]

                    chain_obj["fragments"].append({"fragment_id": fid, "direction": (u, v)})
                    resolved_fragments.add(fid)
                    if v.startswith("?"): defined_variables[v] = fid

                blueprint["main_chains"].append(chain_obj)

                for entity in main_entity_path:
                    if entity in self.constants:
                        processed_starts.add(entity)


        unresolved_fragments = set(fragment_map.keys()) - resolved_fragments
        progress_made = True
        while unresolved_fragments and progress_made:
            progress_made = False
            resolved_this_iteration = set()

            for fid in unresolved_fragments:
                frag = fragment_map[fid]
                s = self.variable_map.get(frag["subject"], frag["subject"])
                o = self.variable_map.get(frag["object"], frag["object"])


                input_vars = set()
                if s.startswith("?") and s not in defined_variables: input_vars.add(s)
                if o.startswith("?") and o not in defined_variables: input_vars.add(o)


                attach_var = None
                if s in defined_variables:
                    attach_var = s
                elif o in defined_variables:
                    attach_var = o


                if attach_var:
                    parent_fid = None


                    if attach_var.startswith("?"):
                        parent_fid = defined_variables.get(attach_var)

                    side_constraint_obj = {
                        "constraint_id": f"SC{len(blueprint['side_constraints']) + 1}",
                        "fragment_id": fid,
                        "attach_to_variable": attach_var,
                        "attach_to_fragment_id": parent_fid
                    }

                    direction = (s, o) if s == attach_var else (o, s)
                    side_constraint_obj["direction"] = direction
                    blueprint["side_constraints"].append(side_constraint_obj)

                    output_var = o if s == attach_var else s
                    if output_var.startswith("?"): defined_variables[output_var] = fid

                    resolved_this_iteration.add(fid)
                    progress_made = True

            unresolved_fragments -= resolved_this_iteration

        if unresolved_fragments:
            return None

        return blueprint

    def _optimize_and_rewrite(self, blueprint):

        if len(blueprint.get("main_chains", [])) <= 1:
            return blueprint

        scored_chains = []
        for chain in blueprint["main_chains"]:
            cost = self._calculate_cost(chain)
            scored_chains.append((cost, chain))
        scored_chains.sort(key=lambda x: x[0])

        winner_chain = scored_chains[0][1]
        demoted_chains = [chain for cost, chain in scored_chains[1:]]


        new_blueprint = {
            "main_chains": [winner_chain],
            "side_constraints": blueprint.get("side_constraints", [])
        }


        last_frag_in_winner = winner_chain["fragments"][-1]
        target_entity_in_path = last_frag_in_winner["direction"][1]
        binding_variable = f"?{target_entity_in_path.replace('?', '')}"


        last_frag_in_winner["direction"] = (last_frag_in_winner["direction"][0], binding_variable)
        defining_fragment_id = last_frag_in_winner["fragment_id"]


        for demoted_chain in demoted_chains:

            reversed_fragments = list(reversed(demoted_chain["fragments"]))
            last_parent_original_fid = defining_fragment_id

            for i, frag_info in enumerate(reversed_fragments):
                original_fid = frag_info["fragment_id"]
                original_dir = frag_info["direction"]

                attach_var = binding_variable if i == 0 else reversed_fragments[i - 1]["direction"][0]

                new_side_constraint = {
                    "constraint_id": f"SC_demoted_{demoted_chain['chain_id']}_{original_fid}",
                    "fragment_id": original_fid,
                    "attach_to_variable": attach_var,
                    "attach_to_fragment_id": last_parent_original_fid,
                    "direction": (original_dir[1], original_dir[0])
                }
                new_blueprint["side_constraints"].append(new_side_constraint)

                last_parent_original_fid = original_fid

        return new_blueprint

    def _get_expected_fan_out(self, abstract_relation, all_concrete_relations, concrete_rel_embs, top_n=3):

        if not all_concrete_relations:
            return 1.0

        abstract_emb = self.text_encoder.embed([abstract_relation])[0]

        similarities = util.cos_sim(abstract_emb, concrete_rel_embs)[0]

        top_scores, top_indices = torch.topk(similarities, k=min(top_n, len(all_concrete_relations)))

        weighted_sum_fan_out = 0.0
        total_weights = 0.0

        for score, index in zip(top_scores, top_indices):
            concrete_relation = all_concrete_relations[index.item()]
            fan_out = self.local_fan_outs.get(concrete_relation, 1.0)
            weight = score.item()
            weighted_sum_fan_out += weight * fan_out
            total_weights += weight

        if total_weights == 0:
            return 1.0

        return weighted_sum_fan_out / total_weights

    def _calculate_cost(self, chain_obj):

        all_concrete_relations = list(self.local_fan_outs.keys())
        concrete_rel_embs = self.text_encoder.embed(all_concrete_relations)


        estimated_candidate_size = 1.0


        for frag_info in chain_obj["fragments"]:
            fid = frag_info["fragment_id"]
            frag = next(f for f in self.logical_fragments if f"F{self.logical_fragments.index(f)}" == fid)
            abstract_relation = frag["relation"]

            expected_fan_out = self._get_expected_fan_out(abstract_relation, all_concrete_relations, concrete_rel_embs)
            estimated_candidate_size *= expected_fan_out

        return estimated_candidate_size


    def _calculate_constraint_cost(self, constraint_obj):

        indicator_path = constraint_obj.get("indicator_sub_path", "")
        if not indicator_path:
            return float('inf')

        abstract_relations = indicator_path.split(' -> ')[1::2]

        all_concrete_relations = list(self.local_fan_outs.keys())

        if not hasattr(self, 'concrete_rel_embs'):
            self.concrete_rel_embs = self.text_encoder.embed(all_concrete_relations)


        estimated_candidate_size = 1.0


        for abstract_relation in abstract_relations:
            expected_fan_out = self._get_expected_fan_out(
                abstract_relation,
                all_concrete_relations,
                self.concrete_rel_embs
            )

            estimated_candidate_size *= expected_fan_out

        return estimated_candidate_size

    def _generate_final_json(self, blueprint):

        if not blueprint or not blueprint.get("main_chains"):

            return None

        final_constraints = []

        fragment_to_constraint_map = {}


        main_chain = blueprint["main_chains"][0]
        main_constraint_id = main_chain["chain_id"]


        main_path_parts = []
        start_entity = main_chain["fragments"][0]["direction"][0]
        main_path_parts.append(start_entity)

        for frag_info in main_chain["fragments"]:
            fid = frag_info["fragment_id"]
            direction = frag_info["direction"]
            frag = self.logical_fragments[int(fid.replace('F', ''))]

            main_path_parts.append(frag["relation"])
            main_path_parts.append(direction[1])
            fragment_to_constraint_map[fid] = main_constraint_id


        main_constraint_obj = {
            "constraint_id": main_constraint_id,
            "description": f"Main generative path starting from '{start_entity}'.",
            "reasoning": "This is the primary reasoning chain identified as the most efficient path to generate candidates for the target variable.",
            "dependencies": [],
            "start_entity": start_entity,
            "indicator_sub_path": " -> ".join(main_path_parts)
        }
        final_constraints.append(main_constraint_obj)


        side_constraints_pool = blueprint.get("side_constraints", [])
        if side_constraints_pool:
            side_graph = nx.DiGraph()
            side_frag_map = {sc["fragment_id"]: sc for sc in side_constraints_pool}
            side_graph.add_nodes_from(side_frag_map.keys())

            for sc in side_constraints_pool:
                parent_fid = sc["attach_to_fragment_id"]
                if parent_fid in side_graph:
                    side_graph.add_edge(parent_fid, sc["fragment_id"])

            roots = [n for n, d in side_graph.in_degree() if d == 0]
            leaves = [n for n, d in side_graph.out_degree() if d == 0]

            all_paths = []
            for root in roots:
                for leaf in leaves:
                    if nx.has_path(side_graph, root, leaf):
                        all_paths.extend(nx.all_simple_paths(side_graph, root, leaf))

            maximal_paths = []
            all_paths.sort(key=len, reverse=True)
            for path in all_paths:
                is_subpath = False
                for max_path in maximal_paths:
                    if max_path[:len(path)] == path:
                        is_subpath = True
                        break
                if not is_subpath:
                    maximal_paths.append(path)


            preliminary_side_constraints = []
            for i, path in enumerate(maximal_paths):
                first_frag_id = path[0]
                first_frag_info = side_frag_map.get(first_frag_id)
                if not first_frag_info: continue

                start_entity = first_frag_info["attach_to_variable"]
                parent_fid = first_frag_info["attach_to_fragment_id"]

                dependencies = []
                if parent_fid:
                    if parent_fid in fragment_to_constraint_map:
                        dependencies.append(fragment_to_constraint_map[parent_fid])
                else:
                    dependencies.append(main_constraint_id)

                path_parts = []
                path_parts.append(start_entity)
                for fid in path:
                    frag_info = side_frag_map.get(fid)
                    if not frag_info: continue
                    direction = frag_info["direction"]

                    try:
                        frag = self.logical_fragments[int(fid.replace('F', ''))]
                        path_parts.append(frag["relation"])
                        path_parts.append(direction[1])
                    except (ValueError, IndexError):
                        continue

                verify_constraint_id = f"VC{i + 1}"
                verify_constraint_obj = {
                    "constraint_id": verify_constraint_id,
                    "description": f"Verification path starting from variable '{start_entity}'.",
                    "reasoning": "This path represents a necessary logical condition...",
                    "dependencies": list(set(dependencies)),
                    "start_entity": start_entity,
                    "indicator_sub_path": " -> ".join(path_parts)
                }
                preliminary_side_constraints.append(verify_constraint_obj)

                constraints_type_A = []
                constraints_type_B = []

                for const in preliminary_side_constraints:
                    last_entity = const["indicator_sub_path"].split(' -> ')[-1]
                    if last_entity.startswith('?'):
                        constraints_type_B.append(const)
                    else:
                        constraints_type_A.append(const)

                constraints_type_A.sort(key=self._calculate_constraint_cost)

                constraints_type_B.sort(key=self._calculate_constraint_cost)


                sorted_side_constraints = constraints_type_A + constraints_type_B


                final_constraints.extend(sorted_side_constraints)

        query_type = "Constraint Intersection" if len(final_constraints) > 1 else "Exploratory Flow"
        final_plan = {
            "target_variable": self.parsed_plan["target_variable"],
            "query_type": query_type,
            "constraint_indicators": final_constraints,
            "co_reference_groups": self.parsed_plan["co_reference_groups"],

        }
        return final_plan


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p','--parser_output_path', type=str, default="")
    parser.add_argument('-g','--subgraph_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = GTELargeEN(device)

    with open(args.parser_output_path, 'r', encoding='utf-8') as f:
        parsed_plans = [json.loads(line) for line in f]

    subgraph_data = torch.load(args.subgraph_path, map_location='cpu')

    if args.output_path is None:
        dir_name = os.path.dirname(args.parser_output_path)
        name, _ = os.path.splitext(os.path.basename(args.parser_output_path))
        args.output_path = os.path.join(dir_name, f"{name.replace('_semantic_plans', '')}_query_plans.jsonl")

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        pbar = tqdm(parsed_plans, desc="Constructing Query Plans")
        for plan in pbar:
            sample_id = plan['id']
            if sample_id in processed_ids:
                continue

            pbar.set_description(f"Processing {sample_id}")

            subgraph_info = subgraph_data.get(sample_id)
            if not subgraph_info or 'scored_triples' not in subgraph_info:
                continue

            constructor = QueryConstructor(
                parsed_plan=plan,
                topic_entities=plan['topic_entities'],
                subgraph_triples=subgraph_info['scored_triples'],
                text_encoder=text_encoder
            )
            final_query_plan = constructor.run()

            if final_query_plan:
                output_entry = {
                    "id": sample_id,
                    "question": plan['question'],
                    "topic_entities": plan['topic_entities'],
                    "answer_entities": plan.get('answer_entities', []),
                    **final_query_plan
                }
                f_out.write(json.dumps(output_entry) + '\n')
                f_out.flush()


if __name__ == '__main__':
    main()
