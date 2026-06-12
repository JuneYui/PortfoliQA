

import argparse
import json
import os
import torch
from collections import defaultdict
from itertools import product
from tqdm import tqdm
import numpy as np
import networkx as nx
from src.model.text_encoders import GTELargeEN
from sentence_transformers import util




class FeatureAugmentor:

    def __init__(self, semantic_plan, query_plan, text_encoder):
        self.semantic_plan = semantic_plan
        self.query_plan = query_plan
        self.text_encoder = text_encoder

    def run(self):
        indicator_embeddings = self._compute_indicator_embeddings()
        return {
            "indicator_embeddings": indicator_embeddings
        }

    def _textualize_indicator_path(self, indicator_path_str: str) -> str:
        parts = indicator_path_str.split(' -> ')
        return " ".join(parts).replace('?', '')

    def _compute_indicator_embeddings(self):
        embeddings = {}
        paths_to_embed = []
        ids_for_paths = []

        for constraint in self.query_plan.get("constraint_indicators", []):
            cid = constraint["constraint_id"]
            path_str = constraint["indicator_sub_path"]
            paths_to_embed.append(self._textualize_indicator_path(path_str))
            ids_for_paths.append(cid)

        if paths_to_embed:
            path_vectors = self.text_encoder.embed(paths_to_embed)
            for cid, vec in zip(ids_for_paths, path_vectors):
                embeddings[cid] = vec

        return embeddings

    def _compute_endpoint_variable_target_vectors(self):
        target_vectors = {}
        processed_vars = set()

        for constraint in self.query_plan.get("constraint_indicators", []):
            path_str = constraint["indicator_sub_path"]

            last_entity = path_str.split(' -> ')[-1]

            if not last_entity.startswith('?'):
                continue

            if last_entity in processed_vars:
                continue

            var_name_no_prefix = last_entity.replace('?', '')

            context_sentences = []
            for frag in self.semantic_plan.get("logical_fragments", []):
                s, r, o = frag["subject"], frag["relation"], frag["object"]

                if s == var_name_no_prefix:

                    context_sentences.append(f"a {s} that {r} {o}")
                elif o == var_name_no_prefix:

                    context_sentences.append(f"a {o} that is the {r} of {s}")

            if context_sentences:
                full_context = f"A {var_name_no_prefix} which is described as: " + ", and ".join(context_sentences)
                var_vector = self.text_encoder.embed([full_context])[0]
                target_vectors[last_entity] = var_vector
                processed_vars.add(last_entity)

        return target_vectors

class DSRGProcessor:

    def __init__(self, subgraph_triples, text_encoder):
        self.text_encoder = text_encoder
        self.triples = [(h, r, t) for h, r, t, s in subgraph_triples]

        self.entities = sorted(list(set([h for h, r, t in self.triples] + [t for h, r, t in self.triples])))
        self.relations = sorted(list(set([r for h, r, t in self.triples])))
        self.relation_embeddings = {}

        self.adj = defaultdict(list)
        for h, r, t in self.triples:
            self.adj[h].append(t)
            self.adj[t].append(h)

        self.entity_embeddings = {}
        self.contextual_type_vectors = {}
        self.relation_signatures = {}
        self.dsrg = {}

    def run(self):
        if not self.triples:
            return {}

        self._embed_all_entities()
        self._embed_all_relations()
        self._compute_contextual_type_vectors()
        self._compute_relation_signatures()
        self._normalize_confidences()
        self._build_dsrg()
        # shortest_path_distances = self._compute_shortest_path_distances()


        return {
            # "contextual_type_vectors": self.contextual_type_vectors,
            # "relation_signatures": self.relation_signatures,
            "dsrg": self.dsrg
            # "entity_embeddings": self.entity_embeddings,
            # "relation_embeddings": self.relation_embeddings,
            # "shortest_path_distances": shortest_path_distances
        }

    def _compute_shortest_path_distances(self):
        if not self.triples:
            return {}

        G = nx.Graph()
        for h, r, t in self.triples:
            G.add_edge(h, t)

        distances = {source: targets for source, targets in nx.all_pairs_shortest_path_length(G)}

        return distances

    def _embed_all_entities(self):
        if self.entities:
            embeddings = self.text_encoder.embed(self.entities)
            self.entity_embeddings = {entity: emb for entity, emb in zip(self.entities, embeddings)}

    def _embed_all_relations(self):
        if self.relations:
            embeddings = self.text_encoder.embed(self.relations)
            self.relation_embeddings = {rel: emb for rel, emb in zip(self.relations, embeddings)}

    def _compute_contextual_type_vectors(self):
        for entity in self.entities:
            v_self = self.entity_embeddings.get(entity, torch.zeros_like(next(iter(self.entity_embeddings.values()))))

            neighbors = self.adj.get(entity, [])
            if not neighbors:
                self.contextual_type_vectors[entity] = {"vector": v_self, "confidence": 0.0}
                continue

            neighbor_embs = torch.stack([self.entity_embeddings[n] for n in neighbors if n in self.entity_embeddings])
            if neighbor_embs.shape[0] == 0:
                self.contextual_type_vectors[entity] = {"vector": v_self, "confidence": 0.0}
                continue

            v_context = torch.mean(neighbor_embs, dim=0)
            cohesion = util.cos_sim(v_context, neighbor_embs).mean().item()

            alpha = cohesion
            smoothed_vector = alpha * v_context + (1 - alpha) * v_self

            confidence = np.log1p(len(neighbors)) * cohesion

            self.contextual_type_vectors[entity] = {
                "vector": smoothed_vector,
                "confidence": confidence
            }

    def _compute_relation_signatures(self):
        relation_subjects = defaultdict(list)
        relation_objects = defaultdict(list)
        for h, r, t in self.triples:
            relation_subjects[r].append(h)
            relation_objects[r].append(t)

        for rel in self.relations:
            subject_type_vectors = [self.contextual_type_vectors[s]["vector"] for s in relation_subjects[rel]]
            vs_centroid = torch.mean(torch.stack(subject_type_vectors), dim=0)
            vs_cohesion = util.cos_sim(vs_centroid, torch.stack(subject_type_vectors)).mean().item()
            vs_confidence = np.log1p(len(subject_type_vectors)) * vs_cohesion

            object_type_vectors = [self.contextual_type_vectors[o]["vector"] for o in relation_objects[rel]]
            vo_centroid = torch.mean(torch.stack(object_type_vectors), dim=0)
            vo_cohesion = util.cos_sim(vo_centroid, torch.stack(object_type_vectors)).mean().item()
            vo_confidence = np.log1p(len(object_type_vectors)) * vo_cohesion

            self.relation_signatures[rel] = {
                "subject_type": {"vector": vs_centroid, "confidence": vs_confidence},
                "object_type": {"vector": vo_centroid, "confidence": vo_confidence}
            }

    def _build_dsrg(self):
        for r1, r2 in product(self.relations, repeat=2):
            if r1 == r2: continue

            sig1 = self.relation_signatures[r1]
            sig2 = self.relation_signatures[r2]

            contextual_fit = util.cos_sim(sig1["object_type"]["vector"], sig2["subject_type"]["vector"]).item()

            emb1 = self.relation_embeddings.get(r1)
            emb2 = self.relation_embeddings.get(r2)
            lexical_similarity = 0.0
            if emb1 is not None and emb2 is not None:
                lexical_similarity = util.cos_sim(emb1, emb2).item()


            combined_confidence = sig1["object_type"]["confidence"] * sig2["subject_type"]["confidence"]
            alpha = combined_confidence

            hybrid_weight = alpha * contextual_fit + (1 - alpha) * lexical_similarity

            final_confidence = combined_confidence

            self.dsrg[(r1, r2)] = {"weight": hybrid_weight, "confidence": final_confidence}

    def _normalize_confidences(self):
        all_confidences = []
        for entity, data in self.contextual_type_vectors.items():
            all_confidences.append(data['confidence'])
        for rel, data in self.relation_signatures.items():
            all_confidences.append(data['subject_type']['confidence'])
            all_confidences.append(data['object_type']['confidence'])

        if not all_confidences: return

        min_conf, max_conf = min(all_confidences), max(all_confidences)

        if max_conf == min_conf: return

        for entity, data in self.contextual_type_vectors.items():
            data['confidence'] = (data['confidence'] - min_conf) / (max_conf - min_conf)
        for rel, data in self.relation_signatures.items():
            data['subject_type']['confidence'] = (data['subject_type']['confidence'] - min_conf) / (max_conf - min_conf)
            data['object_type']['confidence'] = (data['object_type']['confidence'] - min_conf) / (max_conf - min_conf)

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subgraph_path', type=str, default="")
    parser.add_argument('--semantic_plans_path', type=str, default="")
    parser.add_argument('--query_plans_path', type=str, default="")
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = GTELargeEN(device=device)



    subgraph_data = torch.load(args.subgraph_path, map_location='cpu')
    with open(args.semantic_plans_path, 'r', encoding='utf-8') as f:
        semantic_plans = {item['id']: item for item in [json.loads(line) for line in f]}
    with open(args.query_plans_path, 'r', encoding='utf-8') as f:
        query_plans = {item['id']: item for item in [json.loads(line) for line in f]}

    output_data = {}

    for sample_id, sample_info in tqdm(subgraph_data.items(), desc="Processing Subgraphs"):

        # dsrg_processor = DSRGProcessor(sample_info.get('scored_triples', []), text_encoder)
        # computed_dsrg_data = dsrg_processor.run()

        new_sample_info = sample_info.copy()
        # new_sample_info.update(computed_dsrg_data)

        semantic_plan = semantic_plans.get(sample_id)
        query_plan = query_plans.get(sample_id)

        if semantic_plan and query_plan:
            feature_augmentor = FeatureAugmentor(semantic_plan, query_plan, text_encoder)
            computed_feature_data = feature_augmentor.run()
            new_sample_info.update(computed_feature_data)

        output_data[sample_id] = new_sample_info


    if args.output_path is None:
        dir_name = os.path.dirname(args.subgraph_path)
        name, _ = os.path.splitext(os.path.basename(args.subgraph_path))
        args.output_path = os.path.join(dir_name, f"{name}_features_augmented.pth")

    torch.save(output_data, args.output_path)



if __name__ == '__main__':
    main()







