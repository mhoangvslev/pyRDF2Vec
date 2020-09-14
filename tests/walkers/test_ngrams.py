import random
from typing import List

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import NGramWalker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE]
)


def generate_entities() -> List[rdflib.URIRef]:
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestNGramWalker:
    def test_extract(self):
        canonical_walks = NGramWalker(2, 5, UniformSampler()).extract(
            KNOWLEDGE_GRAPH, generate_entities()
        )
        assert type(canonical_walks) == set
