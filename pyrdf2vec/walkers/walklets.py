from typing import Any, Dict, Tuple

import attr
import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WalkletWalker(RandomWalker):
    """Defines the walklet walking strategy.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        seed: The seed to use to ensure ensure random determinism to generate
            the same walks for entities.
            Defaults to None.

    """

    def _extract(
        self, kg: KG, instance: rdflib.URIRef
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks = set()
        walks = self.extract_walks(kg, str(instance))
        for walk in walks:
            if len(walk) == 1:  # type:ignore
                canonical_walks.add((str(walk[0]),))  # type:ignore
            for n in range(1, len(walk)):  # type:ignore
                canonical_walks.add(
                    (str(walk[0]), str(walk[n]))  # type: ignore
                )
        return {instance: tuple(canonical_walks)}
