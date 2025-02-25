from __future__ import annotations

import abc
from typing import List

import attr

from pyrdf2vec.typings import Embeddings, Entities, SWalk


@attr.s
class Embedder(metaclass=abc.ABCMeta):
    """Base class of the embedding techniques."""

    @abc.abstractmethod
    def fit(
        self, corpus: List[List[SWalk]], is_update: bool = False
    ) -> Embedder:
        """Fits a model based on the provided corpus.

        Args:
            corpus: The corpus to fit the model.

        Returns:
            The fitted model according to an embedding technique.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This has to be implemented")

    @abc.abstractmethod
    def transform(self, entities: Entities) -> Embeddings:
        """Constructs a features vector of the provided entities.

        Args:
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The features vector of the provided entities.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This has to be implemented")
    
    @abc.abstractmethod
    def predict(self, walks: List[List[SWalk]]) -> Embeddings:
        """Predicts the embeddings of the provided entities.

        Args:
            entities: The entities to predict the embeddings.

        Returns:
            The embeddings of the provided entities.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This has to be implemented")
    
    @abc.abstractmethod
    def init(self, walks: List[List[SWalk]]) -> Embedder:
        """Initializes the model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.

        Returns:
            The initialized model.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This has to be implemented")
