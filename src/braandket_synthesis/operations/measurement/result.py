from typing import Generic, Union

from braandket_synthesis import Op


class MeasurementResult(Generic[Op]):
    def __init__(self, *,
            values: Union[int, tuple],
            probability: float,
            operation: Op,
    ):
        self._value = values
        self._probability = probability
        self._operation = operation

    @property
    def value(self) -> int:
        return self._value

    @property
    def probability(self) -> float:
        return self._probability

    @property
    def operation(self) -> Op:
        return self._operation

    def __str__(self):
        return f"<MeasurementResult value={self.value}, probability={self.probability}, operation={self.operation}>"
