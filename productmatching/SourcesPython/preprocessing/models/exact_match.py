import typing as t

from utilities.preprocessing import Pipeline


class ExactMatchModel:
    def __init__(
        self,
        tok_norm_args: t.Union[str, dict],
    ):
        super().__init__()
        self.pipeline = Pipeline.create(tok_norm_args)

    def __call__(
        self,
        inputs: t.Tuple[str, str],
    ) -> float:
        left, right = inputs

        left = self.pipeline(left).split(" ")
        right = self.pipeline(right).split(" ")

        # exact name match
        if left == right:
            return 1.3

        # sorted name match
        if sorted(left) == sorted(right):
            return 1.2

        left_joined = "".join(left)
        right_joined = "".join(right)

        # match if names without spaces are the same
        if left_joined == right_joined:
            return 1.1

        return 0
