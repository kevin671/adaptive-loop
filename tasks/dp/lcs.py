import random

from tasks import task

chars = "abcdefghijklmnopqrstuvwxyz"
dictionary = {c: i for i, c in enumerate(chars)}


def lcs_length(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i][j - 1], dp[i - 1][j])
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(str1)][len(str2)]


class LCS(task.GeneralizationTask):
    def __init__(self, max_length: int, using: int):
        self.max_length = max_length
        self.using = using

    def sample_batch(self, batch_size: int, length: int) -> task.Batch:
        """Returns a batch of strings and the expected class."""
        available_chars = random.choices(chars, k=self.using)
        strings = ["".join(random.choices(available_chars, k=length)) for _ in range(batch_size)]
        strings2 = ["".join(random.choices(available_chars, k=length)) for _ in range(batch_size)]
        n_b = [lcs_length(strings[i], strings2[i]) for i in range(batch_size)]
        # input: strings "|" strings2
        # output: n_b
        inputs = [s1 + "|" + s2 for s1, s2 in zip(strings, strings2)]
        # to

    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        return self.using

    @property
    def output_size(self) -> int:
        """Returns the output size for the models."""
        return self.max_length + 1


if __name__ == "__main__":
    task = LCS(10, 2)
