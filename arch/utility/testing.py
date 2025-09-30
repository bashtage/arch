from functools import cached_property

from scipy.stats import chi2

__all__ = ["WaldTestStatistic"]


class WaldTestStatistic:
    """
    Test statistic holder for Wald-type tests

    Parameters
    ----------
    stat : float
        The test statistic
    df : int
        Degree of freedom.
    null : str
        A statement of the test's null hypothesis
    alternative : str
        A statement of the test's alternative hypothesis
    name : str, default "" (empty)
        Name of test
    """

    def __init__(
        self,
        stat: float,
        df: int,
        null: str,
        alternative: str,
        name: str = "",
    ) -> None:
        self._stat = stat
        self._null = null
        self._alternative = alternative
        self.df: int = df
        self._name = name
        self.dist = chi2(df)
        self.dist_name: str = f"chi2({df})"

    @property
    def stat(self) -> float:
        """Test statistic"""
        return self._stat

    @cached_property
    def pval(self) -> float:
        """P-value of test statistic"""
        return 1 - self.dist.cdf(self.stat)

    @cached_property
    def critical_values(self) -> dict[str, float]:
        """Critical values test for common test sizes"""
        return dict(
            zip(["10%", "5%", "1%"], self.dist.ppf([0.9, 0.95, 0.99]), strict=False)
        )

    @property
    def null(self) -> str:
        """Null hypothesis"""
        return self._null

    @property
    def alternative(self) -> str:
        return self._alternative

    def __str__(self) -> str:
        name = "" if not self._name else self._name + "\n"
        return (
            f"{name}H0: {self.null}\n{name}H1: {self.alternative}\nStatistic: {self.stat:0.4f}\n"
            f"P-value: {self.pval:0.4f}\nDistributed: {self.dist}"
        )

    def __repr__(self) -> str:
        return (
            self.__str__() + "\n" + self.__class__.__name__ + f", id: {hex(id(self))}"
        )
