# VAR-X API considerations

Api should look something like


```
python
VARX(y:OptionalArrayLike,
     x:Union[OptionalArrayLike,Dict[Hashable,OptionalArrayLike],List[OptionalArrayLike]]
     lags:Union[int,List[int],None],
     constant:bool,
     hold_back:Union[None,int],
     volatiliy:MultivariateVolatilityModel,
     distribution:MultivariateDistribution
)
```

where `OptionalArrayLike = Union[None,DataFrame,ndarray]` allows for the
supported array types or `None`.

`x` can take many forms:

* `None`, for a standard VAR
* `ndarray` or `DataFrame`, regressors to use in all equations, e.g.,
  day of the week dummies.
* `Dict[Hashable, OptionalArrayLike]` will be a dictionary of
  regressors for equation where the equation name is the key
  (`Hashable`). This is flexible when only a subset of equations will
  contain additional regressors.
* `List[OptionalArrayLike]`

`lags` is interpreted as the maximum lag if an integer or the actual
lags to include if a list of integer.

## Zero Coefficients

An important feature is that ability to remove unnecessary coefficient
by setting the value of a coefficient to 0.  This is the same as
dropping the regressor from the model. This will be implemented using a
property called zeros that will look like

```
@property
def zeros(self):
    """Get the location of the zeros coefficient in the model"""
    return self._zero_locs

@zeros.setter
def zeros(self, value):
    # validation here
    self._zeros_locs = values
```

This structure will allow subclasses to be defined, e.g., `DiagonalVAR`
which only contain own lags.

This will likely need a custom class-based implementation, although
it will be simple to implement by backing with a `MultiIndex DataFrame`
where the column names are the dependent variables and the rows are
`(lag, variable_name)`.