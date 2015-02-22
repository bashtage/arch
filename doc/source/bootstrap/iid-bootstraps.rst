.. py:currentmodule:: arch.bootstrap

Independent, Identical Distributed Data (i.i.d.)
------------------------------------------------
:py:class:`~arch.bootstrap.IIDBootstrap` is the standard bootstrap that is appropriate for data that is
either i.i.d. or at least not serially dependant.

.. autoclass:: IIDBootstrap
    :members: conf_int, var, cov, apply, bootstrap, reset, seed, set_state, get_state
