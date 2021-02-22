# -*- coding: utf-8 -*-
"""
Copyright (c) 2018-2019, Martin Larralde
Copyright (c) 2015-2018, Daniel Greenfeld
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of cached-property or property-cached nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import asyncio
import functools
import threading
from time import time
from typing import Any, Callable, Mapping, Optional
import weakref

__author__ = "Martin Larralde"
__email__ = "martin.larralde@ens-paris-saclay.fr"
__license__ = "BSD"
__version__ = "1.6.3"


class cached_property(property):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """  # noqa

    _sentinel = object()

    _update_wrapper = functools.update_wrapper

    def __init__(self, func) -> None:
        self.cache: Mapping[str, Any] = weakref.WeakKeyDictionary()
        self.func: Callable[[], Any] = func
        self._update_wrapper(func)  # type: ignore

    def __get__(self, obj, cls):
        if obj is None:
            return self

        if asyncio and asyncio.iscoroutinefunction(self.func):
            return self._wrap_in_coroutine(obj)

        value = self.cache.get(obj, self._sentinel)
        if value is self._sentinel:
            value = self.cache[obj] = self.func(obj)

        return value

    def __set_name__(self, owner, name):
        self.__name__: str = name

    def __set__(self, obj, value):
        self.cache[obj] = value

    def __delete__(self, obj):
        del self.cache[obj]

    def _wrap_in_coroutine(self, obj):
        @functools.wraps(obj)
        @asyncio.coroutine
        def wrapper():
            value = self.cache.get(obj, self._sentinel)
            if value is self._sentinel:
                self.cache[obj] = value = asyncio.ensure_future(self.func(obj))
            return value

        return wrapper()


class threaded_cached_property(cached_property):
    """
    A cached_property version for use in environments where multiple threads
    might concurrently try to access the property.
    """

    def __init__(self, func) -> None:
        super(threaded_cached_property, self).__init__(func)
        self.lock: threading.RLock = threading.RLock()

    def __get__(self, obj, cls):
        if obj is None:
            return self
        with self.lock:
            return super(threaded_cached_property, self).__get__(obj, cls)

    def __set__(self, obj, value):
        with self.lock:
            super(threaded_cached_property, self).__set__(obj, value)

    def __delete__(self, obj):
        with self.lock:
            super(threaded_cached_property, self).__delete__(obj)


class cached_property_with_ttl(cached_property):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Setting the ttl to a number expresses how long
    the property will last before being timed out.
    """

    def __init__(self, ttl=None) -> None:
        if callable(ttl):
            func = ttl
            ttl = None
        else:
            func = None
        self.ttl: Optional[int] = ttl
        super(cached_property_with_ttl, self).__init__(func)

    def __call__(self, func):
        super(cached_property_with_ttl, self).__init__(func)
        return self

    def __get__(self, obj, cls):
        if obj is None:
            return self

        now = time()
        if obj in self.cache:
            value, last_updated = self.cache[obj]
            if not self.ttl or self.ttl > now - last_updated:
                return value

        value, _ = self.cache[obj] = (self.func(obj), now)
        return value

    def __set__(self, obj, value):
        super(cached_property_with_ttl, self).__set__(obj, (value, time()))


# Aliases to make cached_property_with_ttl easier to use
cached_property_ttl = cached_property_with_ttl
timed_cached_property = cached_property_with_ttl


class threaded_cached_property_with_ttl(
    cached_property_with_ttl, threaded_cached_property
):
    """
    A cached_property version for use in environments where multiple threads
    might concurrently try to access the property.
    """

    def __init__(self, ttl=None) -> None:
        super(threaded_cached_property_with_ttl, self).__init__(ttl)
        self.lock: threading.RLock = threading.RLock()

    def __get__(self, obj, cls):
        with self.lock:
            return super(threaded_cached_property_with_ttl, self).__get__(obj, cls)

    def __set__(self, obj, value):
        with self.lock:
            return super(threaded_cached_property_with_ttl, self).__set__(obj, value)

    def __delete__(self, obj):
        with self.lock:
            return super(threaded_cached_property_with_ttl, self).__delete__(obj)


# Alias to make threaded_cached_property_with_ttl easier to use
threaded_cached_property_ttl = threaded_cached_property_with_ttl
timed_threaded_cached_property = threaded_cached_property_with_ttl
