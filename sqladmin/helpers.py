import csv
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Generator, Generic, List, TypeVar, Union

from sqlalchemy.orm import InstrumentedAttribute


T = TypeVar("T")


_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.-]")
_windows_device_files = (
    "CON",
    "AUX",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "LPT1",
    "LPT2",
    "LPT3",
    "PRN",
    "NUL",
)


def as_str(s: Union[str, bytes]) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")

    return str(s)


def prettify_class_name(name: str) -> str:
    return re.sub(r"(?<=.)([A-Z])", r" \1", name)


def slugify_class_name(name: str) -> str:
    dashed = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", dashed).lower()


def secure_filename(filename: str) -> str:
    """Ported from Werkzeug.

    Pass it a filename and it will return a secure version of it. This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`. The filename returned is an ASCII only string
    for maximum portability.
    On windows systems the function also makes sure that the file is not
    named after one of the special device files.
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.path.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(_filename_ascii_strip_re.sub("", "_".join(filename.split()))).strip(
        "._"
    )

    # on nt a couple of special files are present in each folder.  We
    # have to ensure that the target file is not such a filename.  In
    # this case we prepend an underline
    if (
        os.name == "nt"
        and filename
        and filename.split(".")[0].upper() in _windows_device_files
    ):
        filename = f"_{filename}"  # pragma: no cover

    return filename


class Writer(ABC):
    """https://docs.python.org/3/library/csv.html#writer-objects"""

    @abstractmethod
    def writerow(self, row: List[str]) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def writerows(self, rows: List[List[str]]) -> None:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def dialect(self) -> csv.Dialect:
        pass  # pragma: no cover


class _PseudoBuffer(object):
    """An object that implements just the write method of the file-like
    interface.
    """

    def write(self, value: T) -> T:
        return value


def stream_to_csv(
    callback: Callable[[Writer], Generator[T, None, None]]
) -> Generator[T, None, None]:
    """Function that takes a callable (that yields from a CSV Writer), and
    provides it a writer that streams the output directly instead of
    storing it in a buffer. The direct output stream is intended to go
    inside a `starlette.responses.StreamingResponse`.

    Loosely adapted from here:

    https://docs.djangoproject.com/en/1.8/howto/outputting-csv/
    """
    writer = csv.writer(_PseudoBuffer())
    return callback(writer)


class _Missing:
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


class cached_property(property, Generic[T]):  # noqa
    """A :func:`property` that is only evaluated once. Subsequent access
    returns the cached value. Setting the property sets the cached
    value. Deleting the property clears the cached value, accessing it
    again will evaluate it again.
    .. code-block:: python
        class Example:
            @cached_property
            def value(self):
                # calculate something important here
                return 42
        e = Example()
        e.value  # evaluates
        e.value  # uses cache
        e.value = 16  # sets cache
        del e.value  # clears cache
    If the class defines ``__slots__``, it must add ``_cache_{name}`` as
    a slot. Alternatively, it can add ``__dict__``, but that's usually
    not desirable.
    .. versionchanged:: 2.1
        Works with ``__slots__``.
    .. versionchanged:: 2.0
        ``del obj.name`` clears the cached value.
    """

    def __init__(
        self,
        fget: Callable[[Any], T],
        name: Optional[str] = None,
        doc: Optional[str] = None,
    ) -> None:
        super().__init__(fget, doc=doc)
        self.__name__ = name or fget.__name__
        self.slot_name = f"_cache_{self.__name__}"
        self.__module__ = fget.__module__

    def __set__(self, obj: object, value: T) -> None:
        if hasattr(obj, "__dict__"):
            obj.__dict__[self.__name__] = value
        else:
            setattr(obj, self.slot_name, value)

    def __get__(self, obj: object, type: type = None) -> T:  # type: ignore
        if obj is None:
            return self  # type: ignore

        obj_dict = getattr(obj, "__dict__", None)

        if obj_dict is not None:
            value: T = obj_dict.get(self.__name__, _missing)
        else:
            value = getattr(obj, self.slot_name, _missing)  # type: ignore[arg-type]

        if value is _missing:
            value = self.fget(obj)  # type: ignore

            if obj_dict is not None:
                obj.__dict__[self.__name__] = value
            else:
                setattr(obj, self.slot_name, value)

        return value

    def __delete__(self, obj: object) -> None:
        if hasattr(obj, "__dict__"):
            del obj.__dict__[self.__name__]
        else:
            setattr(obj, self.slot_name, _missing)


def get_key(
        sqla_model: type,
        sqla_attribute: Union[str, InstrumentedAttribute]
) -> str:
    if isinstance(sqla_attribute, str):
        return sqla_attribute
    else:
        return sqla_attribute
