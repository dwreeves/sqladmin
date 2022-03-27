import inspect
from enum import Enum
import warnings
from dataclasses import dataclass, field
from collections import ChainMap
from sqladmin.helpers import cached_property
from typing import Any, List, Generic, Mapping, Callable, FrozenSet, Dict, Optional, Sequence, Type, Union, no_type_check, TypeVar
from typing_extensions import Protocol

import anyio
from sqlalchemy import inspect as sqlalchemy_inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import MapperProperty, ColumnProperty, Mapper, RelationshipProperty, Session, InstrumentedAttribute
from sqlalchemy.sql.schema import Column
from wtforms import (
    BooleanField,
    DateField,
    DateTimeField,
    DecimalField,
    Field,
    Form,
    IntegerField,
    SelectField,
    StringField,
    TextAreaField,
    validators,
)
from wtforms.fields.core import UnboundField

from sqladmin.fields import JSONField, QuerySelectField, QuerySelectMultipleField
from sqladmin.exceptions import NoConverterFound


# Type - Converter Callable
T = TypeVar("T")
Validator = Callable[[Form, Field], None]

T_MP = TypeVar("T_MP", ColumnProperty, RelationshipProperty)
TF = TypeVar("TF", bound=Field)


class ConverterCallable(Protocol):
    def __call__(
            self,
            model: type,
            prop: T_MP,
            kwargs: Dict[str, Any]
    ) -> UnboundField:
        ...


T_CC = TypeVar("T_CC", bound=ConverterCallable)


class AdminAttribute:

    def __init__(
            self,
            sqla_model: type,
            sqla_attribute: Union[str, InstrumentedAttribute],
            *,
            label: Optional[str] = None,
            wtf_field_type_override: Type[Field] = None,
            wtf_extra_field_kwargs: Dict[str, Any] = None,
            wtf_extra_validators: List[Callable[[Form, Field], None]] = None
    ):
        self.sqla_model = sqla_model

        if isinstance(sqla_attribute, str):
            self.sqla_attribute = getattr(sqla_model, sqla_attribute)
        else:
            self.sqla_attribute = sqla_attribute
        assert isinstance(self.sqla_attribute, InstrumentedAttribute)
        assert self.sqla_attribute.class_ is sqla_model

        self.label = label

        self.wtf_extra_field_kwargs = wtf_extra_field_kwargs or {}
        self.wtf_field_type_override = wtf_field_type_override
        self.wtf_extra_validators = wtf_extra_validators or []

    @property
    def is_relationship(self) -> bool:
        return isinstance(self.sqla_attribute.prop, RelationshipProperty)

    @property
    def ensured_label(self) -> str:
        """self.label is allowed to be optional.
        If you definitely want a string to represent the column,
        then you want to use this property.
        """
        return self.label or self.sqla_property.key

    @property
    def sqla_property(self) -> MapperProperty:
        return self.sqla_attribute.prop

    @cached_property
    def sqla_column(self) -> Column:
        assert len(self.sqla_property.columns) == 1, "Multiple-column properties not supported"
        column = self.sqla_property.columns[0]
        return column

    def _get_default_value(self) -> Any:
        default = getattr(self.sqla_column, "default", None)

        if default is not None:
            # Only actually change default if it has an attribute named
            # 'arg' that's callable.
            callable_default = getattr(default, "arg", None)

            if callable_default is not None:
                # ColumnDefault(val).arg can be also a plain value
                default = (
                    callable_default(None)
                    if callable(callable_default)
                    else callable_default
                )

        return default

    @property
    def wtf_base_validators(self) -> List[Validator]:
        if self.is_relationship:
            return []
        li = []
        if self.sqla_column.nullable:
            li.append(validators.Optional())
        else:
            li.append(validators.InputRequired())
        return li

    @property
    def _wtf_default_field_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "label": self.label,
            "validators": [
                *self.wtf_base_validators,
                *self.wtf_extra_validators
            ],
            "description": self.sqla_property.doc,
            "render_kw": {"class": "form-control"},
            "default": self._get_default_value()
        }
        if self.is_relationship:
            allow_blank = True
            for pair in self.sqla_property.local_remote_pairs:
                if not pair[0].nullable:
                    allow_blank = False
            kwargs["allow_blank"] = allow_blank
        return kwargs

    async def get_object_list(
            self,
            engine: Union[Engine, AsyncEngine]
    ) -> List[Any]:
        target_model = self.sqla_property.mapper.class_
        pk = sqlalchemy_inspect(target_model).primary_key[0].name
        stmt = select(target_model)

        if isinstance(engine, Engine):
            with Session(engine) as session:
                objects = await anyio.to_thread.run_sync(session.execute, stmt)
                object_list = [
                    (str(getattr(obj, pk)), obj)
                    for obj in objects.scalars().all()
                ]
        else:
            async with AsyncSession(engine) as session:
                objects = await session.execute(stmt)
                object_list = [
                    (str(getattr(obj, pk)), obj)
                    for obj in objects.scalars().all()
                ]
        return object_list

    @cached_property
    def wtf_field_kwargs(self) -> ChainMap:
        d = ChainMap(
            self.wtf_extra_field_kwargs,
            self._wtf_default_field_kwargs,
        )
        return d


class ModelConverterBase:

    _callbacks: Dict[str, ConverterCallable] = {}

    def __init__(self):
        super().__init__()
        self._register_callbacks()

    def _register_callbacks(self):
        converters: Dict[str, ConverterCallable] = {}
        for name in dir(self):
            obj = getattr(self, name)
            if hasattr(obj, "_converter_for"):
                for classname in obj._converter_for:
                    converters[classname] = obj
        self._callbacks = converters

    def get_callback(
            self,
            model: type,
            prop: T_MP
    ) -> ConverterCallable:
        if isinstance(prop, RelationshipProperty):
            return self._callbacks[prop.direction.name]

        column = prop.columns[0]
        types = inspect.getmro(type(column.type))

        # Search by module + name
        for col_type in types:
            type_string = f"{col_type.__module__}.{col_type.__name__}"

            if type_string in self._callbacks:
                return self._callbacks[type_string]

        # Search by name
        for col_type in types:
            if col_type.__name__ in self._callbacks:
                return self._callbacks[col_type.__name__]

            # Support for custom types like SQLModel which inherit TypeDecorator
            if hasattr(col_type, "impl"):
                if col_type.impl.__name__ in self._callbacks:  # type: ignore
                    return self._callbacks[col_type.impl.__name__]  # type: ignore

        raise NoConverterFound(  # pragma: nocover
            f"Could not find field converter for column {column.name} ({types[0]!r})."
        )

    def convert(
            self,
            model: type,
            prop: T_MP,
            kwargs: Mapping[str, Any]
    ) -> UnboundField:
        callback = self.get_callback(model, prop)
        return callback(
            model=model,
            prop=prop,
            kwargs=kwargs
        )


def converts(*args: str) -> Callable[[T_CC], T_CC]:
    def _inner(func: T_CC) -> T_CC:
        func._converter_for = frozenset(args)
        return func
    return _inner


class ModelConverter(ModelConverterBase):
    @staticmethod
    def _string_common(prop: ColumnProperty) -> List[Validator]:
        li = []
        column = prop.columns[0]
        if isinstance(column.type.length, int) and column.type.length:
            li.append(validators.Length(max=column.type.length))
        return li

    @converts("String")  # includes Unicode
    def conv_String(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("validators", [])
        extra_validators = self._string_common(prop)
        kwargs["validators"].extend(extra_validators)
        return StringField(**kwargs)

    @converts("Text", "LargeBinary", "Binary")  # includes UnicodeText
    def conv_Text(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("validators", [])
        extra_validators = self._string_common(prop)
        kwargs["validators"].extend(extra_validators)
        return TextAreaField(**kwargs)

    @converts("Boolean", "dialects.mssql.base.BIT")
    def conv_Boolean(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("render_kw", {})
        kwargs["render_kw"]["class"] = "form-check-input"
        return BooleanField(**kwargs)

    @converts("Date")
    def conv_Date(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return DateField(**kwargs)

    @converts("DateTime")
    def conv_DateTime(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return DateTimeField(**kwargs)

    @converts("Enum")
    def conv_Enum(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        available_choices = [(e, e) for e in prop.columns[0].type.enums]
        accepted_values = [choice[0] for choice in available_choices]

        kwargs["choices"] = available_choices
        kwargs.setdefault("validators", [])
        kwargs["validators"].append(validators.AnyOf(accepted_values))
        kwargs["coerce"] = lambda v: v.name if isinstance(v, Enum) else str(v)
        return SelectField(**kwargs)

    @converts("Integer")  # includes BigInteger and SmallInteger
    def handle_integer_types(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return IntegerField(**kwargs)

    @converts("Numeric")  # includes DECIMAL, Float/FLOAT, REAL, and DOUBLE
    def handle_decimal_types(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        # override default decimal places limit, use database defaults instead
        kwargs.setdefault("places", None)
        return DecimalField(**kwargs)

    # @converts("dialects.mysql.types.YEAR", "dialects.mysql.base.YEAR")
    # def conv_MSYear(
    #         self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    # ) -> UnboundField:
    #     kwargs.setdefault("validators", [])
    #     kwargs["validators"].append(validators.NumberRange(min=1901, max=2155))
    #     return StringField(**kwargs)

    @converts("sqlalchemy.dialects.postgresql.base.INET")
    def conv_PGInet(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("label", "IP Address")
        kwargs["validators"].append(validators.IPAddress())
        return StringField(**kwargs)

    @converts("sqlalchemy.dialects.postgresql.base.MACADDR")
    def conv_PGMacaddr(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("label", "MAC Address")
        kwargs.setdefault("validators", [])
        kwargs["validators"].append(validators.MacAddress())
        return StringField(**kwargs)

    @converts("sqlalchemy.dialects.postgresql.base.UUID")
    def conv_PgUuid(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        kwargs.setdefault("label", "UUID")
        kwargs.setdefault("validators", [])
        kwargs["validators"].append(validators.UUID())
        return StringField(**kwargs)

    @converts("JSON")
    def convert_JSON(
            self, model: type, prop: ColumnProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return JSONField(**kwargs)

    @converts("MANYTOONE")
    def conv_ManyToOne(
            self, model: type, prop: RelationshipProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return QuerySelectField(**kwargs)

    @converts("MANYTOMANY", "ONETOMANY")
    def conv_ManyToMany(
            self, model: type, prop: RelationshipProperty, kwargs: Dict[str, Any]
    ) -> UnboundField:
        return QuerySelectMultipleField(**kwargs)


async def new_get_model_form(
        attributes: List[AdminAttribute],
        engine: Union[Engine, AsyncEngine],
        *,
        form_class: Type[Form] = Form,
        converter_class: Type[ModelConverterBase] = ModelConverter,
) -> Type[Form]:
    assert len(attributes) > 0
    type_name = attributes[0].sqla_model.__name__ + "Form"
    converter = converter_class()

    field_dict: Dict[str, UnboundField] = {}

    for attr in attributes:
        kwargs = attr.wtf_field_kwargs
        if attr.is_relationship:
            kwargs["object_list"] = await attr.get_object_list(engine=engine)
        if attr.wtf_field_type_override is not None:
            wtf_field = attr.wtf_field_type_override(**kwargs)
        else:
            wtf_field = converter.convert(
                model=attr.sqla_model,
                prop=attr.sqla_property,
                kwargs=kwargs
            )
        field_dict[attr.sqla_property.key] = wtf_field

    return type(type_name, (form_class,), field_dict)


async def get_model_form(
    model: type,
    engine: Union[Engine, AsyncEngine],
    only: Sequence[str] = None,
    exclude: Sequence[str] = None,
    column_labels: Dict[str, str] = None,
    form_args: Dict[str, Dict[str, Any]] = None,
    form_class: Type[Form] = Form,
    form_overrides: Dict[str, Dict[str, Type[Field]]] = None,
) -> Type[Form]:
    type_name = model.__name__ + "Form"
    converter = ModelConverter()
    mapper = sqlalchemy_inspect(model)
    form_args = form_args or {}
    column_labels = column_labels or {}
    form_overrides = form_overrides or {}

    attributes = []
    for name, attr in mapper.attrs.items():
        if only and name not in only:
            continue
        elif exclude and name in exclude:
            continue

        attributes.append((name, attr))

    field_dict = {}
    for name, attr in attributes:
        field_args = form_args.get(name, {})
        label = column_labels.get(name, None)
        override = form_overrides.get(name, None)
        field = converter.convert(
            model, attr, field_args
        )
        if field is not None:
            field_dict[name] = field

    return type(type_name, (form_class,), field_dict)
