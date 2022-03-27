class SqlAdminException(Exception):
    pass


class InvalidModelError(SqlAdminException):
    pass


class InvalidColumnError(SqlAdminException):
    pass


class NoConverterFound(SqlAdminException):
    pass
