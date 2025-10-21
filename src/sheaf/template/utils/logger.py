import logging
from pprint import pformat
from string.templatelib import Interpolation, Template, convert

COLOR_RESET = "\x1b[0m"
COLOR_GREEN = "\x1b[32m"
COLOR_BLUE = "\x1b[34m"
COLOR_YELLOW = "\x1b[33m"
COLOR_RED = "\x1b[31m"
COLOR_MAGENTA = "\x1b[35m"


class CustomFormatter(logging.Formatter):
    FORMAT = "%(asctime)s %(levelname)s %(message)s"

    FORMATS = {
        logging.DEBUG: COLOR_GREEN + FORMAT + COLOR_RESET,
        logging.INFO: COLOR_BLUE + FORMAT + COLOR_RESET,
        logging.WARNING: COLOR_YELLOW + FORMAT + COLOR_RESET,
        logging.ERROR: COLOR_RED + FORMAT + COLOR_RESET,
        logging.CRITICAL: COLOR_MAGENTA + FORMAT + COLOR_RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        return logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


def render_template(template: Template) -> str:
    """Turn a t-string Template into a string.

    - applies !s/!r/!a conversions via string.templatelib.convert()
    - applies :format_spec when present
    - pretty-prints non-strings by default when no explicit formatting
    """
    parts: list[str] = []
    for item in template:
        if isinstance(item, Interpolation):
            val = item.value
            if item.conversion is not None:  # like f-strings' !s / !r / !a
                val = convert(val, item.conversion)
            if item.format_spec:  # like f-strings' :spec
                try:
                    val = format(val, item.format_spec)
                except Exception:
                    val = str(val)
            if (
                (item.conversion is None)
                and (not item.format_spec)
                and not isinstance(val, str)
            ):
                val = pformat(val)
            parts.append(str(val))
        else:
            parts.append(item)
    return "".join(parts)


class PrettyLogger(logging.Logger):
    """Logger that understands t-strings and pretty-prints other objects."""

    def _render(self, obj) -> str:
        if isinstance(obj, Template):
            return render_template(obj)
        if isinstance(obj, str):
            return obj
        return pformat(obj)

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        # Supports: logger.info(t"..."), logger.info("...", obj), etc.
        pieces = (
            [msg]
            if not args
            else [msg, *(args if isinstance(args, tuple) else (args,))]
        )
        pretty = " ".join(self._render(p) for p in pieces if p is not None)
        super()._log(
            level,
            pretty,
            (),
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )


logging.setLoggerClass(PrettyLogger)
logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)
