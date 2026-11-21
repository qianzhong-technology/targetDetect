"""统一的控制台状态日志格式。"""


def _format_value(value):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "{:.2f}".format(value)
    text = str(value).replace("\r", "").replace("\n", " ")
    return text.replace(" ", "_")


def print_status(module, status, **fields):
    """打印一行结构化状态日志。"""
    parts = ["status={}".format(status)]
    for name, value in fields.items():
        parts.append(
            "{}={}".format(name, _format_value(value))
        )
    print("[{}] {}".format(module, " ".join(parts)))
