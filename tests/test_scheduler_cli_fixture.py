"""CPU-only source injection preserves scheduler identity and entry ordering."""
import importlib
import importlib.util

import pytest


def injector():
    assert importlib.util.find_spec("scheduler_cli_fixture") is not None
    return importlib.import_module("scheduler_cli_fixture").inject_before_cli_main


def test_two_guards_preserve_alias_and_inject_once_before_actual_main():
    source="""events=[]
if __name__ == '__main__':
    events.append('alias')
def main():
    events.append('main')
if __name__ == '__main__':
    main()
"""
    prefix=source[:source.rindex("if __name__")]
    result=injector()(source,"events.append('stub')")
    assert result.startswith(prefix)
    assert result==prefix+"events.append('stub')\n"+source[len(prefix):]
    namespace={"__name__":"__main__"}
    exec(compile(result,"<cpu-fixture>","exec"),namespace)
    assert namespace["events"]==["alias","stub","main"]


@pytest.mark.parametrize("source",[
    "if __name__ == '__main__':\n    alias=True\n",
    "if __name__ == '__main__':\n    def deferred():\n        main()\n",
    "if __name__ == '__main__': main()\nif __name__ == '__main__': main()\n",
])
def test_missing_or_ambiguous_actual_cli_entry_is_rejected(source):
    with pytest.raises(ValueError,match="one guarded CLI main"):
        injector()(source,"stub=True")
