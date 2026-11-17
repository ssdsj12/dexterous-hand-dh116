"""Inject CPU stubs into private scheduler copies without moving module aliases."""
import ast


def inject_before_cli_main(source, stub):
    """Require one top-level __main__ guard that actually calls main()."""
    main_test=ast.dump(ast.parse("__name__ == '__main__'",mode="eval").body)

    def calls_main(guard):
        pending=list(guard.body)
        while pending:
            node=pending.pop()
            # Definitions do not execute their bodies when the guard runs.
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):
                continue
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                    and node.func.id=="main"):
                return True
            pending.extend(ast.iter_child_nodes(node))
        return False

    guards=[node for node in ast.parse(source).body
        if isinstance(node,ast.If) and ast.dump(node.test)==main_test and calls_main(node)]
    if len(guards)!=1:
        raise ValueError("one guarded CLI main() call required")
    offset=sum(len(line) for line in source.splitlines(keepends=True)[:guards[0].lineno-1])
    return source[:offset]+stub+"\n"+source[offset:]
