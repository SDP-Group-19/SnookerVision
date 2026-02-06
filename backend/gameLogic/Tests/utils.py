def run(engine, *events):
    out = []
    for e in events:
        out += engine.on_event(e)
    return out