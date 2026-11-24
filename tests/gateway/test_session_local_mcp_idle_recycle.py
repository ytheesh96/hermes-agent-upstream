"""Editor stdio MCP servers recycle when idle instead of holding a Node child for the session's life.

An editor's stdio MCP server is a local child process (~45-70 MB for a Node server) that a chat uses
in bursts, yet an ACP session (a Buzz chat, a long-lived editor connection) is resident for hours.
Hermes already recycles an idle stdio child (``idle_timeout_seconds``) and transparently reconnects
it on the next call, so the session layer only has to (a) configure the limit on the editor's stdio
specs and (b) keep counting a recycled server as connected — its registry tools stay frozen and its
next call wakes it. Otherwise ``_borrowed`` would treat the recycled server as missing, call
``register_mcp_servers`` (which only signals an asynchronous reconnect) and refuse the turn with
``acp_mcp_discovery_failed``.
"""
import json
import sys
import time
from types import SimpleNamespace

import pytest


STDIO_SPEC = {'name': 'probe', 'command': 'npx', 'args': ['-y', 'editor-mcp'],
              'env': [{'name': 'TOKEN', 'value': 'secret'}]}
HTTP_SPEC = {'name': 'remote', 'type': 'http', 'url': 'https://editor.test/mcp',
             'headers': [{'name': 'Authorization', 'value': 'Bearer x'}]}
SSE_SPEC = {'name': 'legacy', 'type': 'sse', 'url': 'https://editor.test/sse'}


def _configs(monkeypatch, servers=(), cwd='/workspace'):
    from gateway import session_local_mcp
    servers = list(servers)
    names = {i: f'editor_{spec["name"]}_0' for i, spec in enumerate(servers)}
    return session_local_mcp._configs(servers, names, cwd)


def _fake_server(*, session=None, recycled=False, name='editor_a'):
    """A held server stand-in: a live session, or a recycled stdio server (no session, intentional)."""
    return SimpleNamespace(name=name, session=session, _is_recycled_stdio=lambda: recycled,
                           _rpc_lock=None, _tools=[])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_stdio_servers_get_the_default_idle_timeout(monkeypatch):
    monkeypatch.delenv('HERMES_EDITOR_MCP_IDLE_TIMEOUT_SECONDS', raising=False)
    configs = _configs(monkeypatch, [STDIO_SPEC, HTTP_SPEC, SSE_SPEC])

    stdio = configs['editor_probe_0']
    assert stdio['idle_timeout_seconds'] == 1200
    # The transport config itself is untouched.
    assert stdio['command'] == 'npx' and stdio['args'] == ['-y', 'editor-mcp']
    assert stdio['env'] == {'TOKEN': 'secret'} and stdio['cwd'] == '/workspace'
    # No local child to recycle on a remote transport: leave those specs exactly as before.
    assert 'idle_timeout_seconds' not in configs['editor_remote_0']
    assert 'idle_timeout_seconds' not in configs['editor_legacy_0']
    assert configs['editor_legacy_0']['transport'] == 'sse'


def test_env_overrides_the_idle_timeout(monkeypatch):
    monkeypatch.setenv('HERMES_EDITOR_MCP_IDLE_TIMEOUT_SECONDS', '90')
    configs = _configs(monkeypatch, [STDIO_SPEC, HTTP_SPEC])

    assert configs['editor_probe_0']['idle_timeout_seconds'] == 90
    assert 'idle_timeout_seconds' not in configs['editor_remote_0']


@pytest.mark.parametrize('value', ['0', '-30', '0.0'])
def test_zero_or_negative_env_disables_recycling(monkeypatch, value):
    monkeypatch.setenv('HERMES_EDITOR_MCP_IDLE_TIMEOUT_SECONDS', value)
    configs = _configs(monkeypatch, [STDIO_SPEC])

    assert 'idle_timeout_seconds' not in configs['editor_probe_0']


@pytest.mark.parametrize('value', ['', '  ', 'soon', 'nan', 'inf'])
def test_unusable_env_falls_back_to_the_default(monkeypatch, value):
    """Only an explicit 0/negative turns recycling off: a typo must not pin every child resident."""
    monkeypatch.setenv('HERMES_EDITOR_MCP_IDLE_TIMEOUT_SECONDS', value)
    configs = _configs(monkeypatch, [STDIO_SPEC])

    assert configs['editor_probe_0']['idle_timeout_seconds'] == 1200


# ---------------------------------------------------------------------------
# Availability: a recycled stdio server is still connected
# ---------------------------------------------------------------------------

def test_connected_counts_a_recycled_stdio_server(monkeypatch):
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    scope, names = 'editor-session:availability', {0: 'editor_live', 1: 'editor_idle',
                                                  2: 'editor_down', 3: 'editor_unknown'}
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_live', scope),
                        _fake_server(session=object(), name='editor_live'))
    # Recycled: no session, but it kept its tools and wakes on the next call.
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_idle', scope),
                        _fake_server(session=None, recycled=True, name='editor_idle'))
    # Genuinely disconnected (reconnect/park, never recycled): not part of a frozen manifest's tools.
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_down', scope),
                        _fake_server(session=None, recycled=False, name='editor_down'))
    # A server object without the recycle predicate (an older/foreign task) is not assumed available.
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_unknown', scope),
                        SimpleNamespace(name='editor_unknown', session=None))

    assert session_local_mcp._connected(scope, names) == [0, 1]
    assert session_local_mcp._connected(scope, {9: 'editor_absent'}) == []


def test_live_tools_changed_ignores_a_recycled_server(monkeypatch):
    """A recycled server cannot answer tools/list; it was never asked and counts as unchanged."""
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    scope, names = 'editor-session:idle-relist', {0: 'editor_idle'}
    server = _fake_server(session=None, recycled=True, name='editor_idle')
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_idle', scope), server)

    assert session_local_mcp._live_tools_changed(scope, names) is False


# ---------------------------------------------------------------------------
# Borrowing: a recycled server is reused, never re-discovered
# ---------------------------------------------------------------------------

@pytest.fixture
def frozen_registry(monkeypatch):
    """One registered tool of the held toolset, with a stable schema (no MCP child involved)."""
    from tools.registry import registry
    monkeypatch.setattr(registry, 'get_all_tool_names', lambda: ['mcp__editor_a_0__ping_editor'])
    monkeypatch.setattr(registry, 'get_toolset_for_tool', lambda tool: 'mcp-editor_a_0')
    monkeypatch.setattr(registry, 'get_schema', lambda tool: {'name': tool, 'description': 'pong'})
    return {'mcp__editor_a_0__ping_editor': {'name': 'mcp__editor_a_0__ping_editor',
                                             'description': 'pong'}}


@pytest.fixture
def discovery_spy(monkeypatch):
    from tools import mcp_tool_discovery
    calls = []

    def register(configs):
        calls.append(configs)
        return []
    monkeypatch.setattr(mcp_tool_discovery, 'register_mcp_servers', register)
    return calls


@pytest.fixture
def no_teardown(monkeypatch):
    """Keep ``_borrowed``'s exit from stopping the shared MCP loop of unrelated tests."""
    from gateway import session_local_mcp
    monkeypatch.setattr(session_local_mcp, '_disconnect', lambda held, scope: None)


def _authority(**attributes):
    return SimpleNamespace(sessions={}, **attributes)


def test_borrowed_reuses_a_recycled_editor_server(monkeypatch, frozen_registry, discovery_spy, no_teardown):
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    scope, names = 'editor-session:recycled', {0: 'editor_a_0'}
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_a_0', scope),
                        _fake_server(session=None, recycled=True, name='editor_a_0'))
    servers = [{'name': 'a'}]

    with session_local_mcp._borrowed(_authority(), scope, servers, names, '/workspace') as (schemas, connected):
        assert connected == [0]
        # The frozen manifest still matches: the recycled server kept its registry tools.
        assert schemas == frozen_registry
    assert discovery_spy == [], 'a recycled server must not be re-registered'


def test_borrowed_still_discovers_a_genuinely_missing_server(
        monkeypatch, frozen_registry, discovery_spy, no_teardown):
    """The recycled server alone is not a reason to connect anything; a missing one still is."""
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    scope, names = 'editor-session:mixed', {0: 'editor_a_0', 1: 'editor_b_0'}
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_a_0', scope),
                        _fake_server(session=None, recycled=True, name='editor_a_0'))
    servers = [{'name': 'a', 'command': 'a-mcp', 'args': []},
               {'name': 'b', 'command': 'b-mcp', 'args': []}]

    with pytest.raises(Exception, match='acp_mcp_discovery_failed'):
        with session_local_mcp._borrowed(_authority(), scope, servers, names, '/workspace'):
            pass
    assert len(discovery_spy) == 1
    assert set(discovery_spy[0]) == {'editor_a_0', 'editor_b_0'}
    # The recycled server's config keeps its idle limit, so a re-discovery cannot pin it resident.
    assert discovery_spy[0]['editor_a_0']['idle_timeout_seconds'] == 1200


def test_editor_mcp_scope_accepts_a_recycled_server(
        monkeypatch, frozen_registry, discovery_spy, no_teardown):
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    scope, name = 'editor-session:scope-recycled', 'editor_a_0'
    monkeypatch.setitem(mcp_tool._servers, _server_key(name, scope),
                        _fake_server(session=None, recycled=True, name=name))
    data = {'scope': scope, 'fingerprint': 'frozen', 'schemas': frozen_registry, 'names': {'0': name}}
    policy = SimpleNamespace(editor_mcp_json=json.dumps(data), cwd='/workspace')
    servers = [{'name': 'a'}]
    authority = _authority(_local_editor_mcp={scope: servers})

    with session_local_mcp.editor_mcp_scope(authority, policy) as entered:
        assert entered is None  # the execution runs; no acp_mcp_discovery_failed / schema_changed
    assert discovery_spy == []


def test_select_new_servers_does_not_wake_a_recycled_server(monkeypatch):
    """Discovery must not respawn the child a recycle just released: the next call wakes it."""
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery
    from tools.mcp_tool_scope import _server_key
    scope = 'editor-session:discovery'
    recycled = _fake_server(session=None, recycled=True, name='editor_idle')
    parked = _fake_server(session=None, recycled=False, name='editor_parked')
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_idle', scope), recycled)
    monkeypatch.setitem(mcp_tool._servers, _server_key('editor_parked', scope), parked)
    woken = []
    monkeypatch.setattr(discovery._loop, '_signal_reconnect', woken.append)

    from tools.registry import session_tool_scope
    with session_tool_scope(scope):
        new_servers = discovery._select_new_servers({
            'editor_idle': {'command': 'npx'},
            'editor_parked': {'command': 'npx'},
        })

    assert new_servers == {}
    assert [server.name for server in woken] == ['editor_parked']


# ---------------------------------------------------------------------------
# Real recycle + lazy reconnect
# ---------------------------------------------------------------------------

STDIO_SERVER_SOURCE = '''
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

TOOLS = [types.Tool(name="ping_editor", description="reply pong",
                    inputSchema={"type": "object", "properties": {}})]


async def list_tools(ctx, params):
    return types.ListToolsResult(tools=TOOLS)


async def call_tool(ctx, params):
    return types.CallToolResult(content=[types.TextContent(type="text", text="pong:" + params.name)])


async def main():
    server = Server("idle-editor", on_list_tools=list_tools, on_call_tool=call_tool)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''


@pytest.fixture
def mcp_loop():
    from tools import mcp_tool_loop
    mcp_tool_loop._ensure_mcp_loop()
    yield
    mcp_tool_loop._stop_mcp_loop()


@pytest.fixture
def editor_stdio_spec(tmp_path):
    pytest.importorskip('mcp')
    script = tmp_path / 'idle_editor_server.py'
    script.write_text(STDIO_SERVER_SOURCE)
    return {'name': 'probe', 'command': sys.executable, 'args': [str(script)], 'env': []}


def _registry_schemas(name):
    from tools.registry import registry
    tools = [tool for tool in registry.get_all_tool_names()
             if registry.get_toolset_for_tool(tool) == f'mcp-{name}']
    assert tools, 'the editor server registered no tools'
    return {tool: registry.get_schema(tool) for tool in sorted(tools)}


def test_recycled_editor_stdio_server_keeps_its_tools_and_reconnects_on_the_next_call(
        monkeypatch, mcp_loop, editor_stdio_spec, tmp_path):
    """End to end: an editor stdio server configured through ``_configs`` recycles once idle, stays
    available (frozen manifest intact, no discovery), and a tool call inside ``editor_mcp_scope``
    transparently respawns the child and succeeds."""
    from gateway import session_local_mcp
    from tools import mcp_tool, mcp_tool_discovery
    from tools.mcp_tool_handlers import _make_check_fn, _make_tool_handler
    from tools.registry import session_tool_scope
    from tools.mcp_tool_scope import _server_key

    monkeypatch.setenv('HERMES_EDITOR_MCP_IDLE_TIMEOUT_SECONDS', '0.5')
    scope = 'editor-session:real-recycle'
    servers = [editor_stdio_spec]
    names = session_local_mcp._server_names(scope, servers)
    name = names[0]
    configs = session_local_mcp._configs(servers, names, str(tmp_path))
    assert configs[name]['idle_timeout_seconds'] == 0.5

    registrations = []
    real_register = mcp_tool_discovery.register_mcp_servers

    def counting_register(config):
        registrations.append(config)
        return real_register(config)
    monkeypatch.setattr(mcp_tool_discovery, 'register_mcp_servers', counting_register)

    try:
        with session_tool_scope(scope):
            real_register(configs)
            assert session_local_mcp._connected(scope, names) == [0]
            frozen = _registry_schemas(name)
            # Editor stdio servers are not in mcp_servers config: pin the trust tier the harness uses.
            monkeypatch.setitem(mcp_tool._server_trust_levels, _server_key(name, scope, current=False), 'full')

        task = mcp_tool._servers[_server_key(name, scope)]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not task._is_recycled_stdio():
            time.sleep(0.1)
        assert task._is_recycled_stdio(), 'the idle child was never recycled'
        assert task._recycled_reason == 'idle_timeout_seconds'
        assert task.session is None, 'a recycled server keeps no live session'

        data = {'scope': scope, 'fingerprint': session_local_mcp.fingerprint(servers),
                'schemas': frozen, 'names': {str(i): n for i, n in names.items()}}
        policy = SimpleNamespace(editor_mcp_json=json.dumps(data), cwd=str(tmp_path))
        authority = SimpleNamespace(sessions={}, _local_editor_mcp={scope: servers})

        with session_tool_scope(scope):
            # The frozen manifest still matches and nothing is re-registered: the turn would run.
            with session_local_mcp.editor_mcp_scope(authority, policy):
                assert session_local_mcp._connected(scope, names) == [0]
                assert _make_check_fn(name)() is True
                assert session_local_mcp._live_tools_changed(scope, names) is False
                assert mcp_tool._servers[_server_key(name, scope)].session is None
                handler = _make_tool_handler(name, 'ping_editor', 30)
                result = json.loads(handler({}))

                assert result.get('result') == 'pong:ping_editor', result
                assert registrations == [], 'the recycled server was re-registered instead of reused'
                # The lazy reconnect brought the same task back with a live session.
                revived = mcp_tool._servers[_server_key(name, scope)]
                assert revived is task
                assert revived.session is not None and not revived._is_recycled_stdio()
                assert session_local_mcp._connected(scope, names) == [0]
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers(scope=scope)
