"""Starts mcp_server.py and calls its tools, like an AI app would.

Run:  python test_mcp_client.py
Needs:  pip install "mcp[cli]"
The first three checks need no API keys. The last two use your RAG setup.
"""
import asyncio
import sys
from pathlib import Path
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).resolve().parent / "mcp_server.py")


async def call(session, name, args):
    print(f"\n> {name}({args})  (the first RAG call can take up to a minute)")
    try:
        result = await asyncio.wait_for(session.call_tool(name, args), timeout=120)
        print(result.content[0].text if result.content else "(no content)")
    except asyncio.TimeoutError:
        print("TIMED OUT after 120 s")


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER],
        env=dict(os.environ),
        cwd=str(Path(SERVER).parent),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools found:", [t.name for t in tools.tools])

            await call(session, "get_leave_policy", {"leave_type": "annual"})
            await call(session, "get_leave_policy", {"leave_type": "comp off"})
            await call(session, "get_leave_policy", {"leave_type": "maternity"})

            # These two need OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
            await call(session, "search_hr_policy",
                       {"query": "How many days of earned leave can I carry forward?"})
            await call(session, "ask_hr_policy",
                       {"question": "How many days of earned leave do I get per year?"})


if __name__ == "__main__":
    asyncio.run(main())
