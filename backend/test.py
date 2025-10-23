from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import get_settings
import asyncio

# 单例客户端
_amap_mcp_client = None
_tools_cache = None


async def get_amap_mcp_tools():
    """
    使用 LangChain MCPClient 获取高德地图 MCP 工具列表
    """
    global _amap_mcp_client, _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    settings = get_settings()
    if not settings.amap_api_key:
        raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")

    if _amap_mcp_client is None:
        # 创建 MCP 客户端
        _amap_mcp_client = MultiServerMCPClient({
            "amap": {
                "command": "uvx",
                "args": ["amap-mcp-server"],
                "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
                "transport": "stdio",
            }
        })
    
    # 获取工具列表
    _tools_cache = await _amap_mcp_client.get_tools()
    
    print(f"✅ 高德地图MCP工具初始化成功")
    print(f"   工具数量: {len(_tools_cache)}")
    for tool_name in list(_tools_cache.keys())[:5]:
        print(f"     - {tool_name}")
    if len(_tools_cache) > 5:
        print(f"     ... 还有 {len(_tools_cache) - 5} 个工具")
    
    return _tools_cache

asyncio.run(get_amap_mcp_tools())