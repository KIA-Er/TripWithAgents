from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import get_settings
import asyncio
from langchain.agents import initialize_agent
from app.services.unsplash_service import get_unsplash_service

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
    
    # print(f"✅ 高德地图MCP工具初始化成功")
    # print(f"   工具数量: {len(_tools_cache)}")
    # for tool_name in _tools_cache[:5]:
    #     print(f"     - {tool_name.name}")
    #     print(f"     - {tool_name.description}")
    # if len(_tools_cache) > 5:
    #     print(f"     ... 还有 {len(_tools_cache) - 5} 个工具")
    
    return _tools_cache

async def test_mcp_search():
    tools = await get_amap_mcp_tools()
    search_tool = next(tool for tool in tools if tool.name == "maps_text_search")

    result = await search_tool.ainvoke({
        "keywords": "景点",
        "city": "北京",
        "citylimit": "true"
    })

    print(f"POI搜索结果: {str(result)}...")

async def get_poi_detail(poi_id: str):
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            tools = await get_amap_mcp_tools()
            detail_tool = next(tool for tool in tools if tool.name == "maps_search_detail")
            result = await detail_tool.ainvoke({
                "id": 213123
            })

            print(f"POI详情结果: {result[:200]}...")

            # 解析结果并提取图片
            import json
            import re

            # 尝试从结果中提取JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data

            return {"raw": result}

        except Exception as e:
            print(f"❌ 获取POI详情失败: {str(e)}")
            return {}
async def geocode(self, address: str, city: str = None):
        """
        地理编码(地址转坐标)

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            tools = await get_amap_mcp_tools()
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            geocode_tool = next(tool for tool in tools if tool.name == "maps_geo")
            result = await geocode_tool.ainvoke(arguments)

            print(f'''
            name: {geocode_tool.name}
            description: {geocode_tool.description}
            arguments: {geocode_tool.args}''')
            print(f"地理编码结果: {result[:200]}...")

            import json
            data = json.loads(result)
            print(data["return"][0])#将MCP返回的Json转换为python对象

            # TODO: 解析实际的坐标数据
            return None

        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None
# 在异步上下文中运行
asyncio.run(geocode("龙城广场", "深圳"))
# uns = get_unsplash_service()
# result = uns.search_photos("深圳夜景")
# print(result)