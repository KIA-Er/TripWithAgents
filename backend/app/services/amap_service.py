"""高德地图MCP服务封装"""

from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo
import asyncio

# 全局MCP工具实例
_amap_mcp_tool = None


async def get_amap_mcp_tool() -> list[BaseTool]:
    """
    使用 LangChain MCPClient 获取高德地图 MCP 工具列表
    异步获取高德地图MCP工具实例(单例模式)
    
    Returns:
        BaseTool列表
    """
    global _amap_mcp_tool
    
    if _amap_mcp_tool is not None:
        return _amap_mcp_tool
    else:
        settings = get_settings()
        
        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
        
        # 创建MCP工具
        _amap_mcp_client = MultiServerMCPClient(
            {
                "amap":{
                    "command": "uvx",
                    "args": ["amap-mcp-server"],
                    "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
                    "transport": "stdio",
                }
            }
        )
        
        _amap_mcp_tool = await _amap_mcp_client.get_tools()
        print(f"✅ 高德地图MCP工具初始化成功")
        print(f"   工具数量: {len(_amap_mcp_tool)}")
        
        # 打印可用工具列表
        if len(_amap_mcp_tool):
            print("   可用工具:")
            for tool in _amap_mcp_tool[:5]:  # 只打印前5个
                print(f"     - {tool.name}")
            if len(_amap_mcp_tool) > 5:
                print(f"     ... 还有 {len(_amap_mcp_tool) - 5} 个工具")
    
    return _amap_mcp_tool


class AmapService:
    """高德地图服务封装类"""
    
    def __init__(self):
        """初始化服务"""
        self.mcp_tool = None
    
    async def init(self):
        """异步初始化MCP工具"""
        self.mcp_tool = await get_amap_mcp_tool()

    
    async def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI
        
        Args:
            keywords: 搜索关键词
            city: 城市
            citylimit: 是否限制在城市范围内
            
        Returns:
            POI信息列表
        """
        try:
            # 调用MCP工具
            amap = AmapService()
            await amap.init()
            search_tool = next(tool for tool in self.mcp_tool if tool.name == "maps_text_search")
            
            result = await search_tool.ainvoke({
                "keywords": keywords,
                    "city": city,
                    "citylimit": str(citylimit).lower()
            })
            
            # 解析结果
            # 注意: MCP工具返回的是字符串,需要解析
            # 这里简化处理,实际应该解析JSON
            print(f"POI搜索结果: {result[:200]}...")  # 打印前200字符
            
            # TODO: 解析实际的POI数据
            return []
            
        except Exception as e:
            print(f"❌ POI搜索失败: {str(e)}")
            return []
    
    async def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息列表
        """
        try:
            # 调用MCP工具
            amap = AmapService()
            await amap.init()
            weather_tool = next(tool for tool in self.mcp_tool if tool.name == "maps_weather")

            result = await weather_tool.ainvoke({
                "city": city
            })
            
            print(f"天气查询结果: {result[:200]}...")
            
            # TODO: 解析实际的天气数据
            return []
            
        except Exception as e:
            print(f"❌ 天气查询失败: {str(e)}")
            return []
    
    async def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """
        规划路线
        
        Args:
            origin_address: 起点地址
            destination_address: 终点地址
            origin_city: 起点城市
            destination_city: 终点城市
            route_type: 路线类型 (walking/driving/transit)
            
        Returns:
            路线信息
        """
        try:
            amap = AmapService()
            await amap.init()
            # 根据路线类型选择工具
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }
            
            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")
            
            # 构建参数
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }
            
            # 公共交通需要城市参数
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                # 其他路线类型也可以提供城市参数提高准确性
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            
            # 调用MCP工具
            route_tool = next(tool for tool in self.mcp_tool if tool.name == tool_name)
            result = await route_tool.ainvoke(arguments)
            
            print(f"路线规划结果: {result[:200]}...")
            
            # TODO: 解析实际的路线数据
            return {}
            
        except Exception as e:
            print(f"❌ 路线规划失败: {str(e)}")
            return {}
    
    async def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码(地址转坐标)

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            amap = AmapService()
            await amap.init()

            arguments = {"address": address}
            if city:
                arguments["city"] = city

            geocode_tool = next(tool for tool in self.mcp_tool if tool.name == "maps_geo")
            result = await geocode_tool.ainvoke(arguments)

            print(f"地理编码结果: {result[:200]}...")

            # TODO: 解析实际的坐标数据
            return None

        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None

    async def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            amap = AmapService()
            await amap.init()
            detail_tool = next(tool for tool in self.mcp_tool if tool.name == "maps_search_detail")
            result = await detail_tool.ainvoke({
                "id": poi_id
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


# 创建全局服务实例
_amap_service = None


async def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    
    if _amap_service is None:
        _amap_service = AmapService()
        await _amap_service.init()
    
    return _amap_service

