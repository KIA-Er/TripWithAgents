"""多智能体旅行规划系统"""

from .prompt import ATTRACTION_AGENT_PROMPT,PLANNER_AGENT_PROMPT,HOTEL_AGENT_PROMPT,WEATHER_AGENT_PROMPT
import json
from typing import Dict, Any, List, TypedDict
from ..services.llm_service import get_llm
from ..services.amap_service import get_amap_service
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class MultiAgentTripPlanner:
    """基于LangGraph的多智能体旅行规划系统"""

    # 定义状态结构：保存对话和任务上下文
    class TripPlannerState(TypedDict):
        request: TripRequest
        attractions: str
        weather: str
        hotels: str
        plan: str
        messages: List[BaseMessage]

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统...")

        try:
            self.settings = None
            self.llm = None
            self.attraction_agent =None
            self.weather_agent =None
            self.hotel_agent =None
            self.planner_agent =None
        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            raise

    async def init(self):
        """异步初始化方法"""
        try:
            # 创建共享的MCP工具(只创建一次)
            print("  - 创建共享MCP工具...")
            self.amap_service = await get_amap_service()
            self.amap_tool = self.amap_service.mcp_tool
            self.settings = get_settings()
            self.llm = get_llm()


            # 实例化子Agent
            self.attraction_agent = create_react_agent(
                model=self.llm,
                tools=self.amap_tool,
                prompt=ATTRACTION_AGENT_PROMPT,
                name="attraction_expert",
            )
            self.weather_agent = create_react_agent(
                model=self.llm,
                tools=self.amap_tool,
                prompt=WEATHER_AGENT_PROMPT,
                name="weather_expert",
            )
            self.hotel_agent = create_react_agent(
                model=self.llm,
                tools=self.amap_tool,
                prompt=HOTEL_AGENT_PROMPT,
                name="hotel_expert",
            )
            self.planner_agent = create_supervisor(
                agents=[self.attraction_agent,self.weather_agent, self.hotel_agent],
                model=self.llm,
                prompt=PLANNER_AGENT_PROMPT,
                name="planner_supervisor",
                output_mode="last_message"
            )

            self.planner_agent = self.planner_agent.compile()
            print("✅ 多智能体系统初始化完成")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            raise

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")
            
            
            trip_plan = await self._build_planner_query(request)
            final_plan = self._parse_response(trip_plan,request)
            
            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return final_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
    
    def _build_attraction_query(self, request: TripRequest) -> str:
        """构建景点搜索查询 - 直接包含工具调用"""
        keywords = []
        if request.preferences:
            # 只取第一个偏好作为关键词
            keywords = request.preferences[0]
        else:
            keywords = "景点"

        # 直接返回工具调用格式
        query = f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]"
        return query

    async def _build_planner_query(self, request: TripRequest) -> str:
        """
        使用 supervisor 多智能体系统自动生成旅行计划
        """
        if self.planner_agent is None:
                raise RuntimeError("请先运行 await init() 初始化")
        
        attraction_query = self._build_attraction_query(request)

        query = f"""你是一个旅行规划协调专家，你可以指挥三个子智能体：
        - 景点搜索专家：负责根据城市与偏好搜索景点。
        - 天气查询专家：负责查询该城市的天气。
        - 酒店推荐专家：负责推荐合适酒店。

        请协调它们完成以下任务：
        1. {attraction_query}；
        2. 查询当地未来{request.travel_days}天的天气；
        3. 推荐合适的{request.accommodation}酒店；
        4. 综合所有结果，规划出{request.travel_days}天的旅行计划，每天安排2-3个景点、早中晚三餐及推荐酒店；
        5. 输出完整 JSON 格式结果（含景点坐标、时间安排、住宿和交通建议）。

        **基本信息:**
        - 城市: {request.city}
        - 日期: {request.start_date} 至 {request.end_date}
        - 天数: {request.travel_days}天
        - 交通方式: {request.transportation}
        - 住宿: {request.accommodation}
        - 偏好: {', '.join(request.preferences) if request.preferences else '无'}


        **要求:**
        1. 每天安排2-3个景点
        2. 每天必须包含早中晚三餐
        3. 每天推荐一个具体的酒店(从酒店信息中选择)
        3. 考虑景点之间的距离和交通方式
        4. 返回完整的JSON格式数据
        5. 景点的经纬度坐标要真实准确
        """
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        config = {
                "configurable": {
                    "verbose": True,
                    "thread_id": 1,
                }
        }
        print("\n📋 启动多智能体协作生成旅行计划...\n")
        # 🧠 用于存储最终planner输出
        final_output = None
        async for chunk in self.planner_agent.astream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode=["values"],# 
                config=config
        ):
                    if "messages" in chunk[1]:
                        message = chunk[1]["messages"][-1]
                        role = message.__class__.__name__
                        content = message.content
                        print(f"[{role}] {content[:]}...\n")  # 打印前400字符，避免过长
                        final_output = content  # 🔥 保留最后一次AI输出

        return final_output
    
    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取JSON
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # 直接查找JSON对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request)
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta
        
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4 + i*0.01 + j*0.005, latitude=39.9 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


async def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
        await _multi_agent_planner.init()


    return _multi_agent_planner

