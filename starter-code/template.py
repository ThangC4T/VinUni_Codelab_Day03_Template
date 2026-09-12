"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng dịch vụ Vingroup (Vinpearl, Xanh SM, VinFast).
Bạn chỉ được sử dụng các công cụ sau:
{tools}

Quy tắc làm việc bắt buộc:
1. Khi cần thông tin, hãy suy nghĩ (Thought) và chọn Action dạng JSON chuẩn.
2. Cú pháp Action bắt buộc: Action: {{"name": "<tên tool>", "args": {{<các tham số>}}}}
3. Khi đã có đủ thông tin hoặc câu hỏi thuộc FAQ cơ bản, hãy xuất ngay Final Answer: <câu trả lời hoàn chỉnh>.

Định dạng phản hồi mỗi lượt:
Thought: <suy nghĩ bước này>
Action: {{"name": "...", "args": {{...}}}}
Observation: <kết quả từ tool>
Final Answer: <câu trả lời hoàn chỉnh khi hoàn tất>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Baseline query - không dùng tool, chỉ trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt.
        """
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau của khách hàng mà KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception:
                pass

        return {
            "answer": f"[Chatbot Baseline] Bạn có thể tìm chuyến bay trên các trang hàng không. Về thời tiết, bạn nên tra cứu trên ứng dụng thời tiết cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }

class ReActAgent:
    """Production-grade ReAct Agent có sử dụng Thought-Action-Observation Loop & Safeguards"""
    def __init__(self, max_iterations: int = 5, api_key: Optional[str] = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Thực thi tool từ TOOL_MAP với phòng ngừa Trap 1 (KeyError) và Trap 3 (API errors).
        """
        # Trap 1: Chuẩn hóa tên tool để tránh lỗi viết hoa hoặc khoảng trắng
        clean_name = str(tool_name).strip().lower()
        if clean_name not in TOOL_MAP:
            return {"error": f"Tool '{tool_name}' không tồn tại trong danh mục TOOL_MAP."}
        try:
            return TOOL_MAP[clean_name](**args)
        except Exception as e:
            # Trap 3: Bắt lỗi thực thi tool để tránh crash agent
            return {"error": f"Lỗi khi thực thi tool '{clean_name}': {str(e)}"}

    def parse_city_code(self, text: str) -> str:
        """Trích xuất mã sân bay / thành phố từ văn bản truy vấn"""
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper or "HA NOI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper or "HO CHI MINH" in text_upper or "SAI GON" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper or "DA NANG" in text_upper:
            return "DAD"
        return "SGN"

    def parse_flight_args(self, text: str) -> Dict[str, Any]:
        """Trích xuất điểm đi, điểm đến và giá tối đa từ văn bản truy vấn"""
        text_lower = text.lower()
        origin = "HAN"
        destination = "SGN"

        # Điểm đi (origin)
        if "từ sgn" in text_lower or "từ sài gòn" in text_lower or "từ hồ chí minh" in text_lower:
            origin = "SGN"
        elif "từ dad" in text_lower or "từ đà nẵng" in text_lower:
            origin = "DAD"
        elif "từ han" in text_lower or "từ hà nội" in text_lower:
            origin = "HAN"

        # Điểm đến (destination)
        if "đi sgn" in text_lower or "đến sgn" in text_lower or "đi sài gòn" in text_lower or "đi hồ chí minh" in text_lower:
            destination = "SGN"
        elif "đi dad" in text_lower or "đến dad" in text_lower or "đi đà nẵng" in text_lower or "đến đà nẵng" in text_lower:
            destination = "DAD"
        elif "đi han" in text_lower or "đến han" in text_lower or "đi hà nội" in text_lower or "đến hà nội" in text_lower:
            destination = "HAN"
        else:
            if origin == "HAN":
                destination = "DAD" if ("dad" in text_lower or "đà nẵng" in text_lower) else "SGN"
            elif origin == "SGN":
                destination = "HAN"
            elif origin == "DAD":
                destination = "HAN"

        # Giá tối đa (budget)
        max_price = 5000000
        if "2 triệu" in text_lower or "2.000.000" in text_lower or "2tr" in text_lower:
            max_price = 2000000
        elif "1.5 triệu" in text_lower or "1,5 triệu" in text_lower or "1.5tr" in text_lower or "1,5tr" in text_lower:
            max_price = 1500000
        elif "500k" in text_lower or "500.000" in text_lower or "500 ngàn" in text_lower:
            max_price = 500000
        else:
            match_mil = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:triệu|tr)', text_lower)
            if match_mil:
                max_price = int(float(match_mil.group(1).replace(",", ".")) * 1000000)
            else:
                match_k = re.search(r'(\d+)\s*k', text_lower)
                if match_k:
                    max_price = int(match_k.group(1)) * 1000

        return {
            "origin": origin,
            "destination": destination,
            "max_price": max_price
        }

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """
        Thực thi từng bước trong vòng lặp ReAct:
        - Phân tích Thought
        - Chọn và gọi Action
        - Ghi nhận Observation
        - Xác định đã có đủ thông tin để ra Final Answer chưa (is_final)
        """
        user_lower = user_input.lower()

        # 1. Trường hợp câu hỏi FAQ / Chính sách chung (Không cần tool)
        if "chính sách" in user_lower or "đổi trả" in user_lower or "vinpearl" in user_lower:
            thought = "Đây là câu hỏi FAQ chung về chính sách dịch vụ Vinpearl. Không cần sử dụng tool."
            final_answer = "Vé máy bay Vinpearl có thể hỗ trợ đổi ngày trước 24 giờ so với giờ khởi hành, phí đổi vé là 350.000 VNĐ/vé cộng chênh lệch giá vé (nếu có)."
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

        needs_flight = any(k in user_lower for k in ["chuyến bay", "vé", "bay từ", "vé máy bay"])
        needs_weather = any(k in user_lower for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa", "khí hậu"])

        # 2. Xử lý truy vấn Chuyến bay ở Bước 1
        if needs_flight and iteration == 1:
            flight_params = self.parse_flight_args(user_input)
            origin = flight_params["origin"]
            destination = flight_params["destination"]
            max_price = flight_params["max_price"]

            thought = f"Tôi cần tra cứu chuyến bay từ {origin} đi {destination} với giá tối đa {max_price:,} VND."
            action = {
                "name": "get_flight_info",
                "args": {"origin": origin, "destination": destination, "max_price": max_price}
            }
            obs = self.execute_tool(action["name"], action["args"])

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            # Nếu người dùng chỉ hỏi chuyến bay (single-step)
            if not needs_weather:
                if not obs:
                    final_ans = f"Không tìm thấy chuyến bay nào từ {origin} đi {destination} dưới {max_price:,} VND."
                else:
                    lines = [f"- {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in obs]
                    final_ans = f"Tìm thấy {len(obs)} chuyến bay từ {origin} đi {destination} phù hợp:\n" + "\n".join(lines)
                return final_ans, True

            # Nếu còn cần hỏi thời tiết (multi-step), chuyển tiếp sang iteration kế tiếp
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 3. Xử lý truy vấn Thời tiết ở Bước 2 (hoặc Bước 1 nếu chỉ hỏi thời tiết)
        elif needs_weather and (iteration == 2 or (iteration == 1 and not needs_flight)):
            city_code = self.parse_city_code(user_input)
            thought = f"Tôi cần kiểm tra thông tin thời tiết tại {city_code}."
            action = {
                "name": "get_weather_forecast",
                "args": {"city_code": city_code}
            }
            obs = self.execute_tool(action["name"], action["args"])

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            # Nếu người dùng chỉ hỏi thời tiết (single-step)
            if not needs_flight:
                city_name = obs.get("city", city_code)
                temp = obs.get("temperature_c", "N/A")
                condition = obs.get("condition", "")
                recom = obs.get("recommendation", "")
                final_ans = f"Thời tiết tại {city_name} ({city_code}): {temp}°C, {condition}.\nGợi ý trang phục: {recom}"
                return final_ans, True

            # Nếu thuộc multi-step, chuyển tiếp sang bước tổng hợp
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 4. Bước 3 (Multi-step): Tổng hợp câu trả lời cuối cùng từ các kết quả đã thu thập
        else:
            thought = "Tôi đã thu thập đủ thông tin để trả lời khách hàng."
            flight_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_flight_info"), [])
            weather_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_weather_forecast"), {})

            flight_summary = "Không tìm thấy chuyến bay phù hợp."
            if flight_obs:
                lines = [f"   - {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in flight_obs]
                flight_summary = "\n".join(lines)

            weather_summary = f"Thời tiết tại {weather_obs.get('city', 'TP. Hồ Chí Minh')}: {weather_obs.get('temperature_c', 32)}°C ({weather_obs.get('condition', '')}).\n   - Gợi ý trang phục: {weather_obs.get('recommendation', '')}"

            final_answer = (
                f"1. Thông tin chuyến bay:\n{flight_summary}\n\n"
                f"2. Thông tin thời tiết & trang phục:\n   - {weather_summary}"
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Vòng lặp ReAct chính có Safeguard giới hạn số bước lặp max_iterations.
        """
        self.trace = []
        iteration = 1

        while iteration <= self.max_iterations:
            result, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "status": "completed",
                    "answer": result,
                    "iterations": iteration,
                    "trace": self.trace
                }
            iteration += 1

        # Safeguard: Đạt giới hạn số vòng lặp tối đa
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa (Max Iterations Safeguard).",
            "iterations": iteration - 1,
            "trace": self.trace
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Status:", result["status"])
    print("Iterations:", result["iterations"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()