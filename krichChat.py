import os
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY ,TAVILY_API_KEY,LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"]=TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError

from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
import json
import myPackage.GetRaw 
import myPackage.PartDataFrame
import myPackage.APIToDF

class EnquiryAnalysis(BaseModel):
    """Combined extracted information and completeness verification."""
    enquiry_type: str = Field(description="query or other")
    part_list: str = Field(description="Extracted auto parts from message")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_analyze = llm.with_structured_output(EnquiryAnalysis)

analyzer_system = """You're an expert in the auto parts business with extensive knowledge of car brands and models in Thailand. Your task is to analyze the user's enquiry about auto parts and provide specific information. 

First, determine if the message is a general greeting or an actual parts inquiry:

If it's a general greeting (e.g., สวัสดี, hello, hi), respond with:
"Greeting: [original greeting]" and set Enquiry_type='other'\n\n
Customer: {enquiry}"""

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", analyzer_system),
    ("human", """จาก Customer ด้านบน ให้ส่งผลลัพธ์ออกมาในรูปแบบ Enquiry_type และ Part_list ถ้ามีการถามถึงอะไหล่ ให้ส่งค่า Enquiry_type='query' แต่ถ้าไม่ได้ถามถึงอะไหล่
     ให้ส่ง Enquiry_type='other' """)
])

Analyze_data = analyze_prompt | llm_analyze

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List,Dict,Optional
from langchain_core.messages import HumanMessage,AIMessage

class AgentState(TypedDict):
    messages: List[HumanMessage]
    evaluator_result: Optional[Dict[str, str]]
    waiting_for_human_input: Optional[bool]
    user_id: str

def evaluator(state: AgentState):
    messages = state["messages"]
    # Initialize current_result as an empty dictionary if it doesn't exist
    current_result = state["evaluator_result"] or {}
    
    Analyze_data = analyze_prompt | llm_analyze
    enquiry_text = " ".join([message.content for message in messages if isinstance(message, HumanMessage)])
    result = Analyze_data.invoke({"enquiry": enquiry_text})

    new_result = {
            "enquiry_type": getattr(result, 'enquiry_type', 'Unknown') or current_result.get("enquiry_type", "Unknown"),
            "part_list": getattr(result, 'part_list', 'Unknown') or current_result.get("part_list", "Unknown")            
        }
    state["evaluator_result"] = new_result
    print(result)
    return state

def CanQuery_Router(state: AgentState):
    evaluator = state["evaluator_result"]
    Enquiry_type = evaluator["enquiry_type"]
    if Enquiry_type == 'query' :
        return('Query')
    else:
        return('Other')

def ResponseWaitMessage(state: AgentState):
    #evaluator = state["evaluator_result"]
    messages = state["messages"]
    messages = messages[-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณเป็น AI สาว ไม่ต้องถามกลับลูกค้า แค่บอกลูกค้าว่าให้รอสักครู่"),
        ("human", "{messages}")
    ])
    llm = ChatOpenAI(temperature=0.1)
    response = llm.invoke(prompt.format(messages=str(messages)))
    response_text = response.content.split(':')[1].strip() if ':' in response.content else response.content         
    state["messages"].append(AIMessage(content=f"{response_text}"))
    return state

def ResponseWaitMessageShort(state: AgentState):
    evaluator = state["evaluator_result"]
    partSearch = evaluator['part_list']
    user_id = state.get("user_id", None)
    if user_id:
        waitmessage = f"กรุณารอสักครู่ค่ะ\nกำลังค้นหา​ '{partSearch}'"
        state["messages"].append(AIMessage(content=f"{waitmessage}"))
        line_bot_api.push_message(user_id, TextSendMessage(text=waitmessage))
    return state

def QueryFromTable(state: AgentState):
    evaluator = state["evaluator_result"]
    result = PartDataFrame.QueryDF(evaluator)
    state["messages"].append(AIMessage(content=f"{result}"))    
    return state

def QueryFromAPI(state: AgentState):
    evaluator = state["evaluator_result"]

    parameter = evaluator['part_list']
    result = APIToDF.GetQuery(parameter)    
    state["messages"].append(AIMessage(content=f"{result}"))    
    return state

def greeting(state: AgentState):    
    state["messages"] = [] 
    state["messages"].append(AIMessage(content="สวัสดีค่ะ ถ้าต้องการค้นหาอะไหล่ สามารถใส่ชื่ออะไหล่มาได้เลยค่ะ"))
    return state

workflow = StateGraph(AgentState)

workflow.add_node("FirstNode", evaluator)
workflow.add_node("WaitMessage", ResponseWaitMessageShort)
workflow.add_node("queryProcess",QueryFromAPI)
workflow.add_node("Greeting",greeting)
workflow.set_entry_point("FirstNode")
workflow.add_conditional_edges(
    "FirstNode",
    CanQuery_Router,
    {"Query":"WaitMessage",
    "Other": "Greeting"})

workflow.add_edge("WaitMessage","queryProcess")
workflow.add_edge("queryProcess",END)
workflow.add_edge("Greeting",END)

app = workflow.compile()
line_bot_api = LineBotApi('7GCEmFcjHYe0893CchShMcV/yh/b1ZFZgsn20/H+BB5mdnXpyBBNN6hCeTSqXcomnhbsHl22vJQKLPwbd+kxlCxwDCVb+xuZiePWuYOEzGidE9n/EYUXdnsyePoLJ9OOg1BKj++PpMCL0J8gOQ8fZwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('52841806b289e22472e45a372b82d04a')
# Flask app initialization
flask_app = Flask(__name__)
# Store the state for each user
user_states = {}

chat_histories = {}  # Global variable

class Chatbot:
    def __init__(self, user_id=None):
        self.user_id = user_id
        if user_id:
            if user_id not in chat_histories:
                chat_histories[user_id] = {"messages": [], "evaluator_result": None, "user_id": user_id}
            self.state = chat_histories[user_id]
        else:
            self.state = {"messages": [], "evaluator_result": None, "user_id": None}
        self.app = app

    def reset(self):
        if self.user_id:
            chat_histories[self.user_id] = {"messages": [], "evaluator_result": None, "user_id": self.user_id}
        self.state = {"messages": [], "evaluator_result": None, "user_id": self.user_id}

    def invoke_app(self):
        next_state = self.app.invoke(self.state)
        if self.user_id:
            chat_histories[self.user_id] = next_state  # Update global state
        self.state = next_state

    def process_input(self, user_input: str):
        if self.user_id:
            self.state["messages"].append(HumanMessage(content=user_input))
            self.invoke_app()


#chatbot = Chatbot()

@flask_app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    user_message = event.message.text
    
    # Create Chatbot instance with user_id
    chatbot = Chatbot(user_id)
    
    if user_message.lower() == 'clear':
        chatbot.reset()
        response_text = "Chat history has been cleared."
    else:
        chatbot.process_input(user_message)
        latest_messages = chatbot.state["messages"]
        response_text = ""
        for message in latest_messages:
            if isinstance(message, AIMessage):
                response_text = message.content 
        
        # ตรวจสอบผลลัพธ์ล่าสุดใน state และตอบกลับผู้ใช้
        evaluator_result = chatbot.state["evaluator_result"]
        if evaluator_result:
            response_eval = (
                f"Part Name: {evaluator_result.get('part_name', 'Unknown')}\n"
                f"Car Brand: {evaluator_result.get('car_brand', 'Unknown')}\n"
                f"Car Model: {evaluator_result.get('car_model', 'Unknown')}\n"
                f"VIN Code: {evaluator_result.get('vin_code', 'Unknown')}\n"
                f"Year: {evaluator_result.get('year', 'Unknown')}\n"
                f"part_id: {evaluator_result.get('part_id', 'Unknown')}"
            )
        else:
            response_eval = "Sorry, I couldn't process your enquiry."
        
        #response_text = response_eval+'\n\n'+response_text
    
    if response_text != '':
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_text))
        print('\n\n'+response_text+'\n\n')


@flask_app.route("/", methods=['GET'])
def home():
    return "Hello, this is the Flask app root."


if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=int(os.environ.get('PORT','5001'))) #(os.environ.get('PORT,'5000')))